// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ModelInfoExtractor.h"
#include "ParseTreeNavigator.h"
#include "Utils.hpp"
#include <iostream>

namespace lacemodelica {

ModelInfo ModelInfoExtractor::extract(basemodelica::BaseModelicaParser::BaseModelicaContext* tree, const std::string& sourceFile) {
    info = ModelInfo();
    this->sourceFile = sourceFile;

    extractPackageAndModelName(tree);
    extractRecordDefinitions(tree);
    extractGlobalConstants(tree);
    extractVariables(tree);
    extractEquations(tree);
    extractFunctions(tree);
    identifyDerivatives();

    return info;
}

void ModelInfoExtractor::extractPackageAndModelName(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx) {
    // Get package name (first IDENT after 'package')
    auto tokens = ctx->children;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (auto terminal = dynamic_cast<antlr4::tree::TerminalNode*>(tokens[i])) {
            if (terminal->getSymbol()->getType() == basemodelica::BaseModelicaParser::IDENT) {
                info.packageName = stripQuotes(terminal->getText());
                break;
            }
        }
    }

    // Get model name from longClassSpecifier
    auto longClass = ctx->longClassSpecifier();
    if (longClass && longClass->IDENT().size() > 0) {
        info.modelName = stripQuotes(longClass->IDENT(0)->getText());
    }

    // Get description from comment
    if (longClass && longClass->stringComment()) {
        info.description = longClass->stringComment()->getText();
        // Remove quotes
        if (info.description.size() >= 2) {
            info.description = info.description.substr(1, info.description.size() - 2);
        }
    }
}

void ModelInfoExtractor::extractRecordDefinitions(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx) {
    // Look for record definitions in classDefinition elements at package level
    for (auto classDef : ctx->classDefinition()) {
        // Check if this is a record definition
        if (classDef->classSpecifier() && classDef->classPrefixes()) {
            std::string classPrefix = classDef->classPrefixes()->getText();
            if (classPrefix.find("record") != std::string::npos) {
                // This is a record definition
                auto classSpec = classDef->classSpecifier();
                std::string recordName;

                // Get record name from longClassSpecifier
                if (classSpec->longClassSpecifier() && classSpec->longClassSpecifier()->IDENT().size() > 0) {
                    recordName = stripQuotes(classSpec->longClassSpecifier()->IDENT(0)->getText());

                    std::vector<Variable> fields;

                    // Extract field variables from composition
                    if (classSpec->longClassSpecifier()->composition()) {
                        auto composition = classSpec->longClassSpecifier()->composition();

                        for (auto genericElem : composition->genericElement()) {
                            if (auto normalElem = genericElem->normalElement()) {
                                if (auto compClause = normalElem->componentClause()) {
                                    std::string fieldType = stripQuotes(compClause->typeSpecifier()->getText());

                                    for (auto compDecl : compClause->componentList()->componentDeclaration()) {
                                        Variable field;
                                        field.name = stripQuotes(compDecl->declaration()->IDENT()->getText());
                                        field.type = fieldType;
                                        fields.push_back(field);
                                    }
                                }
                            }
                        }
                    }

                    if (!fields.empty()) {
                        recordDefinitions[recordName] = fields;
                    }
                }
            }
        }
    }

    // Also extract enum type definitions
    extractEnumDefinitions(ctx);
}

void ModelInfoExtractor::extractEnumDefinitions(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx) {
    // Look for type definitions with enumeration in classDefinition elements
    for (auto classDef : ctx->classDefinition()) {
        if (!classDef->classPrefixes() || !classDef->classSpecifier()) continue;

        std::string classPrefix = classDef->classPrefixes()->getText();
        if (classPrefix != "type") continue;

        // Check for shortClassSpecifier with enumeration
        auto classSpec = classDef->classSpecifier();
        auto shortClassSpec = classSpec->shortClassSpecifier();
        if (!shortClassSpec) continue;

        // Get the type name (IDENT before '=')
        if (!shortClassSpec->IDENT()) continue;
        std::string enumName = stripQuotes(shortClassSpec->IDENT()->getText());

        // Check if this has an enumList (enumeration definition)
        auto enumList = shortClassSpec->enumList();
        if (!enumList) continue;

        EnumType enumType;
        enumType.name = enumName;

        // Extract description from comment
        if (shortClassSpec->comment() && shortClassSpec->comment()->stringComment()) {
            auto stringComment = shortClassSpec->comment()->stringComment();
            if (!stringComment->STRING().empty()) {
                enumType.description = stripQuotes(stringComment->STRING(0)->getText());
            }
        }

        // Extract literals
        int value = 0;
        for (auto enumLiteral : enumList->enumerationLiteral()) {
            EnumLiteral lit;
            lit.name = stripQuotes(enumLiteral->IDENT()->getText());
            lit.value = value++;

            // Extract literal description from comment
            if (enumLiteral->comment() && enumLiteral->comment()->stringComment()) {
                auto stringComment = enumLiteral->comment()->stringComment();
                if (!stringComment->STRING().empty()) {
                    lit.description = stripQuotes(stringComment->STRING(0)->getText());
                }
            }

            enumType.literals.push_back(lit);
        }

        if (!enumType.literals.empty()) {
            info.addEnumType(enumType);
        }
    }
}

void ModelInfoExtractor::extractGlobalConstants(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx) {
    // Extract package-level constants
    for (auto globalConst : ctx->globalConstant()) {
        Variable var;
        var.name = stripQuotes(globalConst->declaration()->IDENT()->getText());
        var.type = stripQuotes(globalConst->typeSpecifier()->getText());
        var.causality = "parameter";
        var.variability = "fixed";
        var.initial = "exact";
        var.valueReference = info.nextValueReference++;
        var.sourceFile = sourceFile;
        var.sourceLine = globalConst->getStart()->getLine();

        // Extract value from modification
        if (auto mod = globalConst->declaration()->modification()) {
            var.startValue = extractStartValue(mod);
            var.bindingContext = extractBindingContext(mod);
        }

        // Extract dimensions
        if (globalConst->arraySubscripts()) {
            // Handle array dimensions if needed
        }

        // Extract description from comment
        if (globalConst->comment()) {
            var.description = extractDescription(globalConst->comment());
        }

        info.addVariable(var);
    }
}

void ModelInfoExtractor::extractVariables(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx) {
    auto longClass = ctx->longClassSpecifier();
    if (!longClass || !longClass->composition()) return;

    auto composition = longClass->composition();

    // Extract from genericElement list
    for (auto genericElem : composition->genericElement()) {
        // Check for normalElement (variables and parameters)
        if (auto normalElem = genericElem->normalElement()) {
            if (auto compClause = normalElem->componentClause()) {
                std::string typePrefix = "";
                if (compClause->typePrefix()) {
                    typePrefix = compClause->typePrefix()->getText();
                }

                std::string typeSpec = compClause->typeSpecifier()->getText();

                // Process each component in the list
                for (auto compDecl : compClause->componentList()->componentDeclaration()) {
                    Variable var;
                    var.name = stripQuotes(compDecl->declaration()->IDENT()->getText());
                    var.type = stripQuotes(typeSpec);
                    var.valueReference = info.nextValueReference++;
                    var.sourceFile = sourceFile;
                    var.sourceLine = compDecl->getStart()->getLine();

                    // Check for annotation(Evaluate=true) for structural parameters
                    if (compDecl->comment() && compDecl->comment()->annotationComment()) {
                        std::string annotation = compDecl->comment()->annotationComment()->getText();
                        var.isStructuralParameter = annotation.find("Evaluate") != std::string::npos &&
                                                   annotation.find("true") != std::string::npos;
                    }

                    // Determine causality and variability from typePrefix
                    if (typePrefix.find("parameter") != std::string::npos) {
                        if (var.isStructuralParameter) {
                            var.causality = "structuralParameter";
                            var.variability = "fixed";
                        } else {
                            var.causality = "parameter";
                            var.variability = "tunable";
                        }
                        var.initial = "exact";
                    } else {
                        var.causality = "local";
                        var.variability = "continuous";
                        var.initial = "exact";
                    }

                    // Extract dimensions
                    var.dimensions = extractDimensions(compDecl->declaration());

                    // Extract start value and binding context from modification
                    if (auto mod = compDecl->declaration()->modification()) {
                        var.startValue = extractStartValue(mod);
                        var.bindingContext = extractBindingContext(mod);
                        var.minValue = extractMinValue(mod);
                        var.minContext = extractMinContext(mod);
                        var.maxValue = extractMaxValue(mod);
                        var.maxContext = extractMaxContext(mod);

                        // If any variable has a non-const start/binding expression,
                        // it should have initial="calculated"
                        if (!var.startValue.empty() &&
                            !isConstExpression(var.startValue)) {
                            var.initial = "calculated";
                        }
                    }

                    // Extract description from comment
                    if (compDecl->comment()) {
                        var.description = extractDescription(compDecl->comment());
                    }

                    // Check if this is a record type and flatten it
                    auto recordIt = recordDefinitions.find(var.type);
                    if (recordIt != recordDefinitions.end()) {
                        // This is a record type - flatten into individual fields
                        const std::vector<Variable>& fields = recordIt->second;

                        // Parse field modifications from the record instance modification
                        std::map<std::string, std::string> fieldStartValues;
                        if (auto mod = compDecl->declaration()->modification()) {
                            // Look for class modifications like p(x(start = 1.0), y(start = 2.0))
                            if (mod->classModification() && mod->classModification()->argumentList()) {
                                for (auto arg : mod->classModification()->argumentList()->argument()) {
                                    if (auto elemModOrRepl = arg->elementModificationOrReplaceable()) {
                                        auto elemMod = elemModOrRepl->elementModification();
                                        if (elemMod) {
                                            if (elemMod->name()) {
                                                std::string fieldName = stripQuotes(elemMod->name()->getText());
                                                if (elemMod->modification()) {
                                                    std::string fieldStart = extractStartValue(elemMod->modification());
                                                    if (!fieldStart.empty()) {
                                                        fieldStartValues[fieldName] = fieldStart;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Create flattened variables for each field
                        for (const Variable& field : fields) {
                            Variable flatVar = var;  // Copy base properties
                            flatVar.name = var.name + "." + field.name;
                            flatVar.type = field.type;
                            flatVar.valueReference = info.nextValueReference++;

                            // Apply field-specific start value if present
                            auto fieldStartIt = fieldStartValues.find(field.name);
                            if (fieldStartIt != fieldStartValues.end()) {
                                flatVar.startValue = fieldStartIt->second;
                            } else {
                                flatVar.startValue = "";
                            }

                            info.addVariable(flatVar);
                        }
                    } else {
                        // Not a record type, add normally
                        info.addVariable(var);
                    }
                }
            }
        }
    }
}

void ModelInfoExtractor::extractEquations(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx) {
    auto longClass = ctx->longClassSpecifier();
    if (!longClass || !longClass->composition()) return;

    auto composition = longClass->composition();

    // Extract regular equations
    for (auto equation : composition->equation()) {
        processEquation(equation, info.equations);
    }

    // Extract initial equations
    for (auto initialEq : composition->initialEquation()) {
        // initialEquation can be either equation or prioritizeEquation
        if (initialEq->equation()) {
            processEquation(initialEq->equation(), info.initialEquations);
        }
    }
}

void ModelInfoExtractor::scanForDerivativeCalls(const std::string& equationText) {
    size_t pos = 0;
    while ((pos = equationText.find("der(", pos)) != std::string::npos) {
        size_t start = pos + 4;
        size_t end = equationText.find(")", start);
        if (end != std::string::npos) {
            std::string varName = equationText.substr(start, end - start);
            derivativeCalls.insert(varName);
        }
        pos = end;
    }
}

void ModelInfoExtractor::processEquation(basemodelica::BaseModelicaParser::EquationContext* equation, std::vector<Equation>& target) {
    // Common setup for all equation types
    Equation eq;
    eq.sourceFile = sourceFile;
    eq.sourceLine = equation->getStart()->getLine();
    if (equation->comment()) {
        eq.comment = extractDescription(equation->comment());
    }

    // Handle if-equations
    if (equation->ifEquation()) {
        eq.ifEquationContext = equation->ifEquation();
        target.push_back(eq);
        scanForDerivativeCalls(equation->getText());
        return;
    }

    // Handle for-equations
    if (equation->forEquation()) {
        eq.forEquationContext = equation->forEquation();
        target.push_back(eq);
        scanForDerivativeCalls(equation->getText());
        return;
    }

    if (equation->whenEquation()) {
        throw std::runtime_error("When-equations are not supported (" + sourceFile + ":" +
                                 std::to_string(equation->getStart()->getLine()) + ")");
    }

    // Handle simple equations (lhs = rhs)
    auto simpleExpr = equation->simpleExpression();
    auto fullExpr = equation->expression();

    if (simpleExpr && fullExpr) {
        eq.lhsContext = simpleExpr;
        eq.rhsContext = fullExpr;
        target.push_back(eq);
        scanForDerivativeCalls(equation->getText());
        return;
    }

    // Handle assert statements: assert(condition, message, level)
    if (simpleExpr && !fullExpr) {
        auto primary = ParseTreeNavigator::findPrimary(simpleExpr);
        if (primary && primary->componentReference() && primary->functionCallArgs()) {
            std::string funcName = stripQuotes(primary->componentReference()->IDENT(0)->getText());
            if (funcName == "assert") {
                Assertion assertion;
                assertion.sourceFile = sourceFile;
                assertion.sourceLine = equation->getStart()->getLine();

                auto funcArgs = primary->functionCallArgs()->functionArguments();
                if (funcArgs) {
                    // First argument: condition
                    if (funcArgs->expression()) {
                        assertion.conditionContext = funcArgs->expression();
                    }

                    // Second and third arguments
                    auto nonFirst = funcArgs->functionArgumentsNonFirst();
                    if (nonFirst && nonFirst->functionArgument()) {
                        // Second argument: message string
                        auto msgExpr = nonFirst->functionArgument()->expression();
                        if (msgExpr) {
                            assertion.message = stripQuotes(msgExpr->getText());
                        }

                        // Third argument: assertion level
                        nonFirst = nonFirst->functionArgumentsNonFirst();
                        if (nonFirst && nonFirst->functionArgument()) {
                            auto levelExpr = nonFirst->functionArgument()->expression();
                            if (levelExpr) {
                                std::string levelText = levelExpr->getText();
                                // Extract level from "AssertionLevel.error" or "AssertionLevel.warning"
                                size_t dotPos = levelText.find('.');
                                if (dotPos != std::string::npos) {
                                    assertion.level = levelText.substr(dotPos + 1);
                                } else {
                                    assertion.level = levelText;
                                }
                            }
                        }
                    }
                }

                info.assertions.push_back(assertion);
                return;
            }
        }
    }

    throw std::runtime_error("Unsupported equation type (" + sourceFile + ":" +
                             std::to_string(equation->getStart()->getLine()) + ")");
}

void ModelInfoExtractor::identifyDerivatives() {
    // For each variable that appears in der(), create a derivative variable
    for (const auto& derVar : derivativeCalls) {
        std::string cleanDerVar = stripQuotes(derVar);
        auto* stateVar = info.findVariable(cleanDerVar);
        if (stateVar) {
            Variable derVariable;
            derVariable.name = "der(" + cleanDerVar + ")";
            derVariable.type = "Real";
            derVariable.causality = "local";
            derVariable.variability = "continuous";
            derVariable.isDerivative = true;
            derVariable.derivativeOf = stateVar->valueReference;
            derVariable.valueReference = info.nextValueReference++;

            info.addVariable(derVariable);
        }
    }
}


// Helper to find named attribute in modification's class modification argument list
antlr4::tree::ParseTree* ModelInfoExtractor::findModificationAttribute(
    basemodelica::BaseModelicaParser::ModificationContext* ctx,
    const std::string& attrName) {

    if (!ctx) return nullptr;

    // Check class modification for named attributes
    if (ctx->classModification()) {
        auto argList = ctx->classModification()->argumentList();
        if (argList) {
            for (auto arg : argList->argument()) {
                if (auto elemMod = arg->elementModificationOrReplaceable()) {
                    if (auto mod = elemMod->elementModification()) {
                        std::string name = mod->name()->getText();
                        if (name == attrName && mod->modification() && mod->modification()->expression()) {
                            return mod->modification()->expression();
                        }
                    }
                }
            }
        }
    }

    return nullptr;
}

std::string ModelInfoExtractor::extractStartValue(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (ctx->expression()) {
        return ctx->expression()->getText();
    }

    if (auto expr = findModificationAttribute(ctx, "start")) {
        return expr->getText();
    }

    return "";
}

std::string ModelInfoExtractor::extractMinValue(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (auto expr = findModificationAttribute(ctx, "min")) {
        return expr->getText();
    }
    return "";
}

std::string ModelInfoExtractor::extractMaxValue(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (auto expr = findModificationAttribute(ctx, "max")) {
        return expr->getText();
    }
    return "";
}

antlr4::ParserRuleContext* ModelInfoExtractor::extractBindingContext(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (ctx->expression()) {
        return ctx->expression();
    }

    return dynamic_cast<antlr4::ParserRuleContext*>(findModificationAttribute(ctx, "start"));
}

antlr4::ParserRuleContext* ModelInfoExtractor::extractMinContext(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    return dynamic_cast<antlr4::ParserRuleContext*>(findModificationAttribute(ctx, "min"));
}

antlr4::ParserRuleContext* ModelInfoExtractor::extractMaxContext(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    return dynamic_cast<antlr4::ParserRuleContext*>(findModificationAttribute(ctx, "max"));
}

bool ModelInfoExtractor::isConstExpression(const std::string& expr) {
    if (expr.empty()) {
        return true;
    }

    // Check for boolean literals
    if (expr == "true" || expr == "false") {
        return true;
    }

    // Check for array literals like {1.0, 2.0, 3.0} or {{1, 2}, {3, 4}}
    // An array literal is constant if all its elements are constants
    if (expr.size() >= 2 && expr.front() == '{' && expr.back() == '}') {
        // Extract content between braces
        std::string content = expr.substr(1, expr.size() - 2);

        // Parse elements (handle nested braces)
        int braceDepth = 0;
        size_t elementStart = 0;
        for (size_t i = 0; i <= content.size(); i++) {
            char c = (i < content.size()) ? content[i] : ',';  // Treat end as comma
            if (c == '{') {
                braceDepth++;
            } else if (c == '}') {
                braceDepth--;
            } else if (c == ',' && braceDepth == 0) {
                // Found element boundary
                std::string element = content.substr(elementStart, i - elementStart);
                // Trim whitespace
                size_t start = element.find_first_not_of(" \t\n\r");
                size_t end = element.find_last_not_of(" \t\n\r");
                if (start != std::string::npos && end != std::string::npos) {
                    element = element.substr(start, end - start + 1);
                    // Recursively check if element is constant
                    if (!isConstExpression(element)) {
                        return false;
                    }
                }
                elementStart = i + 1;
            }
        }
        return true;  // All elements are constants
    }

    try {
        size_t pos = 0;
        std::stod(expr, &pos);
        // Check if the entire string was consumed (it's a pure number)
        // Skip trailing whitespace
        while (pos < expr.size() && std::isspace(expr[pos])) {
            pos++;
        }
        return pos == expr.size();
    } catch (...) {
        return false;
    }
}

std::string ModelInfoExtractor::extractDescription(basemodelica::BaseModelicaParser::CommentContext* ctx) {
    if (ctx && ctx->stringComment()) {
        std::string desc = ctx->stringComment()->getText();
        // Remove surrounding quotes
        if (desc.size() >= 2 && desc.front() == '"' && desc.back() == '"') {
            return desc.substr(1, desc.size() - 2);
        }
        return desc;
    }
    return "";
}

std::vector<std::string> ModelInfoExtractor::extractDimensions(basemodelica::BaseModelicaParser::DeclarationContext* ctx) {
    std::vector<std::string> dims;

    if (ctx->arraySubscripts()) {
        for (auto subscript : ctx->arraySubscripts()->subscript()) {
            if (subscript->expression()) {
                dims.push_back(stripQuotes(subscript->expression()->getText()));
            } else {
                dims.push_back(":");  // Unknown dimension
            }
        }
    }

    return dims;
}

void ModelInfoExtractor::extractFunctions(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx) {
    // Iterate through all class definitions to find functions
    for (auto classDefCtx : ctx->classDefinition()) {
        // Check if this is a function
        auto classPrefixes = classDefCtx->classPrefixes();
        if (!classPrefixes) continue;

        std::string prefixText = classPrefixes->getText();
        if (prefixText.find("function") == std::string::npos) {
            continue;  // Not a function
        }

        // Extract function details
        Function func;

        auto classSpec = classDefCtx->classSpecifier();
        if (!classSpec) continue;

        auto longClassSpec = classSpec->longClassSpecifier();
        if (!longClassSpec) continue;

        // Get function name
        if (longClassSpec->IDENT().size() > 0) {
            func.name = stripQuotes(longClassSpec->IDENT(0)->getText());
        }

        // Get description from string comment
        if (longClassSpec->stringComment()) {
            func.description = longClassSpec->stringComment()->getText();
            // Remove quotes
            if (func.description.size() >= 2 &&
                func.description.front() == '"' &&
                func.description.back() == '"') {
                func.description = func.description.substr(1, func.description.size() - 2);
            }
        }

        func.sourceFile = sourceFile;
        func.sourceLine = classDefCtx->getStart()->getLine();

        // Extract function composition (inputs, outputs, algorithm)
        auto composition = longClassSpec->composition();
        if (!composition) continue;

        // Extract input/output variables from genericElements
        for (auto genericElem : composition->genericElement()) {
            if (auto normalElem = genericElem->normalElement()) {
                if (auto compClause = normalElem->componentClause()) {
                    std::string typePrefix = "";
                    if (compClause->typePrefix()) {
                        typePrefix = compClause->typePrefix()->getText();
                    }

                    std::string typeSpec = compClause->typeSpecifier()->getText();

                    for (auto compDecl : compClause->componentList()->componentDeclaration()) {
                        Variable var;
                        var.name = stripQuotes(compDecl->declaration()->IDENT()->getText());
                        var.type = stripQuotes(typeSpec);
                        var.sourceFile = sourceFile;
                        var.sourceLine = compDecl->getStart()->getLine();

                        // Extract dimensions for arrays
                        var.dimensions = extractDimensions(compDecl->declaration());

                        // Determine if input or output
                        if (typePrefix.find("input") != std::string::npos) {
                            func.inputs.push_back(var);
                        } else if (typePrefix.find("output") != std::string::npos) {
                            func.outputs.push_back(var);
                        }
                    }
                }
            }
        }

        // Extract algorithm statements
        // Look for 'algorithm' keyword in composition
        // According to grammar: composition has ('initial'? 'algorithm' (statement ';')*)*
        // We need to access the raw children to find algorithm sections

        // Use the composition context to find statement() nodes
        for (auto statement : composition->statement()) {
            Statement stmt;
            stmt.sourceFile = sourceFile;
            stmt.sourceLine = statement->getStart()->getLine();

            // According to grammar line 184:
            // statement: decoration? (componentReference (':=' expression | functionCallArgs) | ...)

            // Check if this is a simple assignment statement (componentReference ':=' expression)
            if (statement->componentReference() && statement->expression()) {
                stmt.lhsContext = statement->componentReference();
                stmt.rhsContext = statement->expression();
                func.algorithmStatements.push_back(stmt);
            }
            // Check if this is a multi-output assignment: (a, b) := func(x)
            // Grammar: '(' outputExpressionList ')' ':=' componentReference functionCallArgs
            else if (statement->outputExpressionList() && statement->componentReference() && statement->functionCallArgs()) {
                stmt.lhsContext = statement->outputExpressionList();
                stmt.rhsContext = statement;  // Store the whole statement to access function call info
                func.algorithmStatements.push_back(stmt);
            }
            // Check if this is a for-statement
            else if (statement->forStatement()) {
                stmt.forStatementContext = statement->forStatement();
                func.algorithmStatements.push_back(stmt);
            }
            // Check if this is a while-statement
            else if (statement->whileStatement()) {
                stmt.whileStatementContext = statement->whileStatement();
                func.algorithmStatements.push_back(stmt);
            }
            // Check if this is an if-statement
            else if (statement->ifStatement()) {
                stmt.ifStatementContext = statement->ifStatement();
                func.algorithmStatements.push_back(stmt);
            }
            // Check if this is a standalone function call: funcName(args) without assignment
            else if (statement->componentReference() && statement->functionCallArgs() && !statement->expression()) {
                stmt.functionCallContext = statement;
                func.algorithmStatements.push_back(stmt);
            }
        }

        info.addFunction(func);
    }
}

} // namespace lacemodelica
