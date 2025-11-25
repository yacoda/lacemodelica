// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ModelInfoExtractor.h"
#include <iostream>

namespace lacemodelica {

// Helper function to strip quotes from BaseModelica quoted identifiers
static std::string stripQuotes(const std::string& str) {
    if (str.size() >= 2 && str.front() == '\'' && str.back() == '\'') {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

ModelInfo ModelInfoExtractor::extract(basemodelica::BaseModelicaParser::BaseModelicaContext* tree, const std::string& sourceFile) {
    info = ModelInfo();
    this->sourceFile = sourceFile;

    extractPackageAndModelName(tree);
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

                    info.addVariable(var);
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

void ModelInfoExtractor::processEquation(basemodelica::BaseModelicaParser::EquationContext* equation, std::vector<Equation>& target) {
    // Check for unsupported equation types first
    if (equation->ifEquation()) {
        throw std::runtime_error("If-equations are not supported (" + sourceFile + ":" +
                                 std::to_string(equation->getStart()->getLine()) + ")");
    }
    if (equation->forEquation()) {
        throw std::runtime_error("For-equations are not supported (" + sourceFile + ":" +
                                 std::to_string(equation->getStart()->getLine()) + ")");
    }
    if (equation->whenEquation()) {
        throw std::runtime_error("When-equations are not supported (" + sourceFile + ":" +
                                 std::to_string(equation->getStart()->getLine()) + ")");
    }

    // Check if it's a simple equation (lhs = rhs)
    auto simpleExpr = equation->simpleExpression();
    auto fullExpr = equation->expression();

    if (simpleExpr && fullExpr) {
        // This is an equation with = sign
        Equation eq;
        eq.lhsContext = simpleExpr;  // Store AST node
        eq.rhsContext = fullExpr;    // Store AST node
        eq.sourceFile = sourceFile;
        eq.sourceLine = equation->getStart()->getLine();

        // Extract string comment
        if (equation->comment()) {
            eq.comment = extractDescription(equation->comment());
        }

        target.push_back(eq);

        // Scan equation text for der() calls to identify derivatives
        std::string eqText = equation->getText();
        size_t pos = 0;
        while ((pos = eqText.find("der(", pos)) != std::string::npos) {
            // Extract the variable name inside der()
            size_t start = pos + 4;
            size_t end = eqText.find(")", start);
            if (end != std::string::npos) {
                std::string varName = eqText.substr(start, end - start);
                derivativeCalls.insert(varName);
            }
            pos = end;
        }
        return;
    }

    // If we get here, equation type is unknown
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


std::string ModelInfoExtractor::extractStartValue(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (ctx->expression()) {
        return ctx->expression()->getText();
    }

    // Check class modification for start= or similar
    if (ctx->classModification()) {
        auto argList = ctx->classModification()->argumentList();
        if (argList) {
            for (auto arg : argList->argument()) {
                if (auto elemMod = arg->elementModificationOrReplaceable()) {
                    if (auto mod = elemMod->elementModification()) {
                        std::string name = mod->name()->getText();
                        if (name == "start" && mod->modification() && mod->modification()->expression()) {
                            return mod->modification()->expression()->getText();
                        }
                    }
                }
            }
        }
    }

    return "";
}

std::string ModelInfoExtractor::extractMinValue(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (ctx->classModification()) {
        auto argList = ctx->classModification()->argumentList();
        if (argList) {
            for (auto arg : argList->argument()) {
                if (auto elemMod = arg->elementModificationOrReplaceable()) {
                    if (auto mod = elemMod->elementModification()) {
                        std::string name = mod->name()->getText();
                        if (name == "min" && mod->modification() && mod->modification()->expression()) {
                            return mod->modification()->expression()->getText();
                        }
                    }
                }
            }
        }
    }
    return "";
}

std::string ModelInfoExtractor::extractMaxValue(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (ctx->classModification()) {
        auto argList = ctx->classModification()->argumentList();
        if (argList) {
            for (auto arg : argList->argument()) {
                if (auto elemMod = arg->elementModificationOrReplaceable()) {
                    if (auto mod = elemMod->elementModification()) {
                        std::string name = mod->name()->getText();
                        if (name == "max" && mod->modification() && mod->modification()->expression()) {
                            return mod->modification()->expression()->getText();
                        }
                    }
                }
            }
        }
    }
    return "";
}

antlr4::ParserRuleContext* ModelInfoExtractor::extractBindingContext(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (ctx->expression()) {
        return ctx->expression();
    }

    // Check class modification for start= or similar
    if (ctx->classModification()) {
        auto argList = ctx->classModification()->argumentList();
        if (argList) {
            for (auto arg : argList->argument()) {
                if (auto elemMod = arg->elementModificationOrReplaceable()) {
                    if (auto mod = elemMod->elementModification()) {
                        std::string name = mod->name()->getText();
                        if (name == "start" && mod->modification() && mod->modification()->expression()) {
                            return mod->modification()->expression();
                        }
                    }
                }
            }
        }
    }

    return nullptr;
}

antlr4::ParserRuleContext* ModelInfoExtractor::extractMinContext(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (ctx->classModification()) {
        auto argList = ctx->classModification()->argumentList();
        if (argList) {
            for (auto arg : argList->argument()) {
                if (auto elemMod = arg->elementModificationOrReplaceable()) {
                    if (auto mod = elemMod->elementModification()) {
                        std::string name = mod->name()->getText();
                        if (name == "min" && mod->modification() && mod->modification()->expression()) {
                            return mod->modification()->expression();
                        }
                    }
                }
            }
        }
    }
    return nullptr;
}

antlr4::ParserRuleContext* ModelInfoExtractor::extractMaxContext(basemodelica::BaseModelicaParser::ModificationContext* ctx) {
    if (ctx->classModification()) {
        auto argList = ctx->classModification()->argumentList();
        if (argList) {
            for (auto arg : argList->argument()) {
                if (auto elemMod = arg->elementModificationOrReplaceable()) {
                    if (auto mod = elemMod->elementModification()) {
                        std::string name = mod->name()->getText();
                        if (name == "max" && mod->modification() && mod->modification()->expression()) {
                            return mod->modification()->expression();
                        }
                    }
                }
            }
        }
    }
    return nullptr;
}

bool ModelInfoExtractor::isConstExpression(const std::string& expr) {
    if (expr.empty()) {
        return true;
    }

    // Check for boolean literals
    if (expr == "true" || expr == "false") {
        return true;
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

            // Check if this is an assignment statement (componentReference ':=' expression)
            if (statement->componentReference() && statement->expression()) {
                stmt.lhsContext = statement->componentReference();
                stmt.rhsContext = statement->expression();
                func.algorithmStatements.push_back(stmt);
            }
            // Note: We're skipping other statement types (function calls, if, for, while, etc.)
            // These would need special handling in the future
        }

        info.addFunction(func);

        std::cerr << "Extracted function: " << func.name
                  << " with " << func.inputs.size() << " inputs, "
                  << func.outputs.size() << " outputs, "
                  << func.algorithmStatements.size() << " algorithm statements" << std::endl;
    }
}

} // namespace lacemodelica
