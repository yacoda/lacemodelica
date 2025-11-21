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

ModelInfo ModelInfoExtractor::extract(basemodelica::BaseModelicaParser::BaseModelicaContext* tree) {
    info = ModelInfo();

    extractPackageAndModelName(tree);
    extractVariables(tree);
    extractEquations(tree);
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

                    // Extract start value from modification
                    if (auto mod = compDecl->declaration()->modification()) {
                        var.startValue = extractStartValue(mod);
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

    // Scan equations for der() calls
    for (auto equation : composition->equation()) {
        // Look for der() calls in the equation
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
    }
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

} // namespace lacemodelica
