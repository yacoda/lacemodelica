// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "BaseModelicaParser.h"
#include "ModelInfo.h"
#include <set>

namespace lacemodelica {

class ModelInfoExtractor {
public:
    ModelInfo extract(basemodelica::BaseModelicaParser::BaseModelicaContext* tree, const std::string& sourceFile = "");

private:
    ModelInfo info;
    std::string sourceFile;
    std::set<std::string> derivativeCalls;  // Track der() calls

    void extractPackageAndModelName(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void extractVariables(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void extractEquations(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
    void identifyDerivatives();

    void processEquation(basemodelica::BaseModelicaParser::EquationContext* equation, std::vector<Equation>& target);

    std::string extractStartValue(basemodelica::BaseModelicaParser::ModificationContext* ctx);
    std::string extractDescription(basemodelica::BaseModelicaParser::CommentContext* ctx);
    std::vector<std::string> extractDimensions(basemodelica::BaseModelicaParser::DeclarationContext* ctx);
};

} // namespace lacemodelica
