// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "ModelInfo.h"
#include "BaseModelicaParser.h"
#include <string>

// Forward declare ONNX types
namespace onnx {
    class GraphProto;
}

namespace lacemodelica {

class ONNXGenerator {
public:
    // Generate ONNX model file and layered standard manifest
    // Returns the directory path where files were generated
    static std::string generate(const ModelInfo& info, const std::string& outputDir);

private:
    static void generateONNXModel(const ModelInfo& info, const std::string& filepath);
    static void generateManifest(const std::string& filepath);

    // Generate ONNX outputs for a list of equations with given prefix
    static void generateEquationOutputs(
        const std::vector<Equation>& equations,
        const std::string& prefix,
        onnx::GraphProto* graph,
        int& nodeCounter
    );

    // Convert BaseModelica expression AST to ONNX nodes
    // Returns the name of the output tensor
    static std::string convertExpression(
        antlr4::ParserRuleContext* expr,
        onnx::GraphProto* graph,
        int& nodeCounter
    );

    static std::string convertSimpleExpression(
        basemodelica::BaseModelicaParser::SimpleExpressionContext* expr,
        onnx::GraphProto* graph,
        int& nodeCounter
    );

    static std::string convertArithmeticExpression(
        basemodelica::BaseModelicaParser::ArithmeticExpressionContext* expr,
        onnx::GraphProto* graph,
        int& nodeCounter
    );

    static std::string convertTerm(
        basemodelica::BaseModelicaParser::TermContext* expr,
        onnx::GraphProto* graph,
        int& nodeCounter
    );

    static std::string convertFactor(
        basemodelica::BaseModelicaParser::FactorContext* expr,
        onnx::GraphProto* graph,
        int& nodeCounter
    );

    static std::string convertPrimary(
        basemodelica::BaseModelicaParser::PrimaryContext* expr,
        onnx::GraphProto* graph,
        int& nodeCounter
    );
};

} // namespace lacemodelica
