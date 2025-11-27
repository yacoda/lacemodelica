// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "ModelInfo.h"
#include "BaseModelicaParser.h"
#include "ONNXHelpers.hpp"
#include <string>

// Forward declare ONNX types
namespace onnx {
    class GraphProto;
    class ModelProto;
}

namespace lacemodelica {

class ONNXGenerator {
public:
    // Generate ONNX model file and layered standard manifest
    // Returns the directory path where files were generated
    static std::string generate(const ModelInfo& info, const std::string& outputDir);

    // Convert BaseModelica expression AST to ONNX nodes
    // Returns the name of the output tensor
    // Public to allow use by helper functions
    static std::string convertExpression(antlr4::ParserRuleContext* expr, const ConversionContext& ctx);

private:
    static void generateONNXModel(const ModelInfo& info, const std::string& filepath);
    static void generateManifest(const std::string& filepath);

    // Generate ONNX outputs for a list of equations with given prefix
    static void generateEquationOutputs(
        const std::vector<Equation>& equations,
        const std::string& prefix,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        std::map<std::string, std::vector<std::string>>& derivativeInputs
    );

    // Generate ONNX Loop node for a for-equation
    // Returns the number of equation outputs generated
    static size_t generateForEquationLoop(
        const Equation& eq,
        const std::string& prefix,
        size_t equationIndex,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        std::map<std::string, std::vector<std::string>>& derivativeInputs,
        bool isNested = false,
        std::map<std::string, std::string>* parentLoopVarMap = nullptr,
        std::string* outLoopNodeName = nullptr
    );

    // Generate ONNX If node for an if-equation
    // Returns the number of equation outputs generated
    static size_t generateIfEquation(
        const Equation& eq,
        const std::string& prefix,
        size_t equationIndex,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        std::map<std::string, std::vector<std::string>>& derivativeInputs
    );

    // Helper to build nested If structure for if-equation RHS
    static std::string buildIfEquationRhs(
        const std::vector<basemodelica::BaseModelicaParser::ExpressionContext*>& conditions,
        const std::vector<basemodelica::BaseModelicaParser::EquationContext*>& equations,
        size_t branchIndex,
        const ConversionContext& ctx);

    // Create ONNX FunctionProto for a function with algorithm
    static void createFunctionProto(
        const Function& func,
        const ModelInfo& info,
        onnx::ModelProto* model
    );

    static std::string convertSimpleExpression(
        basemodelica::BaseModelicaParser::SimpleExpressionContext* expr,
        const ConversionContext& ctx);

    static std::string convertIfExpression(
        basemodelica::BaseModelicaParser::IfExpressionContext* expr,
        const ConversionContext& ctx);

    // Helper for nested if-elseif-else chains
    static std::string convertNestedIfElse(
        const std::vector<basemodelica::BaseModelicaParser::ExpressionNoDecorationContext*>& expressions,
        size_t startIdx,
        const ConversionContext& ctx);

    static std::string convertRelation(
        basemodelica::BaseModelicaParser::RelationContext* relation,
        const ConversionContext& ctx);

    static std::string convertArithmeticExpression(
        basemodelica::BaseModelicaParser::ArithmeticExpressionContext* expr,
        const ConversionContext& ctx);

    static std::string convertTerm(
        basemodelica::BaseModelicaParser::TermContext* expr,
        const ConversionContext& ctx);

    static std::string convertFactor(
        basemodelica::BaseModelicaParser::FactorContext* expr,
        const ConversionContext& ctx);

    static std::string convertPrimary(
        basemodelica::BaseModelicaParser::PrimaryContext* expr,
        const ConversionContext& ctx);

    // Helper to convert der() function calls
    static std::string convertDerFunctionCall(
        basemodelica::BaseModelicaParser::PrimaryContext* expr,
        const ConversionContext& ctx);

    // Helper to convert user-defined function calls
    static std::string convertUserFunctionCall(
        const std::string& funcName,
        basemodelica::BaseModelicaParser::PrimaryContext* expr,
        const Function* func,
        const ConversionContext& ctx);

    // Convert multi-output function call
    static std::vector<std::string> convertMultiOutputFunctionCall(
        antlr4::ParserRuleContext* expr,
        const ConversionContext& ctx,
        size_t expectedOutputCount);
};

} // namespace lacemodelica
