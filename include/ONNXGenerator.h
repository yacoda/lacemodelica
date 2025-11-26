// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "ModelInfo.h"
#include "BaseModelicaParser.h"
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
        std::map<std::string, std::string>* parentLoopVarMap = nullptr
    );

    // Create ONNX FunctionProto for a function with algorithm
    static void createFunctionProto(
        const Function& func,
        const ModelInfo& info,
        onnx::ModelProto* model
    );

    // Convert BaseModelica expression AST to ONNX nodes
    // Returns the name of the output tensor
    static std::string convertExpression(
        antlr4::ParserRuleContext* expr,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = ""
    );

    static std::string convertSimpleExpression(
        basemodelica::BaseModelicaParser::SimpleExpressionContext* expr,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = ""
    );

    static std::string convertIfExpression(
        basemodelica::BaseModelicaParser::IfExpressionContext* expr,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = ""
    );

    static std::string convertRelation(
        basemodelica::BaseModelicaParser::RelationContext* relation,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = ""
    );

    static std::string convertArithmeticExpression(
        basemodelica::BaseModelicaParser::ArithmeticExpressionContext* expr,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = ""
    );

    static std::string convertTerm(
        basemodelica::BaseModelicaParser::TermContext* expr,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = ""
    );

    static std::string convertFactor(
        basemodelica::BaseModelicaParser::FactorContext* expr,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = ""
    );

    static std::string convertPrimary(
        basemodelica::BaseModelicaParser::PrimaryContext* expr,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = ""
    );

    // Convert multi-output function call
    static std::vector<std::string> convertMultiOutputFunctionCall(
        antlr4::ParserRuleContext* expr,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        std::map<std::string, std::vector<std::string>>* derivativeInputs,
        size_t expectedOutputCount
    );
};

} // namespace lacemodelica
