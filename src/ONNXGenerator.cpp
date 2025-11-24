// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ONNXGenerator.h"
#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>
#include <tinyxml2.h>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace lacemodelica {

std::string ONNXGenerator::generate(const ModelInfo& info, const std::string& outputDir) {
    // Layered standard directory structure
    std::string lsName = "org.lacemodelica.ls-onnx-serialization";
    std::string lsDir = outputDir + "/extra/" + lsName;

    // Create directories
    std::filesystem::create_directories(lsDir);

    // Generate ONNX model file
    std::string modelPath = lsDir + "/model.onnx";
    generateONNXModel(info, modelPath);

    // Generate layered standard manifest
    std::string manifestPath = lsDir + "/fmi-ls-manifest.xml";
    generateManifest(manifestPath);

    std::cout << "Generated ONNX layered standard in " << lsDir << "/" << std::endl;

    return lsDir;
}

void ONNXGenerator::generateONNXModel(const ModelInfo& info, const std::string& filepath) {
    onnx::ModelProto model;

    // Set model metadata
    model.set_ir_version(8);  // ONNX IR version 8
    model.set_producer_name("lacemodelica");
    model.set_producer_version("0.1.0");
    model.set_model_version(1);
    model.set_doc_string("Symbolic representation of " + info.modelName);

    // Add opset import (opset version 18)
    auto* opset = model.add_opset_import();
    opset->set_version(18);

    // Create the graph
    auto* graph = model.mutable_graph();
    graph->set_name(info.modelName);

    // Create ONNX inputs for each variable and parameter (skip derivatives)
    for (const auto& var : info.variables) {
        // Skip derivative variables - der() will be an operator
        if (var.isDerivative) {
            continue;
        }

        auto* input = graph->add_input();
        input->set_name(var.name);
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        input_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* input_shape = input_type->mutable_shape();

        // Handle array dimensions
        if (!var.dimensions.empty()) {
            for (const auto& dim : var.dimensions) {
                auto* shape_dim = input_shape->add_dim();
                // Try to parse as integer, otherwise leave symbolic
                try {
                    shape_dim->set_dim_value(std::stoi(dim));
                } catch (...) {
                    shape_dim->set_dim_param(dim);
                }
            }
        } else {
            // Scalar: shape [1]
            auto* shape_dim = input_shape->add_dim();
            shape_dim->set_dim_value(1);
        }
    }

    // Create ONNX outputs for equations
    int nodeCounter = 0;

    // Generate outputs for regular equations
    generateEquationOutputs(info.equations, "eq", graph, nodeCounter);

    // Generate outputs for initial equations
    generateEquationOutputs(info.initialEquations, "init_eq", graph, nodeCounter);

    // Serialize to file
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to create ONNX file: " + filepath);
    }

    if (!model.SerializeToOstream(&ofs)) {
        throw std::runtime_error("Failed to serialize ONNX model to: " + filepath);
    }

    ofs.close();
}

void ONNXGenerator::generateManifest(const std::string& filepath) {
    using namespace tinyxml2;

    XMLDocument doc;

    // Create root element with FMI layered standard attributes
    XMLElement* root = doc.NewElement("fmiLayeredStandardManifest");
    root->SetAttribute("xmlns:fmi-ls", "http://fmi-standard.org/fmi-ls-manifest");
    root->SetAttribute("fmi-ls:fmi-ls-name", "org.lacemodelica.ls-onnx-serialization");
    root->SetAttribute("fmi-ls:fmi-ls-version", "1.0.0");
    root->SetAttribute("fmi-ls:fmi-ls-description",
        "Layered standard for ONNX-serialized symbolic expressions in FMU");

    doc.InsertFirstChild(root);

    // Save to file
    if (doc.SaveFile(filepath.c_str()) != XML_SUCCESS) {
        throw std::runtime_error("Failed to write manifest file: " + filepath);
    }
}

void ONNXGenerator::generateEquationOutputs(
    const std::vector<Equation>& equations,
    const std::string& prefix,
    onnx::GraphProto* graph,
    int& nodeCounter) {

    for (size_t i = 0; i < equations.size(); i++) {
        const auto& eq = equations[i];

        // Convert LHS expression to ONNX graph
        std::string lhsTensor;
        try {
            lhsTensor = convertExpression(eq.lhsContext, graph, nodeCounter);
        } catch (const std::exception& e) {
            std::cerr << "Error converting LHS of " << prefix << " equation " << i;
            if (!eq.sourceFile.empty()) {
                std::cerr << " (" << eq.sourceFile << ":" << eq.sourceLine << ")";
            }
            std::cerr << ": " << e.what() << std::endl;
            std::cerr << "LHS text: " << eq.lhsContext->getText() << std::endl;
            throw;
        }

        // Convert RHS expression to ONNX graph
        std::string rhsTensor;
        try {
            rhsTensor = convertExpression(eq.rhsContext, graph, nodeCounter);
        } catch (const std::exception& e) {
            std::cerr << "Error converting RHS of " << prefix << " equation " << i;
            if (!eq.sourceFile.empty()) {
                std::cerr << " (" << eq.sourceFile << ":" << eq.sourceLine << ")";
            }
            std::cerr << ": " << e.what() << std::endl;
            std::cerr << "RHS text: " << eq.rhsContext->getText() << std::endl;
            throw;
        }

        // Create an '=' operator node with LHS and RHS as inputs
        std::string eqOutputName = prefix + "[" + std::to_string(i) + "]";

        auto* eq_node = graph->add_node();
        eq_node->set_op_type("Equal");
        eq_node->set_name(prefix + "_equal_" + std::to_string(i));
        eq_node->add_input(lhsTensor);
        eq_node->add_input(rhsTensor);
        eq_node->add_output(eqOutputName);

        // Create output for this equation
        auto* eq_output = graph->add_output();
        eq_output->set_name(eqOutputName);
        auto* eq_type = eq_output->mutable_type()->mutable_tensor_type();
        eq_type->set_elem_type(onnx::TensorProto::BOOL);  // Equal returns boolean
        auto* eq_shape = eq_type->mutable_shape();
        eq_shape->add_dim()->set_dim_value(1);

        // Set the string comment as doc_string on the output
        if (!eq.comment.empty()) {
            eq_output->set_doc_string(eq.comment);
        }
    }
}

// Expression conversion functions

std::string ONNXGenerator::convertExpression(
    antlr4::ParserRuleContext* expr,
    onnx::GraphProto* graph,
    int& nodeCounter) {

    if (!expr) {
        throw std::runtime_error("Null expression context");
    }

    // Try to cast to specific expression types
    if (auto* exprCtx = dynamic_cast<basemodelica::BaseModelicaParser::ExpressionContext*>(expr)) {
        // Expression -> ExpressionNoDecoration -> SimpleExpression
        auto* exprNoDecoration = exprCtx->expressionNoDecoration();
        if (exprNoDecoration) {
            auto* simpleExpr = exprNoDecoration->simpleExpression();
            if (simpleExpr) {
                return convertSimpleExpression(simpleExpr, graph, nodeCounter);
            }
        }
    } else if (auto* simpleExpr = dynamic_cast<basemodelica::BaseModelicaParser::SimpleExpressionContext*>(expr)) {
        return convertSimpleExpression(simpleExpr, graph, nodeCounter);
    }

    // Fallback: return placeholder
    throw std::runtime_error("Unsupported expression type: " + expr->getText());
}

std::string ONNXGenerator::convertSimpleExpression(
    basemodelica::BaseModelicaParser::SimpleExpressionContext* expr,
    onnx::GraphProto* graph,
    int& nodeCounter) {

    // SimpleExpression: logicalExpression+
    // For now, assume single logical expression -> arithmetic expression
    auto logicalExprs = expr->logicalExpression();
    if (logicalExprs.empty()) {
        throw std::runtime_error("Empty simple expression");
    }

    // Get first logical expression -> logical term -> logical factor -> relation -> arithmetic
    auto* logicalExpr = logicalExprs[0];
    auto logicalTerms = logicalExpr->logicalTerm();
    if (logicalTerms.empty()) {
        throw std::runtime_error("Empty logical expression");
    }

    auto* logicalTerm = logicalTerms[0];
    auto logicalFactors = logicalTerm->logicalFactor();
    if (logicalFactors.empty()) {
        throw std::runtime_error("Empty logical term");
    }

    auto* logicalFactor = logicalFactors[0];
    auto* relation = logicalFactor->relation();
    if (!relation) {
        throw std::runtime_error("No relation in logical factor");
    }

    auto arithmeticExprs = relation->arithmeticExpression();
    if (arithmeticExprs.empty()) {
        throw std::runtime_error("No arithmetic expression in relation");
    }

    return convertArithmeticExpression(arithmeticExprs[0], graph, nodeCounter);
}

std::string ONNXGenerator::convertArithmeticExpression(
    basemodelica::BaseModelicaParser::ArithmeticExpressionContext* expr,
    onnx::GraphProto* graph,
    int& nodeCounter) {

    // ArithmeticExpression: addOperator? term (addOperator term)*
    auto terms = expr->term();
    auto addOps = expr->addOperator();

    if (terms.empty()) {
        throw std::runtime_error("Empty arithmetic expression");
    }

    // Check if there's a leading addOperator (e.g., unary minus)
    size_t termIndex = 0;
    size_t opIndex = 0;

    // Convert first term
    std::string result = convertTerm(terms[termIndex++], graph, nodeCounter);

    // If we have more addOps than remaining terms, first addOp was a leading unary
    if (addOps.size() > terms.size() - 1) {
        // Leading operator - apply it to the first term
        std::string opText = addOps[opIndex++]->getText();

        if (opText == "-") {
            auto* node = graph->add_node();
            std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);
            node->set_op_type("Neg");
            node->set_name("Neg_" + std::to_string(nodeCounter));
            node->add_input(result);
            node->add_output(outputTensor);
            result = outputTensor;
        } else if (opText == "+") {
            // Unary plus - no-op
        } else {
            throw std::runtime_error("Unsupported leading operator: " + opText);
        }
    }

    // Process remaining terms with operators
    while (opIndex < addOps.size()) {
        std::string opText = addOps[opIndex]->getText();
        std::string rightTensor = convertTerm(terms[termIndex], graph, nodeCounter);
        termIndex++;
        opIndex++;

        // Create ONNX node for the operation
        auto* node = graph->add_node();
        std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

        if (opText == "+") {
            node->set_op_type("Add");
        } else if (opText == "-") {
            node->set_op_type("Sub");
        } else if (opText == ".+") {
            node->set_op_type("Add");  // Element-wise add
        } else if (opText == ".-") {
            node->set_op_type("Sub");  // Element-wise sub
        } else {
            throw std::runtime_error("Unsupported add operator: " + opText);
        }

        node->set_name(node->op_type() + "_" + std::to_string(nodeCounter));
        node->add_input(result);
        node->add_input(rightTensor);
        node->add_output(outputTensor);

        result = outputTensor;
    }

    return result;
}

std::string ONNXGenerator::convertTerm(
    basemodelica::BaseModelicaParser::TermContext* expr,
    onnx::GraphProto* graph,
    int& nodeCounter) {

    // Term: factor (mulOperator factor)*
    auto factors = expr->factor();
    auto mulOps = expr->mulOperator();

    if (factors.empty()) {
        throw std::runtime_error("Empty term");
    }

    // Convert first factor
    std::string result = convertFactor(factors[0], graph, nodeCounter);

    // Process remaining factors with operators
    for (size_t i = 0; i < mulOps.size(); i++) {
        std::string opText = mulOps[i]->getText();
        std::string rightTensor = convertFactor(factors[i + 1], graph, nodeCounter);

        // Create ONNX node for the operation
        auto* node = graph->add_node();
        std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

        if (opText == "*") {
            node->set_op_type("Mul");
        } else if (opText == "/") {
            node->set_op_type("Div");
        } else if (opText == ".*") {
            node->set_op_type("Mul");  // Element-wise multiply
        } else if (opText == "./") {
            node->set_op_type("Div");  // Element-wise divide
        } else {
            throw std::runtime_error("Unsupported mul operator: " + opText);
        }

        node->set_name(node->op_type() + "_" + std::to_string(nodeCounter));
        node->add_input(result);
        node->add_input(rightTensor);
        node->add_output(outputTensor);

        result = outputTensor;
    }

    return result;
}

std::string ONNXGenerator::convertFactor(
    basemodelica::BaseModelicaParser::FactorContext* expr,
    onnx::GraphProto* graph,
    int& nodeCounter) {

    // Factor: primary (('^' | '.^') primary)?
    auto primaries = expr->primary();

    if (primaries.empty()) {
        throw std::runtime_error("Empty factor");
    }

    std::string result = convertPrimary(primaries[0], graph, nodeCounter);

    // Handle power operator if present
    if (primaries.size() > 1) {
        std::string exponentTensor = convertPrimary(primaries[1], graph, nodeCounter);

        auto* node = graph->add_node();
        std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

        node->set_op_type("Pow");
        node->set_name("Pow_" + std::to_string(nodeCounter));
        node->add_input(result);
        node->add_input(exponentTensor);
        node->add_output(outputTensor);

        result = outputTensor;
    }

    return result;
}

std::string ONNXGenerator::convertPrimary(
    basemodelica::BaseModelicaParser::PrimaryContext* expr,
    onnx::GraphProto* graph,
    int& nodeCounter) {

    // Handle different primary types

    // 1. Number literal
    if (expr->UNSIGNED_NUMBER()) {
        std::string value = expr->UNSIGNED_NUMBER()->getText();

        // Create constant tensor
        auto* constant = graph->add_initializer();
        std::string constName = "const_" + std::to_string(nodeCounter++);
        constant->set_name(constName);
        constant->set_data_type(onnx::TensorProto::FLOAT);
        constant->add_dims(1);
        constant->add_float_data(std::stof(value));

        return constName;
    }

    // 2. Component reference (variable)
    if (expr->componentReference()) {
        std::string varName = expr->componentReference()->getText();
        // Strip quotes if present
        if (varName.front() == '\'' && varName.back() == '\'') {
            varName = varName.substr(1, varName.size() - 2);
        }
        return varName;  // Variable inputs are already in the graph
    }

    // 3. Function call (e.g., der())
    if (expr->functionCallArgs()) {
        // Check for der() function
        std::string text = expr->getText();
        if (text.find("der(") == 0) {
            // Extract variable name from der('varname')
            size_t start = 4;  // After "der("
            size_t end = text.find(")", start);
            std::string varName = text.substr(start, end - start);

            // Strip quotes if present
            if (varName.front() == '\'' && varName.back() == '\'') {
                varName = varName.substr(1, varName.size() - 2);
            }

            // Create a Der operator node
            auto* node = graph->add_node();
            std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

            node->set_op_type("Der");
            node->set_name("Der_" + std::to_string(nodeCounter));
            node->add_input(varName);  // Input is the variable itself
            node->add_output(outputTensor);

            return outputTensor;
        }
        throw std::runtime_error("Unsupported function call: " + text);
    }

    // 4. Parenthesized expression
    if (expr->outputExpressionList()) {
        // (expression) - just convert the expression inside
        auto outputList = expr->outputExpressionList();
        auto expressions = outputList->expression();
        if (!expressions.empty()) {
            return convertExpression(expressions[0], graph, nodeCounter);
        }
    }

    throw std::runtime_error("Unsupported primary expression: " + expr->getText());
}

} // namespace lacemodelica
