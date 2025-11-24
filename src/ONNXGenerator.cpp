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
    // Track mapping from variable index to input index for start[] outputs
    std::map<size_t, int> varIndexToInputIndex;
    int inputIndex = 0;

    for (size_t i = 0; i < info.variables.size(); i++) {
        const auto& var = info.variables[i];

        // Skip derivative variables - der() will be an operator
        if (var.isDerivative) {
            continue;
        }

        varIndexToInputIndex[i] = inputIndex++;

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

        // Add source location metadata
        if (!var.sourceFile.empty()) {
            auto* meta_file = input->add_metadata_props();
            meta_file->set_key("source_file");
            meta_file->set_value(var.sourceFile);

            auto* meta_line = input->add_metadata_props();
            meta_line->set_key("source_line");
            meta_line->set_value(std::to_string(var.sourceLine));
        }
    }

    // Create ONNX outputs for equations
    int nodeCounter = 0;

    // Generate outputs for regular equations
    generateEquationOutputs(info.equations, "eq", graph, nodeCounter);

    // Generate outputs for initial equations
    generateEquationOutputs(info.initialEquations, "init_eq", graph, nodeCounter);

    // Generate outputs for calculated initial values (non-const parameter bindings)
    for (size_t i = 0; i < info.variables.size(); i++) {
        const auto& var = info.variables[i];

        if (var.initial == "calculated" && var.bindingContext != nullptr) {
            // Find the input index for this variable
            auto it = varIndexToInputIndex.find(i);
            if (it == varIndexToInputIndex.end()) {
                std::cerr << "Warning: Could not find input index for variable " << var.name << std::endl;
                continue;
            }
            int inputIdx = it->second;

            // Convert binding expression to ONNX
            std::string exprTensor;
            try {
                exprTensor = convertExpression(var.bindingContext, graph, nodeCounter);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to convert binding expression for " << var.name;
                if (!var.sourceFile.empty()) {
                    std::cerr << " (" << var.sourceFile << ":" << var.sourceLine << ")";
                }
                std::cerr << ": " << e.what() << std::endl;
                continue;
            }

            // Create output for this binding using the input index
            std::string outputName = "start[" + std::to_string(inputIdx) + "]";
            auto* output = graph->add_output();
            output->set_name(outputName);
            auto* output_type = output->mutable_type()->mutable_tensor_type();
            output_type->set_elem_type(onnx::TensorProto::FLOAT);
            auto* output_shape = output_type->mutable_shape();
            output_shape->add_dim()->set_dim_value(1);

            // Set doc_string to variable name for reference
            output->set_doc_string("Initial value for " + var.name);

            // Add source location metadata
            if (!var.sourceFile.empty()) {
                auto* meta_file = output->add_metadata_props();
                meta_file->set_key("source_file");
                meta_file->set_value(var.sourceFile);

                auto* meta_line = output->add_metadata_props();
                meta_line->set_key("source_line");
                meta_line->set_value(std::to_string(var.sourceLine));
            }

            // Add metadata linking to the variable
            auto* meta_var = output->add_metadata_props();
            meta_var->set_key("variable_name");
            meta_var->set_value(var.name);

            auto* meta_vr = output->add_metadata_props();
            meta_vr->set_key("value_reference");
            meta_vr->set_value(std::to_string(var.valueReference));

            auto* meta_idx = output->add_metadata_props();
            meta_idx->set_key("input_index");
            meta_idx->set_value(std::to_string(inputIdx));

            // Rename the expression output tensor to the desired output name
            // Find the node that produces exprTensor and rename its output
            bool foundProducer = false;
            for (int j = graph->node_size() - 1; j >= 0; j--) {
                auto* node = graph->mutable_node(j);
                for (int k = 0; k < node->output_size(); k++) {
                    if (node->output(k) == exprTensor) {
                        node->set_output(k, outputName);
                        foundProducer = true;
                        break;
                    }
                }
                if (foundProducer) break;
            }

            // If no node produces this tensor, it's a direct input reference
            // Create an Identity node to connect it to the output
            if (!foundProducer) {
                auto* identity = graph->add_node();
                identity->set_op_type("Identity");
                identity->set_name("start_identity_" + std::to_string(inputIdx));
                identity->add_input(exprTensor);
                identity->add_output(outputName);
            }
        }
    }

    // Helper lambda to generate min/max outputs
    auto generateBoundOutput = [&](size_t i, const Variable& var, antlr4::ParserRuleContext* context,
                                     const std::string& boundType, const std::string& description) {
        // Find the input index for this variable
        auto it = varIndexToInputIndex.find(i);
        if (it == varIndexToInputIndex.end()) {
            std::cerr << "Warning: Could not find input index for variable " << var.name << std::endl;
            return;
        }
        int inputIdx = it->second;

        // Convert bound expression to ONNX
        std::string exprTensor;
        try {
            exprTensor = convertExpression(context, graph, nodeCounter);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to convert " << boundType << " expression for " << var.name;
            if (!var.sourceFile.empty()) {
                std::cerr << " (" << var.sourceFile << ":" << var.sourceLine << ")";
            }
            std::cerr << ": " << e.what() << std::endl;
            return;
        }

        // Create output for this bound using the input index
        std::string outputName = boundType + "[" + std::to_string(inputIdx) + "]";
        auto* output = graph->add_output();
        output->set_name(outputName);
        auto* output_type = output->mutable_type()->mutable_tensor_type();
        output_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* output_shape = output_type->mutable_shape();
        output_shape->add_dim()->set_dim_value(1);

        // Set doc_string to variable name for reference
        output->set_doc_string(description + " for " + var.name);

        // Add source location metadata
        if (!var.sourceFile.empty()) {
            auto* meta_file = output->add_metadata_props();
            meta_file->set_key("source_file");
            meta_file->set_value(var.sourceFile);

            auto* meta_line = output->add_metadata_props();
            meta_line->set_key("source_line");
            meta_line->set_value(std::to_string(var.sourceLine));
        }

        // Add metadata linking to the variable
        auto* meta_var = output->add_metadata_props();
        meta_var->set_key("variable_name");
        meta_var->set_value(var.name);

        auto* meta_vr = output->add_metadata_props();
        meta_vr->set_key("value_reference");
        meta_vr->set_value(std::to_string(var.valueReference));

        auto* meta_idx = output->add_metadata_props();
        meta_idx->set_key("input_index");
        meta_idx->set_value(std::to_string(inputIdx));

        // Rename the expression output tensor to the desired output name
        // Find the node that produces exprTensor and rename its output
        bool foundProducer = false;
        for (int j = graph->node_size() - 1; j >= 0; j--) {
            auto* node = graph->mutable_node(j);
            for (int k = 0; k < node->output_size(); k++) {
                if (node->output(k) == exprTensor) {
                    node->set_output(k, outputName);
                    foundProducer = true;
                    break;
                }
            }
            if (foundProducer) break;
        }

        // If no node produces this tensor, it's a direct input reference
        // Create an Identity node to connect it to the output
        if (!foundProducer) {
            auto* identity = graph->add_node();
            identity->set_op_type("Identity");
            identity->set_name(boundType + "_identity_" + std::to_string(inputIdx));
            identity->add_input(exprTensor);
            identity->add_output(outputName);
        }
    };

    // Generate outputs for non-const min values
    for (size_t i = 0; i < info.variables.size(); i++) {
        const auto& var = info.variables[i];
        if (var.minContext != nullptr && !var.minValue.empty()) {
            // Check if it's non-const by trying to parse as double
            try {
                std::stod(var.minValue);
                // It's a constant, skip ONNX generation
            } catch (...) {
                // Non-const, generate ONNX output
                generateBoundOutput(i, var, var.minContext, "min", "Minimum value");
            }
        }
    }

    // Generate outputs for non-const max values
    for (size_t i = 0; i < info.variables.size(); i++) {
        const auto& var = info.variables[i];
        if (var.maxContext != nullptr && !var.maxValue.empty()) {
            // Check if it's non-const by trying to parse as double
            try {
                std::stod(var.maxValue);
                // It's a constant, skip ONNX generation
            } catch (...) {
                // Non-const, generate ONNX output
                generateBoundOutput(i, var, var.maxContext, "max", "Maximum value");
            }
        }
    }

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

    std::cerr << "DEBUG: generateEquationOutputs called with " << equations.size() << " " << prefix << " equations" << std::endl;

    for (size_t i = 0; i < equations.size(); i++) {
        const auto& eq = equations[i];

        // Debug: Check if this equation contains tan
        std::string rhsText = eq.rhsContext ? eq.rhsContext->getText() : "";
        if (rhsText.find("tan") != std::string::npos) {
            std::cerr << "DEBUG: Equation " << i << " contains 'tan', RHS: " << rhsText.substr(0, 80) << "..." << std::endl;
        }

        std::string lhsTensor, rhsTensor;
        try {
            // Convert LHS expression to ONNX graph
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
        } catch (const std::exception& e) {
            // Skip equations that fail to convert
            std::cerr << "Warning: Skipping equation " << prefix << "[" << i << "] due to conversion error" << std::endl;
            continue;
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

        // Add source location metadata
        if (!eq.sourceFile.empty()) {
            auto* meta_file = eq_output->add_metadata_props();
            meta_file->set_key("source_file");
            meta_file->set_value(eq.sourceFile);

            auto* meta_line = eq_output->add_metadata_props();
            meta_line->set_key("source_line");
            meta_line->set_value(std::to_string(eq.sourceLine));
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

    // 2. Function call (e.g., der(), sin(), cos()) - check BEFORE plain variable
    // Must check functionCallArgs before treating componentReference as plain variable
    if (expr->functionCallArgs()) {
        // Get function name from componentReference if available,
        // otherwise extract from text (for der/initial/pure keywords)
        std::string funcName;
        if (expr->componentReference()) {
            funcName = expr->componentReference()->getText();
            std::cerr << "DEBUG: Function call with componentReference: " << funcName << std::endl;
        } else {
            // For der(), initial(), pure() which are keywords
            std::string text = expr->getText();
            size_t parenPos = text.find("(");
            if (parenPos == std::string::npos) {
                throw std::runtime_error("Malformed function call: " + text);
            }
            funcName = text.substr(0, parenPos);
            std::cerr << "DEBUG: Function call without componentReference: " << funcName << std::endl;
        }

        // Handle der() specially
        if (funcName == "der") {
            std::string text = expr->getText();
            // Extract variable name from der('varname')
            size_t parenPos = text.find("(");
            size_t start = parenPos + 1;
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

        // Map of supported math functions to ONNX operators
        static const std::map<std::string, std::string> mathFuncMap = {
            {"sin", "Sin"}, {"cos", "Cos"}, {"tan", "Tan"},
            {"asin", "Asin"}, {"acos", "Acos"}, {"atan", "Atan"},
            {"sinh", "Sinh"}, {"cosh", "Cosh"}, {"tanh", "Tanh"},
            {"exp", "Exp"}, {"log", "Log"}, {"sqrt", "Sqrt"},
            {"abs", "Abs"}, {"ceil", "Ceil"}, {"floor", "Floor"},
            {"sign", "Sign"}
        };

        auto it = mathFuncMap.find(funcName);
        if (it != mathFuncMap.end()) {
            // Get function arguments
            auto funcCallArgs = expr->functionCallArgs();
            auto funcArgs = funcCallArgs->functionArguments();
            if (!funcArgs || !funcArgs->expression()) {
                throw std::runtime_error("Function " + funcName + " requires arguments");
            }

            // Convert the first argument (for unary functions)
            std::string argTensor = convertExpression(funcArgs->expression(), graph, nodeCounter);

            // Create ONNX node for the math function
            auto* node = graph->add_node();
            std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

            node->set_op_type(it->second);  // ONNX operator name
            node->set_name(funcName + "_" + std::to_string(nodeCounter));
            node->add_input(argTensor);
            node->add_output(outputTensor);

            return outputTensor;
        }

        throw std::runtime_error("Unsupported function call: " + funcName);
    }

    // 3. Component reference (variable) - check AFTER function calls
    if (expr->componentReference()) {
        std::string varName = expr->componentReference()->getText();
        // Strip quotes if present
        if (varName.front() == '\'' && varName.back() == '\'') {
            varName = varName.substr(1, varName.size() - 2);
        }
        return varName;  // Variable inputs are already in the graph
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
