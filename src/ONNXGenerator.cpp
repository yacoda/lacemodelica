// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ONNXGenerator.h"
#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>
#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>
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

    // Add opset import (opset version 18) for default domain
    auto* opset = model.add_opset_import();
    // Don't set domain - it defaults to empty string
    opset->set_version(18);

    // Add opset import for lacemodelica domain (for custom functions)
    auto* lacemodelica_opset = model.add_opset_import();
    lacemodelica_opset->set_domain("lacemodelica");
    lacemodelica_opset->set_version(1);

    // Create the graph
    auto* graph = model.mutable_graph();
    graph->set_name(info.modelName);

    // Create ONNX inputs for each variable and parameter (skip derivatives and constants)
    // Constants with fixed variability become initializers instead
    // Track mapping from variable index to input index for start[] outputs
    std::map<size_t, int> varIndexToInputIndex;
    int inputIndex = 0;

    for (size_t i = 0; i < info.variables.size(); i++) {
        const auto& var = info.variables[i];

        // Skip derivative variables - der() will be an operator
        if (var.isDerivative) {
            continue;
        }

        // Constants with fixed variability become initializers, not inputs
        if (var.variability == "fixed" && !var.startValue.empty()) {
            auto* initializer = graph->add_initializer();
            initializer->set_name(var.name);
            initializer->set_data_type(onnx::TensorProto::DOUBLE);

            // Add dimensions (empty for scalars)
            for (const auto& dim : var.dimensions) {
                try {
                    initializer->add_dims(std::stoi(dim));
                } catch (...) {
                    // If dimension is symbolic, can't use as initializer
                    std::cerr << "Warning: Cannot create initializer for " << var.name
                              << " with symbolic dimension " << dim << std::endl;
                }
            }

            // Parse and store the constant value
            try {
                double value = std::stod(var.startValue);
                initializer->add_double_data(value);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse constant value for " << var.name
                          << ": " << var.startValue << std::endl;
                // Fallback to 0.0
                initializer->add_double_data(0.0);
            }

            continue;  // Don't add as input
        }

        varIndexToInputIndex[i] = inputIndex++;

        auto* input = graph->add_input();
        input->set_name(var.name);
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        // Set element type based on variable type
        if (var.type == "Boolean") {
            input_type->set_elem_type(onnx::TensorProto::BOOL);
        } else {
            input_type->set_elem_type(onnx::TensorProto::DOUBLE);
        }
        auto* input_shape = input_type->mutable_shape();

        // Handle array dimensions
        for (const auto& dim : var.dimensions) {
            auto* shape_dim = input_shape->add_dim();
            // Try to parse as integer, otherwise leave symbolic
            try {
                shape_dim->set_dim_value(std::stoi(dim));
            } catch (...) {
                shape_dim->set_dim_param(dim);
            }
        }
        // Scalars: empty dimensions results in empty shape []

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

    // Create ONNX FunctionProto for each function with algorithm
    for (const auto& func : info.functions) {
        if (!func.algorithmStatements.empty()) {
            std::cerr << "Creating FunctionProto for: " << func.name << std::endl;
            try {
                createFunctionProto(func, info, &model);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to create FunctionProto for " << func.name
                          << ": " << e.what() << std::endl;
            }
        }
    }

    // Create ONNX outputs for equations
    int nodeCounter = 0;

    // Track derivative inputs that need to be added (map from input name to shape info)
    std::map<std::string, std::vector<std::string>> derivativeInputs;

    // Generate outputs for regular equations
    generateEquationOutputs(info.equations, "eq", info, graph, nodeCounter, derivativeInputs);

    // Generate outputs for initial equations
    generateEquationOutputs(info.initialEquations, "init_eq", info, graph, nodeCounter, derivativeInputs);

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
                exprTensor = convertExpression(var.bindingContext, info, graph, nodeCounter, nullptr, &derivativeInputs);
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
            output_type->set_elem_type(onnx::TensorProto::DOUBLE);
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
            exprTensor = convertExpression(context, info, graph, nodeCounter, nullptr, &derivativeInputs);
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
        output_type->set_elem_type(onnx::TensorProto::DOUBLE);
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

    // Add derivative inputs at the end
    for (const auto& [derName, dimensions] : derivativeInputs) {
        auto* input = graph->add_input();
        input->set_name(derName);
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        input_type->set_elem_type(onnx::TensorProto::DOUBLE);
        auto* input_shape = input_type->mutable_shape();

        // Handle array dimensions
        for (const auto& dim : dimensions) {
            auto* shape_dim = input_shape->add_dim();
            // Try to parse as integer, otherwise leave symbolic
            try {
                shape_dim->set_dim_value(std::stoi(dim));
            } catch (...) {
                shape_dim->set_dim_param(dim);
            }
        }
        // Scalars: empty dimensions results in empty shape []
    }

    // Infer shapes for all tensors in the graph
    std::cout << "Inferring shapes..." << std::endl;
    try {
        onnx::shape_inference::InferShapes(model);
        std::cout << "Shape inference successful" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Shape inference failed: " << e.what() << std::endl;
        std::cerr << "Continuing with partial shape information..." << std::endl;
    }

    // Validate ONNX model before serialization
    std::cout << "Validating ONNX model..." << std::endl;
    try {
        // Don't validate opset compatibility for custom functions
        onnx::checker::check_model(model, false, false, false);
        std::cout << "ONNX model validation successful" << std::endl;
    } catch (const onnx::checker::ValidationError& e) {
        std::cerr << "Warning: ONNX validation failed: " << e.what() << std::endl;
        std::cerr << "Continuing anyway for models with custom functions..." << std::endl;
        // For now, just warn - custom functions aren't fully supported by validator
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
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    std::map<std::string, std::vector<std::string>>& derivativeInputs) {

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
                lhsTensor = convertExpression(eq.lhsContext, info, graph, nodeCounter, nullptr, &derivativeInputs);
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
                rhsTensor = convertExpression(eq.rhsContext, info, graph, nodeCounter, nullptr, &derivativeInputs);
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

        // Create equation residual: LHS - RHS (or LHS == RHS for booleans)
        // For numerical evaluation, we want the residual (should be zero when equation is satisfied)
        std::string eqOutputName = prefix + "[" + std::to_string(i) + "]";

        // Check if LHS is a boolean variable
        std::string lhsText = eq.lhsContext->getText();
        // Strip quotes if present
        if (lhsText.front() == '\'' && lhsText.back() == '\'') {
            lhsText = lhsText.substr(1, lhsText.length() - 2);
        }
        const Variable* lhsVar = info.findVariable(lhsText);
        bool isBooleanEquation = (lhsVar && lhsVar->type == "Boolean");

        if (isBooleanEquation) {
            std::cerr << "DEBUG: Equation " << i << " is boolean: " << lhsText << std::endl;
        }

        auto* eq_node = graph->add_node();
        if (isBooleanEquation) {
            eq_node->set_op_type("Equal");  // For boolean equations, check equality
        } else {
            eq_node->set_op_type("Sub");  // For numeric equations, subtract to get residual
        }
        eq_node->set_name(prefix + "_residual_" + std::to_string(i));
        eq_node->add_input(lhsTensor);
        eq_node->add_input(rhsTensor);
        eq_node->add_output(eqOutputName);

        // Create output for this equation
        auto* eq_output = graph->add_output();
        eq_output->set_name(eqOutputName);
        auto* eq_type = eq_output->mutable_type()->mutable_tensor_type();
        if (isBooleanEquation) {
            eq_type->set_elem_type(onnx::TensorProto::BOOL);  // Boolean equation result
        } else {
            eq_type->set_elem_type(onnx::TensorProto::DOUBLE);  // Numeric residual
        }
        // Don't create shape - let shape inference fill it in

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

// Helper function to add an Identity node if tensor name differs from desired name
// Returns the final tensor name (either original or the new name after Identity)
static std::string ensureTensorName(
    onnx::FunctionProto* functionProto,
    const std::string& tensorName,
    const std::string& desiredName) {

    if (tensorName == desiredName) {
        return tensorName;  // No Identity needed
    }

    // Add Identity node to rename the tensor
    auto* identityNode = functionProto->add_node();
    identityNode->set_op_type("Identity");
    identityNode->set_name("output_" + desiredName);
    identityNode->add_input(tensorName);
    identityNode->add_output(desiredName);

    return desiredName;
}

void ONNXGenerator::createFunctionProto(
    const Function& func,
    const ModelInfo& info,
    onnx::ModelProto* model) {

    std::cerr << "Creating ONNX FunctionProto for '" << func.name << "'" << std::endl;

    // Create a new FunctionProto
    auto* functionProto = model->add_functions();
    functionProto->set_name(func.name);
    functionProto->set_domain("lacemodelica");  // Custom domain for user functions

    // Add opset import for standard ONNX operators used in the function
    auto* func_opset = functionProto->add_opset_import();
    func_opset->set_version(18);  // Default domain (empty string)

    // Add function inputs
    for (const auto& input : func.inputs) {
        functionProto->add_input(input.name);
    }

    // Note: We'll add outputs AFTER processing statements,
    // so we can use the actual tensor names produced

    // Map from variable name to current ONNX tensor name
    std::map<std::string, std::string> variableToTensor;

    // Initialize map with function inputs (they map to themselves in the function scope)
    for (const auto& input : func.inputs) {
        variableToTensor[input.name] = input.name;
    }

    int nodeCounter = 0;

    // Process each algorithm statement in order
    for (size_t stmtIndex = 0; stmtIndex < func.algorithmStatements.size(); stmtIndex++) {
        const auto& stmt = func.algorithmStatements[stmtIndex];

        // Extract LHS variable name from componentReference
        std::string lhsVarName = stmt.lhsContext->getText();
        // Strip quotes if present
        if (lhsVarName.size() >= 2 && lhsVarName.front() == '\'' && lhsVarName.back() == '\'') {
            lhsVarName = lhsVarName.substr(1, lhsVarName.size() - 2);
        }

        std::cerr << "  Processing statement " << stmtIndex << ": " << lhsVarName << " := "
                  << stmt.rhsContext->getText().substr(0, 50) << "..." << std::endl;

        try {
            // Remember the current node count to identify newly created nodes
            int startNodeCount = functionProto->node_size();

            // Create a temporary graph to build the expression nodes
            // We'll add them to the function proto instead
            onnx::GraphProto tempGraph;
            std::map<std::string, std::vector<std::string>> localDerivativeInputs;
            std::string rhsTensor = convertExpression(stmt.rhsContext, info, &tempGraph, nodeCounter, &variableToTensor, &localDerivativeInputs);

            // Move nodes from tempGraph to functionProto
            for (int i = 0; i < tempGraph.node_size(); i++) {
                auto* node = functionProto->add_node();
                node->CopyFrom(tempGraph.node(i));

                // Add source location metadata
                auto* meta_file = node->add_metadata_props();
                meta_file->set_key("source_file");
                meta_file->set_value(stmt.sourceFile);

                auto* meta_line = node->add_metadata_props();
                meta_line->set_key("source_line");
                meta_line->set_value(std::to_string(stmt.sourceLine));

                auto* meta_index = node->add_metadata_props();
                meta_index->set_key("statement_index");
                meta_index->set_value(std::to_string(stmtIndex));

                auto* meta_lhs = node->add_metadata_props();
                meta_lhs->set_key("lhs_variable");
                meta_lhs->set_value(lhsVarName);
            }

            // Note: FunctionProto doesn't support initializers (constants)
            // Constants should be represented as Constant nodes instead

            // Store result in map
            variableToTensor[lhsVarName] = rhsTensor;

            std::cerr << "  Mapped " << lhsVarName << " -> " << rhsTensor << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to convert statement for " << lhsVarName;
            if (!stmt.sourceFile.empty()) {
                std::cerr << " (" << stmt.sourceFile << ":" << stmt.sourceLine << ")";
            }
            std::cerr << ": " << e.what() << std::endl;
            throw;
        }
    }

    // Add function outputs with meaningful names
    for (const auto& output : func.outputs) {
        auto it = variableToTensor.find(output.name);
        if (it == variableToTensor.end()) {
            throw std::runtime_error("Output variable " + output.name + " not computed in algorithm");
        }

        std::string internalTensor = it->second;
        std::string outputName = output.name;

        // Ensure the tensor has the desired output name (adds Identity if needed)
        std::string finalName = ensureTensorName(functionProto, internalTensor, outputName);

        // Add the output variable name as the function output
        functionProto->add_output(finalName);
        std::cerr << "  Function output: " << finalName << std::endl;
    }
}

// Helper function to collect all function arguments from recursive structure
static std::vector<basemodelica::BaseModelicaParser::ExpressionContext*>
collectFunctionArguments(basemodelica::BaseModelicaParser::FunctionArgumentsContext* funcArgs) {
    std::vector<basemodelica::BaseModelicaParser::ExpressionContext*> arguments;

    if (!funcArgs) {
        return arguments;
    }

    // Get the first argument
    auto firstExpr = funcArgs->expression();
    if (firstExpr) {
        arguments.push_back(firstExpr);
    }

    // Recursively collect remaining arguments
    auto nonFirst = funcArgs->functionArgumentsNonFirst();
    while (nonFirst) {
        auto funcArg = nonFirst->functionArgument();
        if (funcArg) {
            auto expr = funcArg->expression();
            if (expr) {
                arguments.push_back(expr);
            }
        }
        nonFirst = nonFirst->functionArgumentsNonFirst();
    }

    return arguments;
}

// Expression conversion functions

std::string ONNXGenerator::convertExpression(
    antlr4::ParserRuleContext* expr,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    const std::map<std::string, std::string>* variableMap,
    std::map<std::string, std::vector<std::string>>* derivativeInputs) {

    if (!expr) {
        throw std::runtime_error("Null expression context");
    }

    // Try to cast to specific expression types
    if (auto* exprCtx = dynamic_cast<basemodelica::BaseModelicaParser::ExpressionContext*>(expr)) {
        // Expression -> ExpressionNoDecoration -> (IfExpression | SimpleExpression)
        auto* exprNoDecoration = exprCtx->expressionNoDecoration();
        if (exprNoDecoration) {
            // Check for if expression first
            auto* ifExpr = exprNoDecoration->ifExpression();
            if (ifExpr) {
                return convertIfExpression(ifExpr, info, graph, nodeCounter, variableMap, derivativeInputs);
            }

            auto* simpleExpr = exprNoDecoration->simpleExpression();
            if (simpleExpr) {
                return convertSimpleExpression(simpleExpr, info, graph, nodeCounter, variableMap, derivativeInputs);
            }
        }
    } else if (auto* exprNoDecoration = dynamic_cast<basemodelica::BaseModelicaParser::ExpressionNoDecorationContext*>(expr)) {
        // Handle ExpressionNoDecoration directly (e.g., from if expression conditions)
        auto* ifExpr = exprNoDecoration->ifExpression();
        if (ifExpr) {
            return convertIfExpression(ifExpr, info, graph, nodeCounter, variableMap, derivativeInputs);
        }

        auto* simpleExpr = exprNoDecoration->simpleExpression();
        if (simpleExpr) {
            return convertSimpleExpression(simpleExpr, info, graph, nodeCounter, variableMap, derivativeInputs);
        }
    } else if (auto* simpleExpr = dynamic_cast<basemodelica::BaseModelicaParser::SimpleExpressionContext*>(expr)) {
        return convertSimpleExpression(simpleExpr, info, graph, nodeCounter, variableMap, derivativeInputs);
    }

    // Fallback: return placeholder
    throw std::runtime_error("Unsupported expression type: " + expr->getText());
}

std::string ONNXGenerator::convertIfExpression(
    basemodelica::BaseModelicaParser::IfExpressionContext* expr,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    const std::map<std::string, std::string>* variableMap,
    std::map<std::string, std::vector<std::string>>* derivativeInputs) {

    std::cerr << "Converting if expression" << std::endl;

    // Get all expressions: [condition, then, [elseif cond, elseif then]*, else]
    auto expressions = expr->expressionNoDecoration();

    // Need at least 3: if condition, then branch, else branch
    if (expressions.size() < 3) {
        throw std::runtime_error("Invalid if expression structure");
    }

    // For now, handle simple if-then-else (no elseif)
    if (expressions.size() > 3) {
        // Has elseif clauses - would need nested If nodes
        throw std::runtime_error("elseif clauses not yet supported in if expressions");
    }

    // Convert condition expression (should produce boolean tensor)
    std::string condTensor = convertExpression(expressions[0], info, graph, nodeCounter, variableMap, derivativeInputs);

    // Create then branch as a subgraph
    // Note: If subgraphs have zero inputs - they access parent scope variables directly
    // IMPORTANT: Share nodeCounter with subgraphs to maintain SSA (no duplicate tensor names)
    onnx::GraphProto thenBranch;
    thenBranch.set_name("then_branch");
    std::string thenResult = convertExpression(expressions[1], info, &thenBranch, nodeCounter, variableMap, derivativeInputs);

    // Add output to then branch
    auto* thenOutput = thenBranch.add_output();
    thenOutput->set_name(thenResult);
    auto* thenType = thenOutput->mutable_type()->mutable_tensor_type();
    thenType->set_elem_type(onnx::TensorProto::DOUBLE);
    auto* thenShape = thenType->mutable_shape();
    thenShape->add_dim()->set_dim_value(1);

    // Create else branch as a subgraph
    // Note: If subgraphs have zero inputs - they access parent scope variables directly
    // IMPORTANT: Share nodeCounter with subgraphs to maintain SSA (no duplicate tensor names)
    onnx::GraphProto elseBranch;
    elseBranch.set_name("else_branch");
    std::string elseResult = convertExpression(expressions[2], info, &elseBranch, nodeCounter, variableMap, derivativeInputs);

    // Add output to else branch
    auto* elseOutput = elseBranch.add_output();
    elseOutput->set_name(elseResult);
    auto* elseType = elseOutput->mutable_type()->mutable_tensor_type();
    elseType->set_elem_type(onnx::TensorProto::DOUBLE);
    auto* elseShape = elseType->mutable_shape();
    elseShape->add_dim()->set_dim_value(1);

    // Create If node
    auto* ifNode = graph->add_node();
    std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

    ifNode->set_op_type("If");
    ifNode->set_name("If_" + std::to_string(nodeCounter));
    ifNode->add_input(condTensor);
    ifNode->add_output(outputTensor);

    // Add then_branch attribute
    auto* thenAttr = ifNode->add_attribute();
    thenAttr->set_name("then_branch");
    thenAttr->set_type(onnx::AttributeProto::GRAPH);
    thenAttr->mutable_g()->CopyFrom(thenBranch);

    // Add else_branch attribute
    auto* elseAttr = ifNode->add_attribute();
    elseAttr->set_name("else_branch");
    elseAttr->set_type(onnx::AttributeProto::GRAPH);
    elseAttr->mutable_g()->CopyFrom(elseBranch);

    std::cerr << "Created If node with output: " << outputTensor << std::endl;

    return outputTensor;
}

std::string ONNXGenerator::convertSimpleExpression(
    basemodelica::BaseModelicaParser::SimpleExpressionContext* expr,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    const std::map<std::string, std::string>* variableMap,
    std::map<std::string, std::vector<std::string>>* derivativeInputs) {

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

    // Handle multiple logical factors connected by 'and'
    std::string resultTensor;
    for (size_t i = 0; i < logicalFactors.size(); i++) {
        auto* logicalFactor = logicalFactors[i];
        auto* relation = logicalFactor->relation();
        if (!relation) {
            throw std::runtime_error("No relation in logical factor");
        }

        // Convert this logical factor to a tensor
        std::string factorTensor = convertRelation(relation, info, graph, nodeCounter, variableMap, derivativeInputs);

        // Check if logical factor has 'not' prefix
        // If 'not' is present, logicalFactor will have 2 children: 'not' token and relation
        // If 'not' is absent, it will have 1 child: just the relation
        if (logicalFactor->children.size() > 1) {
            // Apply NOT operator
            auto* notNode = graph->add_node();
            notNode->set_op_type("Not");
            notNode->set_name("Not_" + std::to_string(nodeCounter));
            notNode->add_input(factorTensor);
            factorTensor = "tensor_" + std::to_string(nodeCounter++);
            notNode->add_output(factorTensor);
        }

        if (i == 0) {
            // First factor
            resultTensor = factorTensor;
        } else {
            // Subsequent factors - AND with previous result
            auto* andNode = graph->add_node();
            andNode->set_op_type("And");
            andNode->set_name("And_" + std::to_string(nodeCounter));
            andNode->add_input(resultTensor);
            andNode->add_input(factorTensor);
            resultTensor = "tensor_" + std::to_string(nodeCounter++);
            andNode->add_output(resultTensor);
        }
    }

    return resultTensor;
}

// Helper function to convert a relation to ONNX
std::string ONNXGenerator::convertRelation(
    basemodelica::BaseModelicaParser::RelationContext* relation,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    const std::map<std::string, std::string>* variableMap,
    std::map<std::string, std::vector<std::string>>* derivativeInputs) {

    auto arithmeticExprs = relation->arithmeticExpression();
    if (arithmeticExprs.empty()) {
        throw std::runtime_error("No arithmetic expression in relation");
    }

    // Check if there's a relational operator (comparison)
    if (arithmeticExprs.size() > 1) {
        // There's a comparison operator
        auto relOp = relation->relationalOperator();
        if (!relOp) {
            throw std::runtime_error("Multiple arithmetic expressions but no relational operator");
        }

        // Convert left and right operands
        std::string leftTensor = convertArithmeticExpression(arithmeticExprs[0], info, graph, nodeCounter, variableMap, derivativeInputs);
        std::string rightTensor = convertArithmeticExpression(arithmeticExprs[1], info, graph, nodeCounter, variableMap, derivativeInputs);

        // Create comparison node
        auto* node = graph->add_node();
        std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

        std::string opText = relOp->getText();
        if (opText == ">") {
            node->set_op_type("Greater");
        } else if (opText == "<") {
            node->set_op_type("Less");
        } else if (opText == ">=") {
            node->set_op_type("GreaterOrEqual");
        } else if (opText == "<=") {
            node->set_op_type("LessOrEqual");
        } else if (opText == "==") {
            node->set_op_type("Equal");
        } else if (opText == "<>") {
            // Not equal: first do Equal, then Not the result
            node->set_op_type("Equal");
            node->set_name("Equal_" + std::to_string(nodeCounter));
            node->add_input(leftTensor);
            node->add_input(rightTensor);
            std::string equalOutput = "tensor_" + std::to_string(nodeCounter++);
            node->add_output(equalOutput);

            // Now negate the result
            auto* notNode = graph->add_node();
            notNode->set_op_type("Not");
            notNode->set_name("Not_" + std::to_string(nodeCounter));
            notNode->add_input(equalOutput);
            notNode->add_output(outputTensor);

            return outputTensor;
        } else {
            throw std::runtime_error("Unsupported relational operator: " + opText);
        }

        node->set_name(node->op_type() + "_" + std::to_string(nodeCounter));
        node->add_input(leftTensor);
        node->add_input(rightTensor);
        node->add_output(outputTensor);

        return outputTensor;
    }

    // No comparison, just return the arithmetic expression
    return convertArithmeticExpression(arithmeticExprs[0], info, graph, nodeCounter, variableMap, derivativeInputs);
}

std::string ONNXGenerator::convertArithmeticExpression(
    basemodelica::BaseModelicaParser::ArithmeticExpressionContext* expr,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    const std::map<std::string, std::string>* variableMap,
    std::map<std::string, std::vector<std::string>>* derivativeInputs) {

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
    std::string result = convertTerm(terms[termIndex++], info, graph, nodeCounter, variableMap, derivativeInputs);

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
        std::string rightTensor = convertTerm(terms[termIndex], info, graph, nodeCounter, variableMap, derivativeInputs);
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

// Helper function to check if a tensor represents a matrix (has 2 dimensions)
static bool isMatrixVariable(const std::string& tensorName, const ModelInfo& info) {
    // Check if tensor name matches a variable name
    auto it = info.variableIndex.find(tensorName);
    if (it != info.variableIndex.end()) {
        const auto& var = info.variables[it->second];
        return var.dimensions.size() == 2;
    }
    return false;
}

std::string ONNXGenerator::convertTerm(
    basemodelica::BaseModelicaParser::TermContext* expr,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    const std::map<std::string, std::string>* variableMap,
    std::map<std::string, std::vector<std::string>>* derivativeInputs) {

    // Term: factor (mulOperator factor)*
    auto factors = expr->factor();
    auto mulOps = expr->mulOperator();

    if (factors.empty()) {
        throw std::runtime_error("Empty term");
    }

    // Convert first factor
    std::string result = convertFactor(factors[0], info, graph, nodeCounter, variableMap, derivativeInputs);

    // Process remaining factors with operators
    for (size_t i = 0; i < mulOps.size(); i++) {
        std::string opText = mulOps[i]->getText();
        std::string rightTensor = convertFactor(factors[i + 1], info, graph, nodeCounter, variableMap, derivativeInputs);

        // Create ONNX node for the operation
        auto* node = graph->add_node();
        std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

        if (opText == "*") {
            // Check if both operands are matrices (2D arrays)
            // If so, use MatMul for matrix multiplication; otherwise use Mul
            bool leftIsMatrix = isMatrixVariable(result, info);
            bool rightIsMatrix = isMatrixVariable(rightTensor, info);

            if (leftIsMatrix && rightIsMatrix) {
                node->set_op_type("MatMul");
            } else {
                node->set_op_type("Mul");
            }
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
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    const std::map<std::string, std::string>* variableMap,
    std::map<std::string, std::vector<std::string>>* derivativeInputs) {

    // Factor: primary (('^' | '.^') primary)?
    auto primaries = expr->primary();

    if (primaries.empty()) {
        throw std::runtime_error("Empty factor");
    }

    std::string result = convertPrimary(primaries[0], info, graph, nodeCounter, variableMap, derivativeInputs);

    // Handle power operator if present
    if (primaries.size() > 1) {
        std::string exponentTensor = convertPrimary(primaries[1], info, graph, nodeCounter, variableMap, derivativeInputs);

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
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    const std::map<std::string, std::string>* variableMap,
    std::map<std::string, std::vector<std::string>>* derivativeInputs) {

    // Handle different primary types

    // 1. Number literal
    if (expr->UNSIGNED_NUMBER()) {
        std::string value = expr->UNSIGNED_NUMBER()->getText();

        // Create constant tensor (scalar - empty dims for rank 0)
        auto* constant = graph->add_initializer();
        std::string constName = "const_" + std::to_string(nodeCounter++);
        constant->set_name(constName);
        constant->set_data_type(onnx::TensorProto::DOUBLE);
        // Scalars have empty dims (rank 0)
        constant->add_double_data(std::stod(value));

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
            // Strip quotes if present
            if (funcName.size() >= 2 && funcName.front() == '\'' && funcName.back() == '\'') {
                funcName = funcName.substr(1, funcName.size() - 2);
            }
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
            // Get the argument expression to der()
            auto funcCallArgs = expr->functionCallArgs();
            if (!funcCallArgs) {
                throw std::runtime_error("der() requires an argument");
            }

            auto funcArgs = funcCallArgs->functionArguments();
            if (!funcArgs) {
                throw std::runtime_error("der() requires an argument");
            }

            // Get the first (and only) argument
            auto argExpr = funcArgs->expression();
            if (!argExpr) {
                throw std::runtime_error("der() argument is missing");
            }

            // Check if the argument is a simple variable reference
            // Use ANTLR tree API: check if this is a simple single-path tree (just an identifier)
            bool isSimpleVariable = false;
            std::string varName;

            // Check if expression has only one child path down to a single token
            if (argExpr->children.size() == 1) {
                antlr4::tree::ParseTree* node = argExpr->children[0];
                // Keep following single-child nodes until we reach a terminal or multiple children
                while (node && node->children.size() == 1) {
                    node = node->children[0];
                }
                // If we end at a terminal (leaf node), it's a simple variable
                if (node && node->children.empty()) {
                    isSimpleVariable = true;
                    varName = argExpr->getText();
                    // Strip quotes if present
                    if (varName.size() >= 2 && varName.front() == '\'' && varName.back() == '\'') {
                        varName = varName.substr(1, varName.size() - 2);
                    }
                }
            }

            if (isSimpleVariable && derivativeInputs) {
                // Create a derivative input name (use quotes for consistency with test format)
                std::string derInputName = "der('" + varName + "')";

                // Add to derivative inputs if not already present
                if (derivativeInputs->find(derInputName) == derivativeInputs->end()) {
                    // Get dimensions from the variable info
                    std::vector<std::string> dimensions;
                    const Variable* var = info.findVariable(varName);
                    if (var) {
                        dimensions = var->dimensions;
                        std::cerr << "DEBUG: der(" << varName << ") has " << dimensions.size() << " dimensions: ";
                        for (const auto& d : dimensions) {
                            std::cerr << d << " ";
                        }
                        std::cerr << std::endl;
                    } else {
                        std::cerr << "DEBUG: Could not find variable " << varName << " for der()" << std::endl;
                    }
                    (*derivativeInputs)[derInputName] = dimensions;
                }

                // Return the input name directly
                return derInputName;
            } else {
                // Not a simple variable - convert the expression and create a Der node
                std::string inputTensor = convertExpression(argExpr, info, graph, nodeCounter, variableMap, derivativeInputs);

                // Create a Der operator node
                auto* node = graph->add_node();
                std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

                node->set_op_type("Der");
                node->set_domain("lacemodelica");  // Custom operator in our domain
                node->set_name("Der_" + std::to_string(nodeCounter));
                node->add_input(inputTensor);  // Input is the tensor from the expression
                node->add_output(outputTensor);

                return outputTensor;
            }
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
            std::string argTensor = convertExpression(funcArgs->expression(), info, graph, nodeCounter, variableMap, derivativeInputs);

            // Create ONNX node for the math function
            auto* node = graph->add_node();
            std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

            node->set_op_type(it->second);  // ONNX operator name
            node->set_name(funcName + "_" + std::to_string(nodeCounter));
            node->add_input(argTensor);
            node->add_output(outputTensor);

            return outputTensor;
        }

        // Check if this is a user-defined function with algorithm
        const Function* func = info.findFunction(funcName);
        if (func && !func->algorithmStatements.empty()) {
            std::cerr << "DEBUG: Found user-defined function: " << funcName << std::endl;

            // Get function arguments from call site
            auto funcCallArgs = expr->functionCallArgs();
            auto funcArgs = funcCallArgs->functionArguments();

            if (!funcArgs) {
                throw std::runtime_error("Function " + funcName + " requires arguments");
            }

            // Collect all function arguments
            auto arguments = collectFunctionArguments(funcArgs);

            // Check argument count
            if (arguments.size() != func->inputs.size()) {
                throw std::runtime_error("Function " + funcName + " expects " +
                    std::to_string(func->inputs.size()) + " arguments, got " +
                    std::to_string(arguments.size()));
            }

            // Convert each argument expression to get tensor names
            std::vector<std::string> argTensors;
            for (size_t i = 0; i < arguments.size(); i++) {
                std::string argTensor = convertExpression(arguments[i], info, graph, nodeCounter, variableMap, derivativeInputs);
                argTensors.push_back(argTensor);
                std::cerr << "  Arg " << i << ": " << func->inputs[i].name << " = " << argTensor << std::endl;
            }

            // Create a function call node that references the FunctionProto
            auto* node = graph->add_node();
            std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);

            node->set_op_type(funcName);  // op_type is the function name
            node->set_domain("lacemodelica");  // Custom domain where function is defined
            node->set_name(funcName + "_call_" + std::to_string(nodeCounter));

            // Add inputs (argument tensors)
            for (const auto& argTensor : argTensors) {
                node->add_input(argTensor);
            }

            // Add outputs (for now, assume single output)
            node->add_output(outputTensor);

            std::cerr << "  Created function call node: " << funcName << " -> " << outputTensor << std::endl;

            return outputTensor;
        }

        throw std::runtime_error("Unsupported function call: " + funcName);
    }

    // 3. Component reference (variable) - check AFTER function calls
    if (expr->componentReference()) {
        auto compRef = expr->componentReference();

        // Get base variable name (first IDENT)
        std::string varName = compRef->IDENT(0)->getText();
        // Strip quotes if present
        if (varName.size() >= 2 && varName.front() == '\'' && varName.back() == '\'') {
            varName = varName.substr(1, varName.size() - 2);
        }

        // Get the base tensor (from inputs or variableMap)
        std::string baseTensor;
        if (variableMap && variableMap->find(varName) != variableMap->end()) {
            baseTensor = variableMap->at(varName);
        } else {
            baseTensor = varName;  // Variable inputs are already in the graph
        }

        // Check for array subscripts
        auto subscripts = compRef->arraySubscripts();
        if (subscripts.empty()) {
            // No subscripts, just return the variable
            return baseTensor;
        }

        // Handle array indexing with subscripts
        // Use GatherND for multi-dimensional indexing
        auto arraySubscript = subscripts[0];  // First (and likely only) arraySubscripts
        auto subscriptList = arraySubscript->subscript();

        // Collect all indices (convert from 1-based Modelica to 0-based ONNX)
        std::vector<int64_t> indices;
        for (auto sub : subscriptList) {
            // Check if it's a colon (full slice) or expression
            if (sub->getText() == ":") {
                throw std::runtime_error("Array slice ':' not yet supported in ONNX conversion");
            }

            // It's an expression - evaluate it
            auto subExpr = sub->expression();
            if (!subExpr) {
                throw std::runtime_error("Invalid array subscript");
            }

            // Convert the subscript expression (should yield a constant)
            std::string indexExpr = subExpr->getText();

            // Try to parse as integer constant (Modelica uses 1-based indexing)
            try {
                int modelicaIndex = std::stoi(indexExpr);
                int onnxIndex = modelicaIndex - 1;  // Convert to 0-based
                indices.push_back(onnxIndex);
            } catch (const std::exception& e) {
                throw std::runtime_error("Array subscript must be constant integer, got: " + indexExpr);
            }
        }

        // Create a Constant node for the indices tensor
        auto* constNode = graph->add_node();
        std::string indexTensor = "const_" + std::to_string(nodeCounter++);
        constNode->set_op_type("Constant");
        constNode->set_name(indexTensor);
        constNode->add_output(indexTensor);

        auto* attr = constNode->add_attribute();
        attr->set_name("value");
        attr->set_type(onnx::AttributeProto::TENSOR);
        auto* t = attr->mutable_t();
        t->set_data_type(onnx::TensorProto::INT64);
        // Shape is [num_indices] for GatherND
        t->add_dims(indices.size());
        for (auto idx : indices) {
            t->add_int64_data(idx);
        }

        // Create GatherND node to extract the element
        auto* gatherNode = graph->add_node();
        std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);
        gatherNode->set_op_type("GatherND");
        gatherNode->set_name("GatherND_" + std::to_string(nodeCounter));
        gatherNode->add_input(baseTensor);
        gatherNode->add_input(indexTensor);
        gatherNode->add_output(outputTensor);

        return outputTensor;
    }

    // 4. Parenthesized expression
    if (expr->outputExpressionList()) {
        // (expression) - just convert the expression inside
        auto outputList = expr->outputExpressionList();
        auto expressions = outputList->expression();
        if (!expressions.empty()) {
            return convertExpression(expressions[0], info, graph, nodeCounter, variableMap, derivativeInputs);
        }
    }

    throw std::runtime_error("Unsupported primary expression: " + expr->getText());
}

} // namespace lacemodelica
