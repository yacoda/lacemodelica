// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ONNXGenerator.h"
#include "ExpressionConverter.h"
#include "EquationGenerator.h"
#include "GraphBuilder.h"
#include "ONNXHelpers.hpp"
#include "Utils.hpp"

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

// -----------------------------------------------------------------------------
// Bound Output Generation
// -----------------------------------------------------------------------------

// Context for generating variable bound outputs (start, min, max)
struct BoundOutputContext {
    onnx::GraphProto* graph;
    const ConversionContext& convCtx;
    const std::map<size_t, int>& varIndexToInputIndex;
};

// Generate an ONNX output for a variable bound (start, min, or max value)
static void generateBoundOutput(
    const BoundOutputContext& ctx,
    size_t varIndex,
    const Variable& var,
    antlr4::ParserRuleContext* exprContext,
    const std::string& boundType,
    const std::string& description) {

    auto it = ctx.varIndexToInputIndex.find(varIndex);
    if (it == ctx.varIndexToInputIndex.end()) {
        std::cerr << "Warning: Could not find input index for variable " << var.name << std::endl;
        return;
    }
    int inputIdx = it->second;

    std::string exprTensor;
    try {
        exprTensor = ExpressionConverter::convert(exprContext, ctx.convCtx);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to convert " << boundType << " expression for " << var.name;
        if (!var.sourceFile.empty()) {
            std::cerr << " (" << var.sourceFile << ":" << var.sourceLine << ")";
        }
        std::cerr << ": " << e.what() << std::endl;
        return;
    }

    std::string outputName = boundType + "[" + std::to_string(inputIdx) + "]";
    auto* output = ctx.graph->add_output();
    output->set_name(outputName);
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(onnx::TensorProto::DOUBLE);
    output_type->mutable_shape()->add_dim()->set_dim_value(1);

    output->set_doc_string(description + " for " + var.name);
    addSourceLocationMetadata(output, var.sourceFile, var.sourceLine);

    auto* meta_var = output->add_metadata_props();
    meta_var->set_key("variable_name");
    meta_var->set_value(var.name);

    auto* meta_vr = output->add_metadata_props();
    meta_vr->set_key("value_reference");
    meta_vr->set_value(std::to_string(var.valueReference));

    auto* meta_idx = output->add_metadata_props();
    meta_idx->set_key("input_index");
    meta_idx->set_value(std::to_string(inputIdx));

    ctx.convCtx.builder().renameTensor(exprTensor, outputName,
                                        boundType + "_identity_" + std::to_string(inputIdx));
}

// -----------------------------------------------------------------------------
// Function Proto Generation
// -----------------------------------------------------------------------------

// Add Identity node if tensor name differs from desired name
static std::string ensureTensorName(
    onnx::FunctionProto* functionProto,
    const std::string& tensorName,
    const std::string& desiredName) {

    if (tensorName == desiredName) {
        return tensorName;
    }

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

    auto* functionProto = model->add_functions();
    functionProto->set_name(func.name);
    functionProto->set_domain("lacemodelica");

    auto* func_opset = functionProto->add_opset_import();
    func_opset->set_version(18);

    // Also import the lacemodelica domain for functions that may call other custom functions
    auto* lace_opset = functionProto->add_opset_import();
    lace_opset->set_domain("lacemodelica");
    lace_opset->set_version(1);

    for (const auto& input : func.inputs) {
        functionProto->add_input(input.name);
    }

    std::map<std::string, std::string> variableToTensor;
    for (const auto& input : func.inputs) {
        variableToTensor[input.name] = input.name;
    }

    int nodeCounter = 0;

    for (size_t stmtIndex = 0; stmtIndex < func.algorithmStatements.size(); stmtIndex++) {
        const auto& stmt = func.algorithmStatements[stmtIndex];

        // Check if this is a multi-output assignment: (a, b) := func(x)
        auto* lhsOutputList = dynamic_cast<basemodelica::BaseModelicaParser::OutputExpressionListContext*>(stmt.lhsContext);
        if (lhsOutputList) {
            // Multi-output assignment
            auto* stmtCtx = dynamic_cast<basemodelica::BaseModelicaParser::StatementContext*>(stmt.rhsContext);
            if (!stmtCtx) {
                throw std::runtime_error("Invalid multi-output statement structure");
            }

            // Extract function name and arguments
            auto* funcCompRef = stmtCtx->componentReference();
            auto* funcCallArgs = stmtCtx->functionCallArgs();
            if (!funcCompRef || !funcCallArgs) {
                throw std::runtime_error("Multi-output statement missing function call");
            }

            std::string funcName = stripQuotes(funcCompRef->IDENT(0)->getText());

            // Extract LHS variable names
            std::vector<std::string> outputVarNames;
            for (auto* expr : lhsOutputList->expression()) {
                outputVarNames.push_back(stripQuotes(expr->getText()));
            }

            try {
                onnx::GraphProto tempGraph;
                std::map<std::string, std::vector<std::string>> localDerivativeInputs;
                ConversionContext funcCtx(info, &tempGraph, nodeCounter, &variableToTensor, &localDerivativeInputs);

                // Convert function arguments
                auto* funcArgs = funcCallArgs->functionArguments();
                std::vector<std::string> argTensors;
                if (funcArgs) {
                    if (auto firstExpr = funcArgs->expression()) {
                        argTensors.push_back(ExpressionConverter::convert(firstExpr, funcCtx));
                    }
                    auto nonFirst = funcArgs->functionArgumentsNonFirst();
                    while (nonFirst) {
                        if (auto funcArg = nonFirst->functionArgument()) {
                            if (auto argExpr = funcArg->expression()) {
                                argTensors.push_back(ExpressionConverter::convert(argExpr, funcCtx));
                            }
                        }
                        nonFirst = nonFirst->functionArgumentsNonFirst();
                    }
                }

                // Create function call node with multiple outputs
                auto* callNode = tempGraph.add_node();
                callNode->set_op_type(funcName);
                callNode->set_domain("lacemodelica");
                callNode->set_name(funcName + "_call_" + std::to_string(nodeCounter++));

                for (const auto& argTensor : argTensors) {
                    callNode->add_input(argTensor);
                }

                std::vector<std::string> outputTensors;
                for (size_t i = 0; i < outputVarNames.size(); i++) {
                    std::string outputTensor = "tensor_" + std::to_string(nodeCounter++);
                    callNode->add_output(outputTensor);
                    outputTensors.push_back(outputTensor);
                }

                // Copy initializers and nodes to functionProto
                for (int i = 0; i < tempGraph.initializer_size(); i++) {
                    const auto& init = tempGraph.initializer(i);
                    auto* constNode = functionProto->add_node();
                    constNode->set_op_type("Constant");
                    constNode->set_name(init.name());
                    constNode->add_output(init.name());

                    auto* attr = constNode->add_attribute();
                    attr->set_name("value");
                    attr->set_type(onnx::AttributeProto::TENSOR);
                    attr->mutable_t()->CopyFrom(init);

                    addSourceLocationMetadata(constNode, stmt.sourceFile, stmt.sourceLine);
                }

                for (int i = 0; i < tempGraph.node_size(); i++) {
                    auto* node = functionProto->add_node();
                    node->CopyFrom(tempGraph.node(i));
                    addSourceLocationMetadata(node, stmt.sourceFile, stmt.sourceLine);
                }

                // Map each output to its variable
                for (size_t i = 0; i < outputVarNames.size(); i++) {
                    variableToTensor[outputVarNames[i]] = outputTensors[i];
                }

            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to convert multi-output statement";
                if (!stmt.sourceFile.empty()) {
                    std::cerr << " (" << stmt.sourceFile << ":" << stmt.sourceLine << ")";
                }
                std::cerr << ": " << e.what() << std::endl;
                throw;
            }

            continue;  // Skip the rest of the loop for multi-output
        }

        // Parse LHS to check for indexed assignment (e.g., result[1] := ...)
        auto* lhsCompRef = dynamic_cast<basemodelica::BaseModelicaParser::ComponentReferenceContext*>(stmt.lhsContext);

        std::string baseVarName;
        std::vector<int64_t> lhsIndices;  // 0-based indices for ScatterND
        bool isIndexedAssignment = false;

        if (lhsCompRef) {
            baseVarName = stripQuotes(lhsCompRef->IDENT(0)->getText());

            auto arraySubscripts = lhsCompRef->arraySubscripts();
            if (!arraySubscripts.empty()) {
                auto subscriptList = arraySubscripts[0]->subscript();
                for (auto sub : subscriptList) {
                    if (auto subExpr = sub->expression()) {
                        try {
                            int modelicaIndex = std::stoi(subExpr->getText());
                            lhsIndices.push_back(modelicaIndex - 1);  // Convert to 0-based
                            isIndexedAssignment = true;
                        } catch (...) {
                            throw std::runtime_error("Dynamic indices in LHS not yet supported: " +
                                                   subExpr->getText());
                        }
                    }
                }
            }
        }

        std::string lhsVarName = isIndexedAssignment ? baseVarName : stripQuotes(stmt.lhsContext->getText());

        try {
            onnx::GraphProto tempGraph;
            std::map<std::string, std::vector<std::string>> localDerivativeInputs;
            ConversionContext funcCtx(info, &tempGraph, nodeCounter, &variableToTensor, &localDerivativeInputs);
            std::string rhsTensor = ExpressionConverter::convert(stmt.rhsContext, funcCtx);

            std::string finalTensor = rhsTensor;

            if (isIndexedAssignment) {
                // Use GraphBuilder to add ScatterND nodes to tempGraph
                GraphBuilder builder(&tempGraph, nodeCounter);

                // Get current tensor for this variable
                std::string currentTensor;
                if (variableToTensor.find(baseVarName) != variableToTensor.end()) {
                    currentTensor = variableToTensor[baseVarName];
                } else {
                    // First use - need to create a zeros tensor for output variables
                    // Look up the output variable to get its dimensions
                    std::vector<int64_t> shape;
                    for (const auto& output : func.outputs) {
                        if (output.name == baseVarName) {
                            for (const auto& dim : output.dimensions) {
                                try {
                                    shape.push_back(std::stoi(dim));
                                } catch (...) {
                                    throw std::runtime_error("Symbolic dimensions not supported for scatter initialization: " + dim);
                                }
                            }
                            break;
                        }
                    }
                    if (shape.empty()) {
                        throw std::runtime_error("Could not find dimensions for output variable: " + baseVarName);
                    }

                    // Create zeros tensor as initial value
                    currentTensor = builder.addDoubleZerosConstant(shape);
                    variableToTensor[baseVarName] = currentTensor;
                }

                // Unsqueeze RHS to be [1] shaped for ScatterND updates
                std::string updatesTensor = builder.addUnsqueeze(rhsTensor, {0});

                // ScatterND: update the specific index of the array
                finalTensor = builder.addScatterND(currentTensor, lhsIndices, updatesTensor);
            }

            // Convert initializers from tempGraph to Constant nodes in functionProto
            // (FunctionProto doesn't support initializers, so we need to create Constant nodes)
            for (int i = 0; i < tempGraph.initializer_size(); i++) {
                const auto& init = tempGraph.initializer(i);
                auto* constNode = functionProto->add_node();
                constNode->set_op_type("Constant");
                constNode->set_name(init.name());
                constNode->add_output(init.name());

                auto* attr = constNode->add_attribute();
                attr->set_name("value");
                attr->set_type(onnx::AttributeProto::TENSOR);
                attr->mutable_t()->CopyFrom(init);

                addSourceLocationMetadata(constNode, stmt.sourceFile, stmt.sourceLine);
            }

            for (int i = 0; i < tempGraph.node_size(); i++) {
                auto* node = functionProto->add_node();
                node->CopyFrom(tempGraph.node(i));

                addSourceLocationMetadata(node, stmt.sourceFile, stmt.sourceLine);

                auto* meta_index = node->add_metadata_props();
                meta_index->set_key("statement_index");
                meta_index->set_value(std::to_string(stmtIndex));

                auto* meta_lhs = node->add_metadata_props();
                meta_lhs->set_key("lhs_variable");
                meta_lhs->set_value(lhsVarName);
            }

            variableToTensor[lhsVarName] = finalTensor;

        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to convert statement for " << lhsVarName;
            if (!stmt.sourceFile.empty()) {
                std::cerr << " (" << stmt.sourceFile << ":" << stmt.sourceLine << ")";
            }
            std::cerr << ": " << e.what() << std::endl;
            throw;
        }
    }

    for (const auto& output : func.outputs) {
        auto it = variableToTensor.find(output.name);
        if (it == variableToTensor.end()) {
            throw std::runtime_error("Output variable " + output.name + " not computed in algorithm");
        }

        std::string internalTensor = it->second;
        std::string finalName = ensureTensorName(functionProto, internalTensor, output.name);
        functionProto->add_output(finalName);
    }
}

// -----------------------------------------------------------------------------
// Public Interface
// -----------------------------------------------------------------------------

std::string ONNXGenerator::generate(const ModelInfo& info, const std::string& outputDir) {
    std::string lsName = "org.lacemodelica.ls-onnx-serialization";
    std::string lsDir = outputDir + "/extra/" + lsName;

    std::filesystem::create_directories(lsDir);

    std::string modelPath = lsDir + "/model.onnx";
    generateONNXModel(info, modelPath);

    std::string manifestPath = lsDir + "/fmi-ls-manifest.xml";
    generateManifest(manifestPath);

    std::cout << "Generated ONNX layered standard in " << lsDir << "/" << std::endl;

    return lsDir;
}

void ONNXGenerator::generateONNXModel(const ModelInfo& info, const std::string& filepath) {
    onnx::ModelProto model;

    // Model metadata
    model.set_ir_version(8);
    model.set_producer_name("lacemodelica");
    model.set_producer_version("0.1.0");
    model.set_model_version(1);
    model.set_doc_string("Symbolic representation of " + info.modelName);

    auto* opset = model.add_opset_import();
    opset->set_version(18);

    auto* lacemodelica_opset = model.add_opset_import();
    lacemodelica_opset->set_domain("lacemodelica");
    lacemodelica_opset->set_version(1);

    auto* graph = model.mutable_graph();
    graph->set_name(info.modelName);

    // Check if 'time' is used in any equation
    bool usesTime = false;
    for (const auto& eq : info.equations) {
        std::string eqText;
        if (eq.lhsContext) eqText += eq.lhsContext->getText();
        if (eq.rhsContext) eqText += eq.rhsContext->getText();
        if (eq.ifEquationContext) eqText += eq.ifEquationContext->getText();
        if (eq.forEquationContext) eqText += eq.forEquationContext->getText();
        if (eqText.find("time") != std::string::npos) {
            usesTime = true;
            break;
        }
    }

    if (usesTime) {
        auto* timeInput = graph->add_input();
        timeInput->set_name("time");
        auto* timeType = timeInput->mutable_type()->mutable_tensor_type();
        timeType->set_elem_type(onnx::TensorProto::DOUBLE);
        timeType->mutable_shape();
    }

    // Create inputs and track mapping for start[] outputs
    std::map<size_t, int> varIndexToInputIndex;
    int inputIndex = 0;

    for (size_t i = 0; i < info.variables.size(); i++) {
        const auto& var = info.variables[i];

        if (var.isDerivative) continue;

        // Fixed constants become initializers
        if (var.variability == "fixed" && !var.startValue.empty()) {
            auto* initializer = graph->add_initializer();
            initializer->set_name(var.name);
            initializer->set_data_type(onnx::TensorProto::DOUBLE);

            for (const auto& dim : var.dimensions) {
                try {
                    initializer->add_dims(std::stoi(dim));
                } catch (...) {
                    std::cerr << "Warning: Cannot create initializer for " << var.name
                              << " with symbolic dimension " << dim << std::endl;
                }
            }

            try {
                initializer->add_double_data(std::stod(var.startValue));
            } catch (...) {
                std::cerr << "Warning: Could not parse constant value for " << var.name
                          << ": " << var.startValue << std::endl;
                initializer->add_double_data(0.0);
            }
            continue;
        }

        varIndexToInputIndex[i] = inputIndex++;

        auto* input = graph->add_input();
        input->set_name(var.name);
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        if (var.type == "Boolean") {
            input_type->set_elem_type(onnx::TensorProto::BOOL);
        } else {
            input_type->set_elem_type(onnx::TensorProto::DOUBLE);
        }
        auto* input_shape = input_type->mutable_shape();
        addShapeDimensions(input_shape, var.dimensions);
        addSourceLocationMetadata(input, var.sourceFile, var.sourceLine);
    }

    // Create FunctionProtos for functions with algorithms
    for (const auto& func : info.functions) {
        if (!func.algorithmStatements.empty()) {
            try {
                createFunctionProto(func, info, &model);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to create FunctionProto for " << func.name
                          << ": " << e.what() << std::endl;
            }
        }
    }

    // Generate equation outputs
    int nodeCounter = 0;
    int loopCounter = 0;  // Separate counter for clean loop naming
    std::map<std::string, std::vector<std::string>> derivativeInputs;

    EquationGenerator::generateOutputs(info.equations, "eq", info, graph, nodeCounter, loopCounter, derivativeInputs);
    EquationGenerator::generateOutputs(info.initialEquations, "init_eq", info, graph, nodeCounter, loopCounter, derivativeInputs);

    // Generate bound outputs (start, min, max)
    ConversionContext ctx(info, graph, nodeCounter, nullptr, &derivativeInputs);
    BoundOutputContext boundCtx{graph, ctx, varIndexToInputIndex};

    for (size_t i = 0; i < info.variables.size(); i++) {
        const auto& var = info.variables[i];

        if (var.initial == "calculated" && var.bindingContext != nullptr) {
            generateBoundOutput(boundCtx, i, var, var.bindingContext, "start", "Initial value");
        }
        if (var.minContext != nullptr && !isConstValue(var.minValue)) {
            generateBoundOutput(boundCtx, i, var, var.minContext, "min", "Minimum value");
        }
        if (var.maxContext != nullptr && !isConstValue(var.maxValue)) {
            generateBoundOutput(boundCtx, i, var, var.maxContext, "max", "Maximum value");
        }
    }

    // Add derivative inputs
    for (const auto& [derName, dimensions] : derivativeInputs) {
        auto* input = graph->add_input();
        input->set_name(derName);
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        input_type->set_elem_type(onnx::TensorProto::DOUBLE);
        auto* input_shape = input_type->mutable_shape();

        for (const auto& dim : dimensions) {
            auto* shape_dim = input_shape->add_dim();
            try {
                shape_dim->set_dim_value(std::stoi(dim));
            } catch (...) {
                shape_dim->set_dim_param(dim);
            }
        }
    }

    // Shape inference
    try {
        onnx::shape_inference::InferShapes(model);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Shape inference failed: " << e.what() << std::endl;
    }

    // Validation
    try {
        onnx::checker::check_model(model, false, false, false);
    } catch (const onnx::checker::ValidationError& e) {
        std::cerr << "Warning: ONNX validation failed: " << e.what() << std::endl;
    }

    // Serialize
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

    XMLElement* root = doc.NewElement("fmiLayeredStandardManifest");
    root->SetAttribute("xmlns:fmi-ls", "http://fmi-standard.org/fmi-ls-manifest");
    root->SetAttribute("fmi-ls:fmi-ls-name", "org.lacemodelica.ls-onnx-serialization");
    root->SetAttribute("fmi-ls:fmi-ls-version", "1.0.0");
    root->SetAttribute("fmi-ls:fmi-ls-description",
        "Layered standard for ONNX-serialized symbolic expressions in FMU");

    doc.InsertFirstChild(root);

    if (doc.SaveFile(filepath.c_str()) != XML_SUCCESS) {
        throw std::runtime_error("Failed to write manifest file: " + filepath);
    }
}

// Keep convertExpression for backward compatibility (delegates to ExpressionConverter)
std::string ONNXGenerator::convertExpression(antlr4::ParserRuleContext* expr, const ConversionContext& ctx) {
    return ExpressionConverter::convert(expr, ctx);
}

} // namespace lacemodelica
