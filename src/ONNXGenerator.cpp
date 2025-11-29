// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ONNXGenerator.h"
#include "ExpressionConverter.h"
#include "EquationGenerator.h"
#include "GraphBuilder.h"
#include "ONNXHelpers.hpp"
#include "ParseTreeNavigator.h"
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
// Algorithm For-Loop Helpers
// -----------------------------------------------------------------------------

// Parse for-loop range from ForStatementContext
// Uses shared parseForLoopRangeFromIndex from EquationGenerator.h
static ForLoopRange parseAlgorithmForLoopRange(
    basemodelica::BaseModelicaParser::ForStatementContext* forStmtCtx) {
    return parseForLoopRangeFromIndex(forStmtCtx->forIndex());
}

// Recursively identify written variables in statements, including nested control flow
static void identifyWrittenVariables(
    const std::vector<basemodelica::BaseModelicaParser::StatementContext*>& statements,
    std::set<std::string>& writtenVars) {

    for (auto* stmt : statements) {
        if (stmt->componentReference() && stmt->expression()) {
            // Direct assignment
            std::string varName = stripQuotes(stmt->componentReference()->IDENT(0)->getText());
            writtenVars.insert(varName);
        } else if (stmt->forStatement()) {
            // Nested for-loop - recurse into its body
            identifyWrittenVariables(stmt->forStatement()->statement(), writtenVars);
        } else if (stmt->whileStatement()) {
            // Nested while-loop - recurse into its body
            identifyWrittenVariables(stmt->whileStatement()->statement(), writtenVars);
        } else if (stmt->ifStatement()) {
            // Nested if-statement - recurse into all branches
            auto* ifStmt = stmt->ifStatement();
            for (auto* block : ifStmt->statementBlock()) {
                identifyWrittenVariables(block->statement(), writtenVars);
            }
        }
    }
}

// Context for recursive loop body processing
struct LoopBodyContext {
    onnx::GraphProto* graph;                              // Graph to add nodes to
    const Function& func;
    const ModelInfo& info;
    std::map<std::string, std::string>& loopVarMap;       // All loop vars (1-based tensors)
    std::map<std::string, std::string>& varMap;           // Variables available in this scope
    int& nodeCounter;
    int& loopCounter;
    std::string iterName;                                  // 0-based iteration counter name ("i", "j", etc.)
    std::string loopName;                                  // Current loop name for prefixing
};

// Forward declaration
static void processLoopBodyStatements(
    const std::vector<basemodelica::BaseModelicaParser::StatementContext*>& statements,
    LoopBodyContext& ctx);

// Process a single indexed assignment with ScatterND
static void processIndexedAssignment(
    basemodelica::BaseModelicaParser::StatementContext* stmt,
    LoopBodyContext& ctx) {

    auto* lhsCompRef = stmt->componentReference();
    std::string baseVarName = stripQuotes(lhsCompRef->IDENT(0)->getText());

    // Collect all indices
    struct IndexInfo {
        bool isDynamic;
        std::string varName;
        int64_t staticValue;
    };
    std::vector<IndexInfo> indexInfos;
    bool hasDynamicIndex = false;

    auto arraySubscripts = lhsCompRef->arraySubscripts();
    if (!arraySubscripts.empty()) {
        auto subscriptList = arraySubscripts[0]->subscript();
        for (auto sub : subscriptList) {
            if (auto subExpr = sub->expression()) {
                std::string indexText = subExpr->getText();
                try {
                    int modelicaIndex = std::stoi(indexText);
                    indexInfos.push_back({false, "", modelicaIndex - 1});
                } catch (...) {
                    indexInfos.push_back({true, indexText, 0});
                    hasDynamicIndex = true;
                }
            }
        }
    }

    // Convert RHS expression
    onnx::GraphProto rhsGraph;
    std::map<std::string, std::vector<std::string>> derivInputs;
    ConversionContext convCtx(ctx.info, &rhsGraph, ctx.nodeCounter, &ctx.loopVarMap, &derivInputs, ctx.loopName);

    // Add current scope variables to conversion context
    for (const auto& [vn, tn] : ctx.varMap) {
        ctx.loopVarMap[vn] = tn;
    }

    std::string rhsTensor = ExpressionConverter::convert(stmt->expression(), convCtx);

    // Copy RHS nodes to body graph
    for (int i = 0; i < rhsGraph.initializer_size(); i++) {
        const auto& init = rhsGraph.initializer(i);
        auto* constNode = ctx.graph->add_node();
        constNode->set_op_type("Constant");
        constNode->set_name(init.name());
        constNode->add_output(init.name());
        auto* attr = constNode->add_attribute();
        attr->set_name("value");
        attr->set_type(onnx::AttributeProto::TENSOR);
        attr->mutable_t()->CopyFrom(init);
    }

    for (int i = 0; i < rhsGraph.node_size(); i++) {
        auto* node = ctx.graph->add_node();
        node->CopyFrom(rhsGraph.node(i));
    }

    // Handle ScatterND for indexed assignments
    std::string finalTensor = rhsTensor;
    if (hasDynamicIndex && ctx.varMap.find(baseVarName) != ctx.varMap.end()) {
        std::string currentArrayTensor = ctx.varMap[baseVarName];
        size_t numIndices = indexInfos.size();

        // Build individual 0-based index tensors
        std::vector<std::string> indexTensors;
        for (size_t idx = 0; idx < numIndices; idx++) {
            const auto& idxInfo = indexInfos[idx];
            std::string indexTensor;

            if (!idxInfo.isDynamic) {
                // Static index
                indexTensor = ctx.loopName + "_static_idx_" + std::to_string(ctx.nodeCounter);
                auto* constNode = ctx.graph->add_node();
                constNode->set_op_type("Constant");
                constNode->set_name(indexTensor);
                constNode->add_output(indexTensor);
                auto* attr = constNode->add_attribute();
                attr->set_name("value");
                attr->set_type(onnx::AttributeProto::TENSOR);
                auto* tensor = attr->mutable_t();
                tensor->set_data_type(onnx::TensorProto::INT64);
                tensor->add_int64_data(idxInfo.staticValue);
                ctx.nodeCounter++;
            } else {
                // Dynamic index - look up in loopVarMap
                auto it = ctx.loopVarMap.find(idxInfo.varName);
                if (it != ctx.loopVarMap.end()) {
                    std::string oneBasedTensor = it->second;
                    // Subtract 1 to get 0-based
                    std::string oneConst = ctx.loopName + "_one_" + std::to_string(ctx.nodeCounter);
                    auto* oneNode = ctx.graph->add_node();
                    oneNode->set_op_type("Constant");
                    oneNode->set_name(oneConst);
                    oneNode->add_output(oneConst);
                    auto* oneAttr = oneNode->add_attribute();
                    oneAttr->set_name("value");
                    oneAttr->set_type(onnx::AttributeProto::TENSOR);
                    auto* oneTensor = oneAttr->mutable_t();
                    oneTensor->set_data_type(onnx::TensorProto::INT64);
                    oneTensor->add_int64_data(1);
                    ctx.nodeCounter++;

                    indexTensor = ctx.loopName + "_idx0_" + std::to_string(ctx.nodeCounter);
                    auto* subNode = ctx.graph->add_node();
                    subNode->set_op_type("Sub");
                    subNode->set_name(ctx.loopName + "_to0based_" + std::to_string(ctx.nodeCounter));
                    subNode->add_input(oneBasedTensor);
                    subNode->add_input(oneConst);
                    subNode->add_output(indexTensor);
                    ctx.nodeCounter++;
                } else {
                    throw std::runtime_error("Unknown loop variable in index: " + idxInfo.varName);
                }
            }
            indexTensors.push_back(indexTensor);
        }

        // Unsqueeze each index and concatenate
        std::vector<std::string> unsqueezedIndices;
        for (size_t idx = 0; idx < numIndices; idx++) {
            std::string axesConst = ctx.loopName + "_axes_" + std::to_string(ctx.nodeCounter);
            auto* axesNode = ctx.graph->add_node();
            axesNode->set_op_type("Constant");
            axesNode->set_name(axesConst);
            axesNode->add_output(axesConst);
            auto* axesAttr = axesNode->add_attribute();
            axesAttr->set_name("value");
            axesAttr->set_type(onnx::AttributeProto::TENSOR);
            auto* axesTensor = axesAttr->mutable_t();
            axesTensor->set_data_type(onnx::TensorProto::INT64);
            axesTensor->add_dims(1);
            axesTensor->add_int64_data(0);
            ctx.nodeCounter++;

            std::string unsqueezed = ctx.loopName + "_unsq_idx_" + std::to_string(ctx.nodeCounter);
            auto* unsqNode = ctx.graph->add_node();
            unsqNode->set_op_type("Unsqueeze");
            unsqNode->set_name(unsqueezed + "_op");
            unsqNode->add_input(indexTensors[idx]);
            unsqNode->add_input(axesConst);
            unsqNode->add_output(unsqueezed);
            ctx.nodeCounter++;

            unsqueezedIndices.push_back(unsqueezed);
        }

        // Concat indices
        std::string concatIndices = ctx.loopName + "_concat_idx_" + std::to_string(ctx.nodeCounter);
        auto* concatNode = ctx.graph->add_node();
        concatNode->set_op_type("Concat");
        concatNode->set_name(concatIndices + "_op");
        for (const auto& t : unsqueezedIndices) {
            concatNode->add_input(t);
        }
        concatNode->add_output(concatIndices);
        auto* axisAttr = concatNode->add_attribute();
        axisAttr->set_name("axis");
        axisAttr->set_type(onnx::AttributeProto::INT);
        axisAttr->set_i(0);
        ctx.nodeCounter++;

        // Unsqueeze to add batch dimension
        std::string batchAxesConst = ctx.loopName + "_batch_axes_" + std::to_string(ctx.nodeCounter);
        auto* batchAxesNode = ctx.graph->add_node();
        batchAxesNode->set_op_type("Constant");
        batchAxesNode->set_name(batchAxesConst);
        batchAxesNode->add_output(batchAxesConst);
        auto* batchAxesAttr = batchAxesNode->add_attribute();
        batchAxesAttr->set_name("value");
        batchAxesAttr->set_type(onnx::AttributeProto::TENSOR);
        auto* batchAxesTensor = batchAxesAttr->mutable_t();
        batchAxesTensor->set_data_type(onnx::TensorProto::INT64);
        batchAxesTensor->add_dims(1);
        batchAxesTensor->add_int64_data(0);
        ctx.nodeCounter++;

        std::string finalIndices = ctx.loopName + "_final_idx_" + std::to_string(ctx.nodeCounter);
        auto* batchUnsqNode = ctx.graph->add_node();
        batchUnsqNode->set_op_type("Unsqueeze");
        batchUnsqNode->set_name(finalIndices + "_op");
        batchUnsqNode->add_input(concatIndices);
        batchUnsqNode->add_input(batchAxesConst);
        batchUnsqNode->add_output(finalIndices);
        ctx.nodeCounter++;

        // Unsqueeze update
        std::string updateAxesConst = ctx.loopName + "_update_axes_" + std::to_string(ctx.nodeCounter);
        auto* updateAxesNode = ctx.graph->add_node();
        updateAxesNode->set_op_type("Constant");
        updateAxesNode->set_name(updateAxesConst);
        updateAxesNode->add_output(updateAxesConst);
        auto* updateAxesAttr = updateAxesNode->add_attribute();
        updateAxesAttr->set_name("value");
        updateAxesAttr->set_type(onnx::AttributeProto::TENSOR);
        auto* updateAxesTensor = updateAxesAttr->mutable_t();
        updateAxesTensor->set_data_type(onnx::TensorProto::INT64);
        updateAxesTensor->add_dims(1);
        updateAxesTensor->add_int64_data(0);
        ctx.nodeCounter++;

        std::string unsqueezedUpdate = ctx.loopName + "_unsq_update_" + std::to_string(ctx.nodeCounter);
        auto* updateUnsqNode = ctx.graph->add_node();
        updateUnsqNode->set_op_type("Unsqueeze");
        updateUnsqNode->set_name(unsqueezedUpdate + "_op");
        updateUnsqNode->add_input(rhsTensor);
        updateUnsqNode->add_input(updateAxesConst);
        updateUnsqNode->add_output(unsqueezedUpdate);
        ctx.nodeCounter++;

        // ScatterND
        finalTensor = "scattered_" + baseVarName + "_" + std::to_string(ctx.nodeCounter);
        auto* scatterNode = ctx.graph->add_node();
        scatterNode->set_op_type("ScatterND");
        scatterNode->set_name("scatter_" + baseVarName + "_" + std::to_string(ctx.nodeCounter));
        scatterNode->add_input(currentArrayTensor);
        scatterNode->add_input(finalIndices);
        scatterNode->add_input(unsqueezedUpdate);
        scatterNode->add_output(finalTensor);
        ctx.nodeCounter++;
    }

    ctx.varMap[baseVarName] = finalTensor;
}

// Process a nested for-loop statement
static void processNestedForLoop(
    basemodelica::BaseModelicaParser::ForStatementContext* forCtx,
    LoopBodyContext& ctx) {

    ForLoopRange range = parseAlgorithmForLoopRange(forCtx);
    auto innerStatements = forCtx->statement();
    std::string innerLoopName = "loop_" + std::to_string(ctx.loopCounter++);

    // Create trip count and condition constants
    std::string tripCountConst = innerLoopName + "_trip";
    auto* tripNode = ctx.graph->add_node();
    tripNode->set_op_type("Constant");
    tripNode->set_name(tripCountConst);
    tripNode->add_output(tripCountConst);
    auto* tripAttr = tripNode->add_attribute();
    tripAttr->set_name("value");
    tripAttr->set_type(onnx::AttributeProto::TENSOR);
    auto* tripTensor = tripAttr->mutable_t();
    tripTensor->set_data_type(onnx::TensorProto::INT64);
    tripTensor->add_int64_data(range.tripCount());

    std::string condConst = innerLoopName + "_cond";
    auto* condNode = ctx.graph->add_node();
    condNode->set_op_type("Constant");
    condNode->set_name(condConst);
    condNode->add_output(condConst);
    auto* condAttr = condNode->add_attribute();
    condAttr->set_name("value");
    condAttr->set_type(onnx::AttributeProto::TENSOR);
    auto* condTensor = condAttr->mutable_t();
    condTensor->set_data_type(onnx::TensorProto::BOOL);
    condTensor->add_int32_data(1);

    // Create Loop node
    auto* loopNode = ctx.graph->add_node();
    loopNode->set_op_type("Loop");
    loopNode->set_name(innerLoopName);
    loopNode->add_input(tripCountConst);
    loopNode->add_input(condConst);

    // Setup loop body
    auto* bodyAttr = loopNode->add_attribute();
    bodyAttr->set_name("body");
    bodyAttr->set_type(onnx::AttributeProto::GRAPH);
    auto* innerBodyGraph = bodyAttr->mutable_g();
    innerBodyGraph->set_name(innerLoopName + "_body");

    // Standard loop body inputs
    std::string innerIterName = innerLoopName + "_iter";
    auto* iterInput = innerBodyGraph->add_input();
    iterInput->set_name(innerIterName);
    auto* iterType = iterInput->mutable_type()->mutable_tensor_type();
    iterType->set_elem_type(onnx::TensorProto::INT64);
    iterType->mutable_shape();

    std::string innerCondIn = innerLoopName + "_cond_in";
    auto* condInput = innerBodyGraph->add_input();
    condInput->set_name(innerCondIn);
    auto* condType = condInput->mutable_type()->mutable_tensor_type();
    condType->set_elem_type(onnx::TensorProto::BOOL);
    condType->mutable_shape();

    // Condition output (first output)
    std::string innerCondOut = innerLoopName + "_cond_out";
    auto* condOutput = innerBodyGraph->add_output();
    condOutput->set_name(innerCondOut);
    auto* condOutType = condOutput->mutable_type()->mutable_tensor_type();
    condOutType->set_elem_type(onnx::TensorProto::BOOL);
    condOutType->mutable_shape();

    auto* condIdentity = innerBodyGraph->add_node();
    condIdentity->set_op_type("Identity");
    condIdentity->set_name(innerLoopName + "_cond_pass");
    condIdentity->add_input(innerCondIn);
    condIdentity->add_output(innerCondOut);

    // Convert 0-based iter to 1-based
    std::string oneConst = innerLoopName + "_one";
    auto* oneNode = innerBodyGraph->add_node();
    oneNode->set_op_type("Constant");
    oneNode->set_name(oneConst);
    oneNode->add_output(oneConst);
    auto* oneAttr = oneNode->add_attribute();
    oneAttr->set_name("value");
    oneAttr->set_type(onnx::AttributeProto::TENSOR);
    auto* oneTensor = oneAttr->mutable_t();
    oneTensor->set_data_type(onnx::TensorProto::INT64);
    oneTensor->add_int64_data(1);

    std::string loopVar1Based = innerLoopName + "_" + range.loopVar + "_1based";
    auto* addNode = innerBodyGraph->add_node();
    addNode->set_op_type("Add");
    addNode->set_name(loopVar1Based + "_add");
    addNode->add_input(innerIterName);
    addNode->add_input(oneConst);
    addNode->add_output(loopVar1Based);

    // Find written variables in inner loop
    std::set<std::string> innerWrittenVars;
    identifyWrittenVariables(innerStatements, innerWrittenVars);

    // Setup carried variables
    std::map<std::string, std::string> innerVarMap;
    std::vector<std::string> carriedVars;

    for (const auto& varName : innerWrittenVars) {
        if (ctx.varMap.find(varName) != ctx.varMap.end()) {
            carriedVars.push_back(varName);

            std::string outerTensor = ctx.varMap[varName];
            loopNode->add_input(outerTensor);

            std::string innerInputName = innerLoopName + "_" + varName + "_in";
            auto* innerInput = innerBodyGraph->add_input();
            innerInput->set_name(innerInputName);
            auto* innerInputType = innerInput->mutable_type()->mutable_tensor_type();
            innerInputType->set_elem_type(onnx::TensorProto::DOUBLE);

            // Set shape
            for (const auto& output : ctx.func.outputs) {
                if (output.name == varName && !output.dimensions.empty()) {
                    auto* shape = innerInputType->mutable_shape();
                    for (const auto& dim : output.dimensions) {
                        try { shape->add_dim()->set_dim_value(std::stoi(dim)); }
                        catch (...) { shape->add_dim()->set_dim_param(dim); }
                    }
                    break;
                }
            }

            innerVarMap[varName] = innerInputName;
        }
    }

    // Pass through all outer loop variables (1-based)
    std::map<std::string, std::string> innerLoopVarMap = ctx.loopVarMap;
    for (const auto& [loopVar, loopVarTensor] : ctx.loopVarMap) {
        std::string passName = innerLoopName + "_" + loopVar + "_pass";
        loopNode->add_input(loopVarTensor);

        auto* passInput = innerBodyGraph->add_input();
        passInput->set_name(passName);
        auto* passType = passInput->mutable_type()->mutable_tensor_type();
        passType->set_elem_type(onnx::TensorProto::INT64);
        passType->mutable_shape();

        innerLoopVarMap[loopVar] = passName;
    }

    // Add current loop variable
    innerLoopVarMap[range.loopVar] = loopVar1Based;

    // Pass through other variables (inputs like factor, mat)
    std::vector<std::string> passthroughVars;
    for (const auto& [varName, tensorName] : ctx.varMap) {
        if (std::find(carriedVars.begin(), carriedVars.end(), varName) == carriedVars.end()) {
            std::string passName = innerLoopName + "_" + varName + "_pass";
            loopNode->add_input(tensorName);

            auto* passInput = innerBodyGraph->add_input();
            passInput->set_name(passName);
            auto* passType = passInput->mutable_type()->mutable_tensor_type();
            passType->set_elem_type(onnx::TensorProto::DOUBLE);

            for (const auto& input : ctx.func.inputs) {
                if (input.name == varName && !input.dimensions.empty()) {
                    auto* shape = passType->mutable_shape();
                    for (const auto& dim : input.dimensions) {
                        try { shape->add_dim()->set_dim_value(std::stoi(dim)); }
                        catch (...) { shape->add_dim()->set_dim_param(dim); }
                    }
                    break;
                }
            }

            innerVarMap[varName] = passName;
            passthroughVars.push_back(varName);
        }
    }

    // Recursively process inner loop statements
    LoopBodyContext innerCtx{
        innerBodyGraph, ctx.func, ctx.info, innerLoopVarMap, innerVarMap,
        ctx.nodeCounter, ctx.loopCounter, innerIterName, innerLoopName
    };
    processLoopBodyStatements(innerStatements, innerCtx);

    // Add carried variable outputs
    for (const auto& varName : carriedVars) {
        std::string finalTensor = innerVarMap[varName];
        std::string outputName = innerLoopName + "_" + varName + "_out";

        auto* output = innerBodyGraph->add_output();
        output->set_name(outputName);
        auto* outputType = output->mutable_type()->mutable_tensor_type();
        outputType->set_elem_type(onnx::TensorProto::DOUBLE);

        for (const auto& out : ctx.func.outputs) {
            if (out.name == varName && !out.dimensions.empty()) {
                auto* shape = outputType->mutable_shape();
                for (const auto& dim : out.dimensions) {
                    try { shape->add_dim()->set_dim_value(std::stoi(dim)); }
                    catch (...) { shape->add_dim()->set_dim_param(dim); }
                }
                break;
            }
        }

        auto* identityNode = innerBodyGraph->add_node();
        identityNode->set_op_type("Identity");
        identityNode->set_name(innerLoopName + "_out_" + varName);
        identityNode->add_input(finalTensor);
        identityNode->add_output(outputName);

        std::string loopOutputName = innerLoopName + "_" + varName + "_result";
        loopNode->add_output(loopOutputName);

        // Update outer scope
        ctx.varMap[varName] = loopOutputName;
    }

    // Add passthrough outputs for loop variables
    for (const auto& [loopVar, loopVarTensor] : ctx.loopVarMap) {
        std::string passInName = innerLoopName + "_" + loopVar + "_pass";
        std::string passOutName = innerLoopName + "_" + loopVar + "_pass_out";

        auto* passOutput = innerBodyGraph->add_output();
        passOutput->set_name(passOutName);
        auto* passOutType = passOutput->mutable_type()->mutable_tensor_type();
        passOutType->set_elem_type(onnx::TensorProto::INT64);
        passOutType->mutable_shape();

        auto* passIdentity = innerBodyGraph->add_node();
        passIdentity->set_op_type("Identity");
        passIdentity->set_name(innerLoopName + "_" + loopVar + "_pass_identity");
        passIdentity->add_input(passInName);
        passIdentity->add_output(passOutName);

        loopNode->add_output(innerLoopName + "_" + loopVar + "_final");
    }

    // Add passthrough outputs for other variables
    for (const auto& varName : passthroughVars) {
        std::string passInName = innerLoopName + "_" + varName + "_pass";
        std::string passOutName = innerLoopName + "_" + varName + "_pass_out";

        auto* passOutput = innerBodyGraph->add_output();
        passOutput->set_name(passOutName);
        auto* passOutType = passOutput->mutable_type()->mutable_tensor_type();
        passOutType->set_elem_type(onnx::TensorProto::DOUBLE);

        for (const auto& input : ctx.func.inputs) {
            if (input.name == varName && !input.dimensions.empty()) {
                auto* shape = passOutType->mutable_shape();
                for (const auto& dim : input.dimensions) {
                    try { shape->add_dim()->set_dim_value(std::stoi(dim)); }
                    catch (...) { shape->add_dim()->set_dim_param(dim); }
                }
                break;
            }
        }

        auto* passIdentity = innerBodyGraph->add_node();
        passIdentity->set_op_type("Identity");
        passIdentity->set_name(innerLoopName + "_" + varName + "_pass_identity");
        passIdentity->add_input(passInName);
        passIdentity->add_output(passOutName);

        loopNode->add_output(innerLoopName + "_" + varName + "_final");
    }
}

// Recursively process statements in a loop body
static void processLoopBodyStatements(
    const std::vector<basemodelica::BaseModelicaParser::StatementContext*>& statements,
    LoopBodyContext& ctx) {

    for (auto* stmt : statements) {
        if (stmt->forStatement()) {
            // Nested for-loop
            processNestedForLoop(stmt->forStatement(), ctx);
        } else if (stmt->componentReference() && stmt->expression()) {
            // Direct assignment
            processIndexedAssignment(stmt, ctx);
        }
        // Skip other statement types for now
    }
}

// Generate ONNX Loop for an algorithm for-statement
// Updates variableToTensor with any modified loop-carried variables
static void generateAlgorithmForLoop(
    basemodelica::BaseModelicaParser::ForStatementContext* forStmtCtx,
    onnx::FunctionProto* functionProto,
    const Function& func,
    const ModelInfo& info,
    std::map<std::string, std::string>& variableToTensor,
    int& nodeCounter,
    int& loopCounter,
    const std::string& sourceFile,
    size_t sourceLine) {

    ForLoopRange range = parseAlgorithmForLoopRange(forStmtCtx);
    auto loopStatements = forStmtCtx->statement();

    // Create a temporary graph to build the loop structure
    onnx::GraphProto tempGraph;
    GraphBuilder builder(&tempGraph, nodeCounter);

    std::string loopNodeName = "loop_" + std::to_string(loopCounter++);

    // Identify loop-carried dependencies FIRST (before creating Loop node)
    // Variables that are both read AND written in the loop body
    std::set<std::string> writtenVars;

    // First pass: identify written variables (recursively through nested loops)
    identifyWrittenVariables(loopStatements, writtenVars);

    // For output variables that are written but don't exist yet, initialize with zeros
    // This MUST happen BEFORE creating the Loop node so the zeros constant is defined first
    for (const auto& varName : writtenVars) {
        if (variableToTensor.find(varName) == variableToTensor.end()) {
            // Check if it's an output variable
            for (const auto& output : func.outputs) {
                if (output.name == varName && !output.dimensions.empty()) {
                    // Create zeros tensor for this output
                    std::vector<int64_t> shape;
                    for (const auto& dim : output.dimensions) {
                        try {
                            shape.push_back(std::stoi(dim));
                        } catch (...) {
                            throw std::runtime_error("Symbolic dimensions not supported for loop output initialization: " + dim);
                        }
                    }
                    std::string zerosTensor = builder.addDoubleZerosConstant(shape);
                    variableToTensor[varName] = zerosTensor;
                    break;
                }
            }
        }
    }

    // Trip count and initial condition
    std::string tripCountTensor = builder.addInt64Constant(range.tripCount(), "n_" + loopNodeName);
    std::string condTensor = builder.addBoolConstant(true);

    auto* loopNode = tempGraph.add_node();
    loopNode->set_op_type("Loop");
    loopNode->set_name(loopNodeName);
    loopNode->add_input(tripCountTensor);
    loopNode->add_input(condTensor);

    auto* bodyAttr = loopNode->add_attribute();
    bodyAttr->set_name("body");
    bodyAttr->set_type(onnx::AttributeProto::GRAPH);
    auto* bodyGraph = bodyAttr->mutable_g();
    bodyGraph->set_name("body");

    // Set up loop body standard inputs (iter, cond)
    setupLoopBodyIO(bodyGraph, loopNodeName);

    // Create body builder
    auto bodyBuilder = builder.forSubgraph(bodyGraph, loopNodeName);

    // Convert 0-based iter to 1-based Modelica index
    std::string constOneTensor = bodyBuilder.addInt64Constant(1, "one");
    std::string loopVarTensor = bodyBuilder.addBinaryOp("Add", "i", constOneTensor);

    // Map loop variable to its tensor
    std::map<std::string, std::string> loopVarMap;
    loopVarMap[range.loopVar] = loopVarTensor;

    // Variables that exist before the loop and are written are loop-carried
    std::vector<std::string> carriedVars;
    for (const auto& varName : writtenVars) {
        if (variableToTensor.find(varName) != variableToTensor.end()) {
            carriedVars.push_back(varName);
        }
    }

    // Set up loop-carried inputs/outputs
    std::map<std::string, std::string> bodyInputTensors;  // varName -> body input tensor name
    std::map<std::string, std::string> bodyOutputTensors; // varName -> body output tensor name

    for (const auto& varName : carriedVars) {
        std::string externalTensor = variableToTensor[varName];

        // Add as loop input
        loopNode->add_input(externalTensor);

        // Add body input
        std::string bodyInputName = loopNodeName + "_" + varName + "_in";
        auto* bodyInput = bodyGraph->add_input();
        bodyInput->set_name(bodyInputName);
        auto* inputType = bodyInput->mutable_type()->mutable_tensor_type();
        inputType->set_elem_type(onnx::TensorProto::DOUBLE);

        // Check if it's an array (from outputs or inputs)
        bool foundShape = false;
        for (const auto& output : func.outputs) {
            if (output.name == varName && !output.dimensions.empty()) {
                auto* shape = inputType->mutable_shape();
                for (const auto& dim : output.dimensions) {
                    try {
                        shape->add_dim()->set_dim_value(std::stoi(dim));
                    } catch (...) {
                        shape->add_dim()->set_dim_param(dim);
                    }
                }
                foundShape = true;
                break;
            }
        }
        if (!foundShape) {
            inputType->mutable_shape();  // Scalar
        }

        bodyInputTensors[varName] = bodyInputName;

        // Prepare output name (will be set during statement processing)
        std::string bodyOutputName = loopNodeName + "_" + varName + "_out";
        bodyOutputTensors[varName] = bodyOutputName;
    }

    // Map for variable tensors inside the loop body
    std::map<std::string, std::string> bodyVariableMap;

    // Initialize with carried variables
    for (const auto& varName : carriedVars) {
        bodyVariableMap[varName] = bodyInputTensors[varName];
    }

    // Track passthrough variables (need to add outputs after carried outputs)
    std::vector<std::string> passthroughVars;

    // Also pass through inputs from function that are needed in loop
    for (const auto& [varName, tensorName] : variableToTensor) {
        if (bodyVariableMap.find(varName) == bodyVariableMap.end()) {
            // Not a carried variable - pass it through
            loopNode->add_input(tensorName);

            std::string bodyInputName = loopNodeName + "_" + varName + "_pass";
            auto* bodyInput = bodyGraph->add_input();
            bodyInput->set_name(bodyInputName);
            auto* inputType = bodyInput->mutable_type()->mutable_tensor_type();
            inputType->set_elem_type(onnx::TensorProto::DOUBLE);
            // Check if it's an array
            for (const auto& input : func.inputs) {
                if (input.name == varName && !input.dimensions.empty()) {
                    auto* shape = inputType->mutable_shape();
                    for (const auto& dim : input.dimensions) {
                        try {
                            shape->add_dim()->set_dim_value(std::stoi(dim));
                        } catch (...) {
                            shape->add_dim()->set_dim_param(dim);
                        }
                    }
                    break;
                }
            }

            bodyVariableMap[varName] = bodyInputName;
            passthroughVars.push_back(varName);
        }
    }


    // Process statements in the loop body using recursive helper
    LoopBodyContext ctx{
        bodyGraph, func, info, loopVarMap, bodyVariableMap,
        nodeCounter, loopCounter, "i", loopNodeName
    };
    processLoopBodyStatements(loopStatements, ctx);


    // Add carried variable outputs
    for (const auto& varName : carriedVars) {
        std::string finalTensor = bodyVariableMap[varName];
        std::string bodyOutputName = bodyOutputTensors[varName];

        // Add output to body graph
        auto* bodyOutput = bodyGraph->add_output();
        bodyOutput->set_name(bodyOutputName);
        auto* outputType = bodyOutput->mutable_type()->mutable_tensor_type();
        outputType->set_elem_type(onnx::TensorProto::DOUBLE);

        // Check if it's an array (from outputs)
        bool foundOutputShape = false;
        for (const auto& output : func.outputs) {
            if (output.name == varName && !output.dimensions.empty()) {
                auto* shape = outputType->mutable_shape();
                for (const auto& dim : output.dimensions) {
                    try {
                        shape->add_dim()->set_dim_value(std::stoi(dim));
                    } catch (...) {
                        shape->add_dim()->set_dim_param(dim);
                    }
                }
                foundOutputShape = true;
                break;
            }
        }
        if (!foundOutputShape) {
            outputType->mutable_shape();  // Scalar
        }

        // Identity to rename final tensor to output
        auto* identityNode = bodyGraph->add_node();
        identityNode->set_op_type("Identity");
        identityNode->set_name(loopNodeName + "_out_" + varName);
        identityNode->add_input(finalTensor);
        identityNode->add_output(bodyOutputName);

        // Add loop output
        std::string loopOutputName = loopNodeName + "_" + varName + "_result";
        loopNode->add_output(loopOutputName);

        // Update external variable mapping
        variableToTensor[varName] = loopOutputName;
    }

    // Add passthrough variable outputs (MUST come after carried outputs in ONNX Loop)
    for (const auto& varName : passthroughVars) {
        std::string bodyInputName = loopNodeName + "_" + varName + "_pass";
        std::string bodyOutputName = loopNodeName + "_" + varName + "_pass_out";

        // Add passthrough output to body graph
        auto* bodyOutput = bodyGraph->add_output();
        bodyOutput->set_name(bodyOutputName);
        auto* outputType = bodyOutput->mutable_type()->mutable_tensor_type();
        outputType->set_elem_type(onnx::TensorProto::DOUBLE);

        // Set shape if it's an array
        for (const auto& input : func.inputs) {
            if (input.name == varName && !input.dimensions.empty()) {
                auto* shape = outputType->mutable_shape();
                for (const auto& dim : input.dimensions) {
                    try {
                        shape->add_dim()->set_dim_value(std::stoi(dim));
                    } catch (...) {
                        shape->add_dim()->set_dim_param(dim);
                    }
                }
                break;
            }
        }

        // Identity to pass through
        auto* identityNode = bodyGraph->add_node();
        identityNode->set_op_type("Identity");
        identityNode->set_name(loopNodeName + "_pass_" + varName);
        identityNode->add_input(bodyInputName);
        identityNode->add_output(bodyOutputName);

        // Add loop output
        std::string loopOutputName = loopNodeName + "_" + varName + "_final";
        loopNode->add_output(loopOutputName);
    }

    // Copy everything from tempGraph to functionProto
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

        addSourceLocationMetadata(constNode, sourceFile, sourceLine);
    }

    for (int i = 0; i < tempGraph.node_size(); i++) {
        auto* node = functionProto->add_node();
        node->CopyFrom(tempGraph.node(i));
        addSourceLocationMetadata(node, sourceFile, sourceLine);
    }
}

// -----------------------------------------------------------------------------
// While Loop ONNX Generation
// -----------------------------------------------------------------------------

// Set up while loop body I/O without condition passthrough
// Returns the condition output name (caller must compute and output the condition)
static std::string setupWhileLoopBodyIO(onnx::GraphProto* bodyGraph, const std::string& loopNodeName) {
    // Add iter input (0-based iteration counter)
    auto* iterInput = bodyGraph->add_input();
    iterInput->set_name("i");
    auto* iterType = iterInput->mutable_type()->mutable_tensor_type();
    iterType->set_elem_type(onnx::TensorProto::INT64);
    iterType->mutable_shape();  // Scalar

    // Add condition input
    auto* condInput = bodyGraph->add_input();
    condInput->set_name("cond");
    auto* condInputType = condInput->mutable_type()->mutable_tensor_type();
    condInputType->set_elem_type(onnx::TensorProto::BOOL);
    condInputType->mutable_shape();  // Scalar

    // Create condition output (will be set by while condition expression)
    std::string condOutName = loopNodeName + "_cond_out";
    auto* condOutput = bodyGraph->add_output();
    condOutput->set_name(condOutName);
    auto* condOutputType = condOutput->mutable_type()->mutable_tensor_type();
    condOutputType->set_elem_type(onnx::TensorProto::BOOL);
    condOutputType->mutable_shape();  // Scalar

    // Note: No Identity passthrough here - caller will compute condition

    return condOutName;
}

static void generateAlgorithmWhileLoop(
    basemodelica::BaseModelicaParser::WhileStatementContext* whileStmtCtx,
    onnx::FunctionProto* functionProto,
    const Function& func,
    const ModelInfo& info,
    std::map<std::string, std::string>& variableToTensor,
    int& nodeCounter,
    int& loopCounter,
    const std::string& sourceFile,
    size_t sourceLine) {

    auto* condExpr = whileStmtCtx->expression();
    auto loopStatements = whileStmtCtx->statement();

    // Create a temporary graph to build the loop structure
    onnx::GraphProto tempGraph;
    GraphBuilder builder(&tempGraph, nodeCounter);

    std::string loopNodeName = "while_" + std::to_string(loopCounter++);

    // Identify written variables
    std::set<std::string> writtenVars;
    identifyWrittenVariables(loopStatements, writtenVars);

    // Initialize output variables that don't exist yet
    for (const auto& varName : writtenVars) {
        if (variableToTensor.find(varName) == variableToTensor.end()) {
            for (const auto& output : func.outputs) {
                if (output.name == varName && !output.dimensions.empty()) {
                    std::vector<int64_t> shape;
                    for (const auto& dim : output.dimensions) {
                        try {
                            shape.push_back(std::stoi(dim));
                        } catch (...) {
                            throw std::runtime_error("Symbolic dimensions not supported: " + dim);
                        }
                    }
                    std::string zerosTensor = builder.addDoubleZerosConstant(shape);
                    variableToTensor[varName] = zerosTensor;
                    break;
                }
            }
        }
    }

    // Max iterations (use a reasonable limit to prevent infinite loops)
    // Use empty string for unlimited in ONNX, but we'll use a large constant for safety
    std::string maxIterTensor = builder.addInt64Constant(1000, "max_" + loopNodeName);
    std::string condTensor = builder.addBoolConstant(true);

    auto* loopNode = tempGraph.add_node();
    loopNode->set_op_type("Loop");
    loopNode->set_name(loopNodeName);
    loopNode->add_input(maxIterTensor);
    loopNode->add_input(condTensor);

    auto* bodyAttr = loopNode->add_attribute();
    bodyAttr->set_name("body");
    bodyAttr->set_type(onnx::AttributeProto::GRAPH);
    auto* bodyGraph = bodyAttr->mutable_g();
    bodyGraph->set_name("while_body");

    // Set up body I/O (without condition passthrough)
    std::string condOutName = setupWhileLoopBodyIO(bodyGraph, loopNodeName);

    auto bodyBuilder = builder.forSubgraph(bodyGraph, loopNodeName);

    // No loop variable for while loops, but we need maps for processing
    std::map<std::string, std::string> loopVarMap;  // Empty - no loop variable

    // Variables that exist before the loop and are written are loop-carried
    std::vector<std::string> carriedVars;
    for (const auto& varName : writtenVars) {
        if (variableToTensor.find(varName) != variableToTensor.end()) {
            carriedVars.push_back(varName);
        }
    }

    // Set up loop-carried inputs/outputs
    std::map<std::string, std::string> bodyInputTensors;
    std::map<std::string, std::string> bodyOutputTensors;

    for (const auto& varName : carriedVars) {
        std::string externalTensor = variableToTensor[varName];
        loopNode->add_input(externalTensor);

        std::string bodyInputName = loopNodeName + "_" + varName + "_in";
        auto* bodyInput = bodyGraph->add_input();
        bodyInput->set_name(bodyInputName);
        auto* inputType = bodyInput->mutable_type()->mutable_tensor_type();
        inputType->set_elem_type(onnx::TensorProto::DOUBLE);
        inputType->mutable_shape();  // Scalar

        bodyInputTensors[varName] = bodyInputName;
        bodyOutputTensors[varName] = loopNodeName + "_" + varName + "_out";
    }

    // Map for variable tensors inside the loop body
    std::map<std::string, std::string> bodyVariableMap;
    for (const auto& varName : carriedVars) {
        bodyVariableMap[varName] = bodyInputTensors[varName];
    }

    // Pass through non-carried variables
    std::vector<std::string> passthroughVars;
    for (const auto& [varName, tensorName] : variableToTensor) {
        if (bodyVariableMap.find(varName) == bodyVariableMap.end()) {
            loopNode->add_input(tensorName);

            std::string bodyInputName = loopNodeName + "_" + varName + "_pass";
            auto* bodyInput = bodyGraph->add_input();
            bodyInput->set_name(bodyInputName);
            auto* inputType = bodyInput->mutable_type()->mutable_tensor_type();
            inputType->set_elem_type(onnx::TensorProto::DOUBLE);

            for (const auto& input : func.inputs) {
                if (input.name == varName && !input.dimensions.empty()) {
                    auto* shape = inputType->mutable_shape();
                    for (const auto& dim : input.dimensions) {
                        try {
                            shape->add_dim()->set_dim_value(std::stoi(dim));
                        } catch (...) {
                            shape->add_dim()->set_dim_param(dim);
                        }
                    }
                    break;
                }
            }

            bodyVariableMap[varName] = bodyInputName;
            passthroughVars.push_back(varName);
        }
    }

    // Process statements in the loop body
    LoopBodyContext ctx{
        bodyGraph, func, info, loopVarMap, bodyVariableMap,
        nodeCounter, loopCounter, "", loopNodeName  // Empty loop var name
    };
    processLoopBodyStatements(loopStatements, ctx);

    // Evaluate while condition expression at end of loop body
    // This determines if the loop should continue
    std::map<std::string, std::vector<std::string>> localDerivativeInputs;
    ConversionContext condCtx(info, bodyGraph, nodeCounter, &bodyVariableMap, &localDerivativeInputs);
    condCtx.tensorPrefix = loopNodeName + "_";
    std::string condResultTensor = ExpressionConverter::convert(condExpr, condCtx);

    // Connect condition result to condition output
    auto* condIdentity = bodyGraph->add_node();
    condIdentity->set_op_type("Identity");
    condIdentity->set_name(loopNodeName + "_cond_eval");
    condIdentity->add_input(condResultTensor);
    condIdentity->add_output(condOutName);

    // Add carried variable outputs
    for (const auto& varName : carriedVars) {
        std::string finalTensor = bodyVariableMap[varName];
        std::string bodyOutputName = bodyOutputTensors[varName];

        auto* bodyOutput = bodyGraph->add_output();
        bodyOutput->set_name(bodyOutputName);
        auto* outputType = bodyOutput->mutable_type()->mutable_tensor_type();
        outputType->set_elem_type(onnx::TensorProto::DOUBLE);
        outputType->mutable_shape();  // Scalar

        auto* identityNode = bodyGraph->add_node();
        identityNode->set_op_type("Identity");
        identityNode->set_name(loopNodeName + "_out_" + varName);
        identityNode->add_input(finalTensor);
        identityNode->add_output(bodyOutputName);

        std::string loopOutputName = loopNodeName + "_" + varName + "_result";
        loopNode->add_output(loopOutputName);
        variableToTensor[varName] = loopOutputName;
    }

    // Add passthrough variable outputs
    for (const auto& varName : passthroughVars) {
        std::string bodyInputName = loopNodeName + "_" + varName + "_pass";
        std::string bodyOutputName = loopNodeName + "_" + varName + "_pass_out";

        auto* bodyOutput = bodyGraph->add_output();
        bodyOutput->set_name(bodyOutputName);
        auto* outputType = bodyOutput->mutable_type()->mutable_tensor_type();
        outputType->set_elem_type(onnx::TensorProto::DOUBLE);

        auto* identityNode = bodyGraph->add_node();
        identityNode->set_op_type("Identity");
        identityNode->set_name(loopNodeName + "_pass_" + varName);
        identityNode->add_input(bodyInputName);
        identityNode->add_output(bodyOutputName);

        std::string loopOutputName = loopNodeName + "_" + varName + "_final";
        loopNode->add_output(loopOutputName);
    }

    // Copy from tempGraph to functionProto
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

        addSourceLocationMetadata(constNode, sourceFile, sourceLine);
    }

    for (int i = 0; i < tempGraph.node_size(); i++) {
        auto* node = functionProto->add_node();
        node->CopyFrom(tempGraph.node(i));
        addSourceLocationMetadata(node, sourceFile, sourceLine);
    }
}

// -----------------------------------------------------------------------------
// If Statement Generation
// -----------------------------------------------------------------------------

// Forward declaration for recursive processing
static void processAlgorithmStatements(
    const std::vector<basemodelica::BaseModelicaParser::StatementContext*>& statements,
    onnx::GraphProto* graph,
    const Function& func,
    const ModelInfo& info,
    std::map<std::string, std::string>& variableToTensor,
    int& nodeCounter,
    int& loopCounter,
    const std::string& sourceFile,
    size_t sourceLine,
    const std::string& prefix);

// Generate ONNX If node for an algorithm if-statement
// Handles unbalanced branches by adding identity nodes for unchanged variables
// The prefix parameter ensures unique tensor names across nested subgraphs
static void generateAlgorithmIfStatement(
    basemodelica::BaseModelicaParser::IfStatementContext* ifStmtCtx,
    onnx::GraphProto* graph,
    const Function& func,
    const ModelInfo& info,
    std::map<std::string, std::string>& variableToTensor,
    int& nodeCounter,
    int& loopCounter,
    const std::string& sourceFile,
    size_t sourceLine,
    const std::string& prefix = "") {

    auto conditions = ifStmtCtx->expression();
    auto blocks = ifStmtCtx->statementBlock();

    if (conditions.empty() || blocks.empty()) {
        throw std::runtime_error("Invalid if-statement structure");
    }

    // Snapshot current variable state
    std::map<std::string, std::string> varMapBefore = variableToTensor;

    // Process true branch (first block) with unique prefix
    std::map<std::string, std::string> varMapTrue = varMapBefore;
    int trueNodeCounter = nodeCounter;
    int trueLoopCounter = loopCounter;
    onnx::GraphProto thenGraph;
    thenGraph.set_name("then_branch_" + std::to_string(nodeCounter));
    std::string thenPrefix = prefix + "then" + std::to_string(nodeCounter) + "_";

    processAlgorithmStatements(
        blocks[0]->statement(), &thenGraph, func, info,
        varMapTrue, trueNodeCounter, trueLoopCounter, sourceFile, sourceLine, thenPrefix);

    // Determine false branch content
    // If there are elseif clauses or else, process them; otherwise empty
    std::map<std::string, std::string> varMapFalse = varMapBefore;
    int falseNodeCounter = nodeCounter;
    int falseLoopCounter = loopCounter;
    onnx::GraphProto elseGraph;
    elseGraph.set_name("else_branch_" + std::to_string(nodeCounter));

    std::string elsePrefix = prefix + "else" + std::to_string(nodeCounter) + "_";

    if (conditions.size() == 1 && blocks.size() >= 2) {
        // Simple if-else (one condition, two blocks)
        processAlgorithmStatements(
            blocks[1]->statement(), &elseGraph, func, info,
            varMapFalse, falseNodeCounter, falseLoopCounter, sourceFile, sourceLine, elsePrefix);
    } else if (conditions.size() > 1) {
        // Has elseif clauses - build nested If structure in the else branch
        // if cond1 then block1 elseif cond2 then block2 else block3 end if
        // becomes: If(cond1, block1, If(cond2, block2, block3))

        std::function<void(size_t, onnx::GraphProto*, std::map<std::string, std::string>&, int&, int&, const std::string&)> buildNestedIf;

        buildNestedIf = [&](size_t condIdx, onnx::GraphProto* targetGraph,
                            std::map<std::string, std::string>& varMap, int& nc, int& lc,
                            const std::string& nestedPrefix) {
            if (condIdx >= conditions.size()) {
                // No more conditions - check if there's a final else block
                if (blocks.size() > conditions.size()) {
                    std::string finalElsePrefix = nestedPrefix + "finalelse_";
                    processAlgorithmStatements(
                        blocks[conditions.size()]->statement(), targetGraph, func, info,
                        varMap, nc, lc, sourceFile, sourceLine, finalElsePrefix);
                }
                return;
            }

            // Snapshot before this level
            std::map<std::string, std::string> varMapBeforeLevel = varMap;

            // Convert condition with unique prefix for tensor names
            std::string condPrefix = nestedPrefix + "cond" + std::to_string(condIdx) + "_";
            std::map<std::string, std::vector<std::string>> derivInputs;
            ConversionContext condCtx(info, targetGraph, nc, &varMap, &derivInputs, condPrefix);
            std::string condTensor = ExpressionConverter::convert(conditions[condIdx], condCtx);

            // Build then branch with unique prefix
            std::map<std::string, std::string> thenVarMap = varMapBeforeLevel;
            onnx::GraphProto innerThenGraph;
            innerThenGraph.set_name("nested_then_" + std::to_string(condIdx));
            int thenNc = nc;
            int thenLc = lc;
            std::string innerThenPrefix = nestedPrefix + "nthen" + std::to_string(condIdx) + "_";
            processAlgorithmStatements(
                blocks[condIdx]->statement(), &innerThenGraph, func, info,
                thenVarMap, thenNc, thenLc, sourceFile, sourceLine, innerThenPrefix);

            // Build else branch (recursively) with different prefix
            std::map<std::string, std::string> elseVarMap = varMapBeforeLevel;
            onnx::GraphProto innerElseGraph;
            innerElseGraph.set_name("nested_else_" + std::to_string(condIdx));
            int elseNc = nc;
            int elseLc = lc;
            std::string innerElsePrefix = nestedPrefix + "nelse" + std::to_string(condIdx) + "_";
            buildNestedIf(condIdx + 1, &innerElseGraph, elseVarMap, elseNc, elseLc, innerElsePrefix);

            // Find union of updated variables at this level
            std::set<std::string> updatedVars;
            for (const auto& [var, tensor] : thenVarMap) {
                if (varMapBeforeLevel.find(var) == varMapBeforeLevel.end() ||
                    varMapBeforeLevel[var] != tensor) {
                    updatedVars.insert(var);
                }
            }
            for (const auto& [var, tensor] : elseVarMap) {
                if (varMapBeforeLevel.find(var) == varMapBeforeLevel.end() ||
                    varMapBeforeLevel[var] != tensor) {
                    updatedVars.insert(var);
                }
            }

            if (updatedVars.empty()) {
                // Nothing changed, no If node needed
                nc = std::max({nc, thenNc, elseNc});
                lc = std::max({lc, thenLc, elseLc});
                return;
            }

            // Add outputs to both branches
            std::vector<std::string> outputOrder(updatedVars.begin(), updatedVars.end());
            int outputIdx = nc;

            for (const auto& var : outputOrder) {
                std::string thenOutput = "nested_then_out_" + var + "_" + std::to_string(outputIdx);
                std::string elseOutput = "nested_else_out_" + var + "_" + std::to_string(outputIdx);

                // Then branch output
                std::string thenTensor = (thenVarMap.find(var) != thenVarMap.end() &&
                                          thenVarMap[var] != varMapBeforeLevel[var])
                    ? thenVarMap[var] : varMapBeforeLevel[var];
                auto* thenId = innerThenGraph.add_node();
                thenId->set_op_type("Identity");
                thenId->set_name("nested_then_id_" + var + "_" + std::to_string(outputIdx));
                thenId->add_input(thenTensor);
                thenId->add_output(thenOutput);

                auto* thenOutSpec = innerThenGraph.add_output();
                thenOutSpec->set_name(thenOutput);
                thenOutSpec->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::DOUBLE);

                // Else branch output
                std::string elseTensor = (elseVarMap.find(var) != elseVarMap.end() &&
                                          elseVarMap[var] != varMapBeforeLevel[var])
                    ? elseVarMap[var] : varMapBeforeLevel[var];
                auto* elseId = innerElseGraph.add_node();
                elseId->set_op_type("Identity");
                elseId->set_name("nested_else_id_" + var + "_" + std::to_string(outputIdx));
                elseId->add_input(elseTensor);
                elseId->add_output(elseOutput);

                auto* elseOutSpec = innerElseGraph.add_output();
                elseOutSpec->set_name(elseOutput);
                elseOutSpec->mutable_type()->mutable_tensor_type()->set_elem_type(onnx::TensorProto::DOUBLE);

                outputIdx++;
            }

            // Create If node in targetGraph
            auto* ifNode = targetGraph->add_node();
            ifNode->set_op_type("If");
            ifNode->set_name("nested_if_" + std::to_string(condIdx) + "_" + std::to_string(nc));
            ifNode->add_input(condTensor);

            auto* thenAttr = ifNode->add_attribute();
            thenAttr->set_name("then_branch");
            thenAttr->set_type(onnx::AttributeProto::GRAPH);
            thenAttr->mutable_g()->CopyFrom(innerThenGraph);

            auto* elseAttr = ifNode->add_attribute();
            elseAttr->set_name("else_branch");
            elseAttr->set_type(onnx::AttributeProto::GRAPH);
            elseAttr->mutable_g()->CopyFrom(innerElseGraph);

            // Add outputs and update varMap
            outputIdx = nc;
            for (const auto& var : outputOrder) {
                std::string outputTensor = "nested_if_out_" + var + "_" + std::to_string(outputIdx++);
                ifNode->add_output(outputTensor);
                varMap[var] = outputTensor;
            }

            nc = std::max({outputIdx, thenNc, elseNc});
            lc = std::max({lc, thenLc, elseLc});
        };

        // Start building nested ifs from condition[1] (condition[0] is handled at the outer level)
        buildNestedIf(1, &elseGraph, varMapFalse, falseNodeCounter, falseLoopCounter, elsePrefix);
    }
    // else: blocks.size() == 1, no else branch, varMapFalse stays unchanged

    // Find union of updated variables
    std::set<std::string> updatedVars;
    for (const auto& [var, tensor] : varMapTrue) {
        if (varMapBefore.find(var) == varMapBefore.end() || varMapBefore[var] != tensor) {
            updatedVars.insert(var);
        }
    }
    for (const auto& [var, tensor] : varMapFalse) {
        if (varMapBefore.find(var) == varMapBefore.end() || varMapBefore[var] != tensor) {
            updatedVars.insert(var);
        }
    }

    if (updatedVars.empty()) {
        // No variables were modified - nothing to do
        nodeCounter = std::max({nodeCounter, trueNodeCounter, falseNodeCounter});
        loopCounter = std::max({loopCounter, trueLoopCounter, falseLoopCounter});
        return;
    }

    // Convert the condition with prefix
    std::map<std::string, std::vector<std::string>> derivInputs;
    std::string mainCondPrefix = prefix + "maincond_";
    ConversionContext condCtx(info, graph, nodeCounter, &variableToTensor, &derivInputs, mainCondPrefix);
    std::string condTensor = ExpressionConverter::convert(conditions[0], condCtx);

    // Add outputs to both branches for all updated variables
    std::vector<std::string> outputOrder(updatedVars.begin(), updatedVars.end());

    for (const auto& var : outputOrder) {
        std::string thenOutput = "then_out_" + var + "_" + std::to_string(nodeCounter);
        std::string elseOutput = "else_out_" + var + "_" + std::to_string(nodeCounter);

        // Then branch: use updated tensor or identity from before
        std::string thenTensor = (varMapTrue.find(var) != varMapTrue.end() && varMapTrue[var] != varMapBefore[var])
            ? varMapTrue[var] : varMapBefore[var];
        auto* thenId = thenGraph.add_node();
        thenId->set_op_type("Identity");
        thenId->set_name("then_id_" + var + "_" + std::to_string(nodeCounter));
        thenId->add_input(thenTensor);
        thenId->add_output(thenOutput);

        auto* thenOutSpec = thenGraph.add_output();
        thenOutSpec->set_name(thenOutput);
        auto* thenOutType = thenOutSpec->mutable_type()->mutable_tensor_type();
        thenOutType->set_elem_type(onnx::TensorProto::DOUBLE);

        // Else branch: use updated tensor or identity from before
        std::string elseTensor = (varMapFalse.find(var) != varMapFalse.end() && varMapFalse[var] != varMapBefore[var])
            ? varMapFalse[var] : varMapBefore[var];
        auto* elseId = elseGraph.add_node();
        elseId->set_op_type("Identity");
        elseId->set_name("else_id_" + var + "_" + std::to_string(nodeCounter));
        elseId->add_input(elseTensor);
        elseId->add_output(elseOutput);

        auto* elseOutSpec = elseGraph.add_output();
        elseOutSpec->set_name(elseOutput);
        auto* elseOutType = elseOutSpec->mutable_type()->mutable_tensor_type();
        elseOutType->set_elem_type(onnx::TensorProto::DOUBLE);
    }

    // Create the If node
    auto* ifNode = graph->add_node();
    ifNode->set_op_type("If");
    ifNode->set_name("if_stmt_" + std::to_string(nodeCounter));
    ifNode->add_input(condTensor);

    auto* thenAttr = ifNode->add_attribute();
    thenAttr->set_name("then_branch");
    thenAttr->set_type(onnx::AttributeProto::GRAPH);
    thenAttr->mutable_g()->CopyFrom(thenGraph);

    auto* elseAttr = ifNode->add_attribute();
    elseAttr->set_name("else_branch");
    elseAttr->set_type(onnx::AttributeProto::GRAPH);
    elseAttr->mutable_g()->CopyFrom(elseGraph);

    // Add outputs and update variableToTensor
    for (const auto& var : outputOrder) {
        std::string outputTensor = "if_out_" + var + "_" + std::to_string(nodeCounter);
        ifNode->add_output(outputTensor);
        variableToTensor[var] = outputTensor;
    }

    nodeCounter = std::max({nodeCounter + 1, trueNodeCounter, falseNodeCounter});
    loopCounter = std::max({loopCounter, trueLoopCounter, falseLoopCounter});
}

// Process statements within an if-statement branch (adds nodes to GraphProto)
// The prefix parameter ensures unique tensor names across different subgraphs
static void processAlgorithmStatements(
    const std::vector<basemodelica::BaseModelicaParser::StatementContext*>& statements,
    onnx::GraphProto* graph,
    const Function& func,
    const ModelInfo& info,
    std::map<std::string, std::string>& variableToTensor,
    int& nodeCounter,
    int& loopCounter,
    const std::string& sourceFile,
    size_t sourceLine,
    const std::string& prefix = "") {

    for (auto* stmt : statements) {
        // Handle if-statement (recursive)
        if (stmt->ifStatement()) {
            generateAlgorithmIfStatement(
                stmt->ifStatement(), graph, func, info,
                variableToTensor, nodeCounter, loopCounter,
                sourceFile, sourceLine, prefix);
            continue;
        }

        // Handle simple assignment: var := expr
        if (stmt->componentReference() && stmt->expression()) {
            auto* compRef = stmt->componentReference();
            std::string varName = stripQuotes(compRef->IDENT(0)->getText());

            // Convert RHS expression with prefix for unique tensor names
            std::map<std::string, std::vector<std::string>> derivInputs;
            ConversionContext ctx(info, graph, nodeCounter, &variableToTensor, &derivInputs, prefix);
            std::string rhsTensor = ExpressionConverter::convert(stmt->expression(), ctx);

            // Update variable mapping
            variableToTensor[varName] = rhsTensor;
            continue;
        }

        // For now, skip for-loops and while-loops inside if-branches
        // (these would need GraphProto versions of the generators)
        if (stmt->forStatement()) {
            throw std::runtime_error("For-loops inside if-statements not yet supported");
        }
        if (stmt->whileStatement()) {
            throw std::runtime_error("While-loops inside if-statements not yet supported");
        }
    }
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
    int loopCounter = 0;

    for (size_t stmtIndex = 0; stmtIndex < func.algorithmStatements.size(); stmtIndex++) {
        const auto& stmt = func.algorithmStatements[stmtIndex];

        // Check if this is a for-statement
        if (stmt.isForStatement()) {
            auto* forStmtCtx = dynamic_cast<basemodelica::BaseModelicaParser::ForStatementContext*>(stmt.forStatementContext);
            if (forStmtCtx) {
                generateAlgorithmForLoop(forStmtCtx, functionProto, func, info,
                                         variableToTensor, nodeCounter, loopCounter,
                                         stmt.sourceFile, stmt.sourceLine);
                continue;
            }
        }

        // Check if this is a while-statement
        if (stmt.isWhileStatement()) {
            auto* whileStmtCtx = dynamic_cast<basemodelica::BaseModelicaParser::WhileStatementContext*>(stmt.whileStatementContext);
            if (whileStmtCtx) {
                generateAlgorithmWhileLoop(whileStmtCtx, functionProto, func, info,
                                           variableToTensor, nodeCounter, loopCounter,
                                           stmt.sourceFile, stmt.sourceLine);
                continue;
            }
        }

        // Check if this is an if-statement
        if (stmt.isIfStatement()) {
            auto* ifStmtCtx = dynamic_cast<basemodelica::BaseModelicaParser::IfStatementContext*>(stmt.ifStatementContext);
            if (ifStmtCtx) {
                // Use a temporary graph, then copy nodes to functionProto
                onnx::GraphProto tempGraph;
                generateAlgorithmIfStatement(ifStmtCtx, &tempGraph, func, info,
                                             variableToTensor, nodeCounter, loopCounter,
                                             stmt.sourceFile, stmt.sourceLine);

                // Copy initializers as Constant nodes
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

                // Copy nodes to functionProto
                for (int i = 0; i < tempGraph.node_size(); i++) {
                    auto* node = functionProto->add_node();
                    node->CopyFrom(tempGraph.node(i));
                    addSourceLocationMetadata(node, stmt.sourceFile, stmt.sourceLine);
                }

                continue;
            }
        }

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
        bool isFullSliceAssignment = false;  // For result[:] := ...
        bool isRowAssignment = false;        // For result[i,:] := ...
        bool isColumnAssignment = false;     // For result[:,j] := ...
        bool isRangeAssignment = false;      // For result[2:3] := ... (1D range)
        int64_t rowOrColIndex = -1;          // The index value for row/column assignment

        if (lhsCompRef) {
            baseVarName = stripQuotes(lhsCompRef->IDENT(0)->getText());

            auto arraySubscripts = lhsCompRef->arraySubscripts();
            if (!arraySubscripts.empty()) {
                auto subscriptList = arraySubscripts[0]->subscript();

                // Check if all subscripts are full slices (":")
                bool allFullSlices = true;
                for (auto sub : subscriptList) {
                    if (sub->getText() != ":") {
                        allFullSlices = false;
                        break;
                    }
                }

                if (allFullSlices) {
                    // result[:] or result[:,:] - assign to whole array
                    isFullSliceAssignment = true;
                } else if (subscriptList.size() == 2) {
                    // 2D case - check for row or column assignment
                    bool firstIsSlice = (subscriptList[0]->getText() == ":");
                    bool secondIsSlice = (subscriptList[1]->getText() == ":");

                    if (!firstIsSlice && secondIsSlice) {
                        // result[i,:] - row assignment
                        auto subExpr = subscriptList[0]->expression();
                        if (subExpr) {
                            try {
                                rowOrColIndex = std::stoi(subExpr->getText()) - 1;  // 0-based
                                isRowAssignment = true;
                            } catch (...) {
                                throw std::runtime_error("Dynamic row index not yet supported: " + subExpr->getText());
                            }
                        }
                    } else if (firstIsSlice && !secondIsSlice) {
                        // result[:,j] - column assignment
                        auto subExpr = subscriptList[1]->expression();
                        if (subExpr) {
                            try {
                                rowOrColIndex = std::stoi(subExpr->getText()) - 1;  // 0-based
                                isColumnAssignment = true;
                            } catch (...) {
                                throw std::runtime_error("Dynamic column index not yet supported: " + subExpr->getText());
                            }
                        }
                    } else if (!firstIsSlice && !secondIsSlice) {
                        // result[i,j] - scalar indexed assignment
                        for (auto sub : subscriptList) {
                            if (auto subExpr = sub->expression()) {
                                try {
                                    int modelicaIndex = std::stoi(subExpr->getText());
                                    lhsIndices.push_back(modelicaIndex - 1);
                                    isIndexedAssignment = true;
                                } catch (...) {
                                    throw std::runtime_error("Dynamic indices not yet supported: " + subExpr->getText());
                                }
                            }
                        }
                    } else {
                        throw std::runtime_error("Unsupported 2D slice pattern on LHS");
                    }
                } else if (subscriptList.size() >= 3) {
                    // N-D case (3D, 4D, etc.): result[i,j,k,...] - scalar indexed assignment
                    // Check that all subscripts are static indices (no slices)
                    bool allStaticIndices = true;
                    for (auto sub : subscriptList) {
                        if (sub->getText() == ":") {
                            allStaticIndices = false;
                            break;
                        }
                    }
                    if (allStaticIndices) {
                        for (auto sub : subscriptList) {
                            if (auto subExpr = sub->expression()) {
                                try {
                                    int modelicaIndex = std::stoi(subExpr->getText());
                                    lhsIndices.push_back(modelicaIndex - 1);
                                    isIndexedAssignment = true;
                                } catch (...) {
                                    throw std::runtime_error("Dynamic indices not yet supported in N-D array: " + subExpr->getText());
                                }
                            }
                        }
                    } else {
                        throw std::runtime_error("Slice patterns not yet supported for 3D+ arrays on LHS");
                    }
                } else if (subscriptList.size() == 1 && subscriptList[0]->getText() != ":") {
                    // 1D case: result[i] or result[i:j]
                    auto subExpr = subscriptList[0]->expression();
                    if (subExpr) {
                        if (ParseTreeNavigator::isRangeExpression(subExpr)) {
                            // Range subscript: result[2:3] - store start and end for later
                            try {
                                auto [start, end] = ParseTreeNavigator::parseRangeBounds(subExpr);
                                // Generate all indices in the range (convert to 0-based)
                                for (int64_t idx = start - 1; idx < end; idx++) {
                                    lhsIndices.push_back(idx);
                                }
                                isIndexedAssignment = true;
                                isRangeAssignment = true;
                            } catch (...) {
                                throw std::runtime_error("Dynamic range index not yet supported: " + subExpr->getText());
                            }
                        } else {
                            // Simple index
                            try {
                                int modelicaIndex = std::stoi(subExpr->getText());
                                lhsIndices.push_back(modelicaIndex - 1);
                                isIndexedAssignment = true;
                            } catch (...) {
                                throw std::runtime_error("Dynamic index not yet supported: " + subExpr->getText());
                            }
                        }
                    }
                } else {
                    throw std::runtime_error("Unsupported LHS subscript pattern");
                }
            }
        }

        std::string lhsVarName = (isIndexedAssignment || isFullSliceAssignment || isRowAssignment || isColumnAssignment)
            ? baseVarName : stripQuotes(stmt.lhsContext->getText());

        try {
            onnx::GraphProto tempGraph;
            std::map<std::string, std::vector<std::string>> localDerivativeInputs;
            ConversionContext funcCtx(info, &tempGraph, nodeCounter, &variableToTensor, &localDerivativeInputs);
            std::string rhsTensor = ExpressionConverter::convert(stmt.rhsContext, funcCtx);

            std::string finalTensor = rhsTensor;

            if (isIndexedAssignment || isRowAssignment || isColumnAssignment) {
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

                if (isRowAssignment) {
                    // result[i,:] := row[:] - update entire row i
                    // Unsqueeze row to [1, n] shape
                    std::string unsqueezedRow = builder.addUnsqueeze(rhsTensor, {0});
                    // ScatterND with index [[i]]
                    finalTensor = builder.addScatterND(currentTensor, {rowOrColIndex}, unsqueezedRow);
                } else if (isColumnAssignment) {
                    // result[:,j] := col[:] - update entire column j
                    // Transpose, scatter as row, transpose back
                    std::string transposed = builder.addTranspose(currentTensor, {1, 0});
                    std::string unsqueezedCol = builder.addUnsqueeze(rhsTensor, {0});
                    std::string scattered = builder.addScatterND(transposed, {rowOrColIndex}, unsqueezedCol);
                    finalTensor = builder.addTranspose(scattered, {1, 0});
                } else {
                    // Indexed assignment (scalar or range)
                    if (!isRangeAssignment) {
                        // Single element (1D, 2D, 3D, etc.): Unsqueeze RHS to be [1] shaped for ScatterND updates
                        std::string updatesTensor = builder.addUnsqueeze(rhsTensor, {0});
                        finalTensor = builder.addScatterND(currentTensor, lhsIndices, updatesTensor);
                    } else {
                        // Range assignment (e.g., result[2:3] := {val1, val2})
                        // RHS is already an array with the right shape, use ScatterND1D
                        finalTensor = builder.addScatterND1D(currentTensor, lhsIndices, rhsTensor);
                    }
                }
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
