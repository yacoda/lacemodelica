// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ONNXHelpers.hpp"
#include "GraphBuilder.h"

#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>

namespace lacemodelica {

// -----------------------------------------------------------------------------
// Free Functions (delegate to GraphBuilder for implementation)
// -----------------------------------------------------------------------------
// These functions provide backward compatibility while GraphBuilder
// provides the authoritative implementation.

std::string makeTensorName(const std::string& prefix, int& counter) {
    GraphBuilder builder(nullptr, counter, prefix);
    // We can't use builder directly since it needs a graph, so inline the logic
    if (prefix.empty()) {
        return "tensor_" + std::to_string(counter++);
    }
    return prefix + "_tensor_" + std::to_string(counter++);
}

std::string createInt64Constant(onnx::GraphProto* graph, int64_t value, int& counter,
                                 const std::string& nameHint) {
    GraphBuilder builder(graph, counter);
    return builder.addInt64Constant(value, nameHint);
}

std::string createDoubleConstant(onnx::GraphProto* graph, double value, int& counter) {
    GraphBuilder builder(graph, counter);
    return builder.addDoubleConstant(value);
}

std::string createBoolConstant(onnx::GraphProto* graph, bool value, int& counter) {
    GraphBuilder builder(graph, counter);
    return builder.addBoolConstant(value);
}

std::string createIdentityNode(onnx::GraphProto* graph, const std::string& inputTensor,
                                const std::string& outputName, const std::string& nodeName) {
    int dummyCounter = 0;
    GraphBuilder builder(graph, dummyCounter);
    return builder.addIdentity(inputTensor, outputName, nodeName);
}

std::string convertTo0BasedIndex(onnx::GraphProto* graph, const std::string& oneBasedTensor,
                                  int& counter, const std::string& prefix) {
    GraphBuilder builder(graph, counter, prefix);
    return builder.convertToZeroBasedIndex(oneBasedTensor);
}

std::string createGatherNode(onnx::GraphProto* graph, const std::string& dataTensor,
                              const std::string& indexTensor, int axis, int& counter,
                              const std::string& prefix) {
    GraphBuilder builder(graph, counter, prefix);
    return builder.addGather(dataTensor, indexTensor, axis);
}

std::string createBinaryOp(onnx::GraphProto* graph, const std::string& opType,
                            const std::string& left, const std::string& right,
                            int& counter, const std::string& prefix) {
    GraphBuilder builder(graph, counter, prefix);
    return builder.addBinaryOp(opType, left, right);
}

std::string createUnaryOp(onnx::GraphProto* graph, const std::string& opType,
                           const std::string& input, int& counter,
                           const std::string& prefix) {
    GraphBuilder builder(graph, counter, prefix);
    return builder.addUnaryOp(opType, input);
}

void addScalarDoubleOutput(onnx::GraphProto* graph, const std::string& tensorName) {
    int dummyCounter = 0;
    GraphBuilder builder(graph, dummyCounter);
    builder.addScalarDoubleOutput(tensorName);
}

std::string createIfNode(onnx::GraphProto* graph, const std::string& condTensor,
                          onnx::GraphProto& thenBranch, onnx::GraphProto& elseBranch,
                          int& counter, const std::string& prefix,
                          const std::string& nameHint) {
    GraphBuilder builder(graph, counter, prefix);
    return builder.addIfNode(condTensor, thenBranch, elseBranch, nameHint);
}

std::string createInt64ArrayConstant(onnx::GraphProto* graph, const std::vector<int64_t>& values, int& counter) {
    GraphBuilder builder(graph, counter);
    return builder.addInt64ArrayConstant(values);
}

std::string createGatherNDNode(onnx::GraphProto* graph, const std::string& dataTensor,
                                const std::vector<int64_t>& indices, int& counter,
                                const std::string& prefix) {
    GraphBuilder builder(graph, counter, prefix);
    return builder.addGatherND(dataTensor, indices);
}

void addShapeDimensions(onnx::TensorShapeProto* shape, const std::vector<std::string>& dimensions) {
    int dummyCounter = 0;
    GraphBuilder builder(nullptr, dummyCounter);
    builder.addShapeDimensions(shape, dimensions);
}

SubscriptAnalysis analyzeSubscripts(
    const std::vector<basemodelica::BaseModelicaParser::SubscriptContext*>& subscriptList,
    const std::map<std::string, std::string>* variableMap) {
    int dummyCounter = 0;
    GraphBuilder builder(nullptr, dummyCounter);
    auto analysis = builder.analyzeSubscripts(subscriptList, variableMap);

    // Convert GraphBuilder::SubscriptAnalysis to SubscriptAnalysis
    SubscriptAnalysis result;
    result.hasLoopVariable = analysis.hasLoopVariable;
    result.staticIndices = analysis.staticIndices;
    return result;
}

std::string applyArraySubscripts(
    onnx::GraphProto* graph,
    const std::string& baseTensor,
    const std::vector<basemodelica::BaseModelicaParser::SubscriptContext*>& subscriptList,
    const std::map<std::string, std::string>* variableMap,
    int& nodeCounter,
    const std::string& tensorPrefix) {
    GraphBuilder builder(graph, nodeCounter, tensorPrefix);
    return builder.applySubscripts(baseTensor, subscriptList, variableMap);
}

bool renameTensorToOutput(
    onnx::GraphProto* graph,
    const std::string& tensorName,
    const std::string& outputName,
    const std::string& identityNamePrefix) {
    int dummyCounter = 0;
    GraphBuilder builder(graph, dummyCounter);
    return builder.renameTensor(tensorName, outputName, identityNamePrefix);
}

void addLoopPassthrough(
    onnx::NodeProto* loopNode,
    onnx::GraphProto* bodyGraph,
    const std::string& loopNodeName,
    const std::string& inputName,
    const std::string& bodyInputName,
    int elemType,
    const std::vector<std::string>& dimensions,
    const std::string& outputSuffix) {
    int dummyCounter = 0;
    GraphBuilder builder(bodyGraph, dummyCounter);
    builder.addLoopPassthrough(loopNode, bodyGraph, loopNodeName, inputName,
                               bodyInputName, elemType, dimensions, outputSuffix);
}

} // namespace lacemodelica
