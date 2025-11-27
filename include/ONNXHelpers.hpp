// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include <string>
#include <vector>
#include <map>

// Forward declarations
namespace onnx {
    class GraphProto;
    class TypeProto_Tensor;
    class ValueInfoProto;
    class TensorShapeProto;
}

namespace lacemodelica {

class ModelInfo;  // Forward declaration

// Context object bundling common parameters passed through expression conversion.
// This reduces the number of parameters from 6-7 down to 1, making function
// signatures cleaner and the code more maintainable.
struct ConversionContext {
    const ModelInfo& info;
    onnx::GraphProto* graph;
    int& nodeCounter;
    const std::map<std::string, std::string>* variableMap;
    std::map<std::string, std::vector<std::string>>* derivativeInputs;
    std::string tensorPrefix;

    ConversionContext(
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        const std::map<std::string, std::string>* variableMap = nullptr,
        std::map<std::string, std::vector<std::string>>* derivativeInputs = nullptr,
        const std::string& tensorPrefix = "")
        : info(info)
        , graph(graph)
        , nodeCounter(nodeCounter)
        , variableMap(variableMap)
        , derivativeInputs(derivativeInputs)
        , tensorPrefix(tensorPrefix)
    {}

    // Create a child context with a different graph (for subgraphs)
    ConversionContext withGraph(onnx::GraphProto* newGraph) const {
        return ConversionContext(info, newGraph, nodeCounter, variableMap, derivativeInputs, tensorPrefix);
    }

    // Create a child context with a different prefix
    ConversionContext withPrefix(const std::string& newPrefix) const {
        return ConversionContext(info, graph, nodeCounter, variableMap, derivativeInputs, newPrefix);
    }
};

// Generate a unique tensor name with optional prefix
std::string makeTensorName(const std::string& prefix, int& counter);

// Create ONNX Constant nodes for scalar values
std::string createInt64Constant(onnx::GraphProto* graph, int64_t value, int& counter,
                                 const std::string& nameHint = "");
std::string createDoubleConstant(onnx::GraphProto* graph, double value, int& counter);
std::string createBoolConstant(onnx::GraphProto* graph, bool value, int& counter);

// Create an Identity node to rename/copy a tensor
std::string createIdentityNode(onnx::GraphProto* graph, const std::string& inputTensor,
                                const std::string& outputName, const std::string& nodeName);

// Convert Modelica 1-based index to ONNX 0-based index
std::string convertTo0BasedIndex(onnx::GraphProto* graph, const std::string& oneBasedTensor,
                                  int& counter, const std::string& prefix = "");

// Create a Gather node for dynamic array indexing
std::string createGatherNode(onnx::GraphProto* graph, const std::string& dataTensor,
                              const std::string& indexTensor, int axis, int& counter,
                              const std::string& prefix = "");

// Add source location metadata to an ONNX value info
void addSourceMetadata(onnx::ValueInfoProto* valueInfo, const std::string& sourceFile, size_t sourceLine);

// Configure tensor type with element type and optional dimensions
void configureTensorType(onnx::TypeProto_Tensor* tensorType, int elemType,
                         const std::vector<std::string>& dimensions = {});

// Create binary operation nodes (Add, Sub, Mul, Div, Pow, etc.)
std::string createBinaryOp(onnx::GraphProto* graph, const std::string& opType,
                            const std::string& left, const std::string& right,
                            int& counter, const std::string& prefix = "");

// Create unary operation nodes (Neg, Not, Sin, Cos, etc.)
std::string createUnaryOp(onnx::GraphProto* graph, const std::string& opType,
                           const std::string& input, int& counter,
                           const std::string& prefix = "");

// Add a scalar double output to a subgraph (used for If branches)
void addScalarDoubleOutput(onnx::GraphProto* graph, const std::string& tensorName);

// Create an If node with then/else branch subgraphs
std::string createIfNode(onnx::GraphProto* graph, const std::string& condTensor,
                          onnx::GraphProto& thenBranch, onnx::GraphProto& elseBranch,
                          int& counter, const std::string& prefix = "",
                          const std::string& nameHint = "If");

// Create a Constant node with an int64 array (for indices)
std::string createInt64ArrayConstant(onnx::GraphProto* graph, const std::vector<int64_t>& values, int& counter);

// Create a GatherND node for multi-dimensional static indexing
std::string createGatherNDNode(onnx::GraphProto* graph, const std::string& dataTensor,
                                const std::vector<int64_t>& indices, int& counter,
                                const std::string& prefix = "");

// Add dimensions to a tensor shape, parsing numeric dimensions as values
// and treating non-numeric ones as symbolic parameters
void addShapeDimensions(onnx::TensorShapeProto* shape, const std::vector<std::string>& dimensions);

} // namespace lacemodelica
