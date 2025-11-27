// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include <string>
#include <vector>

// Forward declare ONNX types
namespace onnx {
    class GraphProto;
    class TypeProto_Tensor;
    class ValueInfoProto;
}

namespace lacemodelica {

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

} // namespace lacemodelica
