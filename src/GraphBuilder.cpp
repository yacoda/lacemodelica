// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "GraphBuilder.h"
#include "ParseTreeNavigator.h"

#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>

#include <stdexcept>

namespace lacemodelica {

GraphBuilder::GraphBuilder(onnx::GraphProto* graph, int& nodeCounter)
    : graph_(graph), nodeCounter_(nodeCounter), prefix_() {}

GraphBuilder::GraphBuilder(onnx::GraphProto* graph, int& nodeCounter, const std::string& prefix)
    : graph_(graph), nodeCounter_(nodeCounter), prefix_(prefix) {}

GraphBuilder GraphBuilder::forSubgraph(onnx::GraphProto* subgraph, const std::string& subPrefix) const {
    return GraphBuilder(subgraph, nodeCounter_, subPrefix);
}

GraphBuilder GraphBuilder::withPrefix(const std::string& newPrefix) const {
    return GraphBuilder(graph_, nodeCounter_, newPrefix);
}

// -----------------------------------------------------------------------------
// Tensor Naming
// -----------------------------------------------------------------------------
// ONNX validates SSA across all subgraphs, so names must be globally unique.
// The prefix_ is used to ensure uniqueness across nested scopes.

std::string GraphBuilder::makeTensorName() {
    if (prefix_.empty()) {
        return "t" + std::to_string(nodeCounter_++);
    }
    return prefix_ + "_t" + std::to_string(nodeCounter_++);
}

std::string GraphBuilder::makeTensorName(const std::string& hint) {
    if (hint.empty()) {
        return makeTensorName();
    }
    if (prefix_.empty()) {
        return hint + "_" + std::to_string(nodeCounter_++);
    }
    return prefix_ + "_" + hint + "_" + std::to_string(nodeCounter_++);
}

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

std::string GraphBuilder::addInt64Constant(int64_t value) {
    return addInt64Constant(value, "");
}

std::string GraphBuilder::addInt64Constant(int64_t value, const std::string& nameHint) {
    // Use prefix for global SSA uniqueness
    std::string base = nameHint.empty() ? "i64" : nameHint;
    std::string name = prefix_.empty()
        ? (base + "_" + std::to_string(nodeCounter_++))
        : (prefix_ + "_" + base + "_" + std::to_string(nodeCounter_++));

    auto* node = graph_->add_node();
    node->set_op_type("Constant");
    node->set_name(name);
    node->add_output(name);

    auto* attr = node->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto::TENSOR);
    auto* tensor = attr->mutable_t();
    tensor->set_data_type(onnx::TensorProto::INT64);
    tensor->add_int64_data(value);

    return name;
}

std::string GraphBuilder::addDoubleConstant(double value) {
    std::string name = prefix_.empty()
        ? ("f64_" + std::to_string(nodeCounter_++))
        : (prefix_ + "_f64_" + std::to_string(nodeCounter_++));

    auto* node = graph_->add_node();
    node->set_op_type("Constant");
    node->set_name(name);
    node->add_output(name);

    auto* attr = node->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto::TENSOR);
    auto* tensor = attr->mutable_t();
    tensor->set_data_type(onnx::TensorProto::DOUBLE);
    tensor->add_double_data(value);

    return name;
}

std::string GraphBuilder::addBoolConstant(bool value) {
    std::string name = prefix_.empty()
        ? ("bool_" + std::to_string(nodeCounter_++))
        : (prefix_ + "_bool_" + std::to_string(nodeCounter_++));

    auto* node = graph_->add_node();
    node->set_op_type("Constant");
    node->set_name(name);
    node->add_output(name);

    auto* attr = node->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto::TENSOR);
    auto* tensor = attr->mutable_t();
    tensor->set_data_type(onnx::TensorProto::BOOL);
    tensor->add_int32_data(value ? 1 : 0);

    return name;
}

std::string GraphBuilder::addInt64ArrayConstant(const std::vector<int64_t>& values) {
    std::string name = prefix_.empty()
        ? ("indices_" + std::to_string(nodeCounter_++))
        : (prefix_ + "_indices_" + std::to_string(nodeCounter_++));

    auto* node = graph_->add_node();
    node->set_op_type("Constant");
    node->set_name(name);
    node->add_output(name);

    auto* attr = node->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto::TENSOR);
    auto* tensor = attr->mutable_t();
    tensor->set_data_type(onnx::TensorProto::INT64);
    tensor->add_dims(values.size());
    for (int64_t val : values) {
        tensor->add_int64_data(val);
    }

    return name;
}

std::string GraphBuilder::addDoubleZerosConstant(const std::vector<int64_t>& shape) {
    std::string name = prefix_.empty()
        ? ("zeros_" + std::to_string(nodeCounter_++))
        : (prefix_ + "_zeros_" + std::to_string(nodeCounter_++));

    auto* node = graph_->add_node();
    node->set_op_type("Constant");
    node->set_name(name);
    node->add_output(name);

    auto* attr = node->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto::TENSOR);
    auto* tensor = attr->mutable_t();
    tensor->set_data_type(onnx::TensorProto::DOUBLE);

    // Calculate total size and add dimensions
    int64_t totalSize = 1;
    for (int64_t dim : shape) {
        tensor->add_dims(dim);
        totalSize *= dim;
    }

    // Initialize with zeros
    for (int64_t i = 0; i < totalSize; i++) {
        tensor->add_double_data(0.0);
    }

    return name;
}

// -----------------------------------------------------------------------------
// Operations
// -----------------------------------------------------------------------------

std::string GraphBuilder::addUnaryOp(const std::string& opType, const std::string& input) {
    std::string outputTensor = makeTensorName();

    auto* node = graph_->add_node();
    node->set_op_type(opType);
    node->set_name(outputTensor + "_" + opType);
    node->add_input(input);
    node->add_output(outputTensor);

    return outputTensor;
}

std::string GraphBuilder::addBinaryOp(const std::string& opType,
                                       const std::string& left,
                                       const std::string& right) {
    std::string outputTensor = makeTensorName();

    auto* node = graph_->add_node();
    node->set_op_type(opType);
    node->set_name(outputTensor + "_" + opType);
    node->add_input(left);
    node->add_input(right);
    node->add_output(outputTensor);

    return outputTensor;
}

// -----------------------------------------------------------------------------
// Indexing
// -----------------------------------------------------------------------------

std::string GraphBuilder::addGather(const std::string& data,
                                     const std::string& indices,
                                     int axis) {
    std::string outputTensor = makeTensorName();

    auto* node = graph_->add_node();
    node->set_op_type("Gather");
    node->set_name(outputTensor + "_Gather");
    node->add_input(data);
    node->add_input(indices);
    node->add_output(outputTensor);

    auto* axisAttr = node->add_attribute();
    axisAttr->set_name("axis");
    axisAttr->set_type(onnx::AttributeProto::INT);
    axisAttr->set_i(axis);

    return outputTensor;
}

std::string GraphBuilder::addGatherND(const std::string& data,
                                       const std::vector<int64_t>& indices) {
    std::string indexTensor = addInt64ArrayConstant(indices);
    std::string outputTensor = makeTensorName();

    auto* node = graph_->add_node();
    node->set_op_type("GatherND");
    node->set_name(outputTensor + "_GatherND");
    node->add_input(data);
    node->add_input(indexTensor);
    node->add_output(outputTensor);

    return outputTensor;
}

std::string GraphBuilder::addScatterND(const std::string& data,
                                        const std::vector<int64_t>& indices,
                                        const std::string& updates) {
    // Create indices tensor with shape [1, rank] for updating a single element
    // For example, to update index [0] in a 1D array, indices should be [[0]]
    std::string name = prefix_.empty()
        ? ("scatter_idx_" + std::to_string(nodeCounter_++))
        : (prefix_ + "_scatter_idx_" + std::to_string(nodeCounter_++));

    auto* constNode = graph_->add_node();
    constNode->set_op_type("Constant");
    constNode->set_name(name);
    constNode->add_output(name);

    auto* attr = constNode->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto::TENSOR);
    auto* tensor = attr->mutable_t();
    tensor->set_data_type(onnx::TensorProto::INT64);
    tensor->add_dims(1);  // batch dimension
    tensor->add_dims(indices.size());  // index rank
    for (int64_t val : indices) {
        tensor->add_int64_data(val);
    }

    std::string outputTensor = makeTensorName();

    auto* node = graph_->add_node();
    node->set_op_type("ScatterND");
    node->set_name(outputTensor + "_ScatterND");
    node->add_input(data);
    node->add_input(name);
    node->add_input(updates);
    node->add_output(outputTensor);

    return outputTensor;
}

std::string GraphBuilder::addScatterND1D(const std::string& data,
                                          const std::vector<int64_t>& indices1D,
                                          const std::string& updates) {
    // Create indices tensor with shape [n, 1] for updating multiple 1D positions
    // For example, to update indices [1, 2] in a 1D array, indices should be [[1], [2]]
    std::string name = prefix_.empty()
        ? ("scatter_idx_" + std::to_string(nodeCounter_++))
        : (prefix_ + "_scatter_idx_" + std::to_string(nodeCounter_++));

    auto* constNode = graph_->add_node();
    constNode->set_op_type("Constant");
    constNode->set_name(name);
    constNode->add_output(name);

    auto* attr = constNode->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto::TENSOR);
    auto* tensor = attr->mutable_t();
    tensor->set_data_type(onnx::TensorProto::INT64);
    tensor->add_dims(indices1D.size());  // number of elements to update
    tensor->add_dims(1);                  // each index is 1D
    for (int64_t val : indices1D) {
        tensor->add_int64_data(val);
    }

    std::string outputTensor = makeTensorName();

    auto* node = graph_->add_node();
    node->set_op_type("ScatterND");
    node->set_name(outputTensor + "_ScatterND");
    node->add_input(data);
    node->add_input(name);
    node->add_input(updates);
    node->add_output(outputTensor);

    return outputTensor;
}

std::string GraphBuilder::addUnsqueeze(const std::string& input,
                                        const std::vector<int64_t>& axes) {
    std::string axesTensor = addInt64ArrayConstant(axes);
    std::string outputTensor = makeTensorName();

    auto* node = graph_->add_node();
    node->set_op_type("Unsqueeze");
    node->set_name(outputTensor + "_Unsqueeze");
    node->add_input(input);
    node->add_input(axesTensor);
    node->add_output(outputTensor);

    return outputTensor;
}

std::string GraphBuilder::addSqueeze(const std::string& input,
                                      const std::vector<int64_t>& axes) {
    std::string axesTensor = addInt64ArrayConstant(axes);
    std::string outputTensor = makeTensorName();

    auto* node = graph_->add_node();
    node->set_op_type("Squeeze");
    node->set_name(outputTensor + "_Squeeze");
    node->add_input(input);
    node->add_input(axesTensor);
    node->add_output(outputTensor);

    return outputTensor;
}

std::string GraphBuilder::addSlice(const std::string& data,
                                    const std::vector<int64_t>& starts,
                                    const std::vector<int64_t>& ends,
                                    const std::vector<int64_t>& axes) {
    std::string startsTensor = addInt64ArrayConstant(starts);
    std::string endsTensor = addInt64ArrayConstant(ends);
    std::string axesTensor = addInt64ArrayConstant(axes);
    std::string outputTensor = makeTensorName();

    auto* node = graph_->add_node();
    node->set_op_type("Slice");
    node->set_name(outputTensor + "_Slice");
    node->add_input(data);
    node->add_input(startsTensor);
    node->add_input(endsTensor);
    node->add_input(axesTensor);
    node->add_output(outputTensor);

    return outputTensor;
}

std::string GraphBuilder::addConcat(const std::vector<std::string>& inputs, int64_t axis) {
    if (inputs.empty()) {
        throw std::runtime_error("Concat requires at least one input");
    }
    if (inputs.size() == 1) {
        return inputs[0];  // No concat needed for single input
    }

    std::string outputTensor = makeTensorName("concat");

    auto* node = graph_->add_node();
    node->set_op_type("Concat");
    node->set_name(outputTensor + "_Concat");
    for (const auto& input : inputs) {
        node->add_input(input);
    }
    node->add_output(outputTensor);

    auto* axisAttr = node->add_attribute();
    axisAttr->set_name("axis");
    axisAttr->set_type(onnx::AttributeProto::INT);
    axisAttr->set_i(axis);

    return outputTensor;
}

std::string GraphBuilder::addTranspose(const std::string& input, const std::vector<int64_t>& perm) {
    std::string outputTensor = makeTensorName("transposed");

    auto* node = graph_->add_node();
    node->set_op_type("Transpose");
    node->set_name(outputTensor + "_Transpose");
    node->add_input(input);
    node->add_output(outputTensor);

    auto* permAttr = node->add_attribute();
    permAttr->set_name("perm");
    permAttr->set_type(onnx::AttributeProto::INTS);
    for (int64_t p : perm) {
        permAttr->add_ints(p);
    }

    return outputTensor;
}

std::string GraphBuilder::convertToZeroBasedIndex(const std::string& oneBasedTensor) {
    std::string constOne = addInt64Constant(1, "one");

    std::string zeroBasedTensor = prefix_.empty()
        ? ("idx0_" + std::to_string(nodeCounter_++))
        : (prefix_ + "_idx0_" + std::to_string(nodeCounter_++));

    auto* subNode = graph_->add_node();
    subNode->set_op_type("Sub");
    std::string nodeName = prefix_.empty()
        ? ("to_0based_" + std::to_string(nodeCounter_))
        : (prefix_ + "_to_0based_" + std::to_string(nodeCounter_));
    subNode->set_name(nodeName);
    subNode->add_input(oneBasedTensor);
    subNode->add_input(constOne);
    subNode->add_output(zeroBasedTensor);

    return zeroBasedTensor;
}

// -----------------------------------------------------------------------------
// Control Flow
// -----------------------------------------------------------------------------

std::string GraphBuilder::addIfNode(const std::string& condition,
                                     onnx::GraphProto& thenBranch,
                                     onnx::GraphProto& elseBranch,
                                     const std::string& nameHint) {
    std::string outputTensor = makeTensorName();

    auto* ifNode = graph_->add_node();
    ifNode->set_op_type("If");
    ifNode->set_name(nameHint + "_" + std::to_string(nodeCounter_));
    ifNode->add_input(condition);
    ifNode->add_output(outputTensor);

    auto* thenAttr = ifNode->add_attribute();
    thenAttr->set_name("then_branch");
    thenAttr->set_type(onnx::AttributeProto::GRAPH);
    thenAttr->mutable_g()->CopyFrom(thenBranch);

    auto* elseAttr = ifNode->add_attribute();
    elseAttr->set_name("else_branch");
    elseAttr->set_type(onnx::AttributeProto::GRAPH);
    elseAttr->mutable_g()->CopyFrom(elseBranch);

    return outputTensor;
}

// -----------------------------------------------------------------------------
// Identity / Renaming
// -----------------------------------------------------------------------------

std::string GraphBuilder::addIdentity(const std::string& input,
                                       const std::string& outputName,
                                       const std::string& nodeName) {
    auto* node = graph_->add_node();
    node->set_op_type("Identity");
    node->set_name(nodeName);
    node->add_input(input);
    node->add_output(outputName);
    return outputName;
}

bool GraphBuilder::renameTensor(const std::string& tensorName,
                                 const std::string& newName,
                                 const std::string& identityPrefix) {
    // Search backwards through nodes to find the producer
    for (int j = graph_->node_size() - 1; j >= 0; j--) {
        auto* node = graph_->mutable_node(j);
        for (int k = 0; k < node->output_size(); k++) {
            if (node->output(k) == tensorName) {
                node->set_output(k, newName);
                return true;
            }
        }
    }

    // No producer found - tensor is a direct input; create Identity
    addIdentity(tensorName, newName, identityPrefix);
    return false;
}

// -----------------------------------------------------------------------------
// Graph Structure
// -----------------------------------------------------------------------------

void GraphBuilder::addScalarDoubleOutput(const std::string& tensorName) {
    auto* output = graph_->add_output();
    output->set_name(tensorName);
    auto* tensorType = output->mutable_type()->mutable_tensor_type();
    tensorType->set_elem_type(onnx::TensorProto::DOUBLE);
    tensorType->mutable_shape()->add_dim()->set_dim_value(1);
}

void GraphBuilder::addShapeDimensions(onnx::TensorShapeProto* shape,
                                       const std::vector<std::string>& dimensions) {
    for (const auto& dim : dimensions) {
        auto* shapeDim = shape->add_dim();
        try {
            shapeDim->set_dim_value(std::stoi(dim));
        } catch (...) {
            shapeDim->set_dim_param(dim);
        }
    }
}

// -----------------------------------------------------------------------------
// Array Subscripts
// -----------------------------------------------------------------------------

GraphBuilder::SubscriptAnalysis GraphBuilder::analyzeSubscripts(
    const std::vector<basemodelica::BaseModelicaParser::SubscriptContext*>& subscripts,
    const std::map<std::string, std::string>* variableMap) {

    SubscriptAnalysis result;

    for (auto sub : subscripts) {
        // Full slice ":"
        if (sub->getText() == ":") {
            result.hasLoopVariable = true;
            return result;
        }

        auto subExpr = sub->expression();
        if (!subExpr) {
            continue;
        }

        // Range expression "2:4"
        if (ParseTreeNavigator::isRangeExpression(subExpr)) {
            result.hasLoopVariable = true;
            return result;
        }

        std::string indexExpr = subExpr->getText();

        if (variableMap && variableMap->count(indexExpr) > 0) {
            result.hasLoopVariable = true;
            return result;
        }

        try {
            int modelicaIndex = std::stoi(indexExpr);
            result.staticIndices.push_back(modelicaIndex - 1);  // Convert to 0-based
        } catch (...) {
            result.hasLoopVariable = true;
            return result;
        }
    }

    return result;
}

std::string GraphBuilder::applySubscripts(
    const std::string& baseTensor,
    const std::vector<basemodelica::BaseModelicaParser::SubscriptContext*>& subscripts,
    const std::map<std::string, std::string>* variableMap) {

    auto analysis = analyzeSubscripts(subscripts, variableMap);

    if (!analysis.hasLoopVariable) {
        return addGatherND(baseTensor, analysis.staticIndices);
    }

    // Handle slices and mixed indexing
    std::string currentTensor = baseTensor;

    // Collect information about each subscript
    struct SubscriptInfo {
        enum Type { FULL_SLICE, RANGE_SLICE, STATIC_INDEX, DYNAMIC_INDEX };
        Type type;
        int64_t start = 0;      // For RANGE_SLICE: 0-based start
        int64_t end = 0;        // For RANGE_SLICE: 0-based exclusive end
        int64_t staticIdx = 0;  // For STATIC_INDEX: 0-based index
        std::string loopVar;    // For DYNAMIC_INDEX: loop variable name
    };

    std::vector<SubscriptInfo> subInfos;
    for (size_t dimIdx = 0; dimIdx < subscripts.size(); dimIdx++) {
        auto sub = subscripts[dimIdx];
        SubscriptInfo info;

        if (sub->getText() == ":") {
            info.type = SubscriptInfo::FULL_SLICE;
        } else {
            auto subExpr = sub->expression();
            if (!subExpr) {
                throw std::runtime_error("Invalid array subscript");
            }

            if (ParseTreeNavigator::isRangeExpression(subExpr)) {
                info.type = SubscriptInfo::RANGE_SLICE;
                auto [start, end] = ParseTreeNavigator::parseRangeBounds(subExpr);
                info.start = start - 1;  // Convert to 0-based
                info.end = end;          // Modelica end is inclusive, ONNX is exclusive, so don't subtract
            } else {
                std::string indexExpr = subExpr->getText();
                if (variableMap && variableMap->count(indexExpr) > 0) {
                    info.type = SubscriptInfo::DYNAMIC_INDEX;
                    info.loopVar = indexExpr;
                } else {
                    try {
                        info.type = SubscriptInfo::STATIC_INDEX;
                        info.staticIdx = std::stoi(indexExpr) - 1;  // Convert to 0-based
                    } catch (...) {
                        throw std::runtime_error("Unsupported subscript expression: " + indexExpr);
                    }
                }
            }
        }
        subInfos.push_back(info);
    }

    // Process subscripts from first to last dimension
    // We need to track axis offset as dimensions get removed by static indexing
    int axisOffset = 0;

    for (size_t dimIdx = 0; dimIdx < subInfos.size(); dimIdx++) {
        const auto& info = subInfos[dimIdx];
        int currentAxis = static_cast<int>(dimIdx) - axisOffset;

        switch (info.type) {
            case SubscriptInfo::FULL_SLICE:
                // Full slice: keep all elements on this dimension - no operation needed
                break;

            case SubscriptInfo::RANGE_SLICE: {
                // Range slice: use Slice operator
                currentTensor = addSlice(currentTensor,
                                         {info.start},
                                         {info.end},
                                         {currentAxis});
                break;
            }

            case SubscriptInfo::STATIC_INDEX: {
                // Static index: use Gather to select one element, then squeeze
                std::string indexTensor = addInt64Constant(info.staticIdx, "idx");
                currentTensor = addGather(currentTensor, indexTensor, currentAxis);
                // Gather removes the dimension, so adjust axis offset for subsequent subscripts
                axisOffset++;
                break;
            }

            case SubscriptInfo::DYNAMIC_INDEX: {
                // Dynamic index with loop variable
                std::string loopVar1Based = variableMap->at(info.loopVar);
                std::string index0Based = convertToZeroBasedIndex(loopVar1Based);
                currentTensor = addGather(currentTensor, index0Based, currentAxis);
                // Gather removes the dimension
                axisOffset++;
                break;
            }
        }
    }

    return currentTensor;
}

// -----------------------------------------------------------------------------
// Loop Helpers
// -----------------------------------------------------------------------------

void GraphBuilder::addLoopPassthrough(
    onnx::NodeProto* loopNode,
    onnx::GraphProto* bodyGraph,
    const std::string& loopNodeName,
    const std::string& inputName,
    const std::string& bodyInputName,
    int elemType,
    const std::vector<std::string>& dimensions,
    const std::string& outputSuffix) {

    // ONNX validates SSA globally across all subgraphs, so we need globally unique names.

    // Add to loop node inputs (outer graph scope)
    loopNode->add_input(inputName);

    // Add to loop body inputs (using loop-prefixed name for global uniqueness)
    auto* bodyInput = bodyGraph->add_input();
    bodyInput->set_name(bodyInputName);
    auto* inputType = bodyInput->mutable_type()->mutable_tensor_type();
    inputType->set_elem_type(elemType);
    addShapeDimensions(inputType->mutable_shape(), dimensions);

    // Add to loop body outputs (globally unique name)
    std::string bodyOutName = loopNodeName + "_" + bodyInputName + "_out";
    auto* bodyOutput = bodyGraph->add_output();
    bodyOutput->set_name(bodyOutName);
    auto* outputType = bodyOutput->mutable_type()->mutable_tensor_type();
    outputType->set_elem_type(elemType);
    addShapeDimensions(outputType->mutable_shape(), dimensions);

    // Create Identity node for passthrough
    auto* identity = bodyGraph->add_node();
    identity->set_op_type("Identity");
    identity->set_name(loopNodeName + "_" + bodyInputName + "_pass");
    identity->add_input(bodyInputName);
    identity->add_output(bodyOutName);

    // Add to loop node outputs (outer graph scope - needs loop name for uniqueness)
    loopNode->add_output(loopNodeName + "_" + bodyInputName + outputSuffix);
}

} // namespace lacemodelica
