// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "GraphBuilder.h"

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

std::string GraphBuilder::makeTensorName() {
    if (prefix_.empty()) {
        return "tensor_" + std::to_string(nodeCounter_++);
    }
    return prefix_ + "_tensor_" + std::to_string(nodeCounter_++);
}

std::string GraphBuilder::makeTensorName(const std::string& hint) {
    if (hint.empty()) {
        return makeTensorName();
    }
    return hint + "_" + std::to_string(nodeCounter_++);
}

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

std::string GraphBuilder::addInt64Constant(int64_t value) {
    return addInt64Constant(value, "");
}

std::string GraphBuilder::addInt64Constant(int64_t value, const std::string& nameHint) {
    std::string name = nameHint.empty()
        ? ("const_i64_" + std::to_string(nodeCounter_++))
        : nameHint;

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
    std::string name = "const_f64_" + std::to_string(nodeCounter_++);

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
    std::string name = "const_bool_" + std::to_string(nodeCounter_++);

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
    std::string name = "const_indices_" + std::to_string(nodeCounter_++);

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

std::string GraphBuilder::convertToZeroBasedIndex(const std::string& oneBasedTensor) {
    int constCounter = nodeCounter_++;
    std::string constOne = addInt64Constant(1,
        (prefix_.empty() ? "" : prefix_ + "_") + "const_one_" + std::to_string(constCounter));

    std::string zeroBasedTensor = (prefix_.empty() ? "" : prefix_ + "_") +
                                   "index_0based_" + std::to_string(nodeCounter_++);

    auto* subNode = graph_->add_node();
    subNode->set_op_type("Sub");
    subNode->set_name(zeroBasedTensor + "_Sub");
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
        if (sub->getText() == ":") {
            result.hasLoopVariable = true;
            return result;
        }

        auto subExpr = sub->expression();
        if (!subExpr) {
            continue;
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

    // Dynamic indexing with loop variables
    std::string currentTensor = baseTensor;

    for (size_t dimIdx = 0; dimIdx < subscripts.size(); dimIdx++) {
        auto sub = subscripts[dimIdx];

        if (sub->getText() == ":") {
            throw std::runtime_error("Array slice ':' not yet supported in ONNX conversion");
        }

        auto subExpr = sub->expression();
        if (!subExpr) {
            throw std::runtime_error("Invalid array subscript");
        }

        std::string indexExpr = subExpr->getText();

        if (variableMap && variableMap->count(indexExpr) > 0) {
            std::string loopVar1Based = variableMap->at(indexExpr);
            std::string index0Based = convertToZeroBasedIndex(loopVar1Based);
            currentTensor = addGather(currentTensor, index0Based, 0);
        } else {
            throw std::runtime_error("Mixed static and dynamic indexing not yet fully supported");
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

    // Add to loop node inputs
    loopNode->add_input(inputName);

    // Add to loop body inputs
    auto* bodyInput = bodyGraph->add_input();
    bodyInput->set_name(bodyInputName);
    auto* inputType = bodyInput->mutable_type()->mutable_tensor_type();
    inputType->set_elem_type(elemType);
    addShapeDimensions(inputType->mutable_shape(), dimensions);

    // Add to loop body outputs (passthrough)
    std::string bodyOutName = loopNodeName + "_" + bodyInputName + "_out";
    auto* bodyOutput = bodyGraph->add_output();
    bodyOutput->set_name(bodyOutName);
    auto* outputType = bodyOutput->mutable_type()->mutable_tensor_type();
    outputType->set_elem_type(elemType);
    addShapeDimensions(outputType->mutable_shape(), dimensions);

    // Create Identity node for passthrough
    auto* identity = bodyGraph->add_node();
    identity->set_op_type("Identity");
    identity->set_name(loopNodeName + "_" + bodyInputName + "_passthrough");
    identity->add_input(bodyInputName);
    identity->add_output(bodyOutName);

    // Add to loop node outputs
    loopNode->add_output(bodyInputName + outputSuffix);
}

} // namespace lacemodelica
