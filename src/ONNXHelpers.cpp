// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ONNXHelpers.hpp"

#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>

namespace lacemodelica {

std::string makeTensorName(const std::string& prefix, int& counter) {
    if (prefix.empty()) {
        return "tensor_" + std::to_string(counter++);
    }
    return prefix + "_tensor_" + std::to_string(counter++);
}

std::string createInt64Constant(onnx::GraphProto* graph, int64_t value, int& counter,
                                 const std::string& nameHint) {
    std::string name = nameHint.empty() ? ("const_i64_" + std::to_string(counter++)) : nameHint;
    auto* node = graph->add_node();
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

std::string createDoubleConstant(onnx::GraphProto* graph, double value, int& counter) {
    std::string name = "const_f64_" + std::to_string(counter++);
    auto* node = graph->add_node();
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

std::string createBoolConstant(onnx::GraphProto* graph, bool value, int& counter) {
    std::string name = "const_bool_" + std::to_string(counter++);
    auto* node = graph->add_node();
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

std::string createIdentityNode(onnx::GraphProto* graph, const std::string& inputTensor,
                                const std::string& outputName, const std::string& nodeName) {
    auto* node = graph->add_node();
    node->set_op_type("Identity");
    node->set_name(nodeName);
    node->add_input(inputTensor);
    node->add_output(outputName);
    return outputName;
}

std::string convertTo0BasedIndex(onnx::GraphProto* graph, const std::string& oneBasedTensor,
                                  int& counter, const std::string& prefix) {
    // Create unique constant name using counter before incrementing
    int constCounter = counter++;
    std::string constOne = createInt64Constant(graph, 1, counter,
        (prefix.empty() ? "" : prefix + "_") + "const_one_" + std::to_string(constCounter));

    std::string zeroBasedTensor = (prefix.empty() ? "" : prefix + "_") + "index_0based_" + std::to_string(counter++);

    auto* subNode = graph->add_node();
    subNode->set_op_type("Sub");
    subNode->set_name(zeroBasedTensor + "_Sub");  // Use output tensor name for uniqueness
    subNode->add_input(oneBasedTensor);
    subNode->add_input(constOne);
    subNode->add_output(zeroBasedTensor);

    return zeroBasedTensor;
}

std::string createGatherNode(onnx::GraphProto* graph, const std::string& dataTensor,
                              const std::string& indexTensor, int axis, int& counter,
                              const std::string& prefix) {
    std::string outputTensor = makeTensorName(prefix, counter);

    auto* node = graph->add_node();
    node->set_op_type("Gather");
    node->set_name(outputTensor + "_Gather");  // Use tensor name for uniqueness
    node->add_input(dataTensor);
    node->add_input(indexTensor);
    node->add_output(outputTensor);

    auto* axisAttr = node->add_attribute();
    axisAttr->set_name("axis");
    axisAttr->set_type(onnx::AttributeProto::INT);
    axisAttr->set_i(axis);

    return outputTensor;
}

void addSourceMetadata(onnx::ValueInfoProto* valueInfo, const std::string& sourceFile, size_t sourceLine) {
    if (sourceFile.empty()) return;

    auto* metaFile = valueInfo->add_metadata_props();
    metaFile->set_key("source_file");
    metaFile->set_value(sourceFile);

    auto* metaLine = valueInfo->add_metadata_props();
    metaLine->set_key("source_line");
    metaLine->set_value(std::to_string(sourceLine));
}

void configureTensorType(onnx::TypeProto_Tensor* tensorType, int elemType,
                         const std::vector<std::string>& dimensions) {
    tensorType->set_elem_type(elemType);
    auto* shape = tensorType->mutable_shape();
    for (const auto& dim : dimensions) {
        auto* shapeDim = shape->add_dim();
        try {
            shapeDim->set_dim_value(std::stoi(dim));
        } catch (...) {
            shapeDim->set_dim_param(dim);
        }
    }
}

std::string createBinaryOp(onnx::GraphProto* graph, const std::string& opType,
                            const std::string& left, const std::string& right,
                            int& counter, const std::string& prefix) {
    std::string outputTensor = makeTensorName(prefix, counter);

    auto* node = graph->add_node();
    node->set_op_type(opType);
    node->set_name(outputTensor + "_" + opType);  // Use tensor name for uniqueness
    node->add_input(left);
    node->add_input(right);
    node->add_output(outputTensor);

    return outputTensor;
}

std::string createUnaryOp(onnx::GraphProto* graph, const std::string& opType,
                           const std::string& input, int& counter,
                           const std::string& prefix) {
    std::string outputTensor = makeTensorName(prefix, counter);

    auto* node = graph->add_node();
    node->set_op_type(opType);
    node->set_name(outputTensor + "_" + opType);  // Use tensor name for uniqueness
    node->add_input(input);
    node->add_output(outputTensor);

    return outputTensor;
}

void addScalarDoubleOutput(onnx::GraphProto* graph, const std::string& tensorName) {
    auto* output = graph->add_output();
    output->set_name(tensorName);
    auto* tensorType = output->mutable_type()->mutable_tensor_type();
    tensorType->set_elem_type(onnx::TensorProto::DOUBLE);
    tensorType->mutable_shape()->add_dim()->set_dim_value(1);
}

std::string createIfNode(onnx::GraphProto* graph, const std::string& condTensor,
                          onnx::GraphProto& thenBranch, onnx::GraphProto& elseBranch,
                          int& counter, const std::string& prefix,
                          const std::string& nameHint) {
    std::string outputTensor = makeTensorName(prefix, counter);

    auto* ifNode = graph->add_node();
    ifNode->set_op_type("If");
    ifNode->set_name(nameHint + "_" + std::to_string(counter));
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

    return outputTensor;
}

} // namespace lacemodelica
