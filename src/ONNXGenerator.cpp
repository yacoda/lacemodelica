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
    // Create a trivial "hello world" ONNX model
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

    // Add a trivial "hello world" computation: output = input + 1.0
    // This is just a placeholder for now

    // Create input tensor
    auto* input = graph->add_input();
    input->set_name("x");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(onnx::TensorProto::FLOAT);
    auto* input_shape = input_type->mutable_shape();
    auto* input_dim = input_shape->add_dim();
    input_dim->set_dim_value(1);

    // Create constant node (value = 1.0)
    auto* constant = graph->add_initializer();
    constant->set_name("one");
    constant->set_data_type(onnx::TensorProto::FLOAT);
    constant->add_dims(1);
    constant->add_float_data(1.0f);

    // Create Add node
    auto* add_node = graph->add_node();
    add_node->set_op_type("Add");
    add_node->set_name("add_one");
    add_node->add_input("x");
    add_node->add_input("one");
    add_node->add_output("y");

    // Create output tensor
    auto* output = graph->add_output();
    output->set_name("y");
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(onnx::TensorProto::FLOAT);
    auto* output_shape = output_type->mutable_shape();
    auto* output_dim = output_shape->add_dim();
    output_dim->set_dim_value(1);

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

} // namespace lacemodelica
