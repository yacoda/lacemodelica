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

    // Create ONNX inputs for each variable and parameter
    for (const auto& var : info.variables) {
        auto* input = graph->add_input();
        input->set_name(var.name);
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        input_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* input_shape = input_type->mutable_shape();

        // Handle array dimensions
        if (!var.dimensions.empty()) {
            for (const auto& dim : var.dimensions) {
                auto* shape_dim = input_shape->add_dim();
                // Try to parse as integer, otherwise leave symbolic
                try {
                    shape_dim->set_dim_value(std::stoi(dim));
                } catch (...) {
                    shape_dim->set_dim_param(dim);
                }
            }
        } else {
            // Scalar: shape [1]
            auto* shape_dim = input_shape->add_dim();
            shape_dim->set_dim_value(1);
        }
    }

    // Create ONNX outputs for each equation (eq_lhs[i], eq_rhs[i])
    for (size_t i = 0; i < info.equations.size(); i++) {
        const auto& eq = info.equations[i];

        // Create Identity nodes as placeholders (TODO: parse expressions into operators)
        // For now, just create outputs with metadata about the equation

        // LHS output
        auto* lhs_output = graph->add_output();
        lhs_output->set_name("eq_lhs[" + std::to_string(i) + "]");
        lhs_output->set_doc_string(eq.lhs);
        auto* lhs_type = lhs_output->mutable_type()->mutable_tensor_type();
        lhs_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* lhs_shape = lhs_type->mutable_shape();
        lhs_shape->add_dim()->set_dim_value(1);

        // RHS output
        auto* rhs_output = graph->add_output();
        rhs_output->set_name("eq_rhs[" + std::to_string(i) + "]");
        rhs_output->set_doc_string(eq.rhs);
        auto* rhs_type = rhs_output->mutable_type()->mutable_tensor_type();
        rhs_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* rhs_shape = rhs_type->mutable_shape();
        rhs_shape->add_dim()->set_dim_value(1);
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

} // namespace lacemodelica
