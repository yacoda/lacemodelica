// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "ModelInfo.h"
#include <string>

namespace lacemodelica {

class ONNXGenerator {
public:
    // Generate ONNX model file and layered standard manifest
    // Returns the directory path where files were generated
    static std::string generate(const ModelInfo& info, const std::string& outputDir);

private:
    static void generateONNXModel(const ModelInfo& info, const std::string& filepath);
    static void generateManifest(const std::string& filepath);
};

} // namespace lacemodelica
