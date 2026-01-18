// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "lacemodelica.h"

#include <iostream>
#include <fstream>
#include <filesystem>

#include "antlr4-runtime.h"
#include "BaseModelicaLexer.h"
#include "BaseModelicaParser.h"
#include "ModelInfoExtractor.h"
#include "FMUGenerator.h"
#include "ONNXGenerator.h"

namespace fs = std::filesystem;

namespace {

class ErrorListener : public antlr4::BaseErrorListener {
public:
    void syntaxError(antlr4::Recognizer* recognizer, antlr4::Token* offendingSymbol,
                     size_t line, size_t charPositionInLine,
                     const std::string& msg, std::exception_ptr e) override {
        std::cerr << "  Error at line " << line << ":" << charPositionInLine
                  << " - " << msg << std::endl;
        hasError = true;
    }
    bool hasError = false;
};

} // anonymous namespace

extern "C" {

lacemodelica_status_t lacemodelica_process_bmo(const char* input_file, const char* output_dir) {
    if (!input_file) {
        return LACEMODELICA_ERROR_FILE_NOT_FOUND;
    }

    std::string filepath(input_file);

    // Open and parse the file
    std::ifstream stream(filepath);
    if (!stream.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return LACEMODELICA_ERROR_FILE_NOT_FOUND;
    }

    antlr4::ANTLRInputStream input(stream);
    basemodelica::BaseModelicaLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    basemodelica::BaseModelicaParser parser(&tokens);

    ErrorListener errorListener;
    parser.removeErrorListeners();
    parser.addErrorListener(&errorListener);

    auto tree = parser.baseModelica();

    if (errorListener.hasError) {
        return LACEMODELICA_ERROR_PARSE_FAILED;
    }

    // Extract model information
    lacemodelica::ModelInfoExtractor extractor;
    lacemodelica::ModelInfo info = extractor.extract(tree, filepath);

    // Determine output path
    std::string outputPath;
    if (output_dir && output_dir[0] != '\0') {
        // Use specified output directory
        try {
            fs::create_directories(output_dir);
        } catch (const std::exception& e) {
            std::cerr << "Failed to create output directory: " << e.what() << std::endl;
            return LACEMODELICA_ERROR_OUTPUT_DIR_CREATION_FAILED;
        }
        outputPath = std::string(output_dir) + "/" + info.modelName + "_fmu";
    } else {
        // Auto-detect: if input is from testfiles, output to sibling output/ directory
        fs::path inputPath(filepath);
        fs::path absInputPath = fs::absolute(inputPath);

        if (absInputPath.string().find("/testfiles/") != std::string::npos) {
            // Replace /testfiles/ with /output/ in the path
            fs::path inputDir = absInputPath.parent_path();
            fs::path outputBaseDir = inputDir.parent_path() / "output";
            try {
                fs::create_directories(outputBaseDir);
            } catch (const std::exception& e) {
                std::cerr << "Failed to create output directory: " << e.what() << std::endl;
                return LACEMODELICA_ERROR_OUTPUT_DIR_CREATION_FAILED;
            }
            outputPath = (outputBaseDir / (info.modelName + "_fmu")).string();
        } else {
            // Regular file: output to current directory
            outputPath = info.modelName + "_fmu";
        }
    }

    // Generate FMU
    lacemodelica::FMUGenerator generator;
    if (!generator.generateFMU(info, outputPath)) {
        return LACEMODELICA_ERROR_FMU_GENERATION_FAILED;
    }

    // Generate ONNX layered standard
    try {
        lacemodelica::ONNXGenerator::generate(info, outputPath);
    } catch (const std::exception& e) {
        std::cerr << "ONNX generation failed: " << e.what() << std::endl;
        return LACEMODELICA_ERROR_ONNX_GENERATION_FAILED;
    }

    return LACEMODELICA_SUCCESS;
}

lacemodelica_status_t lacemodelica_parse_bmo(const char* input_file) {
    if (!input_file) {
        return LACEMODELICA_ERROR_FILE_NOT_FOUND;
    }

    std::string filepath(input_file);

    std::ifstream stream(filepath);
    if (!stream.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return LACEMODELICA_ERROR_FILE_NOT_FOUND;
    }

    antlr4::ANTLRInputStream input(stream);
    basemodelica::BaseModelicaLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    basemodelica::BaseModelicaParser parser(&tokens);

    ErrorListener errorListener;
    parser.removeErrorListeners();
    parser.addErrorListener(&errorListener);

    parser.baseModelica();

    return errorListener.hasError ? LACEMODELICA_ERROR_PARSE_FAILED : LACEMODELICA_SUCCESS;
}

const char* lacemodelica_status_string(lacemodelica_status_t status) {
    switch (status) {
        case LACEMODELICA_SUCCESS:
            return "Success";
        case LACEMODELICA_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case LACEMODELICA_ERROR_PARSE_FAILED:
            return "Parse failed";
        case LACEMODELICA_ERROR_FMU_GENERATION_FAILED:
            return "FMU generation failed";
        case LACEMODELICA_ERROR_ONNX_GENERATION_FAILED:
            return "ONNX generation failed";
        case LACEMODELICA_ERROR_OUTPUT_DIR_CREATION_FAILED:
            return "Failed to create output directory";
        default:
            return "Unknown error";
    }
}

} // extern "C"
