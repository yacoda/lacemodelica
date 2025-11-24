// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include <iostream>
#include <fstream>
#include <filesystem>
#include "antlr4-runtime.h"
#include "BaseModelicaLexer.h"
#include "BaseModelicaParser.h"
#include "ModelInfoExtractor.h"
#include "FMUGenerator.h"
#include "ONNXGenerator.h"

using namespace antlr4;
namespace fs = std::filesystem;

class ErrorListener : public BaseErrorListener {
public:
    void syntaxError(Recognizer *recognizer, Token *offendingSymbol,
                    size_t line, size_t charPositionInLine,
                    const std::string &msg, std::exception_ptr e) override {
        std::cerr << "  Error at line " << line << ":" << charPositionInLine
                  << " - " << msg << std::endl;
        hasError = true;
    }
    bool hasError = false;
};

bool parseFile(const std::string& filepath) {
    std::ifstream stream(filepath);
    if (!stream.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }

    ANTLRInputStream input(stream);
    basemodelica::BaseModelicaLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    basemodelica::BaseModelicaParser parser(&tokens);

    // Add error listener
    ErrorListener errorListener;
    parser.removeErrorListeners();
    parser.addErrorListener(&errorListener);

    // Parse the file
    parser.baseModelica();

    return !errorListener.hasError;
}

int main(int argc, char* argv[]) {
    std::cout << "lacemodelica - BaseModelica to FMU/ONNX converter" << std::endl;

    // If no arguments, parse all test files
    std::string testDir = "test/testfiles";

    if (argc > 1) {
        // Parse command line arguments
        std::string filepath = argv[1];
        std::string outputDir;

        // Check for --output-dir option
        if (argc > 3 && std::string(argv[2]) == "--output-dir") {
            outputDir = argv[3];
        }

        std::cout << "Parsing: " << filepath << std::endl;

        std::ifstream stream(filepath);
        if (!stream.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return 1;
        }

        ANTLRInputStream input(stream);
        basemodelica::BaseModelicaLexer lexer(&input);
        CommonTokenStream tokens(&lexer);
        basemodelica::BaseModelicaParser parser(&tokens);

        ErrorListener errorListener;
        parser.removeErrorListeners();
        parser.addErrorListener(&errorListener);

        auto tree = parser.baseModelica();

        if (errorListener.hasError) {
            std::cout << "✗ Parsing failed" << std::endl;
            return 1;
        }

        std::cout << "✓ Parsing succeeded" << std::endl;

        // Extract model information
        std::cout << "\nExtracting model information..." << std::endl;
        lacemodelica::ModelInfoExtractor extractor;
        lacemodelica::ModelInfo info = extractor.extract(tree, filepath);

        std::cout << "  Model: " << info.modelName << std::endl;
        std::cout << "  Variables: " << info.variables.size() << std::endl;

        // Determine output directory
        std::string outputPath;
        if (!outputDir.empty()) {
            // Use specified output directory
            fs::create_directories(outputDir);
            outputPath = outputDir + "/" + info.modelName + "_fmu";
        } else {
            // Auto-detect: if input is from testfiles, output to sibling output/ directory
            fs::path inputPath(filepath);
            fs::path absInputPath = fs::absolute(inputPath);

            if (absInputPath.string().find("/testfiles/") != std::string::npos) {
                // Replace /testfiles/ with /output/ in the path
                fs::path inputDir = absInputPath.parent_path();
                fs::path outputBaseDir = inputDir.parent_path() / "output";
                fs::create_directories(outputBaseDir);
                outputPath = (outputBaseDir / (info.modelName + "_fmu")).string();
            } else {
                // Regular file: output to current directory
                outputPath = info.modelName + "_fmu";
            }
        }

        // Generate FMU
        std::cout << "\nGenerating FMU..." << std::endl;
        lacemodelica::FMUGenerator generator;
        bool success = generator.generateFMU(info, outputPath);

        if (!success) {
            std::cout << "\nFMU generation failed" << std::endl;
            return 1;
        }

        // Generate ONNX layered standard
        std::cout << "\nGenerating ONNX layered standard..." << std::endl;
        try {
            lacemodelica::ONNXGenerator::generate(info, outputPath);
            std::cout << "\nFMU with ONNX layered standard generated successfully in " << outputPath << "/" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ONNX generation failed: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    } else {
        // Parse all test files
        if (!fs::exists(testDir)) {
            std::cerr << "Test directory not found: " << testDir << std::endl;
            return 1;
        }

        int passed = 0, failed = 0;
        std::cout << "\nParsing test files from " << testDir << ":\n" << std::endl;

        for (const auto& entry : fs::directory_iterator(testDir)) {
            if (entry.path().extension() == ".bmo") {
                std::string filename = entry.path().filename().string();
                std::cout << "Testing " << filename << "... ";
                std::cout.flush();

                bool success = parseFile(entry.path().string());
                if (success) {
                    std::cout << "✓" << std::endl;
                    passed++;
                } else {
                    std::cout << "✗" << std::endl;
                    failed++;
                }
            }
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
        return failed == 0 ? 0 : 1;
    }
}
