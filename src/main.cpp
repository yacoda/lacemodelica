// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include <iostream>
#include <fstream>
#include <filesystem>
#include "antlr4-runtime.h"
#include "BaseModelicaLexer.h"
#include "BaseModelicaParser.h"

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
        // Parse specified file
        std::string filepath = argv[1];
        std::cout << "Parsing: " << filepath << std::endl;
        bool success = parseFile(filepath);
        std::cout << (success ? "✓ Success" : "✗ Failed") << std::endl;
        return success ? 0 : 1;
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
