// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include <iostream>
#include <cstring>
#include "lacemodelica.h"

int main(int argc, char* argv[]) {
    std::cout << "lacemodelica - BaseModelica to FMU/ONNX converter" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.bmo> [--output-dir <dir>]" << std::endl;
        std::cerr << "       " << argv[0] << " <input.bmo> --parse-only" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_dir = nullptr;
    bool parse_only = false;

    // Parse command line options
    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--parse-only") == 0) {
            parse_only = true;
        } else if (std::strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        }
    }

    lacemodelica_status_t status;

    if (parse_only) {
        std::cout << "Parsing: " << input_file << std::endl;
        status = lacemodelica_parse_bmo(input_file);
    } else {
        std::cout << "Processing: " << input_file << " -> " << (output_dir ? output_dir : ".") << std::endl;
        status = lacemodelica_process_bmo(input_file, output_dir);
    }

    if (status != LACEMODELICA_SUCCESS) {
        std::cerr << "Error: " << lacemodelica_status_string(status) << std::endl;
        return 1;
    }

    std::cout << "Done." << std::endl;
    return 0;
}
