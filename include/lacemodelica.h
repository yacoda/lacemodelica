// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#ifndef LACEMODELICA_H
#define LACEMODELICA_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return codes for lacemodelica functions
 */
typedef enum {
    LACEMODELICA_SUCCESS = 0,
    LACEMODELICA_ERROR_FILE_NOT_FOUND = 1,
    LACEMODELICA_ERROR_PARSE_FAILED = 2,
    LACEMODELICA_ERROR_FMU_GENERATION_FAILED = 3,
    LACEMODELICA_ERROR_ONNX_GENERATION_FAILED = 4,
    LACEMODELICA_ERROR_OUTPUT_DIR_CREATION_FAILED = 5
} lacemodelica_status_t;

/**
 * @brief Process a BaseModelica (.bmo) file and generate FMU with ONNX layered standard
 *
 * This is the main entry point for processing Modelica files. It performs:
 * 1. Parsing of the BMO file
 * 2. Model information extraction
 * 3. FMU package generation
 * 4. ONNX layered standard generation
 *
 * @param input_file Path to the input .bmo file
 * @param output_dir Output directory for the generated FMU (can be NULL for auto-detection)
 * @return LACEMODELICA_SUCCESS on success, or an error code on failure
 */
lacemodelica_status_t lacemodelica_process_bmo(const char* input_file, const char* output_dir);

/**
 * @brief Parse a BaseModelica file without generating output
 *
 * Useful for validation/syntax checking only.
 *
 * @param input_file Path to the input .bmo file
 * @return LACEMODELICA_SUCCESS if parsing succeeds, LACEMODELICA_ERROR_PARSE_FAILED otherwise
 */
lacemodelica_status_t lacemodelica_parse_bmo(const char* input_file);

/**
 * @brief Get a human-readable error message for a status code
 *
 * @param status The status code
 * @return A string describing the error (static storage, do not free)
 */
const char* lacemodelica_status_string(lacemodelica_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* LACEMODELICA_H */
