# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Joris Gillis, YACODA

cmake_minimum_required(VERSION 3.14)

project(lacemodelica VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies
include(FetchContent)
include(ExternalProject)

# ANTLR4 runtime
set(ANTLR_VERSION 4.13.2)
set(ANTLR_TAG ${ANTLR_VERSION})

FetchContent_Declare(
    antlr4_runtime
    GIT_REPOSITORY https://github.com/antlr/antlr4.git
    GIT_TAG ${ANTLR_TAG}
    SOURCE_SUBDIR runtime/Cpp
)

# TinyXML-2
FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG 10.0.0
)

FetchContent_MakeAvailable(antlr4_runtime tinyxml2)

# Protobuf (required for ONNX)
ExternalProject_Add(protobuf-external
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
    GIT_SHALLOW ON
    GIT_TAG v31.1
    PREFIX "${CMAKE_BINARY_DIR}/external_projects"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_CXX_STANDARD=17
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -Dprotobuf_BUILD_TESTS=OFF
        -Dprotobuf_ABSL_PROVIDER=module
)

add_library(protobuf::libprotobuf STATIC IMPORTED)
add_dependencies(protobuf::libprotobuf protobuf-external)

# utf8_range (protobuf dependency)
add_library(utf8_range::utf8_range STATIC IMPORTED)
set_target_properties(utf8_range::utf8_range PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}utf8_range${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")
target_link_libraries(protobuf::libprotobuf INTERFACE utf8_range::utf8_range)

# Setup abseil library dependencies
include(${CMAKE_SOURCE_DIR}/cmake/abseil_dependencies.cmake)
setup_abseil_dependencies(protobuf::libprotobuf)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/external_projects/include")
set_target_properties(protobuf::libprotobuf PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}protobuf${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")

# ONNX
ExternalProject_Add(onnx-external
    DEPENDS protobuf::libprotobuf
    GIT_REPOSITORY https://github.com/onnx/onnx
    GIT_SHALLOW ON
    GIT_TAG v1.19.1
    PREFIX "${CMAKE_BINARY_DIR}/external_projects"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_CXX_STANDARD=17
        -DONNX_BUILD_TESTS=OFF
        -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/external_projects
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

add_library(ONNX::onnx STATIC IMPORTED)
add_library(ONNX::onnx_proto STATIC IMPORTED)
add_dependencies(ONNX::onnx onnx-external)
add_dependencies(ONNX::onnx_proto onnx-external)
target_link_libraries(ONNX::onnx INTERFACE protobuf::libprotobuf)
set_target_properties(ONNX::onnx PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}onnx${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")
set_target_properties(ONNX::onnx_proto PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}onnx_proto${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")

# Generated parser location (in source tree so it can be committed)
set(GENERATED_PARSER_DIR ${CMAKE_SOURCE_DIR}/generated)

# Optional: Only regenerate parser if REGENERATE_PARSER is set (for developers)
if(REGENERATE_PARSER)
    # Find ANTLR JAR and CMake scripts
    set(ANTLR4_JAR_LOCATION ${CMAKE_BINARY_DIR}/antlr-${ANTLR_VERSION}-complete.jar)

    # Download ANTLR4 JAR if not present
    if(NOT EXISTS ${ANTLR4_JAR_LOCATION})
        message(STATUS "Downloading ANTLR4 JAR...")
        file(DOWNLOAD
            https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar
            ${ANTLR4_JAR_LOCATION}
            SHOW_PROGRESS
        )
    endif()

    set(ANTLR_EXECUTABLE ${ANTLR4_JAR_LOCATION})
    list(APPEND CMAKE_MODULE_PATH ${antlr4_runtime_SOURCE_DIR}/runtime/Cpp/cmake)
    find_package(ANTLR REQUIRED)

    # Generate parser from BaseModelica.g4
    antlr_target(BaseModelicaParser ${CMAKE_SOURCE_DIR}/grammar/BaseModelica.g4
        PACKAGE basemodelica
        OUTPUT_DIRECTORY ${GENERATED_PARSER_DIR}
    )
    set(GENERATED_SOURCES ${ANTLR_BaseModelicaParser_CXX_OUTPUTS})
else()
    # Use committed generated files
    file(GLOB GENERATED_SOURCES ${GENERATED_PARSER_DIR}/*.cpp)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ANTLR4_INCLUDE_DIRS})
include_directories(${GENERATED_PARSER_DIR})
include_directories(${CMAKE_BINARY_DIR}/external_projects/include)

# Executable
add_executable(lacemodelica
    src/main.cpp
    src/ModelInfoExtractor.cpp
    src/FMUGenerator.cpp
    src/ONNXGenerator.cpp
    ${GENERATED_SOURCES}
)

# Make sure lacemodelica depends on external projects
add_dependencies(lacemodelica protobuf-external onnx-external)

target_link_libraries(lacemodelica
    antlr4_shared
    tinyxml2
    ONNX::onnx
    ONNX::onnx_proto
    protobuf::libprotobuf
)

# Testing
enable_testing()

# Find all test files
file(GLOB TEST_FILES "${CMAKE_SOURCE_DIR}/test/testfiles/*.bmo")

# Create parse-only tests for each .bmo file (tests parsing only)
foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_test(
        NAME parse_${TEST_NAME}
        COMMAND lacemodelica ${TEST_FILE} --parse-only
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set_tests_properties(parse_${TEST_NAME} PROPERTIES LABELS "parse")
endforeach()

# Create full generation tests for each .bmo file (tests full pipeline including ONNX)
foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_test(
        NAME generate_${TEST_NAME}
        COMMAND lacemodelica ${TEST_FILE}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set_tests_properties(generate_${TEST_NAME} PROPERTIES LABELS "generate")
endforeach()

# Add Python FMU import test (runs independently, expects FMUs to exist)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(
        NAME fmu_import_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/test_fmu_import.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(fmu_import_test PROPERTIES LABELS "integration")

    # Add ONNX runtime validation test
    add_test(
        NAME onnx_runtime_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/test_onnx_runtime.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(onnx_runtime_test PROPERTIES
        LABELS "onnx;validation"
        DEPENDS "generate_NewtonCoolingBase"
    )

    # List of test files with ONNX reference implementations
    set(ONNX_REFERENCE_TESTS
        NewtonCoolingBase
        ArithmeticOps
        Comparisons
        DerExpression
        DerVariable
        ElementWiseProduct
        GlobalConst
        IfExpression
        InputOutput
        LogicalOps
        NegativeVariable
        NestedIf
        ParameterBindings
        Precedence
        TrigFunctions
        UnaryOps
    )

    # Tests that are expected to fail (but we still want to run them)
    set(ONNX_EXPECTED_FAILURES
        DerExpression  # Uses custom Der operator for derivative of expressions
    )

    # Create individual ONNX runtime validation tests for each model with reference
    foreach(TEST_NAME ${ONNX_REFERENCE_TESTS})
        add_test(
            NAME onnx_validate_${TEST_NAME}
            COMMAND ${Python3_EXECUTABLE} -c "
import sys
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/test')
from test_onnx_runtime import parse_onnx_test_from_bmo, convert_float64_to_float32, Path
import onnxruntime as ort
import onnx
import numpy as np

# Parse .bmo file
bmo_path = Path('${CMAKE_SOURCE_DIR}/test/testfiles/${TEST_NAME}.bmo')
ref_impl, test_cases = parse_onnx_test_from_bmo(bmo_path)

if not ref_impl or not test_cases:
    print(f'No reference found in {bmo_path}')
    sys.exit(1)

# Load ONNX model and convert float64 to float32 for compatibility
onnx_path = Path('${CMAKE_SOURCE_DIR}/test/output/${TEST_NAME}_fmu/extra/org.lacemodelica.ls-onnx-serialization/model.onnx')
if not onnx_path.exists():
    print(f'ONNX file not found: {onnx_path}')
    sys.exit(1)

model = onnx.load(str(onnx_path))
model = convert_float64_to_float32(model)
session = ort.InferenceSession(model.SerializeToString())
input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]

# Debug: Print ONNX graph structure
import onnx
model = onnx.load(str(onnx_path))
print(f'\\n=== ONNX Model for ${TEST_NAME} ===')
print(f'Inputs: {input_names}')
print(f'Outputs: {output_names}')
print(f'\\nGraph nodes:')
for i, node in enumerate(model.graph.node):
    print(f'  {i}: {node.op_type} ({node.name})')
    print(f'      inputs: {list(node.input)}')
    print(f'      outputs: {list(node.output)}')
print('\\n')

# Get input shapes from model
input_shapes = {}
for inp in session.get_inputs():
    input_shapes[inp.name] = [d if isinstance(d, int) else 1 for d in inp.shape]

# Run all test cases
passed = True
for i, test_case in enumerate(test_cases, 1):
    onnx_inputs = {}
    for k, v in test_case.items():
        if k not in input_names:
            continue
        # All inputs are numpy arrays - convert to float32 to match model
        onnx_inputs[k] = v.astype(np.float32)
    onnx_outputs = session.run(output_names, onnx_inputs)
    onnx_results = dict(zip(output_names, onnx_outputs))
    ref_results = ref_impl(test_case)

    for name in output_names:
        if name in ref_results:
            onnx_val = np.array(onnx_results[name])
            ref_val = np.array(ref_results[name])

            # Check shapes match
            if onnx_val.shape != ref_val.shape:
                print(f'Test {i} FAILED: {name}: shape mismatch ONNX={onnx_val.shape} vs Ref={ref_val.shape}')
                passed = False
                continue

            # Compare values (use 1e-4 tolerance to account for float32 precision loss)
            diff = np.abs(onnx_val - ref_val)
            max_diff = np.max(diff) if diff.size > 0 else abs(diff)
            if max_diff >= 1e-4:
                print(f'Test {i} FAILED: {name}: max diff={max_diff:.2e}, shapes={onnx_val.shape}')
                passed = False

if passed:
    print('All tests passed')
    sys.exit(0)
else:
    sys.exit(1)
"
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        set_tests_properties(onnx_validate_${TEST_NAME} PROPERTIES
            LABELS "onnx;validation"
        )

        # Mark expected failures
        if(${TEST_NAME} IN_LIST ONNX_EXPECTED_FAILURES)
            set_tests_properties(onnx_validate_${TEST_NAME} PROPERTIES
                WILL_FAIL TRUE
            )
        endif()
    endforeach()
endif()
