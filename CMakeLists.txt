# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Joris Gillis, YACODA

cmake_minimum_required(VERSION 3.14)

project(lacemodelica VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# RPATH settings for relocatable install
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

# Dependencies
include(FetchContent)
include(ExternalProject)

# ANTLR4 runtime
set(ANTLR_VERSION 4.13.2)
set(ANTLR_TAG ${ANTLR_VERSION})
set(ANTLR4_INSTALL ON)

# Install targets
include(GNUInstallDirs)

FetchContent_Declare(
    antlr4_runtime
    GIT_REPOSITORY https://github.com/antlr/antlr4.git
    GIT_TAG ${ANTLR_TAG}
    SOURCE_SUBDIR runtime/Cpp
)

# TinyXML-2 - prefer system, fallback to FetchContent
# Use CONFIG mode to require modern CMake config (not FindTinyXML2.cmake modules)
# Temporarily disabled: system tinyxml2 may not be built with -fPIC for shared library use
#find_package(tinyxml2 CONFIG QUIET)
if(FALSE AND tinyxml2_FOUND AND TARGET tinyxml2::tinyxml2)
    message(STATUS "Found system TinyXML-2: ${tinyxml2_VERSION}")
    set(USE_SYSTEM_TINYXML2 TRUE)
    # Alias to match FetchContent target name
    add_library(tinyxml2 ALIAS tinyxml2::tinyxml2)
else()
    message(STATUS "TinyXML-2 not found, building from source")
    set(USE_SYSTEM_TINYXML2 FALSE)
    FetchContent_Declare(
        tinyxml2
        GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
        GIT_TAG 10.0.0
    )
endif()

# Set policy defaults for FetchContent dependencies
# This ensures dependencies like googletest (pulled by antlr4) work with newer CMake
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)  # Use extraction timestamp for downloaded archives
endif()
set(CMAKE_POLICY_DEFAULT_CMP0000 NEW)  # Allow dependencies with old cmake_minimum_required

# ANTLR build options - only build shared library, skip tests
set(ANTLR_BUILD_STATIC OFF CACHE BOOL "" FORCE)
set(ANTLR_BUILD_CPP_TESTS OFF CACHE BOOL "" FORCE)

# Enable position independent code for tinyxml2 (required for shared library linking)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(USE_SYSTEM_TINYXML2)
    FetchContent_MakeAvailable(antlr4_runtime)
else()
    FetchContent_MakeAvailable(antlr4_runtime tinyxml2)
endif()

# Try to find system dependencies
# Dependency chain: ONNX -> Protobuf -> (absl, utf8_range)
find_package(absl QUIET)
find_package(utf8_range QUIET)
find_package(Protobuf QUIET)
find_package(ONNX QUIET)

# Check what we can use from the system
if(ONNX_FOUND AND Protobuf_FOUND AND absl_FOUND AND utf8_range_FOUND)
    # Case 1: Use everything from system
    message(STATUS "Found system ONNX: ${ONNX_VERSION}")
    message(STATUS "Found system Protobuf: ${Protobuf_VERSION}")
    set(USE_SYSTEM_ONNX TRUE)
    set(USE_SYSTEM_PROTOBUF TRUE)
elseif(Protobuf_FOUND AND absl_FOUND AND utf8_range_FOUND)
    # Case 2: Use system Protobuf, build ONNX from source
    message(STATUS "Found system Protobuf: ${Protobuf_VERSION}")
    message(STATUS "ONNX not found, building from source")
    set(USE_SYSTEM_ONNX FALSE)
    set(USE_SYSTEM_PROTOBUF TRUE)
    set(ONNX_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
else()
    # Case 3: Build everything from source
    message(STATUS "System ONNX and Protobuf not found, building from source")
    set(USE_SYSTEM_ONNX FALSE)
    set(USE_SYSTEM_PROTOBUF FALSE)
    set(ONNX_PREFIX_PATH "${CMAKE_BINARY_DIR}/external_projects")

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
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
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
endif()

# ONNX - built from source in Case 2 and Case 3
if(NOT USE_SYSTEM_ONNX)
    set(ONNX_EXTERNAL_DEPENDS "")
    if(NOT USE_SYSTEM_PROTOBUF)
        set(ONNX_EXTERNAL_DEPENDS DEPENDS protobuf::libprotobuf)
    endif()

    ExternalProject_Add(onnx-external
        ${ONNX_EXTERNAL_DEPENDS}
        GIT_REPOSITORY https://github.com/onnx/onnx
        GIT_SHALLOW ON
        GIT_TAG v1.19.1
        PREFIX "${CMAKE_BINARY_DIR}/external_projects"
        UPDATE_COMMAND ""
        CMAKE_ARGS
            -DCMAKE_CXX_STANDARD=17
            -DONNX_BUILD_TESTS=OFF
            -DCMAKE_PREFIX_PATH=${ONNX_PREFIX_PATH}
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    )

    add_library(ONNX::onnx STATIC IMPORTED)
    add_library(ONNX::onnx_proto STATIC IMPORTED)
    add_dependencies(ONNX::onnx onnx-external)
    add_dependencies(ONNX::onnx_proto onnx-external)
    target_link_libraries(ONNX::onnx INTERFACE protobuf::libprotobuf)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/external_projects/include")
    set_target_properties(ONNX::onnx PROPERTIES
        IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}onnx${CMAKE_STATIC_LIBRARY_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")
    set_target_properties(ONNX::onnx_proto PROPERTIES
        IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external_projects/lib/${CMAKE_STATIC_LIBRARY_PREFIX}onnx_proto${CMAKE_STATIC_LIBRARY_SUFFIX}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external_projects/include")
endif()

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
if(NOT USE_SYSTEM_ONNX)
    include_directories(${CMAKE_BINARY_DIR}/external_projects/include)
endif()

# Shared library (core functionality)
add_library(lacemodelica SHARED
    src/lacemodelica_c.cpp
    src/ModelInfoExtractor.cpp
    src/FMUGenerator.cpp
    src/ONNXGenerator.cpp
    src/ONNXHelpers.cpp
    src/GraphBuilder.cpp
    src/ParseTreeNavigator.cpp
    src/ExpressionConverter.cpp
    src/EquationGenerator.cpp
    ${GENERATED_SOURCES}
)

# Export macro and visibility settings for shared library
target_compile_definitions(lacemodelica PRIVATE LACEMODELICA_BUILDING)
set_target_properties(lacemodelica PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Make sure library depends on external projects (only when building from source)
if(NOT USE_SYSTEM_ONNX)
    if(USE_SYSTEM_PROTOBUF)
        # Only ONNX built from source
        add_dependencies(lacemodelica onnx-external)
    else()
        # Both built from source
        add_dependencies(lacemodelica protobuf-external onnx-external)
    endif()
endif()

target_link_libraries(lacemodelica
    PUBLIC
        antlr4_shared
        tinyxml2
        ONNX::onnx
        ONNX::onnx_proto
        protobuf::libprotobuf
)

target_include_directories(lacemodelica
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${GENERATED_PARSER_DIR}
)

# Option to build the executable
option(LACEMODELICA_BUILD_EXE "Build the lacemodelica executable" ON)

if(LACEMODELICA_BUILD_EXE)
    # Executable (links to shared library)
    add_executable(lacemodelica_exe src/main.cpp)
    set_target_properties(lacemodelica_exe PROPERTIES OUTPUT_NAME lacemodelica)
    target_link_libraries(lacemodelica_exe lacemodelica)
endif()

install(TARGETS lacemodelica
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(LACEMODELICA_BUILD_EXE)
    install(TARGETS lacemodelica_exe
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# On Windows, copy DLLs to bin directory for the executable to find them
if(WIN32)
    install(FILES $<TARGET_FILE:antlr4_shared>
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lacemodelica
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Testing
enable_testing()

# Find all test files
file(GLOB TEST_FILES "${CMAKE_SOURCE_DIR}/test/testfiles/*.bmo")

if(LACEMODELICA_BUILD_EXE)
    # Create parse-only tests for each .bmo file (tests parsing only)
    foreach(TEST_FILE ${TEST_FILES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        add_test(
            NAME parse_${TEST_NAME}
            COMMAND lacemodelica_exe ${TEST_FILE} --parse-only
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
        set_tests_properties(parse_${TEST_NAME} PROPERTIES LABELS "parse")
    endforeach()

    # Create full generation tests for each .bmo file (tests full pipeline including ONNX)
    foreach(TEST_FILE ${TEST_FILES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        add_test(
            NAME generate_${TEST_NAME}
            COMMAND lacemodelica_exe ${TEST_FILE} --output-dir ${CMAKE_SOURCE_DIR}/test/output/${TEST_NAME}_fmu
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
        set_tests_properties(generate_${TEST_NAME} PROPERTIES LABELS "generate")
    endforeach()
endif()

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

    # Auto-discover test files with ONNX reference implementations
    # Find all .bmo files that contain @onnx-test-reference marker
    execute_process(
        COMMAND grep -l "@onnx-test-reference" ${TEST_FILES}
        OUTPUT_VARIABLE ONNX_REFERENCE_FILES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Convert file paths to test names
    string(REPLACE "\n" ";" ONNX_REFERENCE_FILES "${ONNX_REFERENCE_FILES}")
    set(ONNX_REFERENCE_TESTS "")
    foreach(FILE_PATH ${ONNX_REFERENCE_FILES})
        get_filename_component(TEST_NAME ${FILE_PATH} NAME_WE)
        list(APPEND ONNX_REFERENCE_TESTS ${TEST_NAME})
    endforeach()

    list(LENGTH ONNX_REFERENCE_TESTS ONNX_TEST_COUNT)
    message(STATUS "Found ${ONNX_TEST_COUNT} ONNX reference tests: ${ONNX_REFERENCE_TESTS}")

    # Tests that are expected to fail (but we still want to run them)
    set(ONNX_EXPECTED_FAILURES
        DerExpression  # Uses custom Der operator for derivative of expressions
        MinimalValid  # Empty model with no equations - validation test not meaningful
    )

    # Create individual ONNX runtime validation tests for each model with reference
    foreach(TEST_NAME ${ONNX_REFERENCE_TESTS})
        add_test(
            NAME onnx_validate_${TEST_NAME}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/validate_onnx_model.py ${TEST_NAME} --verbose
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
