cmake_minimum_required(VERSION 3.14)

project(lacemodelica VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ANTLR4 setup
include(FetchContent)

set(ANTLR_VERSION 4.13.2)
set(ANTLR_TAG ${ANTLR_VERSION})

# Download ANTLR4 runtime for C++
FetchContent_Declare(
    antlr4_runtime
    GIT_REPOSITORY https://github.com/antlr/antlr4.git
    GIT_TAG ${ANTLR_TAG}
    SOURCE_SUBDIR runtime/Cpp
)

FetchContent_MakeAvailable(antlr4_runtime)

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

# Executable
add_executable(lacemodelica
    src/main.cpp
    ${GENERATED_SOURCES}
)
target_link_libraries(lacemodelica antlr4_shared)
