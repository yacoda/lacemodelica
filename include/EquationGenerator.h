// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include "ModelInfo.h"
#include "ONNXHelpers.hpp"
#include "BaseModelicaParser.h"
#include <string>
#include <vector>
#include <map>
#include <set>

namespace onnx {
    class GraphProto;
    class NodeProto;
}

namespace lacemodelica {

/**
 * Generates ONNX outputs for Modelica equations.
 *
 * This class handles the transformation of Modelica equations into
 * ONNX graph outputs. Each equation becomes a residual (LHS - RHS)
 * or equality check for boolean equations.
 *
 * Supports:
 * - Simple equations (lhs = rhs)
 * - For-equations (unrolled into ONNX Loop nodes)
 * - If-equations (converted to ONNX If nodes)
 * - Multi-output function calls
 */
class EquationGenerator {
public:
    // Generate ONNX outputs for a list of equations
    // Parameters:
    //   equations: List of equations to process
    //   prefix: Output name prefix ("eq" or "init_eq")
    //   info: Model information for variable lookups
    //   graph: Target ONNX graph
    //   nodeCounter: Counter for unique tensor/node names (SSA)
    //   loopCounter: Counter for loop names (loop_0, loop_1, ...)
    //   derivativeInputs: Map to track discovered derivative variables
    static void generateOutputs(
        const std::vector<Equation>& equations,
        const std::string& prefix,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        int& loopCounter,
        std::map<std::string, std::vector<std::string>>& derivativeInputs);

private:
    // Generate ONNX Loop node for a for-equation
    // Returns the number of equation outputs generated
    static size_t generateForLoop(
        const Equation& eq,
        const std::string& prefix,
        size_t equationIndex,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        int& loopCounter,
        std::map<std::string, std::vector<std::string>>& derivativeInputs,
        bool isNested = false,
        std::map<std::string, std::string>* parentLoopVarMap = nullptr,
        std::string* outLoopNodeName = nullptr);

    // Generate ONNX If node for an if-equation
    // Returns the number of equation outputs generated
    static size_t generateIfEquation(
        const Equation& eq,
        const std::string& prefix,
        size_t equationIndex,
        const ModelInfo& info,
        onnx::GraphProto* graph,
        int& nodeCounter,
        std::map<std::string, std::vector<std::string>>& derivativeInputs);

    // Build nested If structure for if-equation RHS
    static std::string buildIfEquationRhs(
        const std::vector<basemodelica::BaseModelicaParser::ExpressionContext*>& conditions,
        const std::vector<basemodelica::BaseModelicaParser::EquationContext*>& equations,
        size_t branchIndex,
        const ConversionContext& ctx);
};

// -----------------------------------------------------------------------------
// For-Loop Helper Structures
// -----------------------------------------------------------------------------

// Parsed information about a Modelica for-loop range
struct ForLoopRange {
    std::string loopVar;
    int startVal;
    int endVal;
    int tripCount() const { return endVal - startVal + 1; }
};

// Parse for-loop range from grammar context
ForLoopRange parseForLoopRange(basemodelica::BaseModelicaParser::ForEquationContext* forEqCtx);

// Set up standard ONNX Loop body inputs (iter, cond) and condition passthrough
// Returns the condition output name
std::string setupLoopBodyIO(onnx::GraphProto* bodyGraph, const std::string& loopNodeName);

// Scan equations for der() calls and return the set of derivative variable names
std::set<std::string> scanForDerivatives(
    const std::vector<basemodelica::BaseModelicaParser::EquationContext*>& equations);

// Add all required loop passthroughs for a top-level for-loop
void addTopLevelLoopPassthroughs(
    onnx::NodeProto* loopNode,
    onnx::GraphProto* bodyGraph,
    const std::string& loopNodeName,
    const std::string& outputSuffix,
    const ModelInfo& info,
    const std::set<std::string>& requiredDerivatives,
    std::map<std::string, std::vector<std::string>>& derivativeInputs);

// Add a scan output to a loop body and optionally to the main graph
std::string addLoopScanOutput(
    onnx::NodeProto* loopNode,
    onnx::GraphProto* bodyGraph,
    onnx::GraphProto* mainGraph,
    const std::string& loopNodeName,
    const std::string& residualTensor,
    size_t scanIndex,
    const std::string& prefix,
    size_t equationIndex,
    bool isNested,
    int tripCount = -1);

// Create an equation residual output (LHS - RHS or LHS == RHS for booleans)
void createEquationResidual(
    onnx::GraphProto* graph,
    const std::string& lhsTensor,
    const std::string& rhsTensor,
    const std::string& outputName,
    const std::string& nodeName,
    bool isBoolean,
    const std::string& comment,
    const std::string& sourceFile,
    size_t sourceLine);

} // namespace lacemodelica
