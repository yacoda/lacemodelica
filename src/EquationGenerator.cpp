// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "EquationGenerator.h"
#include "ExpressionConverter.h"
#include "GraphBuilder.h"
#include "ParseTreeNavigator.h"
#include "Utils.hpp"

#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>

#include <iostream>
#include <stdexcept>

namespace lacemodelica {

// -----------------------------------------------------------------------------
// Helper Function Implementations
// -----------------------------------------------------------------------------

ForLoopRange parseForLoopRangeFromIndex(basemodelica::BaseModelicaParser::ForIndexContext* forIndex) {
    ForLoopRange range;
    range.loopVar = stripQuotes(forIndex->IDENT()->getText());
    std::string rangeText = forIndex->expression()->getText();

    size_t colonPos = rangeText.find(':');
    if (colonPos == std::string::npos || rangeText.find(':', colonPos + 1) != std::string::npos) {
        throw std::runtime_error("For-loop range must be in format start:end: " + rangeText);
    }
    try {
        range.startVal = std::stoi(rangeText.substr(0, colonPos));
        range.endVal = std::stoi(rangeText.substr(colonPos + 1));
    } catch (...) {
        throw std::runtime_error("For-loop range must contain constant integers");
    }
    return range;
}

ForLoopRange parseForLoopRange(basemodelica::BaseModelicaParser::ForEquationContext* forEqCtx) {
    return parseForLoopRangeFromIndex(forEqCtx->forIndex());
}

std::string setupLoopBodyIO(onnx::GraphProto* bodyGraph, const std::string& loopNodeName) {
    // ONNX checker validates SSA across all subgraphs, so names must be globally unique.

    // Add iter input (0-based iteration counter)
    auto* iterInput = bodyGraph->add_input();
    iterInput->set_name("i");
    auto* iterType = iterInput->mutable_type()->mutable_tensor_type();
    iterType->set_elem_type(onnx::TensorProto::INT64);
    iterType->mutable_shape();  // Scalar

    // Add condition input
    auto* condInput = bodyGraph->add_input();
    condInput->set_name("cond");
    auto* condInputType = condInput->mutable_type()->mutable_tensor_type();
    condInputType->set_elem_type(onnx::TensorProto::BOOL);
    condInputType->mutable_shape();  // Scalar

    // Add condition output (passthrough) - use loop name for uniqueness
    std::string condOutName = loopNodeName + "_cond_out";
    auto* condOutput = bodyGraph->add_output();
    condOutput->set_name(condOutName);
    auto* condOutputType = condOutput->mutable_type()->mutable_tensor_type();
    condOutputType->set_elem_type(onnx::TensorProto::BOOL);
    condOutputType->mutable_shape();  // Scalar

    // Identity node to pass condition through
    auto* condIdentity = bodyGraph->add_node();
    condIdentity->set_op_type("Identity");
    condIdentity->set_name(loopNodeName + "_cond_pass");
    condIdentity->add_input("cond");
    condIdentity->add_output(condOutName);

    return condOutName;
}

LoopSetupResult createForLoopBase(
    onnx::GraphProto* graph,
    GraphBuilder& builder,
    const ForLoopRange& range,
    const std::string& namePrefix) {

    LoopSetupResult result;
    result.loopNodeName = builder.makeTensorName(namePrefix);

    // Create trip count and condition constants in outer graph
    std::string tripCountTensor = builder.addInt64Constant(range.tripCount(), "n_" + result.loopNodeName);
    std::string condTensor = builder.addBoolConstant(true);

    // Create Loop node
    result.loopNode = graph->add_node();
    result.loopNode->set_op_type("Loop");
    result.loopNode->set_name(result.loopNodeName);
    result.loopNode->add_input(tripCountTensor);
    result.loopNode->add_input(condTensor);

    // Create body graph attribute
    auto* bodyAttr = result.loopNode->add_attribute();
    bodyAttr->set_name("body");
    bodyAttr->set_type(onnx::AttributeProto::GRAPH);
    result.bodyGraph = bodyAttr->mutable_g();
    result.bodyGraph->set_name("body");

    // Set up standard body I/O (iter, cond passthrough)
    setupLoopBodyIO(result.bodyGraph, result.loopNodeName);

    // Convert 0-based iter to 1-based Modelica index using a temporary body builder
    auto bodyBuilder = builder.forSubgraph(result.bodyGraph, result.loopNodeName);
    std::string constOneTensor = bodyBuilder.addInt64Constant(1, "one");
    result.loopVarTensor = bodyBuilder.addBinaryOp("Add", "i", constOneTensor);

    return result;
}

std::set<std::string> scanForDerivatives(
    const std::vector<basemodelica::BaseModelicaParser::EquationContext*>& equations) {

    std::set<std::string> derivatives;
    for (auto* eq : equations) {
        std::string eqText = eq->getText();
        size_t pos = 0;
        while ((pos = eqText.find("der(", pos)) != std::string::npos) {
            size_t start = pos + 4;
            size_t end = eqText.find(")", start);
            if (end != std::string::npos) {
                std::string derArg = eqText.substr(start, end - start);
                size_t bracketPos = derArg.find('[');
                std::string baseVar = stripQuotes(
                    (bracketPos != std::string::npos) ? derArg.substr(0, bracketPos) : derArg);
                derivatives.insert("der(" + baseVar + ")");
            }
            pos = end;
        }
    }
    return derivatives;
}

void addTopLevelLoopPassthroughs(
    onnx::NodeProto* loopNode,
    onnx::GraphProto* bodyGraph,
    const std::string& loopNodeName,
    const std::string& outputSuffix,
    const ModelInfo& info,
    const std::set<std::string>& requiredDerivatives,
    std::map<std::string, std::vector<std::string>>& derivativeInputs) {

    // Add all non-derivative, non-fixed variables as loop passthroughs
    for (const auto& var : info.variables) {
        if (!var.isDerivative && var.variability != "fixed") {
            addLoopPassthrough(loopNode, bodyGraph, loopNodeName,
                               var.name, var.name,
                               onnx::TensorProto::DOUBLE, var.dimensions,
                               outputSuffix);
        }
    }

    // Add pre-discovered derivatives as passthroughs
    for (const std::string& derName : requiredDerivatives) {
        // Extract base variable name from der(x) format
        size_t start = derName.find("(") + 1;
        size_t end = derName.rfind(")");
        std::string baseVarName = derName.substr(start, end - start);
        const Variable* baseVar = info.findVariable(baseVarName);
        std::vector<std::string> dims = baseVar ? baseVar->dimensions : std::vector<std::string>{};

        addLoopPassthrough(loopNode, bodyGraph, loopNodeName,
                           derName, derName,
                           onnx::TensorProto::DOUBLE, dims,
                           outputSuffix);

        if (baseVar) {
            derivativeInputs[derName] = baseVar->dimensions;
        }
    }
}

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
    int tripCount) {

    // ONNX validates SSA globally, so use loop-prefixed names for uniqueness.

    // Add scan output to loop body (globally unique name)
    std::string scanOutName = loopNodeName + "_scan_" + std::to_string(scanIndex);
    auto* scanOutput = bodyGraph->add_output();
    scanOutput->set_name(scanOutName);
    auto* scanType = scanOutput->mutable_type()->mutable_tensor_type();
    scanType->set_elem_type(onnx::TensorProto::DOUBLE);
    scanType->mutable_shape();  // Scalar

    // Identity to connect residual to scan output
    auto* scanIdentity = bodyGraph->add_node();
    scanIdentity->set_op_type("Identity");
    scanIdentity->set_name(loopNodeName + "_to_scan_" + std::to_string(scanIndex));
    scanIdentity->add_input(residualTensor);
    scanIdentity->add_output(scanOutName);

    // Add scan output to loop node (outer graph scope)
    std::string loopOutputName = prefix + "[" + std::to_string(equationIndex + scanIndex) + "]";
    loopNode->add_output(loopOutputName);

    // Add to graph outputs (only if this is a top-level loop)
    if (!isNested && mainGraph) {
        auto* graphOutput = mainGraph->add_output();
        graphOutput->set_name(loopOutputName);
        auto* graphOutputType = graphOutput->mutable_type()->mutable_tensor_type();
        graphOutputType->set_elem_type(onnx::TensorProto::DOUBLE);
        auto* graphOutputShape = graphOutputType->mutable_shape();
        if (tripCount > 0) {
            graphOutputShape->add_dim()->set_dim_value(tripCount);
        } else {
            graphOutputShape->add_dim();  // Unknown size
        }
    }

    return loopOutputName;
}

void createEquationResidual(
    onnx::GraphProto* graph,
    const std::string& lhsTensor,
    const std::string& rhsTensor,
    const std::string& outputName,
    const std::string& nodeName,
    bool isBoolean,
    const std::string& comment,
    const std::string& sourceFile,
    size_t sourceLine) {

    auto* node = graph->add_node();
    node->set_op_type(isBoolean ? "Equal" : "Sub");
    node->set_name(nodeName);
    node->add_input(lhsTensor);
    node->add_input(rhsTensor);
    node->add_output(outputName);

    auto* output = graph->add_output();
    output->set_name(outputName);
    auto* outputType = output->mutable_type()->mutable_tensor_type();
    outputType->set_elem_type(isBoolean ? onnx::TensorProto::BOOL : onnx::TensorProto::DOUBLE);
    outputType->mutable_shape();

    if (!comment.empty()) {
        output->set_doc_string(comment);
    }
    addSourceLocationMetadata(output, sourceFile, sourceLine);
}

// -----------------------------------------------------------------------------
// EquationGenerator Implementation
// -----------------------------------------------------------------------------

void EquationGenerator::generateOutputs(
    const std::vector<Equation>& equations,
    const std::string& prefix,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    int& loopCounter,
    std::map<std::string, std::vector<std::string>>& derivativeInputs) {

    ConversionContext ctx(info, graph, nodeCounter, nullptr, &derivativeInputs);

    size_t equationOutputIndex = 0;
    for (size_t i = 0; i < equations.size(); i++) {
        const auto& eq = equations[i];

        if (eq.isForEquation()) {
            size_t numOutputs = generateForLoop(eq, prefix, equationOutputIndex, info, graph, nodeCounter, loopCounter, derivativeInputs);
            equationOutputIndex += numOutputs;
            continue;
        }

        if (eq.isIfEquation()) {
            size_t numOutputs = generateIfEquation(eq, prefix, equationOutputIndex, info, graph, nodeCounter, derivativeInputs);
            equationOutputIndex += numOutputs;
            continue;
        }

        // Check if LHS is a tuple (outputExpressionList) for multi-output functions
        bool isMultiOutput = false;
        std::vector<std::string> outputVarNames;

        // The first primary of "(a + b) * c = d" is a parenthesized expression, not a tuple:
        // a tuple is all of the left-hand side, and has more than one slot
        auto outExprList = eq.lhsContext ? ParseTreeNavigator::findOutputExpressionList(eq.lhsContext) : nullptr;
        if (outExprList && outExprList->children.size() > 1 &&
            eq.lhsContext->getText() == "(" + outExprList->getText() + ")") {
            isMultiOutput = true;
            for (auto outExpr : outExprList->expression()) {
                if (outExpr) {
                    outputVarNames.push_back(stripQuotes(outExpr->getText()));
                }
            }
        }

        if (isMultiOutput) {
            std::vector<std::string> outputTensors = ExpressionConverter::convertMultiOutput(eq.rhsContext, ctx, outputVarNames.size());

            if (outputTensors.size() != outputVarNames.size()) {
                throw std::runtime_error("Multi-output function returned " + std::to_string(outputTensors.size()) +
                                       " outputs, expected " + std::to_string(outputVarNames.size()));
            }

            // Convert each LHS expression to a tensor name
            auto outExprList = ParseTreeNavigator::findOutputExpressionList(eq.lhsContext);
            auto lhsExprs = outExprList->expression();

            for (size_t j = 0; j < outputVarNames.size(); j++) {
                std::string eqOutputName = prefix + "[" + std::to_string(i + j) + "]";

                // Convert the LHS expression to get the proper tensor name (e.g., der('x') -> der(x))
                std::string lhsTensor = ExpressionConverter::convert(lhsExprs[j], ctx);

                createEquationResidual(graph, lhsTensor, outputTensors[j], eqOutputName,
                                       "eq_residual_" + std::to_string(i + j),
                                       false, "", eq.sourceFile, eq.sourceLine);
            }

            i += outputVarNames.size() - 1;
            continue;
        }

        std::string lhsTensor, rhsTensor;
        try {
            try {
                // The variable of a declaration equation is a graph input of its own name
                lhsTensor = eq.lhsContext ? ExpressionConverter::convert(eq.lhsContext, ctx) : eq.lhsVariable;
            } catch (const std::exception& e) {
                std::cerr << "Error converting LHS of " << prefix << " equation " << i;
                if (!eq.sourceFile.empty()) {
                    std::cerr << " (" << eq.sourceFile << ":" << eq.sourceLine << ")";
                }
                std::cerr << ": " << e.what() << std::endl;
                std::cerr << "LHS text: " << eq.lhsContext->getText() << std::endl;
                throw;
            }

            try {
                rhsTensor = ExpressionConverter::convert(eq.rhsContext, ctx);
            } catch (const std::exception& e) {
                std::cerr << "Error converting RHS of " << prefix << " equation " << i;
                if (!eq.sourceFile.empty()) {
                    std::cerr << " (" << eq.sourceFile << ":" << eq.sourceLine << ")";
                }
                std::cerr << ": " << e.what() << std::endl;
                std::cerr << "RHS text: " << eq.rhsContext->getText() << std::endl;
                throw;
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Cannot convert equation " + prefix + "[" + std::to_string(equationOutputIndex) +
                                     "]: " + e.what());
        }

        std::string eqOutputName = prefix + "[" + std::to_string(equationOutputIndex) + "]";

        std::string lhsText = eq.lhsContext ? stripQuotes(eq.lhsContext->getText()) : eq.lhsVariable;
        const Variable* lhsVar = info.findVariable(lhsText);
        bool isBooleanEquation = (lhsVar && lhsVar->type == "Boolean");

        createEquationResidual(graph, lhsTensor, rhsTensor, eqOutputName,
                               prefix + "_residual_" + std::to_string(i),
                               isBooleanEquation, eq.comment,
                               eq.sourceFile, eq.sourceLine);

        equationOutputIndex++;
    }
}

// -----------------------------------------------------------------------------
// If-Equations
// -----------------------------------------------------------------------------
//
// An if-equation with n equations per branch becomes n outputs. The k-th output is built from the
// k-th equation of every branch, also through nested if-equations (MLS 8.3.4 wants as many equations
// in every branch). If the k-th equations have the same left-hand side, this gives lhs - If(c, r1, r2),
// else If(c, l1 - r1, l2 - r2).

namespace {

using IfEquationCtx = basemodelica::BaseModelicaParser::IfEquationContext;
using EquationBlockCtx = basemodelica::BaseModelicaParser::EquationBlockContext;
using EquationCtx = basemodelica::BaseModelicaParser::EquationContext;

size_t countIfEquations(IfEquationCtx* ifEq);

size_t countBlockEquations(EquationBlockCtx* block) {
    size_t count = 0;
    for (auto* eq : block->equation()) {
        if (eq->ifEquation()) {
            count += countIfEquations(eq->ifEquation());
        } else if (eq->simpleExpression() && eq->expression()) {
            count++;
        } else {
            throw std::runtime_error("Only equations and if-equations are supported in an if-equation: " + eq->getText());
        }
    }
    return count;
}

size_t countIfEquations(IfEquationCtx* ifEq) {
    auto blocks = ifEq->equationBlock();
    size_t count = countBlockEquations(blocks[0]);
    for (auto* block : blocks) {
        if (countBlockEquations(block) != count) {
            throw std::runtime_error("Branches of an if-equation with different numbers of equations: " + ifEq->getText());
        }
    }
    // Without an else, some branch must always hold (elseif true)
    auto conditions = ifEq->expression();
    if (count > 0 && blocks.size() == conditions.size() && conditions.back()->getText() != "true") {
        throw std::runtime_error("If-equation without an else branch: " + ifEq->getText());
    }
    return count;
}

// The k-th equation of a block: an equation, or an if-equation with the index k within it
struct BlockEquation {
    EquationCtx* eq = nullptr;
    IfEquationCtx* nested = nullptr;
    size_t nestedIndex = 0;
};

BlockEquation findBlockEquation(EquationBlockCtx* block, size_t k) {
    for (auto* eq : block->equation()) {
        if (eq->ifEquation()) {
            size_t n = countIfEquations(eq->ifEquation());
            if (k < n) return {nullptr, eq->ifEquation(), k};
            k -= n;
        } else {
            if (k == 0) return {eq, nullptr, 0};
            k--;
        }
    }
    throw std::runtime_error("If-equation branch with too few equations: " + block->getText());
}

// The k-th equations of all branches, through nested if-equations
void collectBranchEquations(IfEquationCtx* ifEq, size_t k, std::vector<EquationCtx*>& out) {
    for (auto* block : ifEq->equationBlock()) {
        BlockEquation found = findBlockEquation(block, k);
        if (found.nested) {
            collectBranchEquations(found.nested, found.nestedIndex, out);
        } else {
            out.push_back(found.eq);
        }
    }
}

std::string buildIfOutput(IfEquationCtx* ifEq, size_t branch, size_t k, bool residual, const ConversionContext& ctx);

// The value of the k-th equation of a block: its right-hand side, or its residual
std::string buildBlockValue(EquationBlockCtx* block, size_t k, bool residual, const ConversionContext& ctx) {
    BlockEquation found = findBlockEquation(block, k);
    if (found.nested) {
        return buildIfOutput(found.nested, 0, found.nestedIndex, residual, ctx);
    }
    std::string rhs = ExpressionConverter::convert(found.eq->expression(), ctx);
    if (!residual) return rhs;
    std::string lhs = ExpressionConverter::convert(found.eq->simpleExpression(), ctx);
    return ctx.builder().addBinaryOp("Sub", lhs, rhs);
}

// An ONNX graph output must be produced by a node of that graph, which excludes a captured tensor
void addBranchOutput(const ConversionContext& ctx, const std::string& tensor) {
    auto builder = ctx.builder();
    builder.addScalarDoubleOutput(builder.addUnaryOp("Identity", tensor));
}

std::string buildIfOutput(IfEquationCtx* ifEq, size_t branch, size_t k, bool residual, const ConversionContext& ctx) {
    auto conditions = ifEq->expression();
    auto blocks = ifEq->equationBlock();
    // The else branch, or a last "elseif true" without one
    if (branch >= conditions.size() ||
        (branch + 1 == conditions.size() && blocks.size() == conditions.size() && conditions[branch]->getText() == "true")) {
        return buildBlockValue(blocks[branch], k, residual, ctx);
    }

    auto builder = ctx.builder();
    std::string condition = ExpressionConverter::convert(conditions[branch], ctx);

    onnx::GraphProto thenBranch;
    thenBranch.set_name("then_branch_" + std::to_string(ctx.nodeCounter++));
    ConversionContext thenCtx = ctx.withGraph(&thenBranch);
    addBranchOutput(thenCtx, buildBlockValue(blocks[branch], k, residual, thenCtx));

    onnx::GraphProto elseBranch;
    elseBranch.set_name("else_branch_" + std::to_string(ctx.nodeCounter++));
    ConversionContext elseCtx = ctx.withGraph(&elseBranch);
    addBranchOutput(elseCtx, buildIfOutput(ifEq, branch + 1, k, residual, elseCtx));

    return builder.addIfNode(condition, thenBranch, elseBranch, "If_eq");
}

}  // namespace

size_t EquationGenerator::generateIfEquation(
    const Equation& eq,
    const std::string& prefix,
    size_t equationIndex,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    std::map<std::string, std::vector<std::string>>& derivativeInputs) {

    auto* ifEqCtx = dynamic_cast<IfEquationCtx*>(eq.ifEquationContext);
    if (!ifEqCtx) {
        throw std::runtime_error("Invalid if-equation context");
    }

    size_t count = countIfEquations(ifEqCtx);
    ConversionContext ctx(info, graph, nodeCounter, nullptr, &derivativeInputs);

    for (size_t k = 0; k < count; k++) {
        std::vector<EquationCtx*> branchEquations;
        collectBranchEquations(ifEqCtx, k, branchEquations);
        bool sameLhs = true;
        for (auto* branchEq : branchEquations) {
            sameLhs = sameLhs && branchEq->simpleExpression()->getText() == branchEquations[0]->simpleExpression()->getText();
        }

        std::string eqOutputName = prefix + "[" + std::to_string(equationIndex + k) + "]";
        auto builder = ctx.builder();
        if (sameLhs) {
            std::string lhs = ExpressionConverter::convert(branchEquations[0]->simpleExpression(), ctx);
            std::string rhs = buildIfOutput(ifEqCtx, 0, k, false, ctx);
            auto* subNode = graph->add_node();
            subNode->set_op_type("Sub");
            subNode->set_name(prefix + "_if_residual_" + std::to_string(equationIndex + k));
            subNode->add_input(lhs);
            subNode->add_input(rhs);
            subNode->add_output(eqOutputName);
        } else {
            builder.addIdentity(buildIfOutput(ifEqCtx, 0, k, true, ctx), eqOutputName,
                                prefix + "_if_residual_" + std::to_string(equationIndex + k));
        }

        auto* output = graph->add_output();
        output->set_name(eqOutputName);
        auto* outputType = output->mutable_type()->mutable_tensor_type();
        outputType->set_elem_type(onnx::TensorProto::DOUBLE);
        outputType->mutable_shape();

        addSourceLocationMetadata(output, eq.sourceFile, eq.sourceLine);
    }

    return count;
}

size_t EquationGenerator::generateForLoop(
    const Equation& eq,
    const std::string& prefix,
    size_t equationIndex,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    int& loopCounter,
    std::map<std::string, std::vector<std::string>>& derivativeInputs,
    bool isNested,
    std::map<std::string, std::string>* parentLoopVarMap,
    std::string* outLoopNodeName) {

    auto* forEqCtx = dynamic_cast<basemodelica::BaseModelicaParser::ForEquationContext*>(eq.forEquationContext);
    if (!forEqCtx) {
        throw std::runtime_error("Invalid for-equation context");
    }

    ForLoopRange range = parseForLoopRange(forEqCtx);
    auto loopEquations = forEqCtx->equation();

    GraphBuilder builder(graph, nodeCounter);
    // Use separate loopCounter for clean loop naming (loop_0, loop_1, ...)
    std::string loopNodeName = "loop_" + std::to_string(loopCounter++);
    if (outLoopNodeName) {
        *outLoopNodeName = loopNodeName;
    }

    // These constants are in the outer graph scope
    std::string tripCountTensor = builder.addInt64Constant(range.tripCount(), "n_" + loopNodeName);
    std::string condTensor = builder.addBoolConstant(true);

    auto* loopNode = graph->add_node();
    loopNode->set_op_type("Loop");
    loopNode->set_name(loopNodeName);
    loopNode->add_input(tripCountTensor);
    loopNode->add_input(condTensor);

    auto* bodyAttr = loopNode->add_attribute();
    bodyAttr->set_name("body");
    bodyAttr->set_type(onnx::AttributeProto::GRAPH);
    auto* bodyGraph = bodyAttr->mutable_g();
    bodyGraph->set_name("body");

    setupLoopBodyIO(bodyGraph, loopNodeName);

    if (isNested && parentLoopVarMap) {
        std::string outputSuffix = "_out";
        for (const auto& [varName, tensorName] : *parentLoopVarMap) {
            builder.addLoopPassthrough(loopNode, bodyGraph, loopNodeName,
                                       tensorName, "parent_" + varName,
                                       onnx::TensorProto::INT64, {},
                                       outputSuffix);
        }
    }

    // Inside loop body - use loop prefix for global SSA uniqueness
    auto bodyBuilder = builder.forSubgraph(bodyGraph, loopNodeName);
    std::string constOneTensor = bodyBuilder.addInt64Constant(1, "one");
    std::string loopVarTensor = bodyBuilder.addBinaryOp("Add", "i", constOneTensor);

    std::set<std::string> requiredDerivatives = scanForDerivatives(loopEquations);

    if (!isNested) {
        std::string outputSuffix = "_out";
        addTopLevelLoopPassthroughs(loopNode, bodyGraph, loopNodeName, outputSuffix,
                                    info, requiredDerivatives, derivativeInputs);
    }

    // Use the same counter for the entire loop body to ensure SSA uniqueness
    size_t scanOutputCount = 0;

    for (size_t eqIdx = 0; eqIdx < loopEquations.size(); eqIdx++) {
        auto* innerEq = loopEquations[eqIdx];

        if (innerEq->forEquation()) {
            Equation nestedEq;
            nestedEq.forEquationContext = innerEq->forEquation();
            nestedEq.sourceFile = eq.sourceFile;
            nestedEq.sourceLine = innerEq->getStart()->getLine();

            std::map<std::string, std::string> combinedLoopVarMap;
            if (parentLoopVarMap) {
                combinedLoopVarMap = *parentLoopVarMap;
            }

            // Copy current loop variable for nested loop (globally unique names)
            std::string currentLoopIterCopy = loopNodeName + "_" + range.loopVar + "_" + std::to_string(nodeCounter++);
            auto* iterCopyNode = bodyGraph->add_node();
            iterCopyNode->set_op_type("Identity");
            iterCopyNode->set_name(loopNodeName + "_copy_" + range.loopVar);
            iterCopyNode->add_input(loopVarTensor);
            iterCopyNode->add_output(currentLoopIterCopy);

            combinedLoopVarMap[range.loopVar] = currentLoopIterCopy;

            std::string nestedLoopNodeName;
            size_t nestedOutputCount = generateForLoop(
                nestedEq,
                prefix,
                equationIndex + scanOutputCount,
                info,
                bodyGraph,
                nodeCounter,
                loopCounter,
                derivativeInputs,
                true,
                &combinedLoopVarMap,
                &nestedLoopNodeName);

            onnx::NodeProto* nestedLoopNode = nullptr;
            for (int ni = bodyGraph->node_size() - 1; ni >= 0; ni--) {
                if (bodyGraph->node(ni).name() == nestedLoopNodeName) {
                    nestedLoopNode = bodyGraph->mutable_node(ni);
                    break;
                }
            }

            size_t nestedCarriedCount = 0;
            if (nestedLoopNode) {
                nestedCarriedCount = nestedLoopNode->output_size() - nestedOutputCount;
            }

            for (size_t i = 0; i < nestedOutputCount; i++) {
                std::string nestedOutputName;
                if (nestedLoopNode) {
                    size_t outputIndex = nestedCarriedCount + i;
                    if (outputIndex < nestedLoopNode->output_size()) {
                        nestedOutputName = nestedLoopNode->output(outputIndex);
                    }
                }

                // Globally unique names (ONNX validates SSA across all subgraphs)
                std::string scanOutName = loopNodeName + "_nested_scan_" + std::to_string(scanOutputCount + i);
                auto* scanOutput = bodyGraph->add_output();
                scanOutput->set_name(scanOutName);
                auto* scanType = scanOutput->mutable_type()->mutable_tensor_type();
                scanType->set_elem_type(onnx::TensorProto::DOUBLE);
                auto* scanShape = scanType->mutable_shape();
                scanShape->add_dim();

                auto* scanIdentity = bodyGraph->add_node();
                scanIdentity->set_op_type("Identity");
                scanIdentity->set_name(loopNodeName + "_to_nested_" + std::to_string(scanOutputCount + i));
                scanIdentity->add_input(nestedOutputName);
                scanIdentity->add_output(scanOutName);

                // Outer graph scope - loop node output
                std::string loopOutputName = prefix + "[" + std::to_string(equationIndex + scanOutputCount + i) + "]_nested";
                loopNode->add_output(loopOutputName);

                if (!isNested) {
                    std::string topLevelEqName = prefix + "[" + std::to_string(equationIndex + scanOutputCount + i) + "]";

                    auto* renameNode = graph->add_node();
                    renameNode->set_op_type("Identity");
                    renameNode->set_name("to_" + topLevelEqName);
                    renameNode->add_input(loopOutputName);
                    renameNode->add_output(topLevelEqName);

                    auto* graphOutput = graph->add_output();
                    graphOutput->set_name(topLevelEqName);
                    auto* graphOutputType = graphOutput->mutable_type()->mutable_tensor_type();
                    graphOutputType->set_elem_type(onnx::TensorProto::DOUBLE);
                    auto* graphShape = graphOutputType->mutable_shape();
                    graphShape->add_dim();
                    graphShape->add_dim();
                }
            }

            scanOutputCount += nestedOutputCount;
            continue;
        }

        auto simpleExpr = innerEq->simpleExpression();
        auto fullExpr = innerEq->expression();

        if (!simpleExpr || !fullExpr) {
            throw std::runtime_error("Only simple equations are supported in a for-loop: " + innerEq->getText());
        }

        std::map<std::string, std::string> loopVarMap;
        if (parentLoopVarMap) {
            for (const auto& [varName, tensorName] : *parentLoopVarMap) {
                loopVarMap[varName] = "parent_" + varName;
            }
        }
        loopVarMap[range.loopVar] = loopVarTensor;

        std::string lhsTensor, rhsTensor;
        try {
            // Use loop prefix for global SSA uniqueness
            ConversionContext bodyCtx(info, bodyGraph, nodeCounter, &loopVarMap, &derivativeInputs, loopNodeName);
            lhsTensor = ExpressionConverter::convert(simpleExpr, bodyCtx);
            rhsTensor = ExpressionConverter::convert(fullExpr, bodyCtx);
        } catch (const std::exception& e) {
            throw std::runtime_error("Cannot convert equation " + std::to_string(eqIdx) + " in a for-loop: " + e.what());
        }

        // Globally unique residual name (ONNX validates SSA across all subgraphs)
        std::string residualTensor = loopNodeName + "_res_" + std::to_string(nodeCounter++);
        auto* subNode = bodyGraph->add_node();
        subNode->set_op_type("Sub");
        subNode->set_name(loopNodeName + "_sub_" + std::to_string(nodeCounter));
        subNode->add_input(lhsTensor);
        subNode->add_input(rhsTensor);
        subNode->add_output(residualTensor);

        addLoopScanOutput(loopNode, bodyGraph, graph, loopNodeName, residualTensor,
                          scanOutputCount, prefix, equationIndex, isNested, range.tripCount());

        scanOutputCount++;
    }

    return scanOutputCount;
}

} // namespace lacemodelica
