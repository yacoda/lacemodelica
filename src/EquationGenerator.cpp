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

ForLoopRange parseForLoopRange(basemodelica::BaseModelicaParser::ForEquationContext* forEqCtx) {
    ForLoopRange range;
    auto forIndex = forEqCtx->forIndex();
    range.loopVar = forIndex->IDENT()->getText();
    std::string rangeText = forIndex->expression()->getText();

    size_t colonPos = rangeText.find(':');
    if (colonPos == std::string::npos) {
        throw std::runtime_error("For-equation range must be in format start:end");
    }
    try {
        range.startVal = std::stoi(rangeText.substr(0, colonPos));
        range.endVal = std::stoi(rangeText.substr(colonPos + 1));
    } catch (...) {
        throw std::runtime_error("For-equation range must contain constant integers");
    }
    return range;
}

std::string setupLoopBodyIO(onnx::GraphProto* bodyGraph, const std::string& loopNodeName) {
    // Add iter input (0-based iteration counter)
    auto* iterInput = bodyGraph->add_input();
    iterInput->set_name("iter");
    auto* iterType = iterInput->mutable_type()->mutable_tensor_type();
    iterType->set_elem_type(onnx::TensorProto::INT64);
    iterType->mutable_shape();  // Scalar

    // Add condition input
    auto* condInput = bodyGraph->add_input();
    condInput->set_name("cond_in");
    auto* condInputType = condInput->mutable_type()->mutable_tensor_type();
    condInputType->set_elem_type(onnx::TensorProto::BOOL);
    condInputType->mutable_shape();  // Scalar

    // Add condition output (passthrough)
    std::string condOutName = loopNodeName + "_cond_out";
    auto* condOutput = bodyGraph->add_output();
    condOutput->set_name(condOutName);
    auto* condOutputType = condOutput->mutable_type()->mutable_tensor_type();
    condOutputType->set_elem_type(onnx::TensorProto::BOOL);
    condOutputType->mutable_shape();  // Scalar

    // Identity node to pass condition through
    auto* condIdentity = bodyGraph->add_node();
    condIdentity->set_op_type("Identity");
    condIdentity->set_name(loopNodeName + "_cond_passthrough");
    condIdentity->add_input("cond_in");
    condIdentity->add_output(condOutName);

    return condOutName;
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
                derivatives.insert("der('" + baseVar + "')");
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
        size_t start = derName.find("'") + 1;
        size_t end = derName.rfind("'");
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

    // Add scan output to loop body
    auto* scanOutput = bodyGraph->add_output();
    std::string scanOutName = loopNodeName + "_scan_" + std::to_string(scanIndex);
    scanOutput->set_name(scanOutName);
    auto* scanType = scanOutput->mutable_type()->mutable_tensor_type();
    scanType->set_elem_type(onnx::TensorProto::DOUBLE);
    scanType->mutable_shape();  // Scalar

    // Identity to connect residual to scan output
    auto* scanIdentity = bodyGraph->add_node();
    scanIdentity->set_op_type("Identity");
    scanIdentity->set_name(loopNodeName + "_scan_identity_" + std::to_string(scanIndex));
    scanIdentity->add_input(residualTensor);
    scanIdentity->add_output(scanOutName);

    // Add scan output to loop node
    std::string loopOutputName = prefix + "[" + std::to_string(equationIndex + scanIndex) + "]";
    if (isNested) {
        loopOutputName += "_" + loopNodeName;
    }
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
    std::map<std::string, std::vector<std::string>>& derivativeInputs) {

    ConversionContext ctx(info, graph, nodeCounter, nullptr, &derivativeInputs);

    size_t equationOutputIndex = 0;
    for (size_t i = 0; i < equations.size(); i++) {
        const auto& eq = equations[i];

        if (eq.isForEquation()) {
            size_t numOutputs = generateForLoop(eq, prefix, equationOutputIndex, info, graph, nodeCounter, derivativeInputs);
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

        if (auto outExprList = ParseTreeNavigator::findOutputExpressionList(eq.lhsContext)) {
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

            for (size_t j = 0; j < outputVarNames.size(); j++) {
                std::string eqOutputName = prefix + "[" + std::to_string(i + j) + "]";
                createEquationResidual(graph, outputVarNames[j], outputTensors[j], eqOutputName,
                                       "eq_residual_" + std::to_string(i + j),
                                       false, "", eq.sourceFile, eq.sourceLine);
            }

            i += outputVarNames.size() - 1;
            continue;
        }

        std::string lhsTensor, rhsTensor;
        try {
            try {
                lhsTensor = ExpressionConverter::convert(eq.lhsContext, ctx);
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
            std::cerr << "Warning: Skipping equation " << prefix << "[" << equationOutputIndex << "] due to conversion error" << std::endl;
            equationOutputIndex++;
            continue;
        }

        std::string eqOutputName = prefix + "[" + std::to_string(equationOutputIndex) + "]";

        std::string lhsText = stripQuotes(eq.lhsContext->getText());
        const Variable* lhsVar = info.findVariable(lhsText);
        bool isBooleanEquation = (lhsVar && lhsVar->type == "Boolean");

        createEquationResidual(graph, lhsTensor, rhsTensor, eqOutputName,
                               prefix + "_residual_" + std::to_string(i),
                               isBooleanEquation, eq.comment,
                               eq.sourceFile, eq.sourceLine);

        equationOutputIndex++;
    }
}

size_t EquationGenerator::generateIfEquation(
    const Equation& eq,
    const std::string& prefix,
    size_t equationIndex,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
    std::map<std::string, std::vector<std::string>>& derivativeInputs) {

    auto* ifEqCtx = dynamic_cast<basemodelica::BaseModelicaParser::IfEquationContext*>(eq.ifEquationContext);
    if (!ifEqCtx) {
        throw std::runtime_error("Invalid if-equation context");
    }

    auto expressions = ifEqCtx->expression();
    auto allEquations = ifEqCtx->equation();

    if (allEquations.empty()) {
        std::cerr << "Warning: Empty if-equation" << std::endl;
        return 0;
    }

    auto firstEq = allEquations[0];
    auto firstSimpleExpr = firstEq->simpleExpression();
    if (!firstSimpleExpr) {
        throw std::runtime_error("If-equation branch must contain simple equation");
    }
    std::string lhsVarName = stripQuotes(firstSimpleExpr->getText());

    ConversionContext ifCtx(info, graph, nodeCounter, nullptr, &derivativeInputs);
    std::string rhsTensor = buildIfEquationRhs(expressions, allEquations, 0, ifCtx);

    std::string eqOutputName = prefix + "[" + std::to_string(equationIndex) + "]";

    auto* subNode = graph->add_node();
    subNode->set_op_type("Sub");
    subNode->set_name(prefix + "_if_residual_" + std::to_string(equationIndex));
    subNode->add_input(lhsVarName);
    subNode->add_input(rhsTensor);
    subNode->add_output(eqOutputName);

    auto* output = graph->add_output();
    output->set_name(eqOutputName);
    auto* outputType = output->mutable_type()->mutable_tensor_type();
    outputType->set_elem_type(onnx::TensorProto::DOUBLE);
    outputType->mutable_shape();

    addSourceLocationMetadata(output, eq.sourceFile, eq.sourceLine);

    return 1;
}

std::string EquationGenerator::buildIfEquationRhs(
    const std::vector<basemodelica::BaseModelicaParser::ExpressionContext*>& conditions,
    const std::vector<basemodelica::BaseModelicaParser::EquationContext*>& equations,
    size_t branchIndex,
    const ConversionContext& ctx) {

    auto builder = ctx.builder();

    if (branchIndex >= conditions.size()) {
        if (branchIndex < equations.size()) {
            auto* eqCtx = equations[branchIndex];
            auto* rhsExpr = eqCtx->expression();
            if (rhsExpr) {
                return ExpressionConverter::convert(rhsExpr, ctx);
            }
        }
        return builder.addDoubleConstant(0.0);
    }

    std::string condTensor = ExpressionConverter::convert(conditions[branchIndex], ctx);

    onnx::GraphProto thenBranch;
    thenBranch.set_name("then_branch_" + std::to_string(branchIndex));
    auto thenBuilder = builder.forSubgraph(&thenBranch);

    std::string thenResult;
    if (branchIndex < equations.size()) {
        auto* eqCtx = equations[branchIndex];
        auto* rhsExpr = eqCtx->expression();
        if (rhsExpr) {
            thenResult = ExpressionConverter::convert(rhsExpr, ctx.withGraph(&thenBranch));
        }
    }
    if (thenResult.empty()) {
        thenResult = thenBuilder.addDoubleConstant(0.0);
    }
    thenBuilder.addScalarDoubleOutput(thenResult);

    onnx::GraphProto elseBranch;
    elseBranch.set_name("else_branch_" + std::to_string(branchIndex));
    std::string elseResult = buildIfEquationRhs(conditions, equations, branchIndex + 1, ctx.withGraph(&elseBranch));
    builder.forSubgraph(&elseBranch).addScalarDoubleOutput(elseResult);

    return builder.addIfNode(condTensor, thenBranch, elseBranch, "If_eq");
}

size_t EquationGenerator::generateForLoop(
    const Equation& eq,
    const std::string& prefix,
    size_t equationIndex,
    const ModelInfo& info,
    onnx::GraphProto* graph,
    int& nodeCounter,
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
    std::string loopNodeName = "for_loop_" + std::to_string(nodeCounter++);
    if (outLoopNodeName) {
        *outLoopNodeName = loopNodeName;
    }

    std::string tripCountTensor = builder.addInt64Constant(range.tripCount(), "trip_count_" + loopNodeName);
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
    bodyGraph->set_name("loop_body_" + std::to_string(nodeCounter++));

    setupLoopBodyIO(bodyGraph, loopNodeName);

    if (isNested && parentLoopVarMap) {
        std::string outputSuffix = "_final_" + std::to_string(equationIndex);
        for (const auto& [varName, tensorName] : *parentLoopVarMap) {
            builder.addLoopPassthrough(loopNode, bodyGraph, loopNodeName,
                                       tensorName, "parent_" + varName,
                                       onnx::TensorProto::INT64, {},
                                       outputSuffix);
        }
    }

    auto bodyBuilder = builder.forSubgraph(bodyGraph, loopNodeName);
    std::string constOneTensor = bodyBuilder.addInt64Constant(1, loopNodeName + "_const_one");
    std::string loopVarTensor = bodyBuilder.withPrefix(loopNodeName + "_" + range.loopVar).addBinaryOp("Add", "iter", constOneTensor);

    std::set<std::string> requiredDerivatives = scanForDerivatives(loopEquations);

    if (!isNested) {
        std::string outputSuffix = "_final_" + std::to_string(equationIndex);
        addTopLevelLoopPassthroughs(loopNode, bodyGraph, loopNodeName, outputSuffix,
                                    info, requiredDerivatives, derivativeInputs);
    }

    int bodyNodeCounter = 0;
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

            std::string currentLoopIterCopy = "loop_iter_" + range.loopVar + "_" + std::to_string(bodyNodeCounter++);
            auto* iterCopyNode = bodyGraph->add_node();
            iterCopyNode->set_op_type("Identity");
            iterCopyNode->set_name("copy_iter_for_nested_" + range.loopVar);
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
                std::string collectedOutputName = "collected_nested_" + std::to_string(scanOutputCount + i);

                std::string nestedOutputName;
                if (nestedLoopNode) {
                    size_t outputIndex = nestedCarriedCount + i;
                    if (outputIndex < nestedLoopNode->output_size()) {
                        nestedOutputName = nestedLoopNode->output(outputIndex);
                    }
                }

                auto* scanOutput = bodyGraph->add_output();
                std::string scanOutName = loopNodeName + "_scan_" + std::to_string(scanOutputCount + i);
                scanOutput->set_name(scanOutName);
                auto* scanType = scanOutput->mutable_type()->mutable_tensor_type();
                scanType->set_elem_type(onnx::TensorProto::DOUBLE);
                auto* scanShape = scanType->mutable_shape();
                scanShape->add_dim();

                auto* scanIdentity = bodyGraph->add_node();
                scanIdentity->set_op_type("Identity");
                scanIdentity->set_name(loopNodeName + "_scan_identity_nested_" + std::to_string(scanOutputCount + i));
                scanIdentity->add_input(nestedOutputName);
                scanIdentity->add_output(scanOutName);

                loopNode->add_output(collectedOutputName);

                if (!isNested) {
                    std::string topLevelEqName = prefix + "[" + std::to_string(equationIndex + scanOutputCount + i) + "]";

                    auto* renameNode = graph->add_node();
                    renameNode->set_op_type("Identity");
                    renameNode->set_name("rename_collected_" + std::to_string(scanOutputCount + i));
                    renameNode->add_input(collectedOutputName);
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
            std::cerr << "Warning: Skipping non-simple equation in for-loop" << std::endl;
            continue;
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
            ConversionContext bodyCtx(info, bodyGraph, bodyNodeCounter, &loopVarMap, &derivativeInputs, loopNodeName);
            lhsTensor = ExpressionConverter::convert(simpleExpr, bodyCtx);
            rhsTensor = ExpressionConverter::convert(fullExpr, bodyCtx);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to convert equation " << eqIdx << " in for-loop: " << e.what() << std::endl;
            continue;
        }

        std::string residualTensor = loopNodeName + "_residual_" + std::to_string(scanOutputCount);
        auto* subNode = bodyGraph->add_node();
        subNode->set_op_type("Sub");
        subNode->set_name(residualTensor);
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
