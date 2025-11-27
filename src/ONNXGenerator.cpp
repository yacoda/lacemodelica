// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ONNXGenerator.h"
#include "ONNXHelpers.hpp"
#include "ParseTreeNavigator.h"
#include "Utils.hpp"
#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>
#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>
#include <tinyxml2.h>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace lacemodelica {

// -----------------------------------------------------------------------------
// Operator Mapping Tables
// These tables map Modelica operators/functions to their ONNX equivalents
// -----------------------------------------------------------------------------

// Modelica relational operators -> ONNX comparison operators
const std::map<std::string, std::string> kRelationalOpMap = {
    {">", "Greater"},
    {"<", "Less"},
    {">=", "GreaterOrEqual"},
    {"<=", "LessOrEqual"},
    {"==", "Equal"}
    // Note: "<>" (not equal) is handled specially as Equal + Not
};

// Modelica math functions -> ONNX unary operators
const std::map<std::string, std::string> kMathFunctionMap = {
    // Trigonometric
    {"sin", "Sin"}, {"cos", "Cos"}, {"tan", "Tan"},
    {"asin", "Asin"}, {"acos", "Acos"}, {"atan", "Atan"},
    // Hyperbolic
    {"sinh", "Sinh"}, {"cosh", "Cosh"}, {"tanh", "Tanh"},
    // Exponential and logarithmic
    {"exp", "Exp"}, {"log", "Log"}, {"sqrt", "Sqrt"},
    // Rounding and sign
    {"abs", "Abs"}, {"ceil", "Ceil"}, {"floor", "Floor"}, {"sign", "Sign"}
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
static ForLoopRange parseForLoopRange(basemodelica::BaseModelicaParser::ForEquationContext* forEqCtx) {
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

// Set up standard ONNX Loop body inputs (iter, cond) and condition passthrough
// Returns the condition output name for reference
static std::string setupLoopBodyIO(onnx::GraphProto* bodyGraph, const std::string& loopNodeName) {
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

// Scan equations for der() calls and return the set of derivative variable names
static std::set<std::string> scanForDerivatives(
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

// Context for generating variable bound outputs (start, min, max)
struct BoundOutputContext {
    onnx::GraphProto* graph;
    const ConversionContext& convCtx;
    const std::map<size_t, int>& varIndexToInputIndex;
};

// Add a scan output to a loop body and optionally to the main graph.
// Returns the loop output name.
static std::string addLoopScanOutput(
    onnx::NodeProto* loopNode,
    onnx::GraphProto* bodyGraph,
    onnx::GraphProto* mainGraph,
    const std::string& loopNodeName,
    const std::string& residualTensor,
    size_t scanIndex,
    const std::string& prefix,
    size_t equationIndex,
    bool isNested,
    int tripCount = -1) {

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

// Add all required loop passthroughs for a top-level for-loop.
// This includes all non-derivative, non-fixed variables and pre-discovered derivatives.
static void addTopLevelLoopPassthroughs(
    onnx::NodeProto* loopNode,
    onnx::GraphProto* bodyGraph,
    const std::string& loopNodeName,
    const std::string& outputSuffix,
    const ModelInfo& info,
    const std::set<std::string>& requiredDerivatives,
    std::map<std::string, std::vector<std::string>>& derivativeInputs) {

    // Add all non-derivative, non-fixed variables as loop passthroughs
    // Note: ONNXRuntime requires symmetric I/O (body outputs must match loop inputs)
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
        // Find the base variable to get dimensions
        size_t start = derName.find("'") + 1;
        size_t end = derName.rfind("'");
        std::string baseVarName = derName.substr(start, end - start);
        const Variable* baseVar = info.findVariable(baseVarName);
        std::vector<std::string> dims = baseVar ? baseVar->dimensions : std::vector<std::string>{};

        addLoopPassthrough(loopNode, bodyGraph, loopNodeName,
                           derName, derName,
                           onnx::TensorProto::DOUBLE, dims,
                           outputSuffix);

        // Also add to derivativeInputs map for later reference
        if (baseVar) {
            derivativeInputs[derName] = baseVar->dimensions;
        }
    }
}

// Create an equation residual output (LHS - RHS for numeric, LHS == RHS for boolean).
// Adds the computation node and output to the graph.
static void createEquationResidual(
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

// Generate an ONNX output for a variable bound (start, min, or max value).
// Creates the output with appropriate metadata linking it to the variable.
static void generateBoundOutput(
    const BoundOutputContext& ctx,
    size_t varIndex,
    const Variable& var,
    antlr4::ParserRuleContext* exprContext,
    const std::string& boundType,
    const std::string& description) {

    auto it = ctx.varIndexToInputIndex.find(varIndex);
    if (it == ctx.varIndexToInputIndex.end()) {
        std::cerr << "Warning: Could not find input index for variable " << var.name << std::endl;
        return;
    }
    int inputIdx = it->second;

    std::string exprTensor;
    try {
        exprTensor = ONNXGenerator::convertExpression(exprContext, ctx.convCtx);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to convert " << boundType << " expression for " << var.name;
        if (!var.sourceFile.empty()) {
            std::cerr << " (" << var.sourceFile << ":" << var.sourceLine << ")";
        }
        std::cerr << ": " << e.what() << std::endl;
        return;
    }

    std::string outputName = boundType + "[" + std::to_string(inputIdx) + "]";
    auto* output = ctx.graph->add_output();
    output->set_name(outputName);
    auto* output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(onnx::TensorProto::DOUBLE);
    output_type->mutable_shape()->add_dim()->set_dim_value(1);

    output->set_doc_string(description + " for " + var.name);
    addSourceLocationMetadata(output, var.sourceFile, var.sourceLine);

    auto* meta_var = output->add_metadata_props();
    meta_var->set_key("variable_name");
    meta_var->set_value(var.name);

    auto* meta_vr = output->add_metadata_props();
    meta_vr->set_key("value_reference");
    meta_vr->set_value(std::to_string(var.valueReference));

    auto* meta_idx = output->add_metadata_props();
    meta_idx->set_key("input_index");
    meta_idx->set_value(std::to_string(inputIdx));

    renameTensorToOutput(ctx.graph, exprTensor, outputName,
                         boundType + "_identity_" + std::to_string(inputIdx));
}

// -----------------------------------------------------------------------------

std::string ONNXGenerator::generate(const ModelInfo& info, const std::string& outputDir) {
    // Layered standard directory structure
    std::string lsName = "org.lacemodelica.ls-onnx-serialization";
    std::string lsDir = outputDir + "/extra/" + lsName;

    // Create directories
    std::filesystem::create_directories(lsDir);

    // Generate ONNX model file
    std::string modelPath = lsDir + "/model.onnx";
    generateONNXModel(info, modelPath);

    // Generate layered standard manifest
    std::string manifestPath = lsDir + "/fmi-ls-manifest.xml";
    generateManifest(manifestPath);

    std::cout << "Generated ONNX layered standard in " << lsDir << "/" << std::endl;

    return lsDir;
}

void ONNXGenerator::generateONNXModel(const ModelInfo& info, const std::string& filepath) {
    onnx::ModelProto model;

    // Set model metadata
    model.set_ir_version(8);  // ONNX IR version 8
    model.set_producer_name("lacemodelica");
    model.set_producer_version("0.1.0");
    model.set_model_version(1);
    model.set_doc_string("Symbolic representation of " + info.modelName);

    // Add opset import (opset version 18) for default domain
    auto* opset = model.add_opset_import();
    // Don't set domain - it defaults to empty string
    opset->set_version(18);

    // Add opset import for lacemodelica domain (for custom functions)
    auto* lacemodelica_opset = model.add_opset_import();
    lacemodelica_opset->set_domain("lacemodelica");
    lacemodelica_opset->set_version(1);

    // Create the graph
    auto* graph = model.mutable_graph();
    graph->set_name(info.modelName);

    // Check if 'time' is used in any equation (Modelica's built-in simulation time variable)
    bool usesTime = false;
    for (const auto& eq : info.equations) {
        std::string eqText;
        if (eq.lhsContext) eqText += eq.lhsContext->getText();
        if (eq.rhsContext) eqText += eq.rhsContext->getText();
        if (eq.ifEquationContext) eqText += eq.ifEquationContext->getText();
        if (eq.forEquationContext) eqText += eq.forEquationContext->getText();
        if (eqText.find("time") != std::string::npos) {
            usesTime = true;
            break;
        }
    }

    // Add 'time' as input only if the model uses it
    if (usesTime) {
        auto* timeInput = graph->add_input();
        timeInput->set_name("time");
        auto* timeType = timeInput->mutable_type()->mutable_tensor_type();
        timeType->set_elem_type(onnx::TensorProto::DOUBLE);
        timeType->mutable_shape();  // Scalar - empty shape
    }

    // Create ONNX inputs for each variable and parameter (skip derivatives and constants)
    // Constants with fixed variability become initializers instead
    // Track mapping from variable index to input index for start[] outputs
    std::map<size_t, int> varIndexToInputIndex;
    int inputIndex = 0;

    for (size_t i = 0; i < info.variables.size(); i++) {
        const auto& var = info.variables[i];

        // Skip derivative variables - der() will be an operator
        if (var.isDerivative) {
            continue;
        }

        // Constants with fixed variability become initializers, not inputs
        if (var.variability == "fixed" && !var.startValue.empty()) {
            auto* initializer = graph->add_initializer();
            initializer->set_name(var.name);
            initializer->set_data_type(onnx::TensorProto::DOUBLE);

            // Add dimensions (empty for scalars)
            for (const auto& dim : var.dimensions) {
                try {
                    initializer->add_dims(std::stoi(dim));
                } catch (...) {
                    // If dimension is symbolic, can't use as initializer
                    std::cerr << "Warning: Cannot create initializer for " << var.name
                              << " with symbolic dimension " << dim << std::endl;
                }
            }

            // Parse and store the constant value
            try {
                double value = std::stod(var.startValue);
                initializer->add_double_data(value);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse constant value for " << var.name
                          << ": " << var.startValue << std::endl;
                // Fallback to 0.0
                initializer->add_double_data(0.0);
            }

            continue;  // Don't add as input
        }

        varIndexToInputIndex[i] = inputIndex++;

        auto* input = graph->add_input();
        input->set_name(var.name);
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        // Set element type based on variable type
        if (var.type == "Boolean") {
            input_type->set_elem_type(onnx::TensorProto::BOOL);
        } else {
            input_type->set_elem_type(onnx::TensorProto::DOUBLE);
        }
        auto* input_shape = input_type->mutable_shape();

        // Handle array dimensions (scalars have empty shape [])
        addShapeDimensions(input_shape, var.dimensions);

        addSourceLocationMetadata(input, var.sourceFile, var.sourceLine);
    }

    // Create ONNX FunctionProto for each function with algorithm
    for (const auto& func : info.functions) {
        if (!func.algorithmStatements.empty()) {
            try {
                createFunctionProto(func, info, &model);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to create FunctionProto for " << func.name
                          << ": " << e.what() << std::endl;
            }
        }
    }

    // Create ONNX outputs for equations
    int nodeCounter = 0;
    std::map<std::string, std::vector<std::string>> derivativeInputs;

    // Create conversion context for expression conversion
    ConversionContext ctx(info, graph, nodeCounter, nullptr, &derivativeInputs);

    // Generate outputs for regular equations
    generateEquationOutputs(info.equations, "eq", info, graph, nodeCounter, derivativeInputs);

    // Generate outputs for initial equations
    generateEquationOutputs(info.initialEquations, "init_eq", info, graph, nodeCounter, derivativeInputs);

    // Generate outputs for calculated initial values, min, and max bounds
    BoundOutputContext boundCtx{graph, ctx, varIndexToInputIndex};
    for (size_t i = 0; i < info.variables.size(); i++) {
        const auto& var = info.variables[i];

        // Start values for calculated initial conditions
        if (var.initial == "calculated" && var.bindingContext != nullptr) {
            generateBoundOutput(boundCtx, i, var, var.bindingContext, "start", "Initial value");
        }

        // Non-const min bounds
        if (var.minContext != nullptr && !isConstValue(var.minValue)) {
            generateBoundOutput(boundCtx, i, var, var.minContext, "min", "Minimum value");
        }

        // Non-const max bounds
        if (var.maxContext != nullptr && !isConstValue(var.maxValue)) {
            generateBoundOutput(boundCtx, i, var, var.maxContext, "max", "Maximum value");
        }
    }

    // Add derivative inputs at the end
    for (const auto& [derName, dimensions] : derivativeInputs) {
        auto* input = graph->add_input();
        input->set_name(derName);
        auto* input_type = input->mutable_type()->mutable_tensor_type();
        input_type->set_elem_type(onnx::TensorProto::DOUBLE);
        auto* input_shape = input_type->mutable_shape();

        // Handle array dimensions
        for (const auto& dim : dimensions) {
            auto* shape_dim = input_shape->add_dim();
            // Try to parse as integer, otherwise leave symbolic
            try {
                shape_dim->set_dim_value(std::stoi(dim));
            } catch (...) {
                shape_dim->set_dim_param(dim);
            }
        }
        // Scalars: empty dimensions results in empty shape []
    }

    // Infer shapes for all tensors in the graph
    try {
        onnx::shape_inference::InferShapes(model);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Shape inference failed: " << e.what() << std::endl;
    }

    // Validate ONNX model before serialization
    try {
        onnx::checker::check_model(model, false, false, false);
    } catch (const onnx::checker::ValidationError& e) {
        std::cerr << "Warning: ONNX validation failed: " << e.what() << std::endl;
    }

    // Serialize to file
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to create ONNX file: " + filepath);
    }

    if (!model.SerializeToOstream(&ofs)) {
        throw std::runtime_error("Failed to serialize ONNX model to: " + filepath);
    }

    ofs.close();
}

void ONNXGenerator::generateManifest(const std::string& filepath) {
    using namespace tinyxml2;

    XMLDocument doc;

    // Create root element with FMI layered standard attributes
    XMLElement* root = doc.NewElement("fmiLayeredStandardManifest");
    root->SetAttribute("xmlns:fmi-ls", "http://fmi-standard.org/fmi-ls-manifest");
    root->SetAttribute("fmi-ls:fmi-ls-name", "org.lacemodelica.ls-onnx-serialization");
    root->SetAttribute("fmi-ls:fmi-ls-version", "1.0.0");
    root->SetAttribute("fmi-ls:fmi-ls-description",
        "Layered standard for ONNX-serialized symbolic expressions in FMU");

    doc.InsertFirstChild(root);

    // Save to file
    if (doc.SaveFile(filepath.c_str()) != XML_SUCCESS) {
        throw std::runtime_error("Failed to write manifest file: " + filepath);
    }
}

void ONNXGenerator::generateEquationOutputs(
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

        // Check if this is a for-equation
        if (eq.isForEquation()) {
            size_t numOutputs = generateForEquationLoop(eq, prefix, equationOutputIndex, info, graph, nodeCounter, derivativeInputs);
            equationOutputIndex += numOutputs;
            continue;
        }

        // Check if this is an if-equation
        if (eq.isIfEquation()) {
            size_t numOutputs = generateIfEquation(eq, prefix, equationOutputIndex, info, graph, nodeCounter, derivativeInputs);
            equationOutputIndex += numOutputs;
            continue;
        }

        // Check if LHS is a tuple (outputExpressionList) for multi-output functions
        bool isMultiOutput = false;
        std::vector<std::string> outputVarNames;

        // Use ParseTreeNavigator to find OutputExpressionList (replaces pyramid of dynamic_casts)
        if (auto outExprList = ParseTreeNavigator::findOutputExpressionList(eq.lhsContext)) {
            // This is a tuple output!
            isMultiOutput = true;
            for (auto outExpr : outExprList->expression()) {
                if (outExpr) {
                    outputVarNames.push_back(stripQuotes(outExpr->getText()));
                }
            }
        }

        if (isMultiOutput) {
            // Handle multi-output function call
            std::vector<std::string> outputTensors = convertMultiOutputFunctionCall(eq.rhsContext, ctx, outputVarNames.size());

            if (outputTensors.size() != outputVarNames.size()) {
                throw std::runtime_error("Multi-output function returned " + std::to_string(outputTensors.size()) +
                                       " outputs, expected " + std::to_string(outputVarNames.size()));
            }

            // Create residual equations for each output
            for (size_t j = 0; j < outputVarNames.size(); j++) {
                std::string eqOutputName = prefix + "[" + std::to_string(i + j) + "]";
                createEquationResidual(graph, outputVarNames[j], outputTensors[j], eqOutputName,
                                       "eq_residual_" + std::to_string(i + j),
                                       false, "", eq.sourceFile, eq.sourceLine);
            }

            // Skip to next iteration (we've handled multiple equations at once)
            i += outputVarNames.size() - 1;
            continue;
        }

        std::string lhsTensor, rhsTensor;
        try {
            // Convert LHS expression to ONNX graph
            try {
                lhsTensor = convertExpression(eq.lhsContext, ctx);
            } catch (const std::exception& e) {
                std::cerr << "Error converting LHS of " << prefix << " equation " << i;
                if (!eq.sourceFile.empty()) {
                    std::cerr << " (" << eq.sourceFile << ":" << eq.sourceLine << ")";
                }
                std::cerr << ": " << e.what() << std::endl;
                std::cerr << "LHS text: " << eq.lhsContext->getText() << std::endl;
                throw;
            }

            // Convert RHS expression to ONNX graph
            try {
                rhsTensor = convertExpression(eq.rhsContext, ctx);
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
            // Skip equations that fail to convert
            std::cerr << "Warning: Skipping equation " << prefix << "[" << equationOutputIndex << "] due to conversion error" << std::endl;
            equationOutputIndex++;
            continue;
        }

        // Create equation residual: LHS - RHS (or LHS == RHS for booleans)
        std::string eqOutputName = prefix + "[" + std::to_string(equationOutputIndex) + "]";

        // Check if LHS is a boolean variable
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

// Helper function to add an Identity node if tensor name differs from desired name
// Returns the final tensor name (either original or the new name after Identity)
static std::string ensureTensorName(
    onnx::FunctionProto* functionProto,
    const std::string& tensorName,
    const std::string& desiredName) {

    if (tensorName == desiredName) {
        return tensorName;  // No Identity needed
    }

    // Add Identity node to rename the tensor
    auto* identityNode = functionProto->add_node();
    identityNode->set_op_type("Identity");
    identityNode->set_name("output_" + desiredName);
    identityNode->add_input(tensorName);
    identityNode->add_output(desiredName);

    return desiredName;
}

size_t ONNXGenerator::generateIfEquation(
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

    // Get all expressions (conditions) and equation lists from branches
    auto expressions = ifEqCtx->expression();  // [if_cond, elseif_cond1, elseif_cond2, ...]

    // Each branch has a list of equations
    // Grammar: 'if' expression 'then' (equation ';')* ('elseif' expression 'then' (equation ';')*)* ('else' (equation ';')*)?
    // We need to get equations from each branch

    // Collect all equations from all branches
    // The structure is: if branch equations, then elseif branches, then else branch
    auto allEquations = ifEqCtx->equation();  // All equations across all branches

    // For now, we assume each branch has exactly one equation that assigns to the same variable
    // The if-equation becomes: var - If(cond1, val1, If(cond2, val2, ..., else_val)) = 0

    if (allEquations.empty()) {
        std::cerr << "Warning: Empty if-equation" << std::endl;
        return 0;
    }

    // Get LHS variable from first equation (assume all branches assign to same var)
    auto firstEq = allEquations[0];
    auto firstSimpleExpr = firstEq->simpleExpression();
    if (!firstSimpleExpr) {
        throw std::runtime_error("If-equation branch must contain simple equation");
    }
    std::string lhsVarName = stripQuotes(firstSimpleExpr->getText());

    // Build nested If structure for RHS values
    // We need to pair conditions with their corresponding equations
    // Structure: if cond1 then eq1; [elseif cond2 then eq2;]* [else eq_else;]? end if;

    // Build the conditional RHS using nested If nodes
    // For: if c1 then v1 elseif c2 then v2 else v3
    // Build: If(c1, v1, If(c2, v2, v3))

    ConversionContext ifCtx(info, graph, nodeCounter, nullptr, &derivativeInputs);
    std::string rhsTensor = buildIfEquationRhs(expressions, allEquations, 0, ifCtx);

    // Create residual: lhs_var - rhs_if_result = 0
    std::string eqOutputName = prefix + "[" + std::to_string(equationIndex) + "]";

    auto* subNode = graph->add_node();
    subNode->set_op_type("Sub");
    subNode->set_name(prefix + "_if_residual_" + std::to_string(equationIndex));
    subNode->add_input(lhsVarName);
    subNode->add_input(rhsTensor);
    subNode->add_output(eqOutputName);

    // Add output to graph
    auto* output = graph->add_output();
    output->set_name(eqOutputName);
    auto* outputType = output->mutable_type()->mutable_tensor_type();
    outputType->set_elem_type(onnx::TensorProto::DOUBLE);
    outputType->mutable_shape();

    addSourceLocationMetadata(output, eq.sourceFile, eq.sourceLine);

    return 1;  // One equation output generated
}

// Helper to build nested If structure for if-equation RHS
std::string ONNXGenerator::buildIfEquationRhs(
    const std::vector<basemodelica::BaseModelicaParser::ExpressionContext*>& conditions,
    const std::vector<basemodelica::BaseModelicaParser::EquationContext*>& equations,
    size_t branchIndex,
    const ConversionContext& ctx) {

    // Base case: else branch (no more conditions)
    if (branchIndex >= conditions.size()) {
        if (branchIndex < equations.size()) {
            auto* eqCtx = equations[branchIndex];
            auto* rhsExpr = eqCtx->expression();
            if (rhsExpr) {
                return convertExpression(rhsExpr, ctx);
            }
        }
        return createDoubleConstant(ctx.graph, 0.0, ctx.nodeCounter);
    }

    std::string condTensor = convertExpression(conditions[branchIndex], ctx);

    // Create then branch subgraph
    onnx::GraphProto thenBranch;
    thenBranch.set_name("then_branch_" + std::to_string(branchIndex));

    std::string thenResult;
    if (branchIndex < equations.size()) {
        auto* eqCtx = equations[branchIndex];
        auto* rhsExpr = eqCtx->expression();
        if (rhsExpr) {
            thenResult = convertExpression(rhsExpr, ctx.withGraph(&thenBranch));
        }
    }
    if (thenResult.empty()) {
        thenResult = createDoubleConstant(&thenBranch, 0.0, ctx.nodeCounter);
    }
    addScalarDoubleOutput(&thenBranch, thenResult);

    // Create else branch subgraph recursively
    onnx::GraphProto elseBranch;
    elseBranch.set_name("else_branch_" + std::to_string(branchIndex));
    std::string elseResult = buildIfEquationRhs(conditions, equations, branchIndex + 1, ctx.withGraph(&elseBranch));
    addScalarDoubleOutput(&elseBranch, elseResult);

    return createIfNode(ctx.graph, condTensor, thenBranch, elseBranch, ctx.nodeCounter, "", "If_eq");
}

size_t ONNXGenerator::generateForEquationLoop(
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

    // Parse loop variable and range using helper
    ForLoopRange range = parseForLoopRange(forEqCtx);
    auto loopEquations = forEqCtx->equation();

    // Create Loop node
    std::string loopNodeName = "for_loop_" + std::to_string(nodeCounter++);
    if (outLoopNodeName) {
        *outLoopNodeName = loopNodeName;
    }

    std::string tripCountTensor = createInt64Constant(graph, range.tripCount(), nodeCounter, "trip_count_" + loopNodeName);
    std::string condTensor = createBoolConstant(graph, true, nodeCounter);

    auto* loopNode = graph->add_node();
    loopNode->set_op_type("Loop");
    loopNode->set_name(loopNodeName);
    loopNode->add_input(tripCountTensor);
    loopNode->add_input(condTensor);

    // Create loop body subgraph with standard I/O
    auto* bodyAttr = loopNode->add_attribute();
    bodyAttr->set_name("body");
    bodyAttr->set_type(onnx::AttributeProto::GRAPH);
    auto* bodyGraph = bodyAttr->mutable_g();
    bodyGraph->set_name("loop_body_" + std::to_string(nodeCounter++));

    setupLoopBodyIO(bodyGraph, loopNodeName);

    // For nested loops, add parent loop variables as inputs
    // This allows inner loop to reference outer loop variables
    if (isNested && parentLoopVarMap) {
        std::string outputSuffix = "_final_" + std::to_string(equationIndex);
        for (const auto& [varName, tensorName] : *parentLoopVarMap) {
            addLoopPassthrough(loopNode, bodyGraph, loopNodeName,
                               tensorName, "parent_" + varName,
                               onnx::TensorProto::INT64, {},
                               outputSuffix);
        }
    }

    // Create 1-based loop variable for Modelica semantics (Modelica uses 1-based, ONNX uses 0-based)
    std::string constOneTensor = createInt64Constant(bodyGraph, 1, nodeCounter, loopNodeName + "_const_one");
    std::string loopVarTensor = createBinaryOp(bodyGraph, "Add", "iter", constOneTensor, nodeCounter, loopNodeName + "_" + range.loopVar);

    // Pre-scan equations to discover which derivatives are needed
    std::set<std::string> requiredDerivatives = scanForDerivatives(loopEquations);

    // For top-level loops, add loop-carried dependencies for all variables
    // For nested loops, skip this - variables are already in scope from parent loop
    if (!isNested) {
        std::string outputSuffix = "_final_" + std::to_string(equationIndex);
        addTopLevelLoopPassthroughs(loopNode, bodyGraph, loopNodeName, outputSuffix,
                                    info, requiredDerivatives, derivativeInputs);
    }

    // Process each equation in the loop body
    int bodyNodeCounter = 0;
    size_t scanOutputCount = 0;  // Track how many scan outputs we've created

    for (size_t eqIdx = 0; eqIdx < loopEquations.size(); eqIdx++) {
        auto* innerEq = loopEquations[eqIdx];

        // Check if this is a nested for-equation
        if (innerEq->forEquation()) {
            // Create an Equation wrapper for the nested for-loop
            Equation nestedEq;
            nestedEq.forEquationContext = innerEq->forEquation();
            nestedEq.sourceFile = eq.sourceFile;
            nestedEq.sourceLine = innerEq->getStart()->getLine();

            // Create combined loop variable map for nested loop
            // This includes the current loop's variable for the inner loop to access
            std::map<std::string, std::string> combinedLoopVarMap;
            if (parentLoopVarMap) {
                // Include grandparent variables
                combinedLoopVarMap = *parentLoopVarMap;
            }

            // Add current loop variable (maps to "iter" in current body)
            // But we can't pass "iter" directly as it's a reserved name in ONNX Loop bodies
            // Create an Identity node to copy iter to a uniquely named tensor
            std::string currentLoopIterCopy = "loop_iter_" + range.loopVar + "_" + std::to_string(bodyNodeCounter++);
            auto* iterCopyNode = bodyGraph->add_node();
            iterCopyNode->set_op_type("Identity");
            iterCopyNode->set_name("copy_iter_for_nested_" + range.loopVar);
            iterCopyNode->add_input(loopVarTensor);
            iterCopyNode->add_output(currentLoopIterCopy);

            combinedLoopVarMap[range.loopVar] = currentLoopIterCopy;

            // Recursively generate nested Loop node inside current loop body
            std::string nestedLoopNodeName;
            size_t nestedOutputCount = generateForEquationLoop(
                nestedEq,
                prefix,
                equationIndex + scanOutputCount,
                info,
                bodyGraph,  // Create nested loop in current body graph!
                nodeCounter,  // Use global counter to ensure unique loop names across all nesting levels
                derivativeInputs,
                true,  // isNested = true
                &combinedLoopVarMap,  // Pass parent context
                &nestedLoopNodeName  // Get the nested loop's node name
            );

            // The nested Loop node's outputs are available in the current body graph
            // Connect the scan outputs from the nested loop to the outer loop's scan outputs
            // The nested loop's outputs are at the end (after all the loop-carried dependencies)
            // Find the nested loop node first to determine how many outputs it has
            onnx::NodeProto* nestedLoopNode = nullptr;
            for (int ni = bodyGraph->node_size() - 1; ni >= 0; ni--) {
                if (bodyGraph->node(ni).name() == nestedLoopNodeName) {
                    nestedLoopNode = bodyGraph->mutable_node(ni);
                    break;
                }
            }

            // The scan outputs start after the carried dependencies
            // total_outputs = carried_deps + scan_outputs
            // So: carried_deps = total_outputs - scan_outputs
            size_t nestedCarriedCount = 0;
            if (nestedLoopNode) {
                nestedCarriedCount = nestedLoopNode->output_size() - nestedOutputCount;
            }

            // Now connect each nested scan output
            // The nested loop's outputs are: [carried deps...] [scan outputs...]
            // We need to find the scan outputs which start after the carried dependencies
            // Note: for nested loops themselves, the scan outputs may be named "collected_nested_X"
            // rather than "eq[X]_loopname", so we must reference by position in the loop node's outputs

            for (size_t i = 0; i < nestedOutputCount; i++) {
                std::string collectedOutputName = "collected_nested_" + std::to_string(scanOutputCount + i);

                // The nested loop node's scan outputs start after carried dependencies
                // Reference the actual output name from the nested loop node
                std::string nestedOutputName;
                if (nestedLoopNode) {
                    size_t outputIndex = nestedCarriedCount + i;
                    if (outputIndex < nestedLoopNode->output_size()) {
                        nestedOutputName = nestedLoopNode->output(outputIndex);
                    }
                }

                // Add as scan output of outer loop body
                auto* scanOutput = bodyGraph->add_output();
                std::string scanOutName = loopNodeName + "_scan_" + std::to_string(scanOutputCount + i);
                scanOutput->set_name(scanOutName);
                auto* scanType = scanOutput->mutable_type()->mutable_tensor_type();
                scanType->set_elem_type(onnx::TensorProto::DOUBLE);
                // The inner loop outputs an array, so this scan output collects arrays
                // The shape is [inner_trip_count], but we don't know the exact count at this point
                // Specify it as a 1D array with unknown size to help shape inference
                auto* scanShape = scanType->mutable_shape();
                auto* dim = scanShape->add_dim();
                // Don't set dim_value or dim_param - leave it empty for unknown/dynamic size

                // Identity to pass through the nested loop's scan output
                auto* scanIdentity = bodyGraph->add_node();
                scanIdentity->set_op_type("Identity");
                scanIdentity->set_name(loopNodeName + "_scan_identity_nested_" + std::to_string(scanOutputCount + i));
                scanIdentity->add_input(nestedOutputName);  // Input from nested loop's actual output
                scanIdentity->add_output(scanOutName);

                // Add this scan output to the outer loop node with collected name
                loopNode->add_output(collectedOutputName);

                // Only add to main graph outputs if we're at top level (not nested ourselves)
                // Map the collected output to the final equation name
                if (!isNested) {
                    // At top level, use plain eq[X] name without loop suffix
                    std::string topLevelEqName = prefix + "[" + std::to_string(equationIndex + scanOutputCount + i) + "]";

                    // Add Identity node in main graph to rename collected output to eq[X]
                    auto* renameNode = graph->add_node();
                    renameNode->set_op_type("Identity");
                    renameNode->set_name("rename_collected_" + std::to_string(scanOutputCount + i));
                    renameNode->add_input(collectedOutputName);
                    renameNode->add_output(topLevelEqName);

                    auto* graphOutput = graph->add_output();
                    graphOutput->set_name(topLevelEqName);
                    auto* graphOutputType = graphOutput->mutable_type()->mutable_tensor_type();
                    graphOutputType->set_elem_type(onnx::TensorProto::DOUBLE);
                    // Multi-dimensional shape from nested loops [outer_dim, inner_dim]
                    // Add two dimensions with unknown sizes for proper 2D shape declaration
                    auto* graphShape = graphOutputType->mutable_shape();
                    graphShape->add_dim();  // Outer loop dimension (unknown size)
                    graphShape->add_dim();  // Inner loop dimension (unknown size)
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

        // Create variable map for loop variable substitution
        // Start with parent loop variables if we're nested
        std::map<std::string, std::string> loopVarMap;
        if (parentLoopVarMap) {
            // Include parent loop variables with their parent_ prefixed names
            for (const auto& [varName, tensorName] : *parentLoopVarMap) {
                loopVarMap[varName] = "parent_" + varName;
            }
        }
        // Add current loop variable
        loopVarMap[range.loopVar] = loopVarTensor;

        // Convert LHS and RHS with loop variable substitution
        std::string lhsTensor, rhsTensor;
        try {
            ConversionContext bodyCtx(info, bodyGraph, bodyNodeCounter, &loopVarMap, &derivativeInputs, loopNodeName);
            lhsTensor = convertExpression(simpleExpr, bodyCtx);
            rhsTensor = convertExpression(fullExpr, bodyCtx);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to convert equation " << eqIdx << " in for-loop: " << e.what() << std::endl;
            continue;
        }

        // Compute residual: LHS - RHS
        std::string residualTensor = loopNodeName + "_residual_" + std::to_string(scanOutputCount);
        auto* subNode = bodyGraph->add_node();
        subNode->set_op_type("Sub");
        subNode->set_name(residualTensor);
        subNode->add_input(lhsTensor);
        subNode->add_input(rhsTensor);
        subNode->add_output(residualTensor);

        // Add scan output to loop body and main graph
        addLoopScanOutput(loopNode, bodyGraph, graph, loopNodeName, residualTensor,
                          scanOutputCount, prefix, equationIndex, isNested, range.tripCount());

        scanOutputCount++;
    }

    return scanOutputCount;
}

void ONNXGenerator::createFunctionProto(
    const Function& func,
    const ModelInfo& info,
    onnx::ModelProto* model) {

    // Create a new FunctionProto
    auto* functionProto = model->add_functions();
    functionProto->set_name(func.name);
    functionProto->set_domain("lacemodelica");  // Custom domain for user functions

    // Add opset import for standard ONNX operators used in the function
    auto* func_opset = functionProto->add_opset_import();
    func_opset->set_version(18);  // Default domain (empty string)

    // Add function inputs
    for (const auto& input : func.inputs) {
        functionProto->add_input(input.name);
    }

    // Note: We'll add outputs AFTER processing statements,
    // so we can use the actual tensor names produced

    // Map from variable name to current ONNX tensor name
    std::map<std::string, std::string> variableToTensor;

    // Initialize map with function inputs (they map to themselves in the function scope)
    for (const auto& input : func.inputs) {
        variableToTensor[input.name] = input.name;
    }

    int nodeCounter = 0;

    // Process each algorithm statement in order
    for (size_t stmtIndex = 0; stmtIndex < func.algorithmStatements.size(); stmtIndex++) {
        const auto& stmt = func.algorithmStatements[stmtIndex];

        // Extract LHS variable name from componentReference
        std::string lhsVarName = stripQuotes(stmt.lhsContext->getText());

        try {
            // Remember the current node count to identify newly created nodes
            int startNodeCount = functionProto->node_size();

            // Create a temporary graph to build the expression nodes
            onnx::GraphProto tempGraph;
            std::map<std::string, std::vector<std::string>> localDerivativeInputs;
            ConversionContext funcCtx(info, &tempGraph, nodeCounter, &variableToTensor, &localDerivativeInputs);
            std::string rhsTensor = convertExpression(stmt.rhsContext, funcCtx);

            // Move nodes from tempGraph to functionProto
            for (int i = 0; i < tempGraph.node_size(); i++) {
                auto* node = functionProto->add_node();
                node->CopyFrom(tempGraph.node(i));

                addSourceLocationMetadata(node, stmt.sourceFile, stmt.sourceLine);

                auto* meta_index = node->add_metadata_props();
                meta_index->set_key("statement_index");
                meta_index->set_value(std::to_string(stmtIndex));

                auto* meta_lhs = node->add_metadata_props();
                meta_lhs->set_key("lhs_variable");
                meta_lhs->set_value(lhsVarName);
            }

            // Note: FunctionProto doesn't support initializers (constants)
            // Constants should be represented as Constant nodes instead

            // Store result in map
            variableToTensor[lhsVarName] = rhsTensor;

        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to convert statement for " << lhsVarName;
            if (!stmt.sourceFile.empty()) {
                std::cerr << " (" << stmt.sourceFile << ":" << stmt.sourceLine << ")";
            }
            std::cerr << ": " << e.what() << std::endl;
            throw;
        }
    }

    // Add function outputs with meaningful names
    for (const auto& output : func.outputs) {
        auto it = variableToTensor.find(output.name);
        if (it == variableToTensor.end()) {
            throw std::runtime_error("Output variable " + output.name + " not computed in algorithm");
        }

        std::string internalTensor = it->second;
        std::string outputName = output.name;

        // Ensure the tensor has the desired output name (adds Identity if needed)
        std::string finalName = ensureTensorName(functionProto, internalTensor, outputName);

        // Add the output variable name as the function output
        functionProto->add_output(finalName);
    }
}

// Helper function to collect all function arguments from recursive structure
static std::vector<basemodelica::BaseModelicaParser::ExpressionContext*>
collectFunctionArguments(basemodelica::BaseModelicaParser::FunctionArgumentsContext* funcArgs) {
    std::vector<basemodelica::BaseModelicaParser::ExpressionContext*> arguments;

    if (!funcArgs) {
        return arguments;
    }

    // Get the first argument
    auto firstExpr = funcArgs->expression();
    if (firstExpr) {
        arguments.push_back(firstExpr);
    }

    // Recursively collect remaining arguments
    auto nonFirst = funcArgs->functionArgumentsNonFirst();
    while (nonFirst) {
        auto funcArg = nonFirst->functionArgument();
        if (funcArg) {
            auto expr = funcArg->expression();
            if (expr) {
                arguments.push_back(expr);
            }
        }
        nonFirst = nonFirst->functionArgumentsNonFirst();
    }

    return arguments;
}

// Expression conversion functions

std::string ONNXGenerator::convertExpression(antlr4::ParserRuleContext* expr, const ConversionContext& ctx) {
    if (!expr) {
        throw std::runtime_error("Null expression context");
    }

    // Try to cast to specific expression types
    if (auto* exprCtx = dynamic_cast<basemodelica::BaseModelicaParser::ExpressionContext*>(expr)) {
        auto* exprNoDecoration = exprCtx->expressionNoDecoration();
        if (exprNoDecoration) {
            if (auto* ifExpr = exprNoDecoration->ifExpression()) {
                return convertIfExpression(ifExpr, ctx);
            }
            if (auto* simpleExpr = exprNoDecoration->simpleExpression()) {
                return convertSimpleExpression(simpleExpr, ctx);
            }
        }
    } else if (auto* exprNoDecoration = dynamic_cast<basemodelica::BaseModelicaParser::ExpressionNoDecorationContext*>(expr)) {
        if (auto* ifExpr = exprNoDecoration->ifExpression()) {
            return convertIfExpression(ifExpr, ctx);
        }
        if (auto* simpleExpr = exprNoDecoration->simpleExpression()) {
            return convertSimpleExpression(simpleExpr, ctx);
        }
    } else if (auto* simpleExpr = dynamic_cast<basemodelica::BaseModelicaParser::SimpleExpressionContext*>(expr)) {
        return convertSimpleExpression(simpleExpr, ctx);
    }

    throw std::runtime_error("Unsupported expression type: " + expr->getText());
}

std::string ONNXGenerator::convertIfExpression(
    basemodelica::BaseModelicaParser::IfExpressionContext* expr,
    const ConversionContext& ctx) {

    auto expressions = expr->expressionNoDecoration();

    if (expressions.size() < 3) {
        throw std::runtime_error("Invalid if expression structure");
    }
    if (expressions.size() % 2 != 1) {
        throw std::runtime_error("Invalid if expression structure: expected odd number of expressions");
    }

    std::string condTensor = convertExpression(expressions[0], ctx);

    // Create then branch subgraph
    onnx::GraphProto thenBranch;
    thenBranch.set_name("then_branch");
    std::string thenResult = convertExpression(expressions[1], ctx.withGraph(&thenBranch));
    addScalarDoubleOutput(&thenBranch, thenResult);

    // Create else branch subgraph
    onnx::GraphProto elseBranch;
    elseBranch.set_name("else_branch");
    auto elseCtx = ctx.withGraph(&elseBranch);
    std::string elseResult = (expressions.size() == 3)
        ? convertExpression(expressions[2], elseCtx)
        : convertNestedIfElse(expressions, 2, elseCtx);
    addScalarDoubleOutput(&elseBranch, elseResult);

    return createIfNode(ctx.graph, condTensor, thenBranch, elseBranch, ctx.nodeCounter, ctx.tensorPrefix);
}

std::string ONNXGenerator::convertNestedIfElse(
    const std::vector<basemodelica::BaseModelicaParser::ExpressionNoDecorationContext*>& expressions,
    size_t startIdx,
    const ConversionContext& ctx) {

    size_t remaining = expressions.size() - startIdx;

    if (remaining == 1) {
        return convertExpression(expressions[startIdx], ctx);
    }

    std::string condTensor = convertExpression(expressions[startIdx], ctx);

    onnx::GraphProto thenBranch;
    thenBranch.set_name("then_branch");
    std::string thenResult = convertExpression(expressions[startIdx + 1], ctx.withGraph(&thenBranch));
    addScalarDoubleOutput(&thenBranch, thenResult);

    onnx::GraphProto elseBranch;
    elseBranch.set_name("else_branch");
    std::string elseResult = convertNestedIfElse(expressions, startIdx + 2, ctx.withGraph(&elseBranch));
    addScalarDoubleOutput(&elseBranch, elseResult);

    return createIfNode(ctx.graph, condTensor, thenBranch, elseBranch, ctx.nodeCounter, ctx.tensorPrefix, "If_nested");
}

std::string ONNXGenerator::convertSimpleExpression(
    basemodelica::BaseModelicaParser::SimpleExpressionContext* expr,
    const ConversionContext& ctx) {

    auto logicalExprs = expr->logicalExpression();
    if (logicalExprs.empty()) {
        throw std::runtime_error("Empty simple expression");
    }

    auto* logicalExpr = logicalExprs[0];
    auto logicalTerms = logicalExpr->logicalTerm();
    if (logicalTerms.empty()) {
        throw std::runtime_error("Empty logical expression");
    }

    auto* logicalTerm = logicalTerms[0];
    auto logicalFactors = logicalTerm->logicalFactor();
    if (logicalFactors.empty()) {
        throw std::runtime_error("Empty logical term");
    }

    std::string resultTensor;
    for (size_t i = 0; i < logicalFactors.size(); i++) {
        auto* logicalFactor = logicalFactors[i];
        auto* relation = logicalFactor->relation();
        if (!relation) {
            throw std::runtime_error("No relation in logical factor");
        }

        std::string factorTensor = convertRelation(relation, ctx);

        if (logicalFactor->children.size() > 1) {
            factorTensor = createUnaryOp(ctx.graph, "Not", factorTensor, ctx.nodeCounter, ctx.tensorPrefix);
        }

        resultTensor = (i == 0) ? factorTensor
            : createBinaryOp(ctx.graph, "And", resultTensor, factorTensor, ctx.nodeCounter, ctx.tensorPrefix);
    }

    return resultTensor;
}

std::string ONNXGenerator::convertRelation(
    basemodelica::BaseModelicaParser::RelationContext* relation,
    const ConversionContext& ctx) {

    auto arithmeticExprs = relation->arithmeticExpression();
    if (arithmeticExprs.empty()) {
        throw std::runtime_error("No arithmetic expression in relation");
    }

    if (arithmeticExprs.size() > 1) {
        auto relOp = relation->relationalOperator();
        if (!relOp) {
            throw std::runtime_error("Multiple arithmetic expressions but no relational operator");
        }

        std::string leftTensor = convertArithmeticExpression(arithmeticExprs[0], ctx);
        std::string rightTensor = convertArithmeticExpression(arithmeticExprs[1], ctx);

        std::string opText = relOp->getText();

        auto it = kRelationalOpMap.find(opText);
        if (it != kRelationalOpMap.end()) {
            return createBinaryOp(ctx.graph, it->second, leftTensor, rightTensor, ctx.nodeCounter, ctx.tensorPrefix);
        } else if (opText == "<>") {
            std::string equalResult = createBinaryOp(ctx.graph, "Equal", leftTensor, rightTensor, ctx.nodeCounter, ctx.tensorPrefix);
            return createUnaryOp(ctx.graph, "Not", equalResult, ctx.nodeCounter, ctx.tensorPrefix);
        } else {
            throw std::runtime_error("Unsupported relational operator: " + opText);
        }
    }

    return convertArithmeticExpression(arithmeticExprs[0], ctx);
}

std::string ONNXGenerator::convertArithmeticExpression(
    basemodelica::BaseModelicaParser::ArithmeticExpressionContext* expr,
    const ConversionContext& ctx) {

    auto terms = expr->term();
    auto addOps = expr->addOperator();

    if (terms.empty()) {
        throw std::runtime_error("Empty arithmetic expression");
    }

    size_t termIndex = 0;
    size_t opIndex = 0;

    std::string result = convertTerm(terms[termIndex++], ctx);

    // Handle leading unary operator
    if (addOps.size() > terms.size() - 1) {
        std::string opText = addOps[opIndex++]->getText();
        if (opText == "-") {
            result = createUnaryOp(ctx.graph, "Neg", result, ctx.nodeCounter, ctx.tensorPrefix);
        } else if (opText != "+") {
            throw std::runtime_error("Unsupported leading operator: " + opText);
        }
    }

    while (opIndex < addOps.size()) {
        std::string opText = addOps[opIndex++]->getText();
        std::string rightTensor = convertTerm(terms[termIndex++], ctx);

        std::string onnxOp;
        if (opText == "+" || opText == ".+") {
            onnxOp = "Add";
        } else if (opText == "-" || opText == ".-") {
            onnxOp = "Sub";
        } else {
            throw std::runtime_error("Unsupported add operator: " + opText);
        }

        result = createBinaryOp(ctx.graph, onnxOp, result, rightTensor, ctx.nodeCounter, ctx.tensorPrefix);
    }

    return result;
}

static bool isMatrixVariable(const std::string& tensorName, const ModelInfo& info) {
    auto it = info.variableIndex.find(tensorName);
    if (it != info.variableIndex.end()) {
        return info.variables[it->second].dimensions.size() == 2;
    }
    return false;
}

std::string ONNXGenerator::convertTerm(
    basemodelica::BaseModelicaParser::TermContext* expr,
    const ConversionContext& ctx) {

    auto factors = expr->factor();
    auto mulOps = expr->mulOperator();

    if (factors.empty()) {
        throw std::runtime_error("Empty term");
    }

    std::string result = convertFactor(factors[0], ctx);

    for (size_t i = 0; i < mulOps.size(); i++) {
        std::string opText = mulOps[i]->getText();
        std::string rightTensor = convertFactor(factors[i + 1], ctx);

        std::string onnxOp;
        if (opText == "*") {
            bool leftIsMatrix = isMatrixVariable(result, ctx.info);
            bool rightIsMatrix = isMatrixVariable(rightTensor, ctx.info);
            onnxOp = (leftIsMatrix && rightIsMatrix) ? "MatMul" : "Mul";
        } else if (opText == "/" || opText == "./") {
            onnxOp = "Div";
        } else if (opText == ".*") {
            onnxOp = "Mul";
        } else {
            throw std::runtime_error("Unsupported mul operator: " + opText);
        }

        result = createBinaryOp(ctx.graph, onnxOp, result, rightTensor, ctx.nodeCounter, ctx.tensorPrefix);
    }

    return result;
}

std::string ONNXGenerator::convertFactor(
    basemodelica::BaseModelicaParser::FactorContext* expr,
    const ConversionContext& ctx) {

    auto primaries = expr->primary();

    if (primaries.empty()) {
        throw std::runtime_error("Empty factor");
    }

    std::string result = convertPrimary(primaries[0], ctx);

    if (primaries.size() > 1) {
        std::string exponentTensor = convertPrimary(primaries[1], ctx);
        result = createBinaryOp(ctx.graph, "Pow", result, exponentTensor, ctx.nodeCounter, ctx.tensorPrefix);
    }

    return result;
}

std::string ONNXGenerator::convertDerFunctionCall(
    basemodelica::BaseModelicaParser::PrimaryContext* expr,
    const ConversionContext& ctx) {

    auto funcCallArgs = expr->functionCallArgs();
    if (!funcCallArgs) {
        throw std::runtime_error("der() requires an argument");
    }

    auto funcArgs = funcCallArgs->functionArguments();
    if (!funcArgs) {
        throw std::runtime_error("der() requires an argument");
    }

    auto argExpr = funcArgs->expression();
    if (!argExpr) {
        throw std::runtime_error("der() argument is missing");
    }

    // Check if the argument is a simple variable reference
    bool isSimpleVariable = false;
    std::string varName;

    if (argExpr->children.size() == 1) {
        antlr4::tree::ParseTree* node = argExpr->children[0];
        while (node && node->children.size() == 1) {
            node = node->children[0];
        }
        if (node && node->children.empty()) {
            isSimpleVariable = true;
            varName = stripQuotes(argExpr->getText());
        }
    }

    // Check if the argument is an array element access
    basemodelica::BaseModelicaParser::ComponentReferenceContext* compRef = nullptr;
    if (!isSimpleVariable) {
        auto primary = ParseTreeNavigator::findPrimary(argExpr);
        compRef = primary ? primary->componentReference() : nullptr;
    }

    // Handle array element derivative: der(x[i]) -> der('x')[i]
    if (compRef && !compRef->arraySubscripts().empty()) {
        std::string baseVarName = stripQuotes(compRef->IDENT(0)->getText());

        if (ctx.derivativeInputs) {
            std::string derInputName = "der('" + baseVarName + "')";

            if (ctx.derivativeInputs->find(derInputName) == ctx.derivativeInputs->end()) {
                std::vector<std::string> dimensions;
                const Variable* var = ctx.info.findVariable(baseVarName);
                if (var) {
                    dimensions = var->dimensions;
                }
                (*ctx.derivativeInputs)[derInputName] = dimensions;
            }

            auto subscriptList = compRef->arraySubscripts()[0]->subscript();
            return applyArraySubscripts(ctx.graph, derInputName, subscriptList, ctx.variableMap, ctx.nodeCounter, ctx.tensorPrefix);
        }
    }

    // Handle simple variable derivative
    if (isSimpleVariable && ctx.derivativeInputs) {
        std::string derInputName = "der('" + varName + "')";

        if (ctx.derivativeInputs->find(derInputName) == ctx.derivativeInputs->end()) {
            std::vector<std::string> dimensions;
            const Variable* var = ctx.info.findVariable(varName);
            if (var) {
                dimensions = var->dimensions;
            }
            (*ctx.derivativeInputs)[derInputName] = dimensions;
        }

        return derInputName;
    }

    // Complex expression derivative: create a Der node
    std::string inputTensor = convertExpression(argExpr, ctx);

    auto* node = ctx.graph->add_node();
    std::string outputTensor = makeTensorName(ctx.tensorPrefix, ctx.nodeCounter);

    node->set_op_type("Der");
    node->set_domain("lacemodelica");
    node->set_name("Der_" + std::to_string(ctx.nodeCounter));
    node->add_input(inputTensor);
    node->add_output(outputTensor);

    return outputTensor;
}

std::string ONNXGenerator::convertUserFunctionCall(
    const std::string& funcName,
    basemodelica::BaseModelicaParser::PrimaryContext* expr,
    const Function* func,
    const ConversionContext& ctx) {

    auto funcCallArgs = expr->functionCallArgs();
    auto funcArgs = funcCallArgs->functionArguments();

    if (!funcArgs) {
        throw std::runtime_error("Function " + funcName + " requires arguments");
    }

    auto arguments = collectFunctionArguments(funcArgs);

    if (arguments.size() != func->inputs.size()) {
        throw std::runtime_error("Function " + funcName + " expects " +
            std::to_string(func->inputs.size()) + " arguments, got " +
            std::to_string(arguments.size()));
    }

    std::vector<std::string> argTensors;
    for (size_t i = 0; i < arguments.size(); i++) {
        argTensors.push_back(convertExpression(arguments[i], ctx));
    }

    auto* node = ctx.graph->add_node();
    std::string outputTensor = makeTensorName(ctx.tensorPrefix, ctx.nodeCounter);

    node->set_op_type(funcName);
    node->set_domain("lacemodelica");
    node->set_name(funcName + "_call_" + std::to_string(ctx.nodeCounter));

    for (const auto& argTensor : argTensors) {
        node->add_input(argTensor);
    }
    node->add_output(outputTensor);

    return outputTensor;
}

std::string ONNXGenerator::convertPrimary(
    basemodelica::BaseModelicaParser::PrimaryContext* expr,
    const ConversionContext& ctx) {

    // 1. Number literal
    if (expr->UNSIGNED_NUMBER()) {
        std::string value = expr->UNSIGNED_NUMBER()->getText();
        auto* constant = ctx.graph->add_initializer();
        std::string constName = "const_" + std::to_string(ctx.nodeCounter++);
        constant->set_name(constName);
        constant->set_data_type(onnx::TensorProto::DOUBLE);
        constant->add_double_data(std::stod(value));
        return constName;
    }

    // 1b. Boolean literals
    std::string text = expr->getText();
    if (text == "true" || text == "false") {
        auto* constant = ctx.graph->add_initializer();
        std::string constName = "const_" + std::to_string(ctx.nodeCounter++);
        constant->set_name(constName);
        constant->set_data_type(onnx::TensorProto::BOOL);
        constant->add_int32_data(text == "true" ? 1 : 0);
        return constName;
    }

    // 2. Function call
    if (expr->functionCallArgs()) {
        std::string funcName;
        if (expr->componentReference()) {
            funcName = stripQuotes(expr->componentReference()->getText());
        } else {
            std::string text = expr->getText();
            size_t parenPos = text.find("(");
            if (parenPos == std::string::npos) {
                throw std::runtime_error("Malformed function call: " + text);
            }
            funcName = text.substr(0, parenPos);
        }

        if (funcName == "der") {
            return convertDerFunctionCall(expr, ctx);
        }

        auto mathIt = kMathFunctionMap.find(funcName);
        if (mathIt != kMathFunctionMap.end()) {
            auto funcArgs = expr->functionCallArgs()->functionArguments();
            if (!funcArgs || !funcArgs->expression()) {
                throw std::runtime_error("Function " + funcName + " requires arguments");
            }
            std::string argTensor = convertExpression(funcArgs->expression(), ctx);
            return createUnaryOp(ctx.graph, mathIt->second, argTensor, ctx.nodeCounter, ctx.tensorPrefix);
        }

        const Function* func = ctx.info.findFunction(funcName);
        if (func && !func->algorithmStatements.empty()) {
            return convertUserFunctionCall(funcName, expr, func, ctx);
        }

        throw std::runtime_error("Unsupported function call: " + funcName);
    }

    // 3. Component reference (variable)
    if (expr->componentReference()) {
        auto compRef = expr->componentReference();
        std::string varName = stripQuotes(compRef->IDENT(0)->getText());

        std::string baseTensor;
        if (ctx.variableMap && ctx.variableMap->find(varName) != ctx.variableMap->end()) {
            baseTensor = ctx.variableMap->at(varName);
        } else {
            baseTensor = varName;
        }

        auto subscripts = compRef->arraySubscripts();
        if (subscripts.empty()) {
            return baseTensor;
        }

        auto subscriptList = subscripts[0]->subscript();
        return applyArraySubscripts(ctx.graph, baseTensor, subscriptList, ctx.variableMap, ctx.nodeCounter, ctx.tensorPrefix);
    }

    // 4. Parenthesized expression
    if (expr->outputExpressionList()) {
        auto outputList = expr->outputExpressionList();
        auto expressions = outputList->expression();
        if (!expressions.empty()) {
            return convertExpression(expressions[0], ctx);
        }
    }

    throw std::runtime_error("Unsupported primary expression: " + expr->getText());
}

std::vector<std::string> ONNXGenerator::convertMultiOutputFunctionCall(
    antlr4::ParserRuleContext* expr,
    const ConversionContext& ctx,
    size_t expectedOutputCount) {

    auto primaryCtx = ParseTreeNavigator::findPrimary(expr);

    if (!primaryCtx || !primaryCtx->componentReference() || !primaryCtx->functionCallArgs()) {
        throw std::runtime_error("Multi-output expression must be a function call");
    }

    auto compRef = primaryCtx->componentReference();
    std::string funcName = stripQuotes(compRef->IDENT(0)->getText());

    const Function* func = ctx.info.findFunction(funcName);
    if (!func) {
        throw std::runtime_error("Function not found: " + funcName);
    }

    if (func->outputs.size() != expectedOutputCount) {
        throw std::runtime_error("Function " + funcName + " has " + std::to_string(func->outputs.size()) +
                               " outputs, expected " + std::to_string(expectedOutputCount));
    }

    auto funcCallArgs = primaryCtx->functionCallArgs();
    auto funcArgs = funcCallArgs->functionArguments();
    if (!funcArgs) {
        throw std::runtime_error("Function " + funcName + " requires arguments");
    }

    auto arguments = collectFunctionArguments(funcArgs);
    if (arguments.size() != func->inputs.size()) {
        throw std::runtime_error("Function " + funcName + " expects " +
            std::to_string(func->inputs.size()) + " arguments, got " +
            std::to_string(arguments.size()));
    }

    std::vector<std::string> argTensors;
    for (size_t i = 0; i < arguments.size(); i++) {
        argTensors.push_back(convertExpression(arguments[i], ctx));
    }

    auto* node = ctx.graph->add_node();
    node->set_op_type(funcName);
    node->set_domain("lacemodelica");
    node->set_name(funcName + "_call_" + std::to_string(ctx.nodeCounter));

    for (const auto& argTensor : argTensors) {
        node->add_input(argTensor);
    }

    std::vector<std::string> outputTensors;
    for (size_t i = 0; i < expectedOutputCount; i++) {
        std::string outputTensor = "tensor_" + std::to_string(ctx.nodeCounter++);
        node->add_output(outputTensor);
        outputTensors.push_back(outputTensor);
    }

    return outputTensors;
}

} // namespace lacemodelica
