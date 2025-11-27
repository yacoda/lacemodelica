// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ExpressionConverter.h"
#include "ParseTreeNavigator.h"
#include "Utils.hpp"

#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>

#include <stdexcept>

namespace lacemodelica {

const std::map<std::string, std::string> kRelationalOpMap = {
    {">", "Greater"},
    {"<", "Less"},
    {">=", "GreaterOrEqual"},
    {"<=", "LessOrEqual"},
    {"==", "Equal"}
    // Note: "<>" (not equal) is handled specially as Equal + Not
};

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

std::vector<basemodelica::BaseModelicaParser::ExpressionContext*>
collectFunctionArguments(basemodelica::BaseModelicaParser::FunctionArgumentsContext* funcArgs) {
    std::vector<basemodelica::BaseModelicaParser::ExpressionContext*> arguments;

    if (!funcArgs) return arguments;

    if (auto firstExpr = funcArgs->expression()) {
        arguments.push_back(firstExpr);
    }

    auto nonFirst = funcArgs->functionArgumentsNonFirst();
    while (nonFirst) {
        if (auto funcArg = nonFirst->functionArgument()) {
            if (auto expr = funcArg->expression()) {
                arguments.push_back(expr);
            }
        }
        nonFirst = nonFirst->functionArgumentsNonFirst();
    }

    return arguments;
}

bool isMatrixVariable(const std::string& tensorName, const ModelInfo& info) {
    auto it = info.variableIndex.find(tensorName);
    if (it != info.variableIndex.end()) {
        return info.variables[it->second].dimensions.size() == 2;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Main Entry Point
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convert(antlr4::ParserRuleContext* expr, const ConversionContext& ctx) {
    if (!expr) {
        throw std::runtime_error("Null expression context");
    }

    if (auto* exprCtx = dynamic_cast<basemodelica::BaseModelicaParser::ExpressionContext*>(expr)) {
        if (auto* exprNoDecoration = exprCtx->expressionNoDecoration()) {
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

// -----------------------------------------------------------------------------
// If Expression
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertIfExpression(
    basemodelica::BaseModelicaParser::IfExpressionContext* expr,
    const ConversionContext& ctx) {

    auto expressions = expr->expressionNoDecoration();

    if (expressions.size() < 3) {
        throw std::runtime_error("Invalid if expression structure");
    }
    if (expressions.size() % 2 != 1) {
        throw std::runtime_error("Invalid if expression structure: expected odd number of expressions");
    }

    auto builder = ctx.builder();
    std::string condTensor = convert(expressions[0], ctx);

    onnx::GraphProto thenBranch;
    thenBranch.set_name("then_branch");
    auto thenBuilder = builder.forSubgraph(&thenBranch);
    std::string thenResult = convert(expressions[1], ctx.withGraph(&thenBranch));
    thenBuilder.addScalarDoubleOutput(thenResult);

    onnx::GraphProto elseBranch;
    elseBranch.set_name("else_branch");
    auto elseBuilder = builder.forSubgraph(&elseBranch);
    auto elseCtx = ctx.withGraph(&elseBranch);
    std::string elseResult = (expressions.size() == 3)
        ? convert(expressions[2], elseCtx)
        : convertNestedIfElse(expressions, 2, elseCtx);
    elseBuilder.addScalarDoubleOutput(elseResult);

    return builder.addIfNode(condTensor, thenBranch, elseBranch);
}

std::string ExpressionConverter::convertNestedIfElse(
    const std::vector<basemodelica::BaseModelicaParser::ExpressionNoDecorationContext*>& expressions,
    size_t startIdx,
    const ConversionContext& ctx) {

    size_t remaining = expressions.size() - startIdx;

    if (remaining == 1) {
        return convert(expressions[startIdx], ctx);
    }

    auto builder = ctx.builder();
    std::string condTensor = convert(expressions[startIdx], ctx);

    onnx::GraphProto thenBranch;
    thenBranch.set_name("then_branch");
    std::string thenResult = convert(expressions[startIdx + 1], ctx.withGraph(&thenBranch));
    builder.forSubgraph(&thenBranch).addScalarDoubleOutput(thenResult);

    onnx::GraphProto elseBranch;
    elseBranch.set_name("else_branch");
    std::string elseResult = convertNestedIfElse(expressions, startIdx + 2, ctx.withGraph(&elseBranch));
    builder.forSubgraph(&elseBranch).addScalarDoubleOutput(elseResult);

    return builder.addIfNode(condTensor, thenBranch, elseBranch, "If_nested");
}

// -----------------------------------------------------------------------------
// Simple Expression (Logical)
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertSimpleExpression(
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

    auto builder = ctx.builder();
    std::string resultTensor;

    for (size_t i = 0; i < logicalFactors.size(); i++) {
        auto* logicalFactor = logicalFactors[i];
        auto* relation = logicalFactor->relation();
        if (!relation) {
            throw std::runtime_error("No relation in logical factor");
        }

        std::string factorTensor = convertRelation(relation, ctx);

        if (logicalFactor->children.size() > 1) {
            factorTensor = builder.addUnaryOp("Not", factorTensor);
        }

        resultTensor = (i == 0) ? factorTensor
            : builder.addBinaryOp("And", resultTensor, factorTensor);
    }

    return resultTensor;
}

// -----------------------------------------------------------------------------
// Relational Expression
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertRelation(
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

        auto builder = ctx.builder();
        std::string leftTensor = convertArithmeticExpression(arithmeticExprs[0], ctx);
        std::string rightTensor = convertArithmeticExpression(arithmeticExprs[1], ctx);
        std::string opText = relOp->getText();

        auto it = kRelationalOpMap.find(opText);
        if (it != kRelationalOpMap.end()) {
            return builder.addBinaryOp(it->second, leftTensor, rightTensor);
        } else if (opText == "<>") {
            std::string equalResult = builder.addBinaryOp("Equal", leftTensor, rightTensor);
            return builder.addUnaryOp("Not", equalResult);
        } else {
            throw std::runtime_error("Unsupported relational operator: " + opText);
        }
    }

    return convertArithmeticExpression(arithmeticExprs[0], ctx);
}

// -----------------------------------------------------------------------------
// Arithmetic Expression
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertArithmeticExpression(
    basemodelica::BaseModelicaParser::ArithmeticExpressionContext* expr,
    const ConversionContext& ctx) {

    auto terms = expr->term();
    auto addOps = expr->addOperator();

    if (terms.empty()) {
        throw std::runtime_error("Empty arithmetic expression");
    }

    auto builder = ctx.builder();
    size_t termIndex = 0;
    size_t opIndex = 0;

    std::string result = convertTerm(terms[termIndex++], ctx);

    // Handle leading unary operator
    if (addOps.size() > terms.size() - 1) {
        std::string opText = addOps[opIndex++]->getText();
        if (opText == "-") {
            result = builder.addUnaryOp("Neg", result);
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

        result = builder.addBinaryOp(onnxOp, result, rightTensor);
    }

    return result;
}

// -----------------------------------------------------------------------------
// Term (Multiplication/Division)
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertTerm(
    basemodelica::BaseModelicaParser::TermContext* expr,
    const ConversionContext& ctx) {

    auto factors = expr->factor();
    auto mulOps = expr->mulOperator();

    if (factors.empty()) {
        throw std::runtime_error("Empty term");
    }

    auto builder = ctx.builder();
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

        result = builder.addBinaryOp(onnxOp, result, rightTensor);
    }

    return result;
}

// -----------------------------------------------------------------------------
// Factor (Exponentiation)
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertFactor(
    basemodelica::BaseModelicaParser::FactorContext* expr,
    const ConversionContext& ctx) {

    auto primaries = expr->primary();

    if (primaries.empty()) {
        throw std::runtime_error("Empty factor");
    }

    std::string result = convertPrimary(primaries[0], ctx);

    if (primaries.size() > 1) {
        std::string exponentTensor = convertPrimary(primaries[1], ctx);
        result = ctx.builder().addBinaryOp("Pow", result, exponentTensor);
    }

    return result;
}

// -----------------------------------------------------------------------------
// Primary (Literals, Variables, Function Calls)
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertPrimary(
    basemodelica::BaseModelicaParser::PrimaryContext* expr,
    const ConversionContext& ctx) {

    auto builder = ctx.builder();

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

    // 2. Boolean literals
    std::string text = expr->getText();
    if (text == "true" || text == "false") {
        auto* constant = ctx.graph->add_initializer();
        std::string constName = "const_" + std::to_string(ctx.nodeCounter++);
        constant->set_name(constName);
        constant->set_data_type(onnx::TensorProto::BOOL);
        constant->add_int32_data(text == "true" ? 1 : 0);
        return constName;
    }

    // 3. Function call
    if (expr->functionCallArgs()) {
        std::string funcName;
        if (expr->componentReference()) {
            funcName = stripQuotes(expr->componentReference()->getText());
        } else {
            size_t parenPos = text.find("(");
            if (parenPos == std::string::npos) {
                throw std::runtime_error("Malformed function call: " + text);
            }
            funcName = text.substr(0, parenPos);
        }

        if (funcName == "der") {
            return convertDerCall(expr, ctx);
        }

        auto mathIt = kMathFunctionMap.find(funcName);
        if (mathIt != kMathFunctionMap.end()) {
            auto funcArgs = expr->functionCallArgs()->functionArguments();
            if (!funcArgs || !funcArgs->expression()) {
                throw std::runtime_error("Function " + funcName + " requires arguments");
            }
            std::string argTensor = convert(funcArgs->expression(), ctx);
            return builder.addUnaryOp(mathIt->second, argTensor);
        }

        const Function* func = ctx.info.findFunction(funcName);
        if (func && !func->algorithmStatements.empty()) {
            return convertUserFunctionCall(funcName, expr, func, ctx);
        }

        throw std::runtime_error("Unsupported function call: " + funcName);
    }

    // 4. Component reference (variable)
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
        return builder.applySubscripts(baseTensor, subscriptList, ctx.variableMap);
    }

    // 5. Parenthesized expression
    if (expr->outputExpressionList()) {
        auto outputList = expr->outputExpressionList();
        auto expressions = outputList->expression();
        if (!expressions.empty()) {
            return convert(expressions[0], ctx);
        }
    }

    throw std::runtime_error("Unsupported primary expression: " + expr->getText());
}

// -----------------------------------------------------------------------------
// Derivative Function Call
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertDerCall(
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

    auto builder = ctx.builder();

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
            return builder.applySubscripts(derInputName, subscriptList, ctx.variableMap);
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
    std::string inputTensor = convert(argExpr, ctx);
    std::string outputTensor = builder.makeTensorName();

    auto* node = ctx.graph->add_node();
    node->set_op_type("Der");
    node->set_domain("lacemodelica");
    node->set_name("Der_" + std::to_string(ctx.nodeCounter));
    node->add_input(inputTensor);
    node->add_output(outputTensor);

    return outputTensor;
}

// -----------------------------------------------------------------------------
// User Function Call
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertUserFunctionCall(
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
        argTensors.push_back(convert(arguments[i], ctx));
    }

    auto builder = ctx.builder();
    std::string outputTensor = builder.makeTensorName();

    auto* node = ctx.graph->add_node();
    node->set_op_type(funcName);
    node->set_domain("lacemodelica");
    node->set_name(funcName + "_call_" + std::to_string(ctx.nodeCounter));

    for (const auto& argTensor : argTensors) {
        node->add_input(argTensor);
    }
    node->add_output(outputTensor);

    return outputTensor;
}

// -----------------------------------------------------------------------------
// Multi-Output Function Call
// -----------------------------------------------------------------------------

std::vector<std::string> ExpressionConverter::convertMultiOutput(
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
        argTensors.push_back(convert(arguments[i], ctx));
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
