// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ExpressionConverter.h"
#include "EquationGenerator.h"
#include "ParseTreeNavigator.h"
#include "Utils.hpp"

#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>

#include <stdexcept>
#include <cctype>

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

// Structure to hold both positional and named arguments
struct FunctionCallArguments {
    std::vector<basemodelica::BaseModelicaParser::ExpressionContext*> positional;
    std::map<std::string, basemodelica::BaseModelicaParser::ExpressionContext*> named;
};

// Collect all function arguments (positional and named)
FunctionCallArguments collectAllFunctionArguments(
    basemodelica::BaseModelicaParser::FunctionArgumentsContext* funcArgs) {
    FunctionCallArguments result;

    if (!funcArgs) return result;

    // Case 1: Only named arguments (no positional first expression)
    if (funcArgs->namedArguments()) {
        auto namedArgs = funcArgs->namedArguments();
        for (auto* namedArg : namedArgs->namedArgument()) {
            std::string paramName = stripQuotes(namedArg->IDENT()->getText());
            if (auto* funcArg = namedArg->functionArgument()) {
                if (auto* expr = funcArg->expression()) {
                    result.named[paramName] = expr;
                }
            }
        }
        return result;
    }

    // Case 2: Starts with positional expression
    if (auto firstExpr = funcArgs->expression()) {
        result.positional.push_back(firstExpr);
    }

    // Process remaining arguments (can be positional or switch to named)
    auto nonFirst = funcArgs->functionArgumentsNonFirst();
    while (nonFirst) {
        // Check if we've switched to named arguments
        if (nonFirst->namedArguments()) {
            auto namedArgs = nonFirst->namedArguments();
            for (auto* namedArg : namedArgs->namedArgument()) {
                std::string paramName = stripQuotes(namedArg->IDENT()->getText());
                if (auto* funcArg = namedArg->functionArgument()) {
                    if (auto* expr = funcArg->expression()) {
                        result.named[paramName] = expr;
                    }
                }
            }
            break;  // Named arguments must be last
        }

        // Still positional argument
        if (auto funcArg = nonFirst->functionArgument()) {
            if (auto expr = funcArg->expression()) {
                result.positional.push_back(expr);
            }
        }
        nonFirst = nonFirst->functionArgumentsNonFirst();
    }

    return result;
}

// Resolve arguments to correct order based on function signature
std::vector<basemodelica::BaseModelicaParser::ExpressionContext*>
resolveArgumentOrder(const FunctionCallArguments& args, const Function* func) {
    std::vector<basemodelica::BaseModelicaParser::ExpressionContext*> result;
    result.resize(func->inputs.size(), nullptr);

    // First, place positional arguments
    for (size_t i = 0; i < args.positional.size() && i < func->inputs.size(); i++) {
        result[i] = args.positional[i];
    }

    // Then, place named arguments by finding their position in the function signature
    for (const auto& [name, expr] : args.named) {
        bool found = false;
        for (size_t i = 0; i < func->inputs.size(); i++) {
            if (func->inputs[i].name == name) {
                if (result[i] != nullptr) {
                    throw std::runtime_error("Parameter '" + name + "' specified both positionally and by name");
                }
                result[i] = expr;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Unknown parameter name: " + name);
        }
    }

    // Check all arguments are provided
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] == nullptr) {
            throw std::runtime_error("Missing argument for parameter: " + func->inputs[i].name);
        }
    }

    return result;
}

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

    // 1. Number literal (UNSIGNED_NUMBER for floats, UNSIGNED_INTEGER for ints)
    if (expr->UNSIGNED_NUMBER() || expr->UNSIGNED_INTEGER()) {
        std::string value = expr->UNSIGNED_NUMBER()
            ? expr->UNSIGNED_NUMBER()->getText()
            : expr->UNSIGNED_INTEGER()->getText();
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

        // Check for reduction functions with for-loop syntax: sum(expr for i in 1:n)
        if (funcName == "sum" || funcName == "product") {
            auto funcArgs = expr->functionCallArgs()->functionArguments();
            if (funcArgs && funcArgs->forIndex()) {
                return convertReductionExpression(funcName, funcArgs->expression(),
                                                   funcArgs->forIndex(), ctx);
            }
            // Fall through to handle sum/product on arrays without for-loop
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

    // 4. Component reference (variable or enum literal)
    if (expr->componentReference()) {
        auto compRef = expr->componentReference();
        std::string varName = stripQuotes(compRef->IDENT(0)->getText());

        // Check if this is an enum literal (e.g., 'State'.'off')
        // Enum literals have multiple IDENTs: TypeName.literalName
        if (compRef->IDENT().size() >= 2) {
            std::string typeName = varName;
            std::string literalName = stripQuotes(compRef->IDENT(1)->getText());

            // Look up enum type
            const EnumType* enumType = ctx.info.findEnumType(typeName);
            if (enumType) {
                int enumValue = enumType->getValue(literalName);
                if (enumValue >= 0) {
                    // Return the enum value as a double constant
                    return builder.addDoubleConstant(static_cast<double>(enumValue));
                }
                throw std::runtime_error("Unknown enum literal: " + typeName + "." + literalName);
            }
        }

        std::string baseTensor;
        bool isLoopIndex = false;
        if (ctx.variableMap && ctx.variableMap->find(varName) != ctx.variableMap->end()) {
            baseTensor = ctx.variableMap->at(varName);
            // Detect loop index variables (single letter like i, j, k in loop context)
            // These are int64 and need to be cast to double for arithmetic compatibility
            if (!ctx.tensorPrefix.empty() && varName.size() == 1 && std::isalpha(varName[0])) {
                isLoopIndex = true;
            }
        } else {
            baseTensor = varName;
        }

        auto subscripts = compRef->arraySubscripts();
        if (subscripts.empty()) {
            // Cast loop index to double for arithmetic compatibility with double constants
            if (isLoopIndex) {
                std::string doubleTensor = builder.makeTensorName("idx_dbl");
                auto* castNode = ctx.graph->add_node();
                castNode->set_op_type("Cast");
                castNode->set_name(doubleTensor + "_Cast");
                castNode->add_input(baseTensor);
                castNode->add_output(doubleTensor);
                auto* toAttr = castNode->add_attribute();
                toAttr->set_name("to");
                toAttr->set_type(onnx::AttributeProto::INT);
                toAttr->set_i(11);  // ONNX TensorProto::DOUBLE
                return doubleTensor;
            }
            return baseTensor;
        }

        auto subscriptList = subscripts[0]->subscript();

        // Check if any subscript contains a complex expression (not just a constant, loop var, or range)
        bool hasComplexSubscript = false;
        for (auto sub : subscriptList) {
            if (sub->getText() == ":") {
                continue;  // Full slice notation - handled by applySubscripts
            }
            auto subExpr = sub->expression();
            if (!subExpr) continue;

            // Check if it's a range expression (e.g., "2:4")
            if (ParseTreeNavigator::isRangeExpression(subExpr)) {
                continue;  // Range expression - handled by applySubscripts
            }

            std::string indexText = subExpr->getText();

            // Check if it's a simple loop variable reference
            if (ctx.variableMap && ctx.variableMap->count(indexText) > 0) {
                continue;  // Simple loop var - handled by applySubscripts
            }

            // Check if it's a constant value (number or boolean)
            if (isConstValue(indexText)) {
                continue;  // Static constant - handled by applySubscripts
            }

            // Complex expression (like "3-i") - needs special handling
            hasComplexSubscript = true;
            break;
        }

        if (!hasComplexSubscript) {
            return builder.applySubscripts(baseTensor, subscriptList, ctx.variableMap);
        }

        // Handle complex subscript expressions by converting them
        std::string currentTensor = baseTensor;
        for (auto sub : subscriptList) {
            auto subExpr = sub->expression();
            if (!subExpr) {
                throw std::runtime_error("Invalid array subscript");
            }

            // Convert the subscript expression to a tensor
            std::string indexTensor = convert(subExpr, ctx);

            // Cast to int64 if needed (subscript expressions may produce double)
            std::string indexInt64 = builder.makeTensorName("idx_int64");
            auto* castNode = ctx.graph->add_node();
            castNode->set_op_type("Cast");
            castNode->set_name(indexInt64 + "_Cast");
            castNode->add_input(indexTensor);
            castNode->add_output(indexInt64);
            auto* toAttr = castNode->add_attribute();
            toAttr->set_name("to");
            toAttr->set_type(onnx::AttributeProto::INT);
            toAttr->set_i(7);  // ONNX TensorProto::INT64

            // Convert from 1-based Modelica to 0-based ONNX
            std::string index0Based = builder.convertToZeroBasedIndex(indexInt64);

            // Use Gather to index the array
            currentTensor = builder.addGather(currentTensor, index0Based, 0);
        }
        return currentTensor;
    }

    // 5. Array constructor {arrayArguments}
    if (expr->arrayArguments()) {
        auto* arrayArgs = expr->arrayArguments();
        if (arrayArgs->forIndex()) {
            return convertArrayConstructor(arrayArgs, ctx);
        }
        // Handle constant/expression array literals like {1, 2, 3} or {val, val*2}
        return convertArrayLiteral(arrayArgs, ctx);
    }

    // 6. Parenthesized expression
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

    // Handle array element derivative: der(x[i]) -> der(x)[i]
    if (compRef && !compRef->arraySubscripts().empty()) {
        std::string baseVarName = stripQuotes(compRef->IDENT(0)->getText());

        if (ctx.derivativeInputs) {
            std::string derInputName = "der(" + baseVarName + ")";

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
        std::string derInputName = "der(" + varName + ")";

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

    // Collect all arguments (positional and named) and resolve to correct order
    auto allArgs = collectAllFunctionArguments(funcArgs);
    auto arguments = resolveArgumentOrder(allArgs, func);

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

    // Collect all arguments (positional and named) and resolve to correct order
    auto allArgs = collectAllFunctionArguments(funcArgs);
    auto arguments = resolveArgumentOrder(allArgs, func);

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

// -----------------------------------------------------------------------------
// Array Constructor
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertArrayConstructor(
    basemodelica::BaseModelicaParser::ArrayArgumentsContext* arrayArgs,
    const ConversionContext& ctx) {

    // This function only handles array comprehensions: {expr for i in 1:n}
    // Regular array literals are handled by convertArrayLiteral
    if (!arrayArgs->forIndex()) {
        throw std::runtime_error("Array literal without comprehension not supported in expressions");
    }

    auto expressions = arrayArgs->expression();
    if (expressions.empty()) {
        throw std::runtime_error("Array comprehension requires a body expression");
    }
    return convertArrayComprehension(expressions[0], arrayArgs->forIndex(), ctx);
}

std::string ExpressionConverter::convertArrayLiteral(
    basemodelica::BaseModelicaParser::ArrayArgumentsContext* arrayArgs,
    const ConversionContext& ctx) {

    auto builder = ctx.builder();
    auto expressions = arrayArgs->expression();

    if (expressions.empty()) {
        throw std::runtime_error("Empty array literal");
    }

    // Check if all elements are pure constant values (numbers).
    // If so, create a constant tensor directly - more efficient than concat.
    bool allConstant = true;
    std::vector<double> constantValues;
    for (auto* expr : expressions) {
        std::string exprText = expr->getText();
        if (!isConstValue(exprText)) {
            allConstant = false;
            break;
        }
        try {
            constantValues.push_back(std::stod(exprText));
        } catch (...) {
            allConstant = false;
            break;
        }
    }
    if (allConstant) {
        return builder.addDoubleArrayConstant(constantValues);
    }

    // Convert and unsqueeze each element/row, then concat along axis 0
    // Works for both 1D {a, b, c} and 2D {{a, b}, {c, d}} cases
    std::vector<std::string> tensors;
    for (auto* expr : expressions) {
        std::string tensor = convert(expr, ctx);
        // Unsqueeze to add leading dimension: scalar->[1] or row->[1,n]
        tensors.push_back(builder.addUnsqueeze(tensor, {0}));
    }

    return builder.addConcat(tensors, 0);
}

// -----------------------------------------------------------------------------
// Array Comprehension
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertArrayComprehension(
    basemodelica::BaseModelicaParser::ExpressionContext* bodyExpr,
    basemodelica::BaseModelicaParser::ForIndexContext* forIndex,
    const ConversionContext& ctx) {

    auto builder = ctx.builder();
    ForLoopRange range = parseForLoopRangeFromIndex(forIndex);

    // Create common loop structure
    auto loop = createForLoopBase(ctx.graph, builder, range, "comprehension");
    auto bodyBuilder = loop.makeBodyBuilder(builder);

    // Create variable map with loop variable
    std::map<std::string, std::string> bodyVarMap;
    if (ctx.variableMap) {
        bodyVarMap = *ctx.variableMap;
    }
    bodyVarMap[range.loopVar] = loop.loopVarTensor;

    // Create context for body expression
    ConversionContext bodyCtx = ctx.withGraph(loop.bodyGraph);
    bodyCtx.variableMap = &bodyVarMap;
    bodyCtx.tensorPrefix = loop.loopNodeName + "_";

    // Convert body expression
    std::string bodyResultTensor = convert(bodyExpr, bodyCtx);

    // Unsqueeze to make it [1] shape for scan output
    std::string unsqueezedResult = bodyBuilder.addUnsqueeze(bodyResultTensor, {0});

    // Add scan output (accumulated across iterations)
    std::string scanOutputName = loop.loopNodeName + "_scan_out";
    auto* scanOutput = loop.bodyGraph->add_output();
    scanOutput->set_name(scanOutputName);
    auto* scanOutType = scanOutput->mutable_type()->mutable_tensor_type();
    scanOutType->set_elem_type(onnx::TensorProto::DOUBLE);
    scanOutType->mutable_shape()->add_dim()->set_dim_value(1);

    // Identity to connect body result to scan output
    auto* identityNode = loop.bodyGraph->add_node();
    identityNode->set_op_type("Identity");
    identityNode->set_name(loop.loopNodeName + "_scan_identity");
    identityNode->add_input(unsqueezedResult);
    identityNode->add_output(scanOutputName);

    // Add loop output (scan outputs become array)
    std::string loopOutputTensor = loop.loopNodeName + "_result";
    loop.loopNode->add_output(loopOutputTensor);

    // Squeeze the result from [n, 1] to [n]
    // Create axes constant before Squeeze node for correct topological order
    std::string axesTensor = builder.addInt64ArrayConstant({1});
    std::string squeezedResult = builder.makeTensorName("array_result");
    auto* squeezeNode = ctx.graph->add_node();
    squeezeNode->set_op_type("Squeeze");
    squeezeNode->set_name(squeezedResult + "_Squeeze");
    squeezeNode->add_input(loopOutputTensor);
    squeezeNode->add_input(axesTensor);
    squeezeNode->add_output(squeezedResult);

    return squeezedResult;
}

// -----------------------------------------------------------------------------
// Reduction Expression
// -----------------------------------------------------------------------------

std::string ExpressionConverter::convertReductionExpression(
    const std::string& reductionOp,
    basemodelica::BaseModelicaParser::ExpressionContext* bodyExpr,
    basemodelica::BaseModelicaParser::ForIndexContext* forIndex,
    const ConversionContext& ctx) {

    auto builder = ctx.builder();
    ForLoopRange range = parseForLoopRangeFromIndex(forIndex);

    // Initial accumulator value (0 for sum, 1 for product) - must be before loop node
    std::string initAccumTensor;
    if (reductionOp == "sum") {
        initAccumTensor = builder.addDoubleConstant(0.0);
    } else if (reductionOp == "product") {
        initAccumTensor = builder.addDoubleConstant(1.0);
    } else {
        throw std::runtime_error("Unknown reduction operation: " + reductionOp);
    }

    // Create common loop structure
    auto loop = createForLoopBase(ctx.graph, builder, range, "reduce_" + reductionOp);
    auto bodyBuilder = loop.makeBodyBuilder(builder);

    // Add loop-carried accumulator input
    loop.loopNode->add_input(initAccumTensor);

    // Add accumulator input to body graph
    std::string accumInputName = loop.loopNodeName + "_accum_in";
    auto* accumInput = loop.bodyGraph->add_input();
    accumInput->set_name(accumInputName);
    auto* accumInputType = accumInput->mutable_type()->mutable_tensor_type();
    accumInputType->set_elem_type(onnx::TensorProto::DOUBLE);
    accumInputType->mutable_shape();  // Scalar

    // Create variable map with loop variable
    std::map<std::string, std::string> bodyVarMap;
    if (ctx.variableMap) {
        bodyVarMap = *ctx.variableMap;
    }
    bodyVarMap[range.loopVar] = loop.loopVarTensor;

    // Create context for body expression
    ConversionContext bodyCtx = ctx.withGraph(loop.bodyGraph);
    bodyCtx.variableMap = &bodyVarMap;
    bodyCtx.tensorPrefix = loop.loopNodeName + "_";

    // Convert body expression
    std::string bodyResultTensor = convert(bodyExpr, bodyCtx);

    // Apply reduction operation: accum_out = accum_in op body_result
    std::string onnxOp = (reductionOp == "sum") ? "Add" : "Mul";
    std::string accumOutputTensor = bodyBuilder.addBinaryOp(onnxOp, accumInputName, bodyResultTensor);

    // Add accumulator output to body
    std::string accumOutputName = loop.loopNodeName + "_accum_out";
    auto* accumOutput = loop.bodyGraph->add_output();
    accumOutput->set_name(accumOutputName);
    auto* accumOutputType = accumOutput->mutable_type()->mutable_tensor_type();
    accumOutputType->set_elem_type(onnx::TensorProto::DOUBLE);
    accumOutputType->mutable_shape();  // Scalar

    // Identity to connect to output
    auto* identityNode = loop.bodyGraph->add_node();
    identityNode->set_op_type("Identity");
    identityNode->set_name(loop.loopNodeName + "_accum_identity");
    identityNode->add_input(accumOutputTensor);
    identityNode->add_output(accumOutputName);

    // Add loop output (final accumulator value)
    std::string loopOutputTensor = loop.loopNodeName + "_result";
    loop.loopNode->add_output(loopOutputTensor);

    return loopOutputTensor;
}

} // namespace lacemodelica
