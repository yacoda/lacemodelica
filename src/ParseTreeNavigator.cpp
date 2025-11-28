// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ParseTreeNavigator.h"
#include <stdexcept>

namespace lacemodelica {

bool ParseTreeNavigator::isRangeExpression(
    basemodelica::BaseModelicaParser::ExpressionContext* expr) {
    if (!expr) return false;
    auto* exprNoDeco = expr->expressionNoDecoration();
    if (!exprNoDeco) return false;
    auto* simpleExpr = exprNoDeco->simpleExpression();
    if (!simpleExpr) return false;
    // Range expression has multiple logicalExpressions separated by ':'
    return simpleExpr->logicalExpression().size() > 1;
}

std::pair<int64_t, int64_t> ParseTreeNavigator::parseRangeBounds(
    basemodelica::BaseModelicaParser::ExpressionContext* expr) {
    auto* simpleExpr = expr->expressionNoDecoration()->simpleExpression();
    auto logExprs = simpleExpr->logicalExpression();

    if (logExprs.size() == 2) {
        // start:end
        int64_t start = std::stoi(logExprs[0]->getText());
        int64_t end = std::stoi(logExprs[1]->getText());
        return {start, end};
    } else if (logExprs.size() == 3) {
        // start:step:end - step is ignored for now (assume step=1)
        int64_t start = std::stoi(logExprs[0]->getText());
        int64_t end = std::stoi(logExprs[2]->getText());
        return {start, end};
    }

    throw std::runtime_error("Invalid range expression: " + expr->getText());
}

std::map<std::type_index, ParseTreeNavigator::NavigationFunc>
    ParseTreeNavigator::navigationMap;

bool ParseTreeNavigator::initialized = false;
const bool ParseTreeNavigator::dummy = ParseTreeNavigator::initialize();

bool ParseTreeNavigator::initialize() {
    if (initialized) return true;

    using namespace basemodelica;

    // Define navigation rules for each context type
    // Each rule says: "given this node type, navigate to this child"

    navigationMap[typeid(BaseModelicaParser::ExpressionContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto expr = static_cast<BaseModelicaParser::ExpressionContext*>(ctx);
            return expr->expressionNoDecoration();
        };

    navigationMap[typeid(BaseModelicaParser::ExpressionNoDecorationContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto exprNoDec = static_cast<BaseModelicaParser::ExpressionNoDecorationContext*>(ctx);
            return exprNoDec->simpleExpression();
        };

    navigationMap[typeid(BaseModelicaParser::SimpleExpressionContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto simpleExpr = static_cast<BaseModelicaParser::SimpleExpressionContext*>(ctx);
            return simpleExpr->logicalExpression().size() > 0 ?
                   simpleExpr->logicalExpression(0) : nullptr;
        };

    navigationMap[typeid(BaseModelicaParser::LogicalExpressionContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto logExpr = static_cast<BaseModelicaParser::LogicalExpressionContext*>(ctx);
            return logExpr->logicalTerm().size() > 0 ?
                   logExpr->logicalTerm(0) : nullptr;
        };

    navigationMap[typeid(BaseModelicaParser::LogicalTermContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto logTerm = static_cast<BaseModelicaParser::LogicalTermContext*>(ctx);
            return logTerm->logicalFactor().size() > 0 ?
                   logTerm->logicalFactor(0) : nullptr;
        };

    navigationMap[typeid(BaseModelicaParser::LogicalFactorContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto logFactor = static_cast<BaseModelicaParser::LogicalFactorContext*>(ctx);
            return logFactor->relation();
        };

    navigationMap[typeid(BaseModelicaParser::RelationContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto relation = static_cast<BaseModelicaParser::RelationContext*>(ctx);
            return relation->arithmeticExpression().size() > 0 ?
                   relation->arithmeticExpression(0) : nullptr;
        };

    navigationMap[typeid(BaseModelicaParser::ArithmeticExpressionContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto arithExpr = static_cast<BaseModelicaParser::ArithmeticExpressionContext*>(ctx);
            return arithExpr->term().size() > 0 ?
                   arithExpr->term(0) : nullptr;
        };

    navigationMap[typeid(BaseModelicaParser::TermContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto term = static_cast<BaseModelicaParser::TermContext*>(ctx);
            return term->factor().size() > 0 ?
                   term->factor(0) : nullptr;
        };

    navigationMap[typeid(BaseModelicaParser::FactorContext)] =
        [](antlr4::ParserRuleContext* ctx) -> antlr4::tree::ParseTree* {
            auto factor = static_cast<BaseModelicaParser::FactorContext*>(ctx);
            return factor->primary().size() > 0 ?
                   factor->primary(0) : nullptr;
        };

    initialized = true;
    return true;
}

} // namespace lacemodelica
