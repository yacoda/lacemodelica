// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ParseTreeNavigator.h"
#include "Utils.hpp"
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
        // start:step:end - only a step of 1 is supported
        if (std::stoi(logExprs[1]->getText()) != 1) {
            throw std::runtime_error("Range with a step other than 1: " + expr->getText());
        }
        int64_t start = std::stoi(logExprs[0]->getText());
        int64_t end = std::stoi(logExprs[2]->getText());
        return {start, end};
    }

    throw std::runtime_error("Invalid range expression: " + expr->getText());
}

bool ParseTreeNavigator::isDerCall(basemodelica::BaseModelicaParser::PrimaryContext* primary) {
    return primary && primary->functionCallArgs() && !primary->componentReference() &&
           primary->getStart()->getText() == "der";
}

basemodelica::BaseModelicaParser::ComponentReferenceContext*
ParseTreeNavigator::derStateReference(basemodelica::BaseModelicaParser::PrimaryContext* derCall) {
    using namespace basemodelica;
    auto* funcArgs = derCall->functionCallArgs()->functionArguments();
    if (!funcArgs || !funcArgs->expression()) {
        return nullptr;
    }

    // The argument is a reference only if the expression is that one primary: each rule
    // between them has a single child, so x in der(x + y) or der(-x) does not count
    antlr4::tree::ParseTree* node = funcArgs->expression();
    while (!dynamic_cast<BaseModelicaParser::PrimaryContext*>(node)) {
        if (node->children.size() != 1 || !dynamic_cast<antlr4::ParserRuleContext*>(node->children[0])) {
            return nullptr;
        }
        node = node->children[0];
    }
    auto* primary = static_cast<BaseModelicaParser::PrimaryContext*>(node);
    auto* ref = primary->componentReference();
    if (!ref || primary->functionCallArgs()) {
        return nullptr;
    }

    // An element of an array of records, 'a'[i].'b', is not a variable of its own
    bool afterSubscripts = false;
    for (auto* child : ref->children) {
        if (dynamic_cast<BaseModelicaParser::ArraySubscriptsContext*>(child)) {
            afterSubscripts = true;
        } else if (afterSubscripts) {
            return nullptr;
        }
    }
    return ref;
}

std::string ParseTreeNavigator::stateName(basemodelica::BaseModelicaParser::ComponentReferenceContext* ref) {
    std::string name;
    for (auto* ident : ref->IDENT()) {
        if (!name.empty()) {
            name += ".";
        }
        name += stripQuotes(ident->getText());
    }
    return name;
}

std::vector<basemodelica::BaseModelicaParser::PrimaryContext*>
ParseTreeNavigator::findDerCalls(antlr4::tree::ParseTree* tree) {
    std::vector<basemodelica::BaseModelicaParser::PrimaryContext*> calls;
    std::vector<antlr4::tree::ParseTree*> stack{tree};
    while (!stack.empty()) {
        auto* node = stack.back();
        stack.pop_back();
        if (!node) continue;
        auto* primary = dynamic_cast<basemodelica::BaseModelicaParser::PrimaryContext*>(node);
        if (isDerCall(primary)) {
            calls.push_back(primary);
        }
        stack.insert(stack.end(), node->children.rbegin(), node->children.rend());
    }
    return calls;
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
