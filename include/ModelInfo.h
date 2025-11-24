// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#pragma once

#include <string>
#include <vector>
#include <map>

// Forward declare ANTLR context types
namespace basemodelica {
    class BaseModelicaParser;
}

namespace antlr4 {
    class ParserRuleContext;
}

namespace lacemodelica {

struct Equation {
    std::string lhs;  // Left-hand side expression as string
    std::string rhs;  // Right-hand side expression as string
    std::string comment;  // String comment / documentation
    antlr4::ParserRuleContext* lhsContext = nullptr;  // AST node for LHS
    antlr4::ParserRuleContext* rhsContext = nullptr;  // AST node for RHS
};

struct Variable {
    std::string name;
    std::string type;  // "Real", "Integer", "Boolean"
    std::string causality;  // "parameter", "structuralParameter", "local", "output"
    std::string variability;  // "fixed", "tunable", "continuous"
    std::string initial;  // "exact", "approx", "calculated"
    std::string startValue;
    std::string description;  // Variable description from comment
    int valueReference;
    int derivativeOf = -1;  // valueReference of state if this is a derivative
    std::vector<std::string> dimensions;  // For arrays
    bool isStructuralParameter = false;
    bool isDerivative = false;
};

class ModelInfo {
public:
    std::string packageName;
    std::string modelName;
    std::string description;

    std::vector<Variable> variables;
    std::map<std::string, int> variableIndex;  // name -> index in variables
    std::vector<Equation> equations;
    std::vector<Equation> initialEquations;

    int nextValueReference = 1;

    void addVariable(const Variable& var) {
        variableIndex[var.name] = variables.size();
        variables.push_back(var);
    }

    Variable* findVariable(const std::string& name) {
        auto it = variableIndex.find(name);
        if (it != variableIndex.end()) {
            return &variables[it->second];
        }
        return nullptr;
    }

    const Variable* findVariable(const std::string& name) const {
        auto it = variableIndex.find(name);
        if (it != variableIndex.end()) {
            return &variables[it->second];
        }
        return nullptr;
    }

    std::vector<Variable> getStates() const {
        std::vector<Variable> states;
        for (const auto& var : variables) {
            if (var.variability == "continuous" && !var.isDerivative && var.causality == "local") {
                states.push_back(var);
            }
        }
        return states;
    }

    std::vector<Variable> getDerivatives() const {
        std::vector<Variable> derivatives;
        for (const auto& var : variables) {
            if (var.isDerivative) {
                derivatives.push_back(var);
            }
        }
        return derivatives;
    }

    std::vector<Variable> getOutputs() const {
        std::vector<Variable> outputs;
        for (const auto& var : variables) {
            if (var.causality == "output") {
                outputs.push_back(var);
            }
        }
        return outputs;
    }
};

} // namespace lacemodelica
