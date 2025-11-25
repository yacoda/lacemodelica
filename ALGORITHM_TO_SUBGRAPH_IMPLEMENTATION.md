# Algorithm to ONNX Subgraph Implementation

## High-Level Overview

### Goal
Transform Modelica functions containing `algorithm` sections into ONNX computational subgraphs. This enables symbolic representation of imperative algorithm code as dataflow graphs.

### Core Concept
Instead of preserving the imperative style of algorithms (sequential assignment statements), we trace through the algorithm execution symbolically to build a dataflow graph:

1. **Parse algorithm statements** - Extract assignment statements `lhs := rhs` from functions
2. **Maintain execution map** - Keep an ordered map from variable names to their current ONNX node/tensor
3. **Trace through assignments** - For each `lhs := rhs`:
   - Evaluate RHS expression by substituting variable references with their current ONNX nodes from the map
   - Create new ONNX nodes for the RHS computation
   - Store resulting tensor in the map at the LHS variable name
4. **Extract outputs** - After processing all statements, output variables contain fully traced expressions in terms of inputs
5. **Create ONNX subgraph** - Package the traced computation as an ONNX subgraph
6. **Function calls** - When the main model calls this function, reference the subgraph

### Example
For the polynomial function in `FunctionInputOutput.bmo`:
```modelica
algorithm
  'sub_result' := 'x' * 'x';           // Step 1: map["sub_result"] = Mul(x, x)
  'sub_result' := 'sub_result'*'a';    // Step 2: map["sub_result"] = Mul(Mul(x,x), a)
  'result' := 'sub_result' + 'b' * 'x' + 'c';  // Step 3: map["result"] = Add(Add(Mul(Mul(x,x),a), Mul(b,x)), c)
```

The final `map["result"]` contains the complete computational graph: `((x * x) * a) + (b * x) + c`

### Key Design Decision: Store ONNX Nodes Directly
We store ONNX node outputs (tensor names) in the map rather than AST nodes. This provides:
- **Single-pass conversion** - Build graph while tracing, no second pass needed
- **Direct graph wiring** - When referencing a variable, we already have the ONNX tensor to connect
- **Less memory** - No intermediate representation
- **Simpler code** - No AST→ONNX translation layer

### Preserving Algorithm Structure through Metadata
Even though we convert the imperative algorithm to a dataflow graph, we preserve traceability by attaching metadata to each ONNX node:

**Metadata attached to each node:**
- **`source_file`** - Source filename where the algorithm statement appears
- **`source_line`** - Line number of the statement in the source file
- **`statement_index`** - Index of the statement within the algorithm (0-based)
- **`lhs_variable`** - Variable name being assigned (left-hand side of `:=`)

**Note:** Function name is not included in per-node metadata since the subgraph itself is already associated with the function.

**Benefits:**
- **Debugging** - Trace ONNX nodes back to original algorithm statements
- **Visualization** - Tools can reconstruct and display the original algorithm structure
- **Error reporting** - Provide meaningful error messages with source locations
- **Analysis** - Understand dataflow dependencies in context of the original algorithm

**Example:**
For the statement `'sub_result' := 'sub_result' * 'a';` at line 11 (statement index 1), the generated Mul node would have:
```
metadata_props {
  key: "source_file"
  value: "FunctionInputOutput.bmo"
}
metadata_props {
  key: "source_line"
  value: "11"
}
metadata_props {
  key: "statement_index"
  value: "1"
}
metadata_props {
  key: "lhs_variable"
  value: "sub_result"
}
```

This metadata allows tools to reconstruct the algorithmic view even though the underlying representation is a pure dataflow graph.

---

## Technical Approach

### Phase 1: Extract Functions and Algorithms
- Parse `function` definitions from the BaseModelica AST
- Extract input/output variable declarations
- Extract `algorithm` section statements
- Store statement AST nodes (preserve `lhsContext` and `rhsContext`)

### Phase 2: Convert Algorithm to Subgraph
- Create ONNX subgraph with function inputs as graph inputs
- Initialize variable→tensor map with input variable names
- For each algorithm statement:
  - Convert RHS expression to ONNX nodes
  - During conversion, when encountering variable references, use the current tensor from the map
  - Store resulting output tensor in map at LHS variable name
- Mark function output variables as subgraph outputs
- Return subgraph reference

### Phase 3: Handle Function Calls
- When converting expressions in the main model, detect function calls
- If function has an algorithm, reference its subgraph
- Connect call-site arguments to subgraph inputs
- Wire subgraph outputs to the call-site output

---

## Detailed Implementation

### 1. Data Structures (include/ModelInfo.h)

**Location:** After `struct Variable` definition (around line 50)

**Add new structures:**
```cpp
struct Statement {
    antlr4::ParserRuleContext* lhsContext;  // componentReference on left of :=
    antlr4::ParserRuleContext* rhsContext;  // expression on right of :=
    std::string sourceFile;
    size_t sourceLine = 0;
};

struct Function {
    std::string name;
    std::string description;
    std::vector<Variable> inputs;
    std::vector<Variable> outputs;
    std::vector<Statement> algorithmStatements;
    std::string sourceFile;
    size_t sourceLine = 0;
};
```

**Location:** Inside `class ModelInfo` (after line 62)

**Add members and methods:**
```cpp
    std::vector<Function> functions;
    std::map<std::string, int> functionIndex;  // name -> index in functions

    void addFunction(const Function& func) {
        functionIndex[func.name] = functions.size();
        functions.push_back(func);
    }

    Function* findFunction(const std::string& name) {
        auto it = functionIndex.find(name);
        if (it != functionIndex.end()) {
            return &functions[it->second];
        }
        return nullptr;
    }

    const Function* findFunction(const std::string& name) const {
        auto it = functionIndex.find(name);
        if (it != functionIndex.end()) {
            return &functions[it->second];
        }
        return nullptr;
    }
```

---

### 2. Function Extraction - Header (include/ModelInfoExtractor.h)

**Location:** In `class ModelInfoExtractor` private section (after line 23)

**Add method declaration:**
```cpp
    void extractFunctions(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx);
```

---

### 3. Function Extraction - Implementation (src/ModelInfoExtractor.cpp)

**Location A:** In `extract()` method (around line 23, after `extractVariables(tree)`)

**Add call:**
```cpp
    extractFunctions(tree);
```

**Location B:** At end of file (after `extractDimensions()` method, around line 411)

**Add implementation:**
```cpp
void ModelInfoExtractor::extractFunctions(basemodelica::BaseModelicaParser::BaseModelicaContext* ctx) {
    // Iterate through all class definitions to find functions
    for (auto classDefCtx : ctx->classDefinition()) {
        // Check if this is a function
        auto classPrefixes = classDefCtx->classPrefixes();
        if (!classPrefixes) continue;

        std::string prefixText = classPrefixes->getText();
        if (prefixText.find("function") == std::string::npos) {
            continue;  // Not a function
        }

        // Extract function details
        Function func;

        auto classSpec = classDefCtx->classSpecifier();
        if (!classSpec) continue;

        auto longClassSpec = classSpec->longClassSpecifier();
        if (!longClassSpec) continue;

        // Get function name
        if (longClassSpec->IDENT().size() > 0) {
            func.name = stripQuotes(longClassSpec->IDENT(0)->getText());
        }

        // Get description from string comment
        if (longClassSpec->stringComment()) {
            func.description = longClassSpec->stringComment()->getText();
            // Remove quotes
            if (func.description.size() >= 2 &&
                func.description.front() == '"' &&
                func.description.back() == '"') {
                func.description = func.description.substr(1, func.description.size() - 2);
            }
        }

        func.sourceFile = sourceFile;
        func.sourceLine = classDefCtx->getStart()->getLine();

        // Extract function composition (inputs, outputs, algorithm)
        auto composition = longClassSpec->composition();
        if (!composition) continue;

        // Extract input/output variables from genericElements
        for (auto genericElem : composition->genericElement()) {
            if (auto normalElem = genericElem->normalElement()) {
                if (auto compClause = normalElem->componentClause()) {
                    std::string typePrefix = "";
                    if (compClause->typePrefix()) {
                        typePrefix = compClause->typePrefix()->getText();
                    }

                    std::string typeSpec = compClause->typeSpecifier()->getText();

                    for (auto compDecl : compClause->componentList()->componentDeclaration()) {
                        Variable var;
                        var.name = stripQuotes(compDecl->declaration()->IDENT()->getText());
                        var.type = stripQuotes(typeSpec);
                        var.sourceFile = sourceFile;
                        var.sourceLine = compDecl->getStart()->getLine();

                        // Determine if input or output
                        if (typePrefix.find("input") != std::string::npos) {
                            func.inputs.push_back(var);
                        } else if (typePrefix.find("output") != std::string::npos) {
                            func.outputs.push_back(var);
                        }
                    }
                }
            }
        }

        // Extract algorithm statements
        // Look for 'algorithm' keyword in composition
        // According to grammar: composition has ('initial'? 'algorithm' (statement ';')*)*
        // We need to access the raw children to find algorithm sections

        // Use the composition context to find statement() nodes
        for (auto statement : composition->statement()) {
            Statement stmt;
            stmt.sourceFile = sourceFile;
            stmt.sourceLine = statement->getStart()->getLine();

            // According to grammar line 184:
            // statement: decoration? (componentReference (':=' expression | functionCallArgs) | ...)

            // Check if this is an assignment statement (componentReference ':=' expression)
            if (statement->componentReference() && statement->expression()) {
                stmt.lhsContext = statement->componentReference();
                stmt.rhsContext = statement->expression();
                func.algorithmStatements.push_back(stmt);
            }
            // Note: We're skipping other statement types (function calls, if, for, while, etc.)
            // These would need special handling in the future
        }

        info.addFunction(func);

        std::cerr << "Extracted function: " << func.name
                  << " with " << func.inputs.size() << " inputs, "
                  << func.outputs.size() << " outputs, "
                  << func.algorithmStatements.size() << " algorithm statements" << std::endl;
    }
}
```

**Note:** You'll need to use the existing `stripQuotes()` helper function that's already defined at the top of the file.

---

### 4. ONNX Subgraph Generation - Header (include/ONNXGenerator.h)

**Location:** In `class ONNXGenerator` private section (after line 33)

**Add method declaration:**
```cpp
    // Convert function algorithm to ONNX subgraph
    // Returns the subgraph name/identifier
    static std::string convertAlgorithmToSubgraph(
        const Function& func,
        onnx::GraphProto* parentGraph,
        int& nodeCounter
    );
```

---

### 5. ONNX Subgraph Generation - Implementation (src/ONNXGenerator.cpp)

**Location:** After `generateEquationOutputs()` method (around line 436)

**Add implementation:**
```cpp
std::string ONNXGenerator::convertAlgorithmToSubgraph(
    const Function& func,
    onnx::GraphProto* parentGraph,
    int& nodeCounter) {

    std::cerr << "Converting function '" << func.name << "' algorithm to ONNX subgraph" << std::endl;

    // Map from variable name to current ONNX tensor name
    std::map<std::string, std::string> variableToTensor;

    // Initialize map with function inputs
    for (const auto& input : func.inputs) {
        // Input variables map directly to their names
        variableToTensor[input.name] = input.name;
    }

    // Process each algorithm statement in order
    for (size_t stmtIndex = 0; stmtIndex < func.algorithmStatements.size(); stmtIndex++) {
        const auto& stmt = func.algorithmStatements[stmtIndex];

        // Extract LHS variable name from componentReference
        std::string lhsVarName = stmt.lhsContext->getText();
        // Strip quotes if present
        if (lhsVarName.front() == '\'' && lhsVarName.back() == '\'') {
            lhsVarName = lhsVarName.substr(1, lhsVarName.size() - 2);
        }

        std::cerr << "  Processing statement " << stmtIndex << ": " << lhsVarName << " := "
                  << stmt.rhsContext->getText().substr(0, 50) << "..." << std::endl;

        // Convert RHS expression to ONNX nodes
        // This will use variableToTensor map via convertPrimary when it encounters variables
        // We need to pass the map context somehow...
        // For now, we'll use a simpler approach: convert expression normally,
        // then the variables will reference their current tensors

        try {
            // Remember the current node count to identify newly created nodes
            int startNodeCount = parentGraph->node_size();

            std::string rhsTensor = convertExpression(stmt.rhsContext, parentGraph, nodeCounter);

            // Add metadata to all nodes created for this statement
            for (int i = startNodeCount; i < parentGraph->node_size(); i++) {
                auto* node = parentGraph->mutable_node(i);

                // Add source location metadata
                auto* meta_file = node->add_metadata_props();
                meta_file->set_key("source_file");
                meta_file->set_value(stmt.sourceFile);

                auto* meta_line = node->add_metadata_props();
                meta_line->set_key("source_line");
                meta_line->set_value(std::to_string(stmt.sourceLine));

                auto* meta_index = node->add_metadata_props();
                meta_index->set_key("statement_index");
                meta_index->set_value(std::to_string(stmtIndex));

                auto* meta_lhs = node->add_metadata_props();
                meta_lhs->set_key("lhs_variable");
                meta_lhs->set_value(lhsVarName);
            }

            // Store result in map
            variableToTensor[lhsVarName] = rhsTensor;

            std::cerr << "  Mapped " << lhsVarName << " -> " << rhsTensor << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to convert statement for " << lhsVarName;
            if (!stmt.sourceFile.empty()) {
                std::cerr << " (" << stmt.sourceFile << ":" << stmt.sourceLine << ")";
            }
            std::cerr << ": " << e.what() << std::endl;
            throw;
        }
    }

    // At this point, all output variables should be in the map
    // We don't create a separate subgraph structure yet - just ensure outputs exist
    for (const auto& output : func.outputs) {
        auto it = variableToTensor.find(output.name);
        if (it == variableToTensor.end()) {
            std::cerr << "Warning: Output variable " << output.name
                      << " not computed in algorithm" << std::endl;
        } else {
            std::cerr << "  Output " << output.name << " = " << it->second << std::endl;
        }
    }

    // Return the function name as the subgraph identifier
    return func.name;
}
```

**Important Note:** The above implementation has a limitation - the `convertExpression()` method doesn't know about the `variableToTensor` map. For a complete implementation, you have two options:

**Option A (Simpler):** Create wrapper versions of the convert methods that accept the map:
```cpp
static std::string convertExpressionWithMap(
    antlr4::ParserRuleContext* expr,
    onnx::GraphProto* graph,
    int& nodeCounter,
    const std::map<std::string, std::string>& variableToTensor
);
```

**Option B (Thread-local):** Use a thread-local or member variable to pass the context through the conversion methods.

For the initial implementation, **you should use Option A** and add a `variableToTensor` parameter to all the convert methods when called from `convertAlgorithmToSubgraph()`.

---

### 6. Generate Subgraphs in Main Flow (src/ONNXGenerator.cpp)

**Location:** In `generateONNXModel()` method, after line 112 (after initial equation generation)

**Add:**
```cpp
    // Generate ONNX subgraphs for functions with algorithms
    for (const auto& func : info.functions) {
        if (!func.algorithmStatements.empty()) {
            std::cerr << "Generating subgraph for function: " << func.name << std::endl;
            try {
                convertAlgorithmToSubgraph(func, graph, nodeCounter);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to generate subgraph for function " << func.name
                          << ": " << e.what() << std::endl;
            }
        }
    }
```

---

### 7. Handle Function Calls in Expressions (src/ONNXGenerator.cpp)

**Location:** In `convertPrimary()` method, around line 763 (after checking mathFuncMap, before the throw statement)

**Add:**
```cpp
        // Check if this is a user-defined function with algorithm
        // We need access to ModelInfo here - this requires refactoring
        // For now, add a TODO comment and throw an error

        // TODO: Check info.findFunction(funcName)
        // If found and has algorithmStatements:
        //   - Get function arguments from funcCallArgs
        //   - Look up the subgraph created for this function
        //   - Create ONNX node that references the subgraph
        //   - Wire arguments to subgraph inputs
        //   - Return subgraph output tensor

        std::cerr << "Warning: User-defined function calls not yet implemented: " << funcName << std::endl;
```

**Note:** To fully implement function calls, you'll need to:
1. Pass `ModelInfo` to the `convertExpression` methods (requires refactoring signatures)
2. Store subgraph metadata during `convertAlgorithmToSubgraph()`
3. Create proper ONNX subgraph structures (not just inline nodes)
4. Use ONNX's function/subgraph calling mechanism

This is the most complex part and can be implemented in a follow-up phase.

---

## Implementation Strategy

### Phase 1: Basic Extraction (Start Here)
1. Implement data structures in `ModelInfo.h`
2. Implement `extractFunctions()` in `ModelInfoExtractor.cpp`
3. Test that functions are correctly extracted by printing them

### Phase 2: Inline Algorithm Conversion
1. Implement `convertAlgorithmToSubgraph()` that creates inline ONNX nodes (not a separate subgraph yet)
2. This will trace through the algorithm and create the full computation graph
3. Test with `FunctionInputOutput.bmo` - the graph should include the expanded polynomial

### Phase 3: Proper Subgraphs (Future)
1. Refactor to create actual ONNX subgraph/function structures
2. Implement function call mechanism
3. Handle control flow (if/for/while in algorithms)

---

## Testing

### Test File
Use `/home/yacoda/programs/lacemodelica/test/testfiles/FunctionInputOutput.bmo`

### Expected Behavior
After Phase 1:
- Console output should show extracted function `polynomial` with 4 inputs, 1 output, 3 statements

After Phase 2:
- When the main model calls `polynomial('x', 'a', 'b', 'c')`, the ONNX graph should contain the expanded computation
- The model output should include nodes for: x*x, (x*x)*a, b*x, and the final addition

---

## Notes and Considerations

### Variable Scoping
- Algorithm-local variables (like `sub_result`) exist only within the function scope
- The `variableToTensor` map handles this naturally - it's local to the subgraph conversion

### Control Flow
- Initial implementation handles only sequential assignments
- `if`, `for`, `while`, `when` statements in algorithms are not supported yet
- Should throw clear error messages when encountered

### Array Operations
- Arrays in function inputs/outputs need special handling
- Consider shape propagation through the algorithm

### Function Purity
- Assumes functions are pure (no side effects)
- This matches Modelica function semantics

### ONNX Limitations
- ONNX subgraphs have specific constraints
- May need to use ONNX's If/Loop operators for control flow
- Version compatibility considerations (using opset 18)

---

## References

- Grammar definition: `/home/yacoda/programs/lacemodelica/grammar/BaseModelica.g4`
  - Statement rule: line 183-192
  - Function definition: line 33 (classPrefixes with 'function')
  - Algorithm section: line 72 in composition rule

- Existing conversion code: `src/ONNXGenerator.cpp`
  - Expression conversion: lines 438-787
  - Primary conversion (variable references): lines 661-787

- Test file: `/home/yacoda/programs/lacemodelica/test/testfiles/FunctionInputOutput.bmo`
