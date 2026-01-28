![lacemodelica logo](docs/logo.svg)

# lacemodelica

A C++17 parser and compiler for BaseModelica that generates Model-Exchange Functional Mock-up Units (FMU v3.0) with symbolic representations in ONNX format.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Idea in two slides

![Slide 1 - BaseModelica to FMU/ONNX conversion](docs/slide-1.png)

![Slide 2 - Advanced features with functions and loops](docs/slide-2.png)

## Features

- **BaseModelica Parser**: Full ANTLR4-based parser for BaseModelica language
- **FMU Generation**: Target output format for Functional Mock-up Units
- **ONNX Symbolic Representation**: Layered standard for symbolic math
- **C++17**: Modern C++ implementation with minimal dependencies

## Installation

### Prerequisites

- CMake 3.14 or higher
- C++17 compatible compiler (GCC 11+, Clang 10+, MSVC 2019+)
- Internet connection (for downloading ANTLR4 runtime)

### Building from Source

```bash
git clone https://github.com/yourusername/lacemodelica.git
cd lacemodelica
mkdir build && cd build
cmake ..
make -j4
```

The executable will be available at `build/lacemodelica`.

### Developer Build (Regenerating Parser)

If you modify the grammar file, regenerate the parser with:

```bash
cmake -DREGENERATE_PARSER=ON ..
make -j4
```

**Note**: Parser regeneration requires Java (for ANTLR4 JAR).

## Usage

### Parse All Test Files

```bash
./build/lacemodelica
```

### Parse a Specific File

```bash
./build/lacemodelica path/to/model.bmo
```

### Example

```bash
./build/lacemodelica_exe test/testfiles/MinimalValid.bmo --output-dir output/MinimalValid_fmu
```

Output:
```
lacemodelica - BaseModelica to FMU/ONNX converter
Processing: test/testfiles/MinimalValid.bmo -> output/MinimalValid_fmu
Generated: output/MinimalValid_fmu/modelDescription.xml
Done.
```

## C API

The shared library exposes a C API for easy integration from any language with C FFI support.

### Header

```c
#include <lacemodelica.h>
```

### Functions

```c
// Process a BMO file and generate FMU with ONNX layered standard
// output_dir is required and specifies where modelDescription.xml will be written
lacemodelica_status_t lacemodelica_process_bmo(const char* input_file, const char* output_dir);

// Parse a BMO file without generating output (validation only)
lacemodelica_status_t lacemodelica_parse_bmo(const char* input_file);

// Get human-readable error message
const char* lacemodelica_status_string(lacemodelica_status_t status);
```

### Status Codes

| Code | Description |
|------|-------------|
| `LACEMODELICA_SUCCESS` | Operation completed successfully |
| `LACEMODELICA_ERROR_FILE_NOT_FOUND` | Input file not found |
| `LACEMODELICA_ERROR_PARSE_FAILED` | Parsing failed (syntax error) |
| `LACEMODELICA_ERROR_FMU_GENERATION_FAILED` | FMU generation failed |
| `LACEMODELICA_ERROR_ONNX_GENERATION_FAILED` | ONNX generation failed |
| `LACEMODELICA_ERROR_OUTPUT_DIR_CREATION_FAILED` | Could not create output directory |

### Example

```c
#include <stdio.h>
#include <lacemodelica.h>

int main() {
    lacemodelica_status_t status = lacemodelica_process_bmo("model.bmo", "model_fmu");
    if (status != LACEMODELICA_SUCCESS) {
        fprintf(stderr, "Error: %s\n", lacemodelica_status_string(status));
        return 1;
    }
    return 0;
}
```

## ONNX Output Structure

The generated ONNX model represents the mathematical equations of a Modelica model as a computational graph. The model has multiple outputs representing different aspects of the equation system.

### Output Categories

| Category | Format | Description |
|----------|--------|-------------|
| **Equation Residuals** | `eq[N]` | Residuals for equations in the `equation` section |
| **Initial Equations** | `init_eq[N]` | Residuals for equations in the `initial equation` section |
| **Start Values** | `start[N]` | Initial/default values for variables |
| **Bounds** | `min[N]`, `max[N]` | Variable bounds (only for non-constant expressions) |

### Equation Residuals (`eq[N]`, `init_eq[N]`)

For each equation of the form `lhs = rhs`, the output computes the **residual**: `lhs - rhs`. At the solution, all residuals should equal zero.

**Example** - Given this Modelica model:

```modelica
model 'NewtonCoolingBase'
  parameter Real 'T_inf' = 25.0;
  parameter Real 'T0' = 90.0;
  parameter Real 'h' = 0.7;
  parameter Real 'A' = 1.0;
  parameter Real 'm' = 0.1;
  parameter Real 'c_p' = 1.2;
  Real 'T';
initial equation
  'T' = 'T0';
equation
  'm' * 'c_p' * der('T') = 'h' * 'A' * ('T_inf' - 'T');
end 'NewtonCoolingBase';
```

The ONNX model will have these outputs:

| Output | Represents | Computed As |
|--------|------------|-------------|
| `eq[0]` | Cooling equation residual | `m * c_p * der(T) - h * A * (T_inf - T)` |
| `init_eq[0]` | Initial condition residual | `T - T0` |
| `start[0]` | Initial value of T | `T0` |

### Interpretation

- **Inputs**: All model variables (states, derivatives, parameters) are inputs to the ONNX graph
- **Outputs**: Equation residuals that should all be zero when the system is correctly solved
- **Solver usage**: A numerical solver can use this representation to find values that make all residuals zero

### Testing with ONNX Runtime

You can validate the ONNX output using Python:

```bash
python test/test_onnx_runtime.py
```

This runs the generated ONNX models against reference Python implementations embedded in the `.bmo` test files.

## Project Structure

```
lacemodelica/
├── CMakeLists.txt          # Build configuration
├── LICENSE                 # MIT license
├── README.md              # This file
├── src/
│   └── main.cpp           # Main parser implementation
├── include/               # Header files
├── grammar/
│   └── BaseModelica.g4    # ANTLR4 grammar (from BaseModelica.jl)
├── generated/             # Generated parser code (committed)
│   ├── BaseModelicaLexer.cpp
│   ├── BaseModelicaLexer.h
│   ├── BaseModelicaParser.cpp
│   └── BaseModelicaParser.h
└── test/
    └── testfiles/         # Test suite from BaseModelica.jl
```

## Testing

```bash
cd build
ctest
```

## Dependencies

### Runtime Dependencies

- ANTLR4 C++ Runtime 4.13.2 (automatically fetched via CMake)
- TinyXML-2 10.0.0 (automatically fetched via CMake)

### Build Dependencies

- CMake 3.14+
- C++17 compiler
- Java (only for regenerating parser from grammar)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Components

- **BaseModelica Grammar**: Copyright (c) 2024 Jadon Clugston (MIT License)
  - Source: https://github.com/SciML/BaseModelica.jl
- **ANTLR4**: Copyright (c) 2012-2024 The ANTLR Project (BSD-3-Clause License)
- **TinyXML-2**: Copyright (c) Lee Thomason (Zlib License)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`./build/lacemodelica`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Integrations

### CasADi

LaceModelica can be used as a plugin for [CasADi](https://web.casadi.org/)'s DaeBuilder to load BaseModelica models directly:

- **Nightly builds**: [nightly-lacemodelica](https://github.com/casadi/casadi/releases/tag/nightly-lacemodelica)
- **Source branch**: [casadi/casadi:lacemodelica](https://github.com/casadi/casadi/tree/lacemodelica)
- **Unit tests**: [daebuilder.py](https://github.com/casadi/casadi/blob/3e84acf45522cd156acf07458507cc525c004edc/test/python/daebuilder.py#L823)

## Acknowledgments

- [BaseModelica.jl](https://github.com/SciML/BaseModelica.jl) - Grammar and test files
- [ANTLR4](https://www.antlr.org/) - Parser generator
- [SciML](https://sciml.ai/) - Scientific machine learning ecosystem

## Related Projects

- [Modelica2Pyomo](https://github.com/looms-polimi/Modelica2Pyomo) - Modelica to Pyomo converter for optimization

## Contact

**Joris Gillis** - YACODA

Project Link: [https://github.com/yacoda/lacemodelica](https://github.com/yacoda/lacemodelica)

## Roadmap

- [ ] Complete FMU generation
- [ ] ONNX symbolic representation layer
- [ ] Optimization passes
- [ ] Documentation and examples
- [ ] CI/CD pipeline
- [ ] Package manager integration (vcpkg, conan)


## Grammar Reference

The BaseModelica grammar (`grammar/BaseModelica.g4`) defines the following rules:

### Parser Rules

| Rule | Definition | Examples | Tests |
|------|------------|----------|-------|
| baseModelica | `: versionHeader 'package' IDENT`<br>`  (decoration? classDefinition ';' \| decoration? globalConstant ';')*`<br>`  decoration? 'model' longClassSpecifier ';'`<br>`  (annotationComment ';')?`<br>`  'end' IDENT ';'` | `//! base 0.1.0 package P model M end M; end P;` | [baseModelica01](test/testfiles/units/baseModelica01.bmo) ✅ |
| versionHeader | `: VERSION_HEADER` | `//! base 0.1.0`<br>`//! base 1.2.3` | [versionHeader01](test/testfiles/units/versionHeader01.bmo) ✅<br>[versionHeader02](test/testfiles/units/versionHeader02.bmo) ✅ |
| classDefinition | `: classPrefixes classSpecifier` | `type X = Real`<br>`function foo end foo`<br>`record Point Real x; end Point` | [classDefinition01](test/testfiles/units/classDefinition01.bmo) ✅<br>[classDefinition02](test/testfiles/units/classDefinition02.bmo) ✅<br>[classDefinition03](test/testfiles/units/classDefinition03.bmo) ❌ |
| classPrefixes | `: 'type'`<br>`\| 'operator'? 'record'`<br>`\| (('pure' 'constant'?) \| 'impure')? 'operator'? 'function'` | `type`<br>`record`<br>`operator record`<br>`function`<br>`pure function`<br>`impure function` | [classPrefixes_01](test/testfiles/units/classPrefixes_01.bmo) ✅<br>[classPrefixes_02](test/testfiles/units/classPrefixes_02.bmo) ✅<br>[classPrefixes_03](test/testfiles/units/classPrefixes_03.bmo) ❌ |
| classSpecifier | `: longClassSpecifier`<br>`\| shortClassSpecifier`<br>`\| derClassSpecifier` | *(see long/short/derClassSpecifier)* | |
| longClassSpecifier | `: IDENT stringComment composition 'end' IDENT` | `Foo "a model" Real x; end Foo`<br>`Bar end Bar` | [longClassSpecifier_01](test/testfiles/units/longClassSpecifier_01.bmo) ❌<br>[longClassSpecifier_02](test/testfiles/units/longClassSpecifier_02.bmo) ❌ |
| shortClassSpecifier | `: IDENT '=' (basePrefix? typeSpecifier classModification?`<br>`             \| 'enumeration' '(' (enumList? \| ':') ')') comment` | `X = Real`<br>`Y = input Real`<br>`Color = enumeration(Red, Green, Blue)` | [shortClassSpecifier_01](test/testfiles/units/shortClassSpecifier_01.bmo) ✅ |
| derClassSpecifier | `: IDENT '=' 'der' '(' typeSpecifier ',' IDENT (',' IDENT)* ')' comment` | `dx = der(x, t)`<br>`dydx = der(f, x, y)` | [derClassSpecifier_01](test/testfiles/units/derClassSpecifier_01.bmo) ❌ |
| basePrefix | `: 'input'`<br>`\| 'output'` | `input`<br>`output` | [basePrefix_01](test/testfiles/units/basePrefix_01.bmo) ❌<br>[basePrefix_02](test/testfiles/units/basePrefix_02.bmo) ❌ |
| enumList | `: enumerationLiteral (',' enumerationLiteral)*` | `Red, Green, Blue`<br>`On, Off` | [enumList_01](test/testfiles/units/enumList_01.bmo) ❌ |
| enumerationLiteral | `: IDENT comment` | `Red`<br>`Green "color"` | [enumerationLiteral_01](test/testfiles/units/enumerationLiteral_01.bmo) ❌ |
| composition | `: (decoration? genericElement ';')*`<br>`  ('equation' (equation ';')*`<br>`  \| 'initial' 'equation' (initialEquation ';')*`<br>`  \| 'initial'? 'algorithm' (statement ';')*)*`<br>`  (decoration? 'external' languageSpecification? externalFunctionCall? annotationComment? ';')?`<br>`  basePartition*`<br>`  (annotationComment ';')?` | `Real x; equation x = 1;`<br>`Real y; initial equation y = 0;` | [composition_01](test/testfiles/units/composition_01.bmo) ✅ |
| languageSpecification | `: STRING` | `"C"`<br>`"FORTRAN 77"` | [languageSpecification_01](test/testfiles/units/languageSpecification_01.bmo) ❌ |
| externalFunctionCall | `: (componentReference '=')? IDENT '(' expressionList? ')'` | `sin(x)`<br>`y = cos(z)`<br>`myFunc()` | [externalFunctionCall_01](test/testfiles/units/externalFunctionCall_01.bmo) ❌ |
| genericElement | `: normalElement`<br>`\| parameterEquation` | *(see normalElement, parameterEquation)* | |
| normalElement | `: componentClause` | *(see componentClause)* | |
| parameterEquation | `: 'parameter' 'equation' guessValue '=' (expression \| prioritizeExpression) comment` | `parameter equation guess(x) = 1.0`<br>`parameter equation guess(T) = prioritize(300, 1)` | [parameterEquation_01](test/testfiles/units/parameterEquation_01.bmo) ✅ |
| guessValue | `: 'guess' '(' componentReference ')'` | `guess(x)`<br>`guess(obj.val)` | [parameterEquation_01](test/testfiles/units/parameterEquation_01.bmo) ✅ |
| basePartition | `: 'partition' stringComment (annotationComment ';')?`<br>`  (clockClause ';')* subPartition*` | `partition "main"` | [basePartition_01](test/testfiles/units/basePartition_01.bmo) ✅ |
| subPartition | `: 'subpartition' '(' argumentList ')' stringComment (annotationComment ';')?`<br>`  ('equation' (equation ';')* \| 'algorithm' (statement ';')*)*` | `subpartition(solver=ImplicitEuler) equation x=1;` | [subPartition_01](test/testfiles/units/subPartition_01.bmo) ✅ |
| clockClause | `: decoration? 'Clock' IDENT '=' expression comment` | `Clock clk = 0.1`<br>`@1 Clock fastClk = 0.01` | [clockClause_01](test/testfiles/units/clockClause_01.bmo) ✅ |
| componentClause | `: typePrefix typeSpecifier componentList` | `Real x`<br>`parameter Integer n = 5`<br>`input Real u, v` | [componentClause_01](test/testfiles/units/componentClause_01.bmo) ✅<br>[componentClause_02](test/testfiles/units/componentClause_02.bmo) ✅ |
| globalConstant | `: 'constant' typeSpecifier arraySubscripts? declaration comment` | `constant Real pi = 3.14159`<br>`constant Integer[3] dims = {1,2,3}` | [globalConstant_01](test/testfiles/units/globalConstant_01.bmo) ✅ |
| typePrefix | `: ('discrete' \| 'parameter' \| 'constant')? ('input' \| 'output')?` | *(empty)*<br>`parameter`<br>`discrete`<br>`input`<br>`constant output` | [typePrefix_01](test/testfiles/units/typePrefix_01.bmo) ✅<br>[typePrefix_02](test/testfiles/units/typePrefix_02.bmo) ✅ |
| componentList | `: componentDeclaration (',' componentDeclaration)*` | `x`<br>`x, y, z`<br>`a = 1, b = 2` | [componentList_01](test/testfiles/units/componentList_01.bmo) ✅ |
| componentDeclaration | `: declaration comment` | `x`<br>`pos = 0 "position"` | [componentDeclaration_01](test/testfiles/units/componentDeclaration_01.bmo) ✅ |
| declaration | `: IDENT arraySubscripts? modification?` | `x`<br>`arr[3]`<br>`val = 5`<br>`mat[2,3]` | [declaration_01](test/testfiles/units/declaration_01.bmo) ✅<br>[declaration_02](test/testfiles/units/declaration_02.bmo) ❌ |
| modification | `: classModification ('=' expression)?`<br>`\| '=' expression`<br>`\| ':=' expression` | `= 5`<br>`:= x + 1`<br>`(start=0) = 1`<br>`(fixed=true)` | [modification_01](test/testfiles/units/modification_01.bmo) ✅ |
| classModification | `: '(' argumentList? ')'` | `()`<br>`(start=0)`<br>`(min=0, max=100)` | [classModification_01](test/testfiles/units/classModification_01.bmo) ✅ |
| argumentList | `: argument (',' argument)*` | `start=0`<br>`start=0, fixed=true` | [argumentList_01](test/testfiles/units/argumentList_01.bmo) ✅ |
| argument | `: decoration? elementModificationOrReplaceable` | `start=0`<br>`@1 fixed=true` | |
| elementModificationOrReplaceable | `: elementModification` | *(see elementModification)* | |
| elementModification | `: name modification? stringComment` | `start = 0`<br>`fixed`<br>`unit = "m/s" "velocity"` | [elementModification_01](test/testfiles/units/elementModification_01.bmo) ✅ |
| equation | `: decoration? (simpleExpression decoration? ('=' expression)?`<br>`              \| ifEquation`<br>`              \| forEquation`<br>`              \| whenEquation) comment` | `x = y + 1`<br>`der(x) = -x`<br>`@1 a = b` | [equation_01](test/testfiles/units/equation_01.bmo) ✅<br>[equation_02](test/testfiles/units/equation_02.bmo) ✅ |
| initialEquation | `: equation`<br>`\| prioritizeEquation` | `x = 0`<br>`prioritize(x, 1)` | [initialEquation_01](test/testfiles/units/initialEquation_01.bmo) ✅ |
| statement | `: decoration? (componentReference (':=' expression \| functionCallArgs)`<br>`              \| '(' outputExpressionList ')' ':=' componentReference functionCallArgs`<br>`              \| 'break'`<br>`              \| 'return'`<br>`              \| ifStatement`<br>`              \| forStatement`<br>`              \| whileStatement`<br>`              \| whenStatement) comment` | `x := 5`<br>`print("hi")`<br>`(a, b) := foo()`<br>`return`<br>`break` | [statement_01](test/testfiles/units/statement_01.bmo) ✅ |
| ifEquation | `: 'if' expression 'then' equationBlock`<br>`  ('elseif' expression 'then' equationBlock)*`<br>`  ('else' equationBlock)?`<br>`  'end' 'if'` | `if x > 0 then y = 1; end if`<br>`if a then b=1; else b=0; end if` | [ifEquation_01](test/testfiles/units/ifEquation_01.bmo) ✅ |
| equationBlock | `: (equation ';')*` | `x = 1;`<br>`x = 1; y = 2;` | [equationBlock_01](test/testfiles/units/equationBlock_01.bmo) ❌ |
| ifStatement | `: 'if' expression 'then' statementBlock`<br>`  ('elseif' expression 'then' statementBlock)*`<br>`  ('else' statementBlock)?`<br>`  'end' 'if'` | `if x > 0 then y := 1; end if`<br>`if a then b:=1; else b:=0; end if` | [ifStatement_01](test/testfiles/units/ifStatement_01.bmo) ✅ |
| statementBlock | `: (statement ';')*` | `x := 1;`<br>`x := 1; y := 2;` | [statementBlock_01](test/testfiles/units/statementBlock_01.bmo) ❌ |
| forEquation | `: 'for' forIndex 'loop' (equation ';')* 'end' 'for'` | `for i in 1:n loop x[i] = i; end for` | [forEquation_01](test/testfiles/units/forEquation_01.bmo) ❌ |
| forStatement | `: 'for' forIndex 'loop' (statement ';')* 'end' 'for'` | `for i in 1:n loop x[i] := i; end for` | [forStatement_01](test/testfiles/units/forStatement_01.bmo) ❌ |
| forIndex | `: IDENT 'in' expression` | `i in 1:10`<br>`k in arr`<br>`idx in {1,2,3}` | [forIndex_01](test/testfiles/units/forIndex_01.bmo) ❌ |
| whileStatement | `: 'while' expression 'loop' (statement ';')* 'end' 'while'` | `while x > 0 loop x := x-1; end while` | [whileStatement_01](test/testfiles/units/whileStatement_01.bmo) ❌ |
| whenEquation | `: 'when' expression 'then' (equation ';')*`<br>`  ('elsewhen' expression 'then' (equation ';')*)*`<br>`  'end' 'when'` | `when x > 0 then y = 1; end when`<br>`when e1 then a=1; elsewhen e2 then a=2; end when` | [whenEquation_01](test/testfiles/units/whenEquation_01.bmo) ❌ |
| whenStatement | `: 'when' expression 'then' (statement ';')*`<br>`  ('elsewhen' expression 'then' (statement ';')*)*`<br>`  'end' 'when'` | `when x > 0 then y := 1; end when` | [whenStatement_01](test/testfiles/units/whenStatement_01.bmo) ❌ |
| prioritizeEquation | `: 'prioritize' '(' componentReference ',' priority ')'` | `prioritize(x, 1)`<br>`prioritize(obj.val, 2)` | [prioritizeEquation_01](test/testfiles/units/prioritizeEquation_01.bmo) ✅ |
| prioritizeExpression | `: 'prioritize' '(' expression ',' priority ')'` | `prioritize(x + 1, 2)`<br>`prioritize(sin(t), 1)` | [prioritizeExpression_01](test/testfiles/units/prioritizeExpression_01.bmo) ✅ |
| priority | `: expression` | `1`<br>`n + 1` | [priority_01](test/testfiles/units/priority_01.bmo) ✅ |
| decoration | `: '@' UNSIGNED_INTEGER` | `@1`<br>`@42`<br>`@0` | [decoration_01](test/testfiles/units/decoration_01.bmo) ✅ |
| expression | `: expressionNoDecoration decoration?` | `x + 1`<br>`x @1`<br>`a * b @2` | [expression_01](test/testfiles/units/expression_01.bmo) ✅ |
| expressionNoDecoration | `: simpleExpression`<br>`\| ifExpression` | `x + 1`<br>`if a then b else c` | |
| ifExpression | `: 'if' expressionNoDecoration 'then' expressionNoDecoration`<br>`  ('elseif' expressionNoDecoration 'then' expressionNoDecoration)*`<br>`  'else' expressionNoDecoration` | `if x > 0 then 1 else -1`<br>`if a then b elseif c then d else e` | [ifExpression_01](test/testfiles/units/ifExpression_01.bmo) ✅<br>[ifExpression_02](test/testfiles/units/ifExpression_02.bmo) ✅ |
| simpleExpression | `: logicalExpression (':' logicalExpression (':' logicalExpression)?)?` | `x + 1`<br>`1:10`<br>`1:2:10` | [simpleExpression_01](test/testfiles/units/simpleExpression_01.bmo) ✅<br>[simpleExpression_02](test/testfiles/units/simpleExpression_02.bmo) ❌<br>[simpleExpression_03](test/testfiles/units/simpleExpression_03.bmo) ❌ |
| logicalExpression | `: logicalTerm ('or' logicalTerm)*` | `a or b`<br>`x > 0 or y < 0` | [logicalExpression_01](test/testfiles/units/logicalExpression_01.bmo) ✅<br>[logicalExpression_02](test/testfiles/units/logicalExpression_02.bmo) ✅ |
| logicalTerm | `: logicalFactor ('and' logicalFactor)*` | `a and b`<br>`x > 0 and y > 0` | [logicalTerm_01](test/testfiles/units/logicalTerm_01.bmo) ✅<br>[logicalTerm_02](test/testfiles/units/logicalTerm_02.bmo) ✅ |
| logicalFactor | `: 'not'? relation` | `x > 0`<br>`not done`<br>`not (a and b)` | [logicalFactor_01](test/testfiles/units/logicalFactor_01.bmo) ✅<br>[logicalFactor_02](test/testfiles/units/logicalFactor_02.bmo) ✅<br>[logicalFactor_03](test/testfiles/units/logicalFactor_03.bmo) ✅ |
| relation | `: arithmeticExpression (relationalOperator arithmeticExpression)?` | `x`<br>`x < 5`<br>`a == b` | [relation_01](test/testfiles/units/relation_01.bmo) ✅<br>[relation_02](test/testfiles/units/relation_02.bmo) ✅<br>[relation_03](test/testfiles/units/relation_03.bmo) ✅ |
| relationalOperator | `: '<' \| '<=' \| '>' \| '>=' \| '==' \| '<>'` | `<`<br>`<=`<br>`>`<br>`>=`<br>`==`<br>`<>` | [relationalOperator_01](test/testfiles/units/relationalOperator_01.bmo) ✅<br>[relationalOperator_02](test/testfiles/units/relationalOperator_02.bmo) ✅ |
| arithmeticExpression | `: addOperator? term (addOperator term)*` | `x + y`<br>`-x`<br>`a - b + c` | [arithmeticExpression_01](test/testfiles/units/arithmeticExpression_01.bmo) ✅<br>[arithmeticExpression_02](test/testfiles/units/arithmeticExpression_02.bmo) ✅ |
| addOperator | `: '+' \| '-' \| '.+' \| '.-'` | `+`<br>`-`<br>`.+`<br>`.-` | [addOperator_01](test/testfiles/units/addOperator_01.bmo) ✅ |
| term | `: factor (mulOperator factor)*` | `x * y`<br>`a / b`<br>`x * y / z` | [term_01](test/testfiles/units/term_01.bmo) ✅ |
| mulOperator | `: '*' \| '/' \| '.*' \| './'` | `*`<br>`/`<br>`.*`<br>`./` | [mulOperator_01](test/testfiles/units/mulOperator_01.bmo) ✅ |
| factor | `: primary (('^' \| '.^') primary)?` | `x`<br>`x^2`<br>`a.^b` | [factor_01](test/testfiles/units/factor_01.bmo) ✅<br>[factor_02](test/testfiles/units/factor_02.bmo) ✅<br>[factor_03](test/testfiles/units/factor_03.bmo) ❌ |
| primary | `: UNSIGNED_NUMBER`<br>`\| STRING`<br>`\| 'false'`<br>`\| 'true'`<br>`\| ('der' \| 'initial' \| 'pure') functionCallArgs`<br>`\| componentReference functionCallArgs?`<br>`\| '(' outputExpressionList ')' arraySubscripts?`<br>`\| '[' expressionList (';' expressionList)* ']'`<br>`\| '{' arrayArguments '}'`<br>`\| 'end'` | `42`<br>`3.14`<br>`"hello"`<br>`true`<br>`false`<br>`der(x)`<br>`sin(x)`<br>`{1, 2, 3}`<br>`[1, 2; 3, 4]`<br>`(x, y)`<br>`end` | [primary_01](test/testfiles/units/primary_01.bmo) ✅<br>[primary_02](test/testfiles/units/primary_02.bmo) ✅ |
| typeSpecifier | `: '.'? name` | `Real`<br>`Integer`<br>`.Modelica.SIunits.Time` | [typeSpecifier_01](test/testfiles/units/typeSpecifier_01.bmo) ✅ |
| name | `: IDENT ('.' IDENT)*` | `Real`<br>`Modelica.SIunits.Time` | [name_01](test/testfiles/units/name_01.bmo) ❌ |
| componentReference | `: '.'? IDENT arraySubscripts? ('.' IDENT arraySubscripts?)*` | `x`<br>`arr[1]`<br>`obj.field`<br>`.global.var` | [componentReference_01](test/testfiles/units/componentReference_01.bmo) ✅ |
| functionCallArgs | `: '(' functionArguments? ')'` | `()`<br>`(x)`<br>`(x, y)`<br>`(n=5)` | [functionCallArgs_01](test/testfiles/units/functionCallArgs_01.bmo) ✅ |
| functionArguments | `: expression (',' functionArgumentsNonFirst \| 'for' forIndex)?`<br>`\| functionPartialApplication (',' functionArgumentsNonFirst)?`<br>`\| namedArguments` | `x, y`<br>`i for i in 1:n`<br>`n=5` | |
| functionArgumentsNonFirst | `: functionArgument (',' functionArgumentsNonFirst)?`<br>`\| namedArguments` | `y, z`<br>`n=5, m=10` | |
| arrayArguments | `: expression ((',' expression)* \| 'for' forIndex)` | `1, 2, 3`<br>`i^2 for i in 1:n` | [arrayArguments_01](test/testfiles/units/arrayArguments_01.bmo) ❌ |
| namedArguments | `: namedArgument (',' namedArgument)*` | `start=0`<br>`start=0, fixed=true` | [namedArguments_01](test/testfiles/units/namedArguments_01.bmo) ✅ |
| namedArgument | `: IDENT '=' functionArgument` | `start = 0`<br>`n = 5` | [namedArgument_01](test/testfiles/units/namedArgument_01.bmo) ✅ |
| functionArgument | `: functionPartialApplication`<br>`\| expression` | `x`<br>`x + 1`<br>`function sin` | [functionArgument_01](test/testfiles/units/functionArgument_01.bmo) ✅ |
| functionPartialApplication | `: 'function' typeSpecifier '(' namedArguments? ')'` | `function sin()`<br>`function Modelica.Math.atan2(y=1)` | [functionPartialApplication_01](test/testfiles/units/functionPartialApplication_01.bmo) ❌ |
| outputExpressionList | `: expression? (',' expression?)*` | `x`<br>`x, y`<br>`x, , z`<br>`, y` | [outputExpressionList_01](test/testfiles/units/outputExpressionList_01.bmo) ✅ |
| expressionList | `: expression (',' expression)*` | `1, 2, 3`<br>`x, y` | [expressionList_01](test/testfiles/units/expressionList_01.bmo) ❌ |
| arraySubscripts | `: '[' subscript (',' subscript)* ']'` | `[1]`<br>`[1, 2]`<br>`[:]`<br>`[:, 1]` | [arraySubscripts_01](test/testfiles/units/arraySubscripts_01.bmo) ✅ |
| subscript | `: ':'`<br>`\| expression` | `:`<br>`1`<br>`n+1` | [subscript_01](test/testfiles/units/subscript_01.bmo) ✅ |
| comment | `: stringComment annotationComment?` | *(empty)*<br>`"description"`<br>`"doc" annotation(Icon())` | [comment_01](test/testfiles/units/comment_01.bmo) ✅ |
| stringComment | `: (STRING ('+' STRING)*)?` | *(empty)*<br>`"a string"`<br>`"part1" + "part2"` | [stringComment_01](test/testfiles/units/stringComment_01.bmo) ✅ |
| annotationComment | `: 'annotation' classModification` | `annotation()`<br>`annotation(Icon(graphics={}))` | [annotationComment_01](test/testfiles/units/annotationComment_01.bmo) ✅ |

### Lexer Rules

| Rule | Definition | Examples | Tests |
|------|------------|----------|-------|
| VERSION_HEADER | `: '//!' ' ' 'base' ' ' [0-9]+ '.' [0-9]+ ('.' [0-9]+)? ~[\r\n]*` | `//! base 0.1.0`<br>`//! base 1.2.3 some extra text` | [versionHeader01](test/testfiles/units/versionHeader01.bmo) ✅<br>[versionHeader02](test/testfiles/units/versionHeader02.bmo) ✅ |
| IDENT | `: NONDIGIT (DIGIT \| NONDIGIT)*`<br>`\| Q_IDENT` | `x`<br>`myVar`<br>`_temp`<br>`'special name'`<br>`'has spaces'` | [IDENT_01](test/testfiles/units/IDENT_01.bmo) ✅<br>[IDENT_02](test/testfiles/units/IDENT_02.bmo) ✅ |
| UNSIGNED_NUMBER | `: DIGIT+ ('.' DIGIT*)? EXPONENT?` | `42`<br>`3.14`<br>`1e-5`<br>`2.5E10`<br>`0.` | [UNSIGNED_NUMBER_01](test/testfiles/units/UNSIGNED_NUMBER_01.bmo) ✅ |
| UNSIGNED_INTEGER | `: DIGIT+` | `0`<br>`42`<br>`100` | [UNSIGNED_INTEGER_01](test/testfiles/units/UNSIGNED_INTEGER_01.bmo) ✅ |
| STRING | `: '"' (S_CHAR \| S_ESCAPE)* '"'` | `"hello"`<br>`"line1\nline2"`<br>`""` | [STRING_01](test/testfiles/units/STRING_01.bmo) ✅ |
| WS | `: ([ \t] \| NL)+ -> skip` | *(whitespace, skipped)* |
| LINE_COMMENT | `: '//' ~[!\r\n] ~[\r\n]* (NL \| EOF) -> skip` | `// this is a comment` | [LINE_COMMENT_01](test/testfiles/units/LINE_COMMENT_01.bmo) ✅ |
| ML_COMMENT | `: '/*' .*? '*/' -> skip` | `/* comment */`<br>`/* multi\nline */` | [ML_COMMENT_01](test/testfiles/units/ML_COMMENT_01.bmo) ✅ |
