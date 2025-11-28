# Grammar Test Coverage Report

This report identifies which grammar constructs from `grammar/BaseModelica.g4` are covered by unit tests in `test/testfiles/`.

## Summary

| Metric | Value |
|--------|-------|
| Total Grammar Rules | 81 |
| Rules with Test Coverage | 46 |
| Rules without Coverage | 35 |
| **Coverage Percentage** | **56.8%** |

---

## Recently Implemented (2025-11-28)

| Construct | Test File | Status |
|-----------|-----------|--------|
| Named arguments | `NamedArguments.bmo` | ✓ Implemented |
| Enumerations | `Enumeration.bmo` | ✓ Implemented |
| Parameter equations | `ParameterEquation.bmo` | ✓ Syntax parsed |
| String comments | `StringComment.bmo` | ✓ Already working |

---

## Remaining High Priority Gaps

| Construct | Grammar Rule | Description |
|-----------|--------------|-------------|
| When equations | `whenEquation` | Event-driven simulation equations |
| When statements | `whenStatement` | Event-driven algorithm statements |
| If statements | `ifStatement` | Procedural if/else in algorithms |

---

## Medium Priority Gaps

| Construct | Grammar Rule | Description |
|-----------|--------------|-------------|
| Partial function application | `functionPartialApplication` | Higher-order functions (requires grammar changes for function-typed parameters) |
| Decorations/Priority | `decoration`, `prioritize` | `@priority` syntax (grammar parsing issue) |

---

## Low Priority Gaps

| Construct | Grammar Rule | Description |
|-----------|--------------|-------------|
| Clock partitions | `clockPartition`, `baseClock` | Synchronous modeling |
| Connect clauses | `connectClause` | Component connections |
| External functions | `externalFunctionCall` | External function declarations |

---

## Coverage by Category

### Excellent Coverage (100%)

- **Expressions**: Arithmetic, logical, relational, array expressions
- **Operators**: All arithmetic and logical operators
- **Type System**: Real, Integer, Boolean, String types
- **References**: Component and type references

### Good Coverage (75%+)

- **Function Calls**: Positional arguments, named arguments
- **Type Definitions**: Enumerations, records

### Partial Coverage (50-75%)

- **Control Flow**: For loops, while loops, if expressions
  - Missing: `ifStatement`, `whenEquation`, `whenStatement`
- **Class Definitions**: Models, functions, records

### Needs Implementation

- **Higher-order Functions**: Partial application, function-typed parameters
- **Decorations**: `@N` annotation syntax
- **Synchronous Features**: Clock partitions

---

## Implementation Details

### Named Arguments (`namedArgument`)

Added support in `ExpressionConverter.cpp`:
- `collectAllFunctionArguments()` - collects positional and named arguments
- `resolveArgumentOrder()` - reorders arguments based on function signature

### Enumerations (`enumList`, `enumerationLiteral`)

Added support in:
- `ModelInfo.h` - `EnumType` and `EnumLiteral` structs
- `ModelInfoExtractor.cpp` - `extractEnumDefinitions()` extracts enum type definitions
- `ExpressionConverter.cpp` - enum literals (`Type.value`) converted to integer constants

---

## Well-Covered Constructs

| Category | Test Count | Examples |
|----------|------------|----------|
| Arithmetic operations | 56 | `+`, `-`, `*`, `/`, `^` |
| Array operations | 47 | Indexing, slicing, comprehensions |
| Type qualifiers | 44+ | `parameter`, `constant`, `input`, `output` |
| For loops | 12 | Iterative constructs |
| If expressions | 9 | Conditional expressions |
| While loops | 4 | While loop constructs |
| Function definitions | 10+ | Algorithm functions, named args |
| Enumerations | 1 | Enum type definitions and literals |

---

## Notes

- Report updated: 2025-11-28
- Grammar file: `grammar/BaseModelica.g4`
- Test directory: `test/testfiles/`
- Test count: 237 tests (227 passing, 10 pre-existing failures)
