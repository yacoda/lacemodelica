# lacemodelica

A C++17 parser and compiler for BaseModelica that generates Functional Mock-up Units (FMUs) with symbolic representations in ONNX format.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
./build/lacemodelica test/testfiles/MinimalValid.bmo
```

Output:
```
lacemodelica - BaseModelica to FMU/ONNX converter
Parsing: test/testfiles/MinimalValid.bmo
✓ Success
```

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

## Development Status

🚧 **Early Development** - Currently implements:

- ✅ BaseModelica parsing (12/12 test files passing)
- ✅ ANTLR4 integration with C++ runtime
- ✅ Error reporting and validation
- 🚧 FMU generation (planned)
- 🚧 ONNX symbolic output (planned)

## Testing

The project includes 12 test files from the [BaseModelica.jl](https://github.com/SciML/BaseModelica.jl) repository:

```bash
cd build
./lacemodelica
```

All tests currently pass with successful parsing.

## Dependencies

### Runtime Dependencies

- ANTLR4 C++ Runtime 4.13.2 (automatically fetched via CMake)

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

## Acknowledgments

- [BaseModelica.jl](https://github.com/SciML/BaseModelica.jl) - Grammar and test files
- [ANTLR4](https://www.antlr.org/) - Parser generator
- [SciML](https://sciml.ai/) - Scientific machine learning ecosystem

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
