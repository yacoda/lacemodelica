#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Joris Gillis, YACODA

"""
Test ONNX runtime evaluation against reference Python implementations.
"""

import os
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import json
import re


def parse_onnx_test_from_bmo(bmo_path):
    """
    Parse ONNX test reference implementation and test cases from .bmo file.

    Returns:
        tuple: (reference_implementation_func, test_cases_list)
    """
    with open(bmo_path, 'r') as f:
        content = f.read()

    # Extract Python reference implementation (stop at next @ marker)
    ref_pattern = r'// @onnx-test-reference\n((?:// (?!@).*\n|//\n)+)'
    ref_match = re.search(ref_pattern, content)

    if not ref_match:
        return None, []

    # Extract Python code by removing comment prefixes
    python_lines = []
    lines = ref_match.group(1).split('\n')
    capturing = False

    for line in lines:
        # Stop at next @ marker
        if line.startswith('// @'):
            break

        # Start capturing when we see the def line
        if line.startswith('// def '):
            capturing = True

        if capturing:
            if line.startswith('// '):
                python_lines.append(line[3:])  # Remove '// ' prefix
            elif line == '//':
                python_lines.append('')  # Empty line

    python_code = '\n'.join(python_lines)

    # Execute Python code to get the reference_implementation function
    namespace = {}
    exec(python_code, namespace)
    reference_implementation = namespace.get('reference_implementation')

    # Extract test cases
    test_case_pattern = r'// @onnx-test-case\n// .*\n((?:// .*\n)+)'
    test_cases = []

    for match in re.finditer(test_case_pattern, content):
        # Extract JSON by removing comment prefixes
        json_lines = []
        for line in match.group(1).split('\n'):
            if line.startswith('// '):
                json_lines.append(line[3:])
            elif line == '//':
                json_lines.append('')

        json_str = '\n'.join(json_lines).strip()
        if json_str:
            test_case = json.loads(json_str)
            test_cases.append(test_case)

    return reference_implementation, test_cases


def test_newton_cooling_base():
    """Test NewtonCoolingBase ONNX model against Python reference."""
    print("Testing NewtonCoolingBase...", end=" ", flush=True)

    # Find the .bmo file and parse reference implementation
    script_dir = Path(__file__).parent
    bmo_path = script_dir / "testfiles" / "NewtonCoolingBase.bmo"

    if not bmo_path.exists():
        print("FAILED")
        print(f"  .bmo file not found: {bmo_path}")
        return False

    # Parse reference implementation and test cases from .bmo file
    reference_implementation, test_cases = parse_onnx_test_from_bmo(bmo_path)

    if not reference_implementation:
        print("FAILED")
        print(f"  No @onnx-test-reference found in {bmo_path}")
        return False

    if not test_cases:
        print("FAILED")
        print(f"  No @onnx-test-case found in {bmo_path}")
        return False

    print(f"[{len(test_cases)} test cases]")

    # Find the ONNX file
    output_dir = script_dir / "output" / "NewtonCoolingBase_fmu" / "extra" / "org.lacemodelica.ls-onnx-serialization"
    onnx_path = output_dir / "model.onnx"

    if not onnx_path.exists():
        print("  SKIPPED")
        print(f"  ONNX file not found: {onnx_path}")
        print("  Run the C++ tests first to generate ONNX models.")
        return None

    try:
        # Load ONNX model
        session = ort.InferenceSession(str(onnx_path))

        # Get input/output names
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        print(f"\n  Inputs: {input_names}")
        print(f"  Outputs: {output_names}")

        passed = True

        # Run each test case
        for i, test_case in enumerate(test_cases, 1):
            # Convert test case to numpy arrays
            onnx_inputs = {}
            for key, value in test_case.items():
                if key in input_names:
                    onnx_inputs[key] = np.array([value], dtype=np.float32)

            # Run ONNX model
            onnx_outputs = session.run(output_names, onnx_inputs)
            onnx_results = dict(zip(output_names, onnx_outputs))

            # Run reference
            ref_results = reference_implementation(test_case)

            # Compare results
            print(f"\n  Test case {i}:")
            # Print a few key inputs for context
            if 'T' in test_case:
                der_T_key = "der('T')"
                der_T_val = test_case.get(der_T_key, 'N/A')
                print(f"    T={test_case['T']}, der('T')={der_T_val}")

            print(f"    Comparing ONNX vs Reference:")

            for name in output_names:
                if name in ref_results:
                    onnx_val = onnx_results[name][0] if hasattr(onnx_results[name], '__len__') else onnx_results[name]
                    ref_val = ref_results[name]
                    diff = abs(onnx_val - ref_val)
                    match = "✓" if diff < 1e-5 else "✗"
                    print(f"      {name}: ONNX={onnx_val:.6f}, Ref={ref_val:.6f}, Diff={diff:.2e} {match}")
                    if diff >= 1e-5:
                        passed = False

        if passed:
            print("\n  PASSED")
            return True
        else:
            print("\n  FAILED - mismatches found")
            return False

    except Exception as e:
        print(f"FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run ONNX runtime tests."""
    print("=" * 60)
    print("Testing ONNX Runtime Evaluation")
    print("=" * 60)
    print()

    # Run test
    result = test_newton_cooling_base()

    if result is None:
        print("\nNo tests run (ONNX files not found)")
        return 0

    # Summary
    print("\n" + "=" * 60)
    if result:
        print("Result: PASSED")
    else:
        print("Result: FAILED")
    print("=" * 60)

    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
