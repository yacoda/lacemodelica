#!/usr/bin/env python3
"""
ONNX model validation script for individual test cases.

Usage:
    python validate_onnx_model.py <test_name>

This script validates a single ONNX model against its reference implementation
defined in the corresponding .bmo file.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

# Import test utilities
sys.path.insert(0, str(Path(__file__).parent))
from test_onnx_runtime import parse_onnx_test_from_bmo, convert_float64_to_float32, parse_onnx_structure_assertions


def validate_onnx_model(test_name: str, verbose: bool = False) -> bool:
    """
    Validate an ONNX model against its reference implementation.

    Args:
        test_name: Name of the test (e.g., 'ArithmeticOps')
        verbose: Print detailed debug information

    Returns:
        True if validation passed, False otherwise
    """
    # Parse .bmo file (search recursively in testfiles)
    testfiles_dir = Path(__file__).parent / 'testfiles'
    bmo_matches = list(testfiles_dir.rglob(f'{test_name}.bmo'))
    if not bmo_matches:
        print(f'Test file not found: {testfiles_dir / f"{test_name}.bmo"}')
        return False
    bmo_path = bmo_matches[0]

    # Check for @onnx-test-known-failure annotation
    with open(bmo_path, 'r') as f:
        content = f.read()
    if '@onnx-test-known-failure' in content and 'validate' in content.split('@onnx-test-known-failure')[1].split('\n')[0]:
        print(f'SKIPPED: {test_name} has @onnx-test-known-failure(validate) annotation')
        return True  # Return success so test passes

    ref_impl, test_cases = parse_onnx_test_from_bmo(bmo_path)

    if not ref_impl or not test_cases:
        print(f'No reference implementation or test cases found in {bmo_path}')
        return False

    # Load ONNX model - try float64 first, fall back to float32 conversion if needed
    onnx_path = Path(__file__).parent / 'output' / f'{test_name}_fmu' / 'extra' / 'org.lacemodelica.ls-onnx-serialization' / 'model.onnx'
    if not onnx_path.exists():
        print(f'ONNX file not found: {onnx_path}')
        return False

    model = onnx.load(str(onnx_path))
    use_float32 = False
    try:
        # Try loading with float64 first
        session = ort.InferenceSession(model.SerializeToString())
    except Exception:
        # Fall back to float32 conversion if needed (e.g., for trig ops)
        model_f32 = convert_float64_to_float32(model)
        session = ort.InferenceSession(model_f32.SerializeToString())
        use_float32 = True
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    # Debug: Print ONNX graph structure
    if verbose:
        print(f'\n=== ONNX Model for {test_name} ===')
        print(f'Inputs: {input_names}')
        print(f'Outputs: {output_names}')
        print(f'\nGraph nodes:')
        for i, node in enumerate(model.graph.node):
            print(f'  {i}: {node.op_type} ({node.name})')
            print(f'      inputs: {list(node.input)}')
            print(f'      outputs: {list(node.output)}')
        print()

    # Check structure assertions (node presence/absence)
    structure_assertions = parse_onnx_structure_assertions(bmo_path)
    node_types = {node.op_type for node in model.graph.node}

    for forbidden_type in structure_assertions['assert_no_nodes']:
        if forbidden_type in node_types:
            print(f'Structure assertion FAILED: Found forbidden node type "{forbidden_type}" in graph')
            return False

    for required_type in structure_assertions['assert_has_nodes']:
        if required_type not in node_types:
            print(f'Structure assertion FAILED: Required node type "{required_type}" not found in graph')
            return False

    # Run all test cases
    passed = True
    for i, test_case in enumerate(test_cases, 1):
        onnx_inputs = {}
        for k, v in test_case.items():
            if k not in input_names:
                continue
            # Convert to appropriate dtype - bool stays bool, others become float32/64
            if v.dtype == np.bool_:
                onnx_inputs[k] = v
            elif use_float32:
                onnx_inputs[k] = v.astype(np.float32)
            else:
                onnx_inputs[k] = v.astype(np.float64)

        onnx_outputs = session.run(output_names, onnx_inputs)
        onnx_results = dict(zip(output_names, onnx_outputs))
        ref_results = ref_impl(test_case)

        # Check for missing equation outputs (eq[X])
        # Only validate equation residuals, not start values or init equations
        ref_eq_outputs = [name for name in ref_results.keys() if name.startswith('eq[')]
        missing_outputs = set(ref_eq_outputs) - set(output_names)
        if missing_outputs:
            print(f'Test {i} FAILED: Missing equation outputs in ONNX model: {sorted(missing_outputs)}')
            passed = False

        for name in output_names:
            if name in ref_results:
                onnx_val = np.array(onnx_results[name])
                ref_val = np.array(ref_results[name])

                # Check shapes match
                if onnx_val.shape != ref_val.shape:
                    print(f'Test {i} FAILED: {name}: shape mismatch ONNX={onnx_val.shape} vs Ref={ref_val.shape}')
                    passed = False
                    continue

                # Compare values (handle both boolean and numeric)
                if onnx_val.dtype == np.bool_ or ref_val.dtype == np.bool_:
                    # Boolean comparison - check exact equality
                    if not np.all(onnx_val == ref_val):
                        print(f'Test {i} FAILED: {name}: boolean mismatch ONNX={onnx_val} vs Ref={ref_val}')
                        passed = False
                else:
                    # Numeric comparison - use 1e-4 tolerance to account for float32 precision loss
                    diff = np.abs(onnx_val - ref_val)
                    max_diff = np.max(diff) if diff.size > 0 else abs(diff)
                    if max_diff >= 1e-4:
                        print(f'Test {i} FAILED: {name}: max diff={max_diff:.2e}, shapes={onnx_val.shape}')
                        passed = False

    if passed:
        if verbose:
            print('All tests passed')
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate ONNX model against reference implementation')
    parser.add_argument('test_name', help='Name of the test to validate (e.g., ArithmeticOps)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed debug information')

    args = parser.parse_args()

    success = validate_onnx_model(args.test_name, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
