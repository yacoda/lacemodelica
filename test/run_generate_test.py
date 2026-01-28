#!/usr/bin/env python3
"""
Wrapper script for generate tests that respects @onnx-test-known-failure annotations.

Usage:
    python run_generate_test.py <lacemodelica_exe> <test_file> --output-dir <dir>
"""

import sys
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 4:
        print("Usage: run_generate_test.py <lacemodelica_exe> <test_file> --output-dir <dir>")
        sys.exit(1)

    exe = sys.argv[1]
    test_file = sys.argv[2]
    remaining_args = sys.argv[3:]

    # Check for @onnx-test-known-failure(generate) annotation
    try:
        with open(test_file, 'r') as f:
            content = f.read()
        if '@onnx-test-known-failure' in content:
            # Check if 'generate' is in the annotation
            for line in content.split('\n'):
                if '@onnx-test-known-failure' in line and 'generate' in line:
                    print(f'SKIPPED: {test_file} has @onnx-test-known-failure(generate) annotation')
                    sys.exit(0)  # Return success
    except Exception as e:
        print(f"Warning: Could not read test file: {e}")

    # Run the actual test
    result = subprocess.run([exe, test_file] + remaining_args)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
