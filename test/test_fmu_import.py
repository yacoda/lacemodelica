#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Joris Gillis, YACODA

"""
Test FMU imports using CasADi DaeBuilder.
Loads each generated FMU from test/output/ and validates it.
"""

import os
import sys
from pathlib import Path
from casadi import *

def test_fmu(fmu_path):
    """Test loading a single FMU with CasADi DaeBuilder."""
    fmu_name = os.path.basename(fmu_path)
    print(f"Testing {fmu_name}...", end=" ", flush=True)

    try:
        # Load FMU with DaeBuilder
        dae = DaeBuilder('model', str(fmu_path))
        dae.disp(True)

        # Print basic info
        print(f"OK")
        print(f"  Variables: {dae.nx()} states, {dae.nz()} algebraics, {dae.np()} parameters")

        return True
    except Exception as e:
        print(f"FAILED")
        print(f"  Error: {e}")
        return False

def main():
    """Run FMU import tests."""
    print("=" * 60)
    print("Testing FMU imports with CasADi DaeBuilder")
    print("=" * 60)
    print()

    # Find test output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Run the C++ tests first to generate FMUs.")
        return 1

    # Find all FMU directories
    fmu_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.endswith("_fmu")])

    if not fmu_dirs:
        print(f"No FMU directories found in {output_dir}")
        return 1

    print(f"Found {len(fmu_dirs)} FMU(s) to test:\n")

    # Test each FMU
    passed = 0
    failed = 0

    for fmu_dir in fmu_dirs:
        if test_fmu(fmu_dir):
            passed += 1
        else:
            failed += 1
        print()

    # Summary
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
