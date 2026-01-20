#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Joris Gillis, YACODA

"""
Test FMU imports using CasADi DaeBuilder.
Loads each generated FMU from test/output/ and validates it.
Skips FMUs that contain structuralParameter (not supported by CasADi).
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from casadi import *
except ImportError:
    print("CasADi not found, skipping FMU import tests")
    sys.exit(0)

def has_structural_parameters(fmu_path):
    """Check if FMU contains structuralParameter causality."""
    model_desc = fmu_path / "modelDescription.xml"
    if not model_desc.exists():
        return False

    try:
        tree = ET.parse(model_desc)
        root = tree.getroot()

        # Check all variables for structuralParameter causality
        for var in root.findall(".//{*}Float64") + root.findall(".//{*}Int32") + root.findall(".//{*}Boolean"):
            causality = var.get("causality")
            if causality == "structuralParameter":
                return True

        return False
    except Exception as e:
        print(f"Warning: Could not parse modelDescription.xml: {e}")
        return False

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
    print("(Skipping FMUs with structuralParameter)")
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
    all_fmu_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.endswith("_fmu")])

    if not all_fmu_dirs:
        print(f"No FMU directories found in {output_dir}")
        return 1

    # Filter out FMUs with structuralParameters
    fmu_dirs = []
    skipped = []
    for fmu_dir in all_fmu_dirs:
        if has_structural_parameters(fmu_dir):
            skipped.append(fmu_dir.name)
        else:
            fmu_dirs.append(fmu_dir)

    print(f"Found {len(all_fmu_dirs)} FMU(s), testing {len(fmu_dirs)} (skipping {len(skipped)} with structuralParameter)\n")

    if skipped:
        print(f"Skipped: {', '.join(skipped)}\n")

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
    print(f"Results: {passed} passed, {failed} failed, {len(skipped)} skipped")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
