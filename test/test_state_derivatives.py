#!/usr/bin/env python3
"""The der(x) inputs of the ONNX graph are the derivatives in modelDescription.xml: each is the
derivative of x, with the dimensions of x, and there are no others.

Usage: test_state_derivatives.py <lacemodelica_exe> <test file>...
"""

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import onnx


def dimensions(variable, by_reference):
    """The dimensions of a variable: a number, or the name of a structural parameter."""
    result = []
    for dim in variable.findall("Dimension"):
        if dim.get("start") is not None:
            result.append(dim.get("start"))
        else:
            result.append(by_reference[dim.get("valueReference")].get("name"))
    return result


def check(exe, test_file):
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([exe, test_file, "--output-dir", folder], check=True)
        root = ET.parse(Path(folder) / "modelDescription.xml").getroot()
        graph = onnx.load(str(Path(folder) / "extra/org.lacemodelica.ls-onnx-serialization/model.onnx")).graph

    variables = {v.get("name"): v for v in root.find("ModelVariables")}
    by_reference = {v.get("valueReference"): v for v in variables.values()}
    derivatives = {d.get("valueReference") for d in root.find("ModelStructure").findall("ContinuousStateDerivative")}

    names = [i.name for i in graph.input if i.name.startswith("der(")]
    assert names, f"{test_file}: no der() input"
    described = sorted(name for name, v in variables.items() if v.get("derivative") is not None)
    assert described == sorted(names), f"{test_file}: derivatives {described}, der() inputs {sorted(names)}"
    for i in graph.input:
        if not i.name.startswith("der("):
            continue
        state = i.name[len("der("):-1]
        der = variables.get(i.name)
        assert der is not None, f"{test_file}: {i.name} is not in modelDescription.xml"
        assert der.get("derivative") == variables[state].get("valueReference"), f"{test_file}: {i.name}"
        assert der.get("valueReference") in derivatives, f"{test_file}: {i.name} is no ContinuousStateDerivative"
        dims = dimensions(der, by_reference)
        assert dims == dimensions(variables[state], by_reference), f"{test_file}: {i.name} has dimensions {dims}"
        shape = [str(d.dim_value) if d.HasField("dim_value") else d.dim_param for d in i.type.tensor_type.shape.dim]
        assert shape == dims, f"{test_file}: {i.name} has shape {shape} in the graph, {dims} in modelDescription.xml"


def main():
    exe = sys.argv[1]
    for test_file in sys.argv[2:]:
        check(exe, test_file)
    print("OK")


if __name__ == "__main__":
    main()
