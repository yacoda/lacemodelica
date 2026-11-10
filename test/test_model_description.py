#!/usr/bin/env python3
"""What modelDescription.xml says of causality (with the Evaluate annotation), nominal and fixed; a bound with an expression is no literal there.

Usage: test_model_description.py <lacemodelica_exe> <test file>
"""

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import onnx


def main():
    exe, test_file = sys.argv[1:3]
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([exe, test_file, "--output-dir", folder], check=True)
        root = ET.parse(Path(folder) / "modelDescription.xml").getroot()
        graph = onnx.load(str(Path(folder) / "extra/org.lacemodelica.ls-onnx-serialization/model.onnx")).graph

    variables = {v.get("name"): v for v in root.find("ModelVariables")}
    inputs = [i.name for i in graph.input]
    outputs = {o.name for o in graph.output}

    def fixed(name):
        found = variables[name].find("Annotations/Annotation[@type='org.lacemodelica.modelica']/Attributes")
        return None if found is None else found.get("fixed")

    assert variables["scale"].get("causality") == "parameter"
    assert variables["evaluated"].get("causality") == "structuralParameter"
    assert "evaluated" not in inputs, "a structural parameter is a constant of the graph"
    assert variables["notEvaluated"].get("causality") == "parameter", "only Evaluate = true counts"
    assert "notEvaluated" in inputs
    assert variables["u"].get("causality") == "input"
    assert variables["u"].get("initial") is None, "FMI does not allow initial on an input"
    assert variables["u"].get("nominal") == "5.0"
    assert fixed("u") == "true"
    assert variables["v"].get("causality") == "input"
    assert fixed("v") == "false"
    assert variables["y"].get("causality") == "output"
    assert variables["y"].get("nominal") is None, "an expression is no literal"
    assert f"nominal[{inputs.index('y')}]" in outputs
    assert variables["x"].get("causality") == "local"
    assert fixed("x") is None
    assert variables["x"].get("min") is None, "an expression is no literal"
    assert f"min[{inputs.index('x')}]" in outputs
    print("OK")


if __name__ == "__main__":
    main()
