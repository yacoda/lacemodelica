#!/usr/bin/env python3
"""What modelDescription.xml says of causality, nominal, fixed and initial; a bound with an expression is no literal there.

Usage: test_model_description.py <lacemodelica_exe> <test file>
"""

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


def main():
    exe, test_file = sys.argv[1:3]
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([exe, test_file, "--output-dir", folder], check=True)
        root = ET.parse(Path(folder) / "modelDescription.xml").getroot()
        model = onnx.load(str(Path(folder) / "extra/org.lacemodelica.ls-onnx-serialization/model.onnx"))
        graph = model.graph

    variables = {v.get("name"): v for v in root.find("ModelVariables")}
    inputs = [i.name for i in graph.input]
    outputs = {o.name for o in graph.output}

    def fixed(name):
        found = variables[name].find("Annotations/Annotation[@type='org.lacemodelica.modelica']/Attributes")
        return None if found is None else found.get("fixed")

    assert variables["scale"].get("causality") == "parameter"
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

    # initial: exact for a start value with fixed = true, approx for one with fixed = false (the
    # default), calculated (the default, so not written) without a start value or with an expression
    def initial(name):
        return variables[name].get("initial"), variables[name].get("start")

    assert initial("scale") == ("exact", "2.0")
    assert initial("u") == (None, None)
    assert initial("x") == (None, None)
    assert initial("y") == (None, None)
    assert initial("a") == ("exact", "1.0")
    assert fixed("a") == "true"
    assert initial("b") == ("approx", "2.0")
    assert fixed("b") == "false"
    assert initial("c") == ("approx", "3.0")
    assert fixed("c") is None
    assert initial("e") == ("calculated", None)
    assert fixed("e") == "true"
    assert f"start[{inputs.index('e')}]" in outputs
    assert initial("z") == ("exact", "0.5")
    assert initial("p.x") == ("exact", "1.0")
    assert fixed("p.x") == "true"
    assert initial("p.y") == ("approx", "2.0")
    assert fixed("p.y") is None
    assert initial("der(a)") == (None, None)

    # start[N], min[N], max[N] and nominal[N] belong to graph input N, which counts time as well
    assert inputs[0] == "time", inputs
    for output in graph.output:
        attribute, _, index = output.name.partition("[")
        if attribute in ("start", "min", "max", "nominal"):
            variable = next(p.value for p in output.metadata_props if p.key == "variable_name")
            assert inputs[int(index[:-1])] == variable, (output.name, variable)

    # fixed = true without a start value: the default start value of the type is the initial value
    assert initial("f") == ("exact", "0")
    assert initial("g") == ("exact", "false")
    assert initial("q.y") == ("exact", "0")

    # fixed = true with an expression as start value: the initial equation x = start, after the
    # model's own
    assert initial("q.x") == ("calculated", None)
    assert fixed("q.x") == "true"
    init_eqs = sorted(o for o in outputs if o.startswith("init_eq["))
    assert init_eqs == ["init_eq[0]", "init_eq[1]", "init_eq[2]"], init_eqs
    values = {i.name: np.array(1.0) for i in graph.input}
    values["scale"] = np.array(2.0)
    values["b"] = np.array(2.5)
    values["e"] = np.array(4.0)  # 2.0 * scale
    values["q.x"] = np.array(6.0)  # 3.0 * scale
    values = {k: v.astype(bool) if k == "g" else v for k, v in values.items()}
    residuals = ReferenceEvaluator(model).run(init_eqs, values)
    assert [float(r) for r in residuals] == [0.0, 0.0, 0.0], residuals
    values["e"] = np.array(5.0)
    values["q.x"] = np.array(8.0)
    residuals = ReferenceEvaluator(model).run(init_eqs, values)
    assert [float(r) for r in residuals] == [0.0, 1.0, 2.0], residuals
    print("OK")


if __name__ == "__main__":
    main()
