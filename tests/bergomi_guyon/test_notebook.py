"""Execute the tutorial and its independent mathematical checks."""

import json
from pathlib import Path


def test_tutorial_executes_independent_checks():
    notebook = json.loads((Path(__file__).resolve().parents[2]
                           / "bergomi_guyon_recursion.ipynb").read_text())
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"notebook cell {index}", "exec"), namespace)
    # The current tutorial works through order three; the coefficient suite
    # separately verifies the publication artifacts through order six.
    assert len(namespace["third_order"].coefficients) == 4
