"""Notebooks are committed without outputs: outputs can contain real user data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_has_no_outputs(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    assert code_cells, "notebook has no code"
    assert all(not c.get("outputs") for c in code_cells), "clear outputs before committing"
    assert all(c.get("execution_count") is None for c in code_cells)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_logic_lives_in_modules(path):
    """Notebooks narrate; they must not define their own analysis functions."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "def " not in source
