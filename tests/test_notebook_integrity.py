###############################################################################
#                       test_notebook_integrity.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Notebook structure and reproducibility guardrails
###############################################################################

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_document_execution_mode_and_do_not_upgrade_environment():
    notebooks = sorted(ROOT.glob("*.ipynb"))
    assert notebooks

    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        assert any("### Execution environment" in source for source in sources), path.name
        assert not any("pip install --upgrade riskoptima" in source for source in sources), path.name


def test_notebooks_do_not_store_error_outputs():
    for path in sorted(ROOT.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        errors = [
            output
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, path.name
