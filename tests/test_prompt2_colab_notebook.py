import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/prompt2_shortcut_probe_colab.ipynb"


def test_prompt2_notebook_is_clean_run_all_contract():
    notebook = json.loads(NOTEBOOK.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    combined = "\n".join("".join(cell.get("source", [])) for cell in code)
    required = (
        "drive.mount", "git', 'clone", "git', 'fetch", "PINNED_COMMIT",
        "pip', 'install", "torch.cuda.is_available", "--readiness-only",
        "pytest", "analysis/shortcut_probe.py", "shortcut_probe.csv",
        "shortcut_probe_summary.json", "fig1_shortcut.pdf", "thickening_probe.csv",
        "shortcut_probe_manifest.json", "manifest['artifacts']", "claim_limit",
        "WORK_DIR", "LOCAL_TEMP_PARENT", "epoch_000.pt", "epoch_025.pt",
        "epoch_125.pt",
    )
    for token in required:
        assert token in combined
    assert "/Users/anil" not in combined
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert all(not cell.get("outputs") for cell in code)
    assert all(cell.get("execution_count") is None for cell in code)
    groups = [cell["metadata"]["prompt2_group"] for cell in code]
    assert groups == ["configuration", "SETUP", "READINESS_AUDIT", "PYTEST",
                      "SHORTCUT_PROBE", "VALIDATE_OUTPUTS", "FINAL_SUMMARY"]


def test_prompt2_first_code_cell_is_only_editable_configuration_and_is_pinned():
    notebook = json.loads(NOTEBOOK.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "".join(code[0]["source"])
    assert code[0]["metadata"]["prompt2_group"] == "configuration"
    assert "b1c294def65d7625e44a9e4d9be280033bd31dcc" in source
    assert "CHECKPOINTS = {0:" in source
    assert "DRIVE_ROOT" in source and "ANALYSIS_OUTPUT" in source
    assert "REPLACE_WITH" not in source


def test_prompt2_code_cells_are_valid_python():
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"prompt2-cell-{index}", "exec")
