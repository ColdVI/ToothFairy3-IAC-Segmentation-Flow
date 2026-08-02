import json
from pathlib import Path


NOTEBOOK = (Path(__file__).resolve().parents[1]
            / "notebooks/prompt2_limited_endpoint_probe_colab.ipynb")


def test_prompt2_notebook_is_clean_run_all_contract():
    notebook = json.loads(NOTEBOOK.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    combined = "\n".join("".join(cell.get("source", [])) for cell in code)
    required = (
        "drive.mount", "git', 'clone", "git', 'fetch", "PINNED_COMMIT",
        "pip', 'install", "torch.cuda.is_available", "--readiness-only",
        "pytest", "analysis/shortcut_probe.py", "shortcut_probe.csv",
        "shortcut_probe_summary.json", "limited_endpoint_diagnostic.pdf",
        "thickening_probe.csv", "shortcut_probe_manifest.json",
        "manifest['artifacts']", "claim_limit", "WORK_DIR", "LOCAL_TEMP_PARENT",
        "best_legacy_unknown_epoch", "epoch_129", "best.pt", "last.pt", "latest.pt",
        "protocol_deviation", "exact_epoch_trajectory_available",
        "historical_per_epoch_checkpoints_were_not_saved", "diagnostic_only",
    )
    for token in required:
        assert token in combined
    assert "/Users/anil" not in combined
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert all(not cell.get("outputs") for cell in code)
    assert all(cell.get("execution_count") is None for cell in code)
    groups = [cell["metadata"]["prompt2_group"] for cell in code]
    assert groups == ["configuration", "SETUP", "READINESS_AUDIT", "PYTEST",
                      "LIMITED_ENDPOINT_PROBE", "VALIDATE_OUTPUTS", "FINAL_SUMMARY"]


def test_prompt2_first_code_cell_is_only_editable_configuration_and_is_pinned():
    notebook = json.loads(NOTEBOOK.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "".join(code[0]["source"])
    assert code[0]["metadata"]["prompt2_group"] == "configuration"
    assert "7b6a32686fab381a0840d6f512a6ca9647f9593a" in source
    assert "BEST_CHECKPOINT" in source and "LAST_CHECKPOINT_CANDIDATES" in source
    assert "DRIVE_ROOT" in source and "ANALYSIS_OUTPUT" in source
    assert "REPLACE_WITH" not in source


def test_prompt2_code_cells_are_valid_python():
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"prompt2-cell-{index}", "exec")


def test_prompt2_notebook_never_requests_historical_epoch_files_or_training():
    notebook = json.loads(NOTEBOOK.read_text())
    combined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for forbidden in ("epoch_000.pt", "epoch_025.pt", "epoch_125.pt", "fig1_shortcut.pdf"):
        assert forbidden not in combined
    assert "train_flow.py" not in combined
