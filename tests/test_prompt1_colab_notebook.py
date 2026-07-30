import json
from pathlib import Path


def test_prompt1_colab_notebook_has_required_contract():
    path = Path(__file__).resolve().parents[1] / "notebooks/prompt1_completion_colab.ipynb"
    notebook = json.loads(path.read_text())
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    combined = "\n".join(sources)
    required = (
        "DRIVE_ROOT", "DATASET_ROOT", "NNUNET_RESULTS", "TRACKB_CACHE_ROOT",
        "OUTPUT_ROOT", "REPO_URL", "PINNED_COMMIT", "NUM_WORKERS", "DEVICE",
        "MAX_CASES", "FORCE_REBUILD", "EXPORT_TRUE_SOFTMAX", "RETRY_FAILED",
        "cache_manifest_480", "identity_prior", "RUN/RESUME", "pytest",
    )
    for token in required:
        assert token in combined
    assert "/Users/anil" not in combined
    assert notebook["metadata"]["accelerator"] == "GPU"
    run_cells = [cell for cell in notebook["cells"]
                 if cell.get("metadata", {}).get("prompt1_group") == "RUN_RESUME"]
    assert len(run_cells) == 1
    assert "bootstrap_repo()" in "".join(run_cells[0]["source"])
    assert "'run'" in "".join(run_cells[0]["source"])


def test_first_code_cell_is_the_single_editable_configuration():
    path = Path(__file__).resolve().parents[1] / "notebooks/prompt1_completion_colab.ipynb"
    notebook = json.loads(path.read_text())
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert first_code["metadata"]["prompt1_group"] == "configuration"
    source = "".join(first_code["source"])
    assert "dataset_cache_colab_v1/Dataset801_IAC_LR" in source
    assert "MAX_CASES = 2" in source
    assert "EXPORT_TRUE_SOFTMAX = True" in source
    assert "FORCE_REBUILD = False" in source
