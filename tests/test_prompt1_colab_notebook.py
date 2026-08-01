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
        "SPLITS_PATH", "configs_cache/splits.json", "QUICK_PREFLIGHT_CASES_PER_FOLD",
        "FULL_PREFLIGHT_CASES", "cache_manifest_480", "identity_prior", "pytest",
        "preflight-quick", "smoke", "preflight-full", "run-full",
    )
    for token in required:
        assert token in combined
    assert "/Users/anil" not in combined
    assert notebook["metadata"]["accelerator"] == "GPU"
    groups = {cell.get("metadata", {}).get("prompt1_group") for cell in notebook["cells"]}
    assert {"SETUP_TESTS", "QUICK_PREFLIGHT", "TWO_CASE_SMOKE",
            "FULL_PREFLIGHT", "FULL_COMPLETION"} <= groups


def test_first_code_cell_is_the_single_editable_configuration():
    path = Path(__file__).resolve().parents[1] / "notebooks/prompt1_completion_colab.ipynb"
    notebook = json.loads(path.read_text())
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert first_code["metadata"]["prompt1_group"] == "configuration"
    source = "".join(first_code["source"])
    assert "dataset_cache_colab_v1/Dataset801_IAC_LR" in source
    assert "MAX_CASES = 2" in source
    assert "SPLITS_PATH = DRIVE_ROOT / 'configs_cache/splits.json'" in source
    assert "QUICK_PREFLIGHT_CASES_PER_FOLD = 1" in source
    assert "FULL_PREFLIGHT_CASES = 40" in source
    assert "EXPORT_TRUE_SOFTMAX = True" in source
    assert "FORCE_REBUILD = False" in source
