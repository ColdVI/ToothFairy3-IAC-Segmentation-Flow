import ast
import json
import re
from pathlib import Path


NOTEBOOK = Path("notebooks/flow_v2_prompt3r_colab.ipynb")


def _load():
    return json.loads(NOTEBOOK.read_text())


def test_notebook_is_clean_run_all_python_and_centrally_configured():
    notebook = _load()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 7
    for cell in code_cells:
        ast.parse("".join(cell["source"]))
    content_cells = [index for index, cell in enumerate(code_cells)
                     if "/content" in "".join(cell["source"])]
    assert content_cells == [0]
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code_cells)


def test_notebook_has_pinned_clean_checkout_and_l4_preflight():
    text = NOTEBOOK.read_text()
    match = re.search(r"PINNED_COMMIT = '([0-9a-f]{40})'", text)
    assert match and set(match.group(1)) != {"0"}
    assert "git', 'checkout', '--detach', PINNED_COMMIT" in text
    assert "assert not dirty" in text
    assert "REQUIRE_GPU_NAME = 'L4'" in text
    assert "prompt3r_preflight.py" in text


def test_training_is_gated_resumable_and_never_launches_deferred_grid():
    notebook = _load()
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")
    assert "pytest', '-q', 'tests/'" in code
    assert "assert PREFLIGHT_PASSED and TEST_GATE_PASSED" in code
    assert "assert PREFLIGHT_PASSED and TEST_GATE_PASSED and SMOKE_PASSED" in code
    assert "--stop-after-epoch', '1'" in code and "--resume" in code
    for forbidden in ("B0", "B1", "B2", "B3", "50 epoch", "Prompt 4", "Prompt 5", "Prompt 6"):
        assert forbidden not in code


def test_notebook_validates_all_required_persistent_outputs():
    text = NOTEBOOK.read_text()
    for name in ("epoch_trajectory.csv", "paired_validation_quick_all_epochs.csv",
                 "paired_validation_full.csv", "pilot_decision.json",
                 "trajectory_metrics.pdf", "geometry_bias.pdf", "manifest.json"):
        assert name in text
    assert "PROMPT3R_ROOT = DRIVE_ROOT / 'outputs/prompt3r'" in text
