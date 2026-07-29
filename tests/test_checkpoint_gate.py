import pytest
import torch

import _pathsetup  # noqa: F401
from flow.train import (atomic_torch_save, checkpoint_is_better, is_safe,
                        migrate_legacy_best, require_noninferiority_margin,
                        require_prior_floor, update_checkpoint_selection)
from scripts.identity_baseline import _write_prior_floor


FLOOR = {"complete_cv": True, "dice": 0.90, "cldice": 0.98,
         "hd95": 0.5, "score": 0.94}
FLOOR_METRICS = {key: FLOOR[key] for key in ("dice", "cldice", "hd95", "score")}
MARGIN = 0.01


def metrics(dice, gap, betti, hd95, cldice, score=None):
    return {"dice": dice, "gap_mm": gap, "betti0": betti, "hd95": hd95,
            "cldice": cldice, "score": score if score is not None else .5 * (dice + cldice)}


def test_checkpoint_policy_fails_closed_when_partial_or_undecided():
    with pytest.raises(ValueError, match="missing"):
        require_prior_floor({})
    with pytest.raises(ValueError, match="partial"):
        require_prior_floor({"prior_floor": {**FLOOR, "complete_cv": False}})
    with pytest.raises(ValueError, match="incomplete"):
        require_prior_floor({"prior_floor": {"complete_cv": True, "dice": 0.9}})
    assert require_prior_floor({"prior_floor": FLOOR}) == FLOOR_METRICS
    with pytest.raises(ValueError, match="explicit user decision"):
        require_noninferiority_margin({"noninferiority_margin": None})
    assert require_noninferiority_margin({"noninferiority_margin": MARGIN}) == MARGIN


def test_safe_gate_uses_dice_margin_not_weighted_score():
    high_score_but_inferior_dice = metrics(.889, 1, 0, .2, 1.0, score=.99)
    eligible_boundary = metrics(.890, 3, 1, .8, .95, score=.1)
    assert not is_safe(high_score_but_inferior_dice, FLOOR_METRICS, MARGIN)
    assert is_safe(eligible_boundary, FLOOR_METRICS, MARGIN)


def test_lexicographic_selection_orders_topology_boundary_and_epoch():
    baseline = {**metrics(.90, 5, 1, 1.0, .95), "epoch": 5}
    better_gap = metrics(.89, 4, 2, 2.0, .90)
    assert checkpoint_is_better(better_gap, 6, baseline, FLOOR_METRICS, MARGIN)

    same_gap_better_betti = metrics(.89, 5, 0, 2.0, .90)
    assert checkpoint_is_better(same_gap_better_betti, 6, baseline, FLOOR_METRICS, MARGIN)
    same_topology_better_hd = metrics(.89, 5, 1, .9, .90)
    assert checkpoint_is_better(same_topology_better_hd, 6, baseline, FLOOR_METRICS, MARGIN)
    same_boundary_better_cl = metrics(.89, 5, 1, 1.0, .96)
    assert checkpoint_is_better(same_boundary_better_cl, 6, baseline, FLOOR_METRICS, MARGIN)
    exact_later = metrics(.90, 5, 1, 1.0, .95)
    assert not checkpoint_is_better(exact_later, 6, baseline, FLOOR_METRICS, MARGIN)


def test_checkpoint_trio_survives_resume_and_safe_gate(tmp_path):
    unsafe = metrics(.88, 3, 0, .8, .96)
    best_any, best_safe, write_any, write_safe = update_checkpoint_selection(
        unsafe, 0, None, None, FLOOR_METRICS, MARGIN)
    assert write_any and not write_safe
    assert best_any["epoch"] == 0 and best_safe is None
    atomic_torch_save({"epoch": 0, "best_any": best_any, "best_safe": best_safe},
                      str(tmp_path / "last.pt"))
    atomic_torch_save({"val": best_any}, str(tmp_path / "best_any.pt"))
    assert not (tmp_path / "best_safe.pt").exists()

    resumed = torch.load(tmp_path / "last.pt", weights_only=False)
    safe = metrics(.895, 2, 0, .7, .95)
    best_any, best_safe, write_any, write_safe = update_checkpoint_selection(
        safe, 1, resumed["best_any"], resumed["best_safe"], FLOOR_METRICS, MARGIN)
    assert write_any and write_safe
    atomic_torch_save({"val": best_any}, str(tmp_path / "best_any.pt"))
    atomic_torch_save({"val": best_safe}, str(tmp_path / "best_safe.pt"))
    atomic_torch_save({"epoch": 1, "best_any": best_any, "best_safe": best_safe},
                      str(tmp_path / "last.pt"))

    final = torch.load(tmp_path / "last.pt", weights_only=False)
    assert final["best_any"]["epoch"] == final["best_safe"]["epoch"] == 1
    assert all((tmp_path / name).is_file()
               for name in ("last.pt", "best_any.pt", "best_safe.pt"))
    assert not list(tmp_path.glob("*.partial"))


def test_legacy_best_is_preserved_as_unsafe_best_any(tmp_path):
    legacy = tmp_path / "best.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])},
                "val": {"dice": .91, "cldice": .98, "hd95": .5, "score": .945}}, legacy)
    record, safe = migrate_legacy_best(str(tmp_path))
    assert legacy.is_file()
    assert (tmp_path / "best_any.pt").is_file()
    assert not (tmp_path / "best_safe.pt").exists()
    assert record["legacy_missing_topology"] and not record["safe_eligible"]
    assert safe is None


def test_identity_writer_preserves_yaml_and_marks_complete(tmp_path):
    config = tmp_path / "flow.yaml"
    config.write_text("# keep me\nbase: 32\nprior_floor:\n  complete_cv: null\n"
                      "  dice: null\n  cldice: null\n  hd95: null\n  score: null\n"
                      "# after\nnoninferiority_margin: null\nepochs: 5\n")
    _write_prior_floor(config, FLOOR_METRICS)
    text = config.read_text()
    assert "# keep me" in text and "# after" in text
    assert "  complete_cv: true" in text
    assert "  score: 0.94" in text
    assert "noninferiority_margin: null" in text
    assert "epochs: 5" in text
