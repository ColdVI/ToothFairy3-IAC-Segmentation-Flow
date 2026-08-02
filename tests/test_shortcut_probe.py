import json

import numpy as np
import pytest
import torch

import _pathsetup  # noqa: F401
from analysis.shortcut_probe import (
    ReadinessError,
    apply_cbct_intervention,
    apply_prior_intervention,
    case_level_bootstrap,
    center_crop_pad,
    cosine_r2,
    generate_smoke_artifacts,
    publish_artifacts,
    relative_delta,
    validate_checkpoint_epoch,
)


def test_identical_prediction_has_zero_relative_delta():
    prediction = torch.randn(2, 4, 4, 4)
    result = relative_delta(prediction, prediction.clone())
    assert result["valid"] is True
    assert result["value"] == 0.0


def test_known_shortcut_has_unit_cosine_and_r2():
    x0 = torch.randn(2, 4, 4, 4)
    x1 = torch.randn(2, 4, 4, 4)
    t = 0.2
    xt = (1 - t) * x0 + t * x1
    shortcut = (xt - x0) / t
    metrics = cosine_r2(shortcut, x1 - x0)
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["flag"] == "ok"


def test_zero_norm_prediction_is_flagged_without_nan_drop():
    metrics = cosine_r2(torch.zeros(8), torch.arange(8.0))
    assert metrics["cosine"] is None
    assert metrics["cosine_valid"] is False
    assert metrics["r2"] is not None
    assert "zero_prediction_norm" in metrics["flag"]


def test_cross_case_crop_pad_matches_target_shape():
    donor = torch.arange(2 * 3 * 5 * 7).reshape(1, 2, 3, 5, 7)
    result = center_crop_pad(donor, (6, 4, 8), fill=9)
    assert result.shape == (1, 2, 6, 4, 8)


def test_bootstrap_resamples_cases_not_patches():
    rows = ([{"case_id": "a", "value": 0.0, "valid": True}] * 100
            + [{"case_id": "b", "value": 10.0, "valid": True}])
    result = case_level_bootstrap(rows, iterations=1000, seed=7)
    assert result["bootstrap_unit"] == "case"
    assert result["n_cases"] == 2
    assert result["n_patches"] == 101
    assert result["mean"] == pytest.approx(5.0)


def test_cbct_interventions_preserve_state_prior_and_time():
    cond = torch.randn(1, 8, 4, 4, 4)
    donor = torch.randn(1, 8, 3, 5, 4)
    xt = torch.randn(1, 2, 4, 4, 4); x0 = torch.randn_like(xt); t = torch.tensor([.2])
    snapshots = (xt.clone(), x0.clone(), t.clone())
    generator = torch.Generator().manual_seed(4)
    for name in ("zero", "noise", "shuffle"):
        changed = apply_cbct_intervention(cond, name, generator=generator,
                                          donor_cond=donor if name == "shuffle" else None)
        assert torch.equal(changed[:, 1:], cond[:, 1:])
        assert torch.equal(xt, snapshots[0]) and torch.equal(x0, snapshots[1])
        assert torch.equal(t, snapshots[2])


def test_prior_interventions_preserve_xt_and_cbct():
    cond = torch.randn(1, 8, 4, 4, 4)
    donor = torch.randn(1, 8, 5, 3, 6)
    xt = torch.randn(1, 2, 4, 4, 4); snapshot = xt.clone()
    for name in ("zero", "swap"):
        changed = apply_prior_intervention(cond, name,
                                           donor_cond=donor if name == "swap" else None)
        assert torch.equal(changed[:, 0], cond[:, 0])
        assert torch.equal(changed[:, 1:3], cond[:, 1:3])
        assert torch.equal(changed[:, 5:], cond[:, 5:])
        assert torch.equal(xt, snapshot)


def test_checkpoint_epoch_validation_rejects_wrong_or_missing_epoch(tmp_path):
    wrong = tmp_path / "wrong.pt"; torch.save({"epoch": 24}, wrong)
    with pytest.raises(ReadinessError, match="expected 25, got 24"):
        validate_checkpoint_epoch(wrong, 25)
    missing = tmp_path / "missing.pt"; torch.save({"val": {}}, missing)
    with pytest.raises(ReadinessError, match="no internal epoch"):
        validate_checkpoint_epoch(missing, 0)


def test_existing_analysis_artifacts_are_never_overwritten(tmp_path):
    staged = tmp_path / "staged"; generate_smoke_artifacts(staged)
    output = tmp_path / "output"; publish_artifacts(staged, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_artifacts(staged, output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_figure_and_summary_smoke_generation(tmp_path):
    generate_smoke_artifacts(tmp_path)
    figure = tmp_path / "fig1_shortcut.pdf"
    summary = json.loads((tmp_path / "shortcut_probe_summary.json").read_text())
    assert figure.stat().st_size > 100
    assert figure.read_bytes().startswith(b"%PDF")
    assert summary["synthetic_smoke_test"] is True
    assert (tmp_path / "shortcut_probe.csv").stat().st_size > 100
    assert (tmp_path / "thickening_probe.csv").stat().st_size > 10
