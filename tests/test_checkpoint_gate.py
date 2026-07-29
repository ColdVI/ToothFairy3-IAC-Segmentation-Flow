from pathlib import Path

import pytest

import _pathsetup  # noqa: F401
from flow.train import checkpoint_is_better, require_prior_floor
from scripts.identity_baseline import _write_prior_floor


FLOOR = {"dice": 0.90, "cldice": 0.98, "hd95": 0.5, "score": 0.94}


def test_prior_floor_fails_closed_when_missing_or_incomplete():
    with pytest.raises(ValueError, match="missing"):
        require_prior_floor({})
    with pytest.raises(ValueError, match="incomplete"):
        require_prior_floor({"prior_floor": {"dice": 0.9}})
    assert require_prior_floor({"prior_floor": FLOOR}) == FLOOR


def test_checkpoint_must_beat_prior_before_tie_breaking():
    best = dict(FLOOR)
    below = {"dice": 0.91, "cldice": 0.96, "hd95": 0.2, "score": 0.935}
    above = {"dice": 0.91, "cldice": 0.98, "hd95": 0.6, "score": 0.945}
    assert not checkpoint_is_better(below, best, FLOOR)
    assert checkpoint_is_better(above, best, FLOOR)

    better_hd95 = dict(above, hd95=0.4)
    assert checkpoint_is_better(better_hd95, above, FLOOR)


def test_identity_writer_preserves_surrounding_yaml(tmp_path):
    config = tmp_path / "flow.yaml"
    config.write_text("# keep me\nbase: 32\nprior_floor:\n  dice: null\n  cldice: null\n"
                      "  hd95: null\n  score: null\n# after\nepochs: 5\n")
    _write_prior_floor(config, FLOOR)
    text = config.read_text()
    assert "# keep me" in text and "# after" in text
    assert "  score: 0.94" in text
    assert "epochs: 5" in text
