import csv

import pytest
import yaml

from flow.checkpointing import CheckpointConflictError
from flow.prompt3r_train import _plot_outputs, append_paired_rows


def test_prompt3r_config_declares_exact_pilot_contract():
    cfg = yaml.safe_load(open("configs/flow_prompt3r.yaml"))
    assert cfg["zero_init_head"] is True
    assert cfg["cond_include_coarse_sdf"] is False
    assert cfg["selection_policy"] == "paired_identity"
    assert cfg["epochs"] == 15 and cfg["checkpoint_every"] == 1
    assert cfg["immutable_epoch_checkpoints_until"] == 15
    assert cfg["pilot_validation"]["full_epochs"] == [0, 5, 10, 15]
    assert cfg["prior_floor"]["score"] is None
    assert cfg["random_t_aux_enabled"] is False


def test_paired_csv_resume_skips_exact_row_and_rejects_conflict(tmp_path):
    path = tmp_path / "paired.csv"
    row = {"epoch": 0, "validation_tier": "quick", "case_id": "c", "side": 1,
           "metric_valid": True, "delta_dice": 0.0}
    append_paired_rows(path, [row])
    append_paired_rows(path, [row])
    assert len(list(csv.DictReader(path.open()))) == 1
    with pytest.raises(CheckpointConflictError):
        append_paired_rows(path, [{**row, "delta_dice": .1}])


def test_paper_ready_plot_outputs_smoke_with_minimal_synthetic_rows(tmp_path):
    trajectory = [
        {"epoch": 0, "quick_mean_delta_dice": 0., "quick_mean_delta_cldice": 0.,
         "quick_mean_delta_hd95_mm": 0., "quick_mean_delta_gap_mm": 0.},
        {"epoch": 1, "quick_mean_delta_dice": -.001,
         "quick_mean_delta_cldice": 0., "quick_mean_delta_hd95_mm": -.1,
         "quick_mean_delta_gap_mm": 0.},
    ]
    full = [{"epoch": 0, "metric_valid": "True", "delta_volume_ratio": "0.0",
             "delta_radius_bias_mm": "0.0"}]
    _plot_outputs(tmp_path, trajectory, full)
    assert (tmp_path / "trajectory_metrics.pdf").stat().st_size > 100
    assert (tmp_path / "geometry_bias.pdf").stat().st_size > 100
