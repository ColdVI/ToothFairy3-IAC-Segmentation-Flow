import json

import numpy as np
import pytest

from evaluation.geometry_metrics import side_geometry_metrics
from flow.pilot_cases import build_pilot_case_manifest, select_pilot_cases
from flow.validate import (CORE_METRICS, GEOMETRY_METRICS, paired_case_rows,
                           paired_identity_safety_gate)


def _paired_row(case_id, side, *, valid=True, dice_delta=-0.001,
                cldice_delta=-0.001, hd95_delta=-0.1, gap_delta=0.0,
                flow_volume_ratio=1.01, flow_radius_bias=0.05):
    identity = {
        "dice": .91, "cldice": .99, "hd95_mm": .83, "gap_mm": .3, "betti0": 0,
        "volume_ratio": 1.0, "signed_surface_mean_mm": 0.0,
        "radius_bias_mm": 0.0, "connected_components": 1,
        "false_positive_component_count": 0,
        "max_false_positive_distance_mm": 0.0,
    }
    flow = dict(identity)
    flow.update({"dice": identity["dice"] + dice_delta,
                 "cldice": identity["cldice"] + cldice_delta,
                 "hd95_mm": identity["hd95_mm"] + hd95_delta,
                 "gap_mm": identity["gap_mm"] + gap_delta,
                 "volume_ratio": flow_volume_ratio,
                 "radius_bias_mm": flow_radius_bias})
    row = {"case_id": case_id, "scanner_group": "F", "side": side,
           "identity_lr_swap": 0.0, "flow_lr_swap": 0.0,
           "delta_lr_swap": 0.0, "metric_valid": valid,
           "exclusion_reason": "" if valid else "synthetic_invalid"}
    for name in CORE_METRICS + GEOMETRY_METRICS:
        row[f"identity_{name}"] = identity[name]
        row[f"flow_{name}"] = flow[name]
        row[f"delta_{name}"] = flow[name] - identity[name]
    return row


def _four_rows(**kwargs):
    return [_paired_row(case, side, **kwargs)
            for case in ("ToothFairy3F_001", "ToothFairy3F_002")
            for side in (1, 2)]


def test_geometry_metrics_report_physical_bias_and_false_components():
    gt = np.zeros((21, 21, 21), bool)
    gt[7:14, 9:12, 9:12] = True
    pred = gt.copy()
    pred[2:4, 2:4, 2:4] = True
    metrics = side_geometry_metrics(pred, gt, spacing=(2.0, 1.0, .5))
    assert metrics["volume_ratio"] > 1
    assert metrics["connected_component_count"] == 2
    assert metrics["fp_component_count"] == 1
    assert metrics["max_fp_distance_mm"] > 0
    assert metrics["geometry_valid"] is True


def test_paired_gate_aggregates_bilateral_cases_and_passes_exact_contract():
    result = paired_identity_safety_gate(_four_rows())
    assert result["safe"] is True
    assert len(result["case_rows"]) == 2
    assert result["aggregate"]["mean_delta_dice"] == pytest.approx(-0.001)
    assert all(result["criteria"].values())


def test_any_invalid_side_fails_closed_without_nan_drop():
    rows = _four_rows()
    rows[0]["metric_valid"] = False
    rows[0]["exclusion_reason"] = "zero_norm"
    result = paired_identity_safety_gate(rows)
    assert result["safe"] is False
    assert result["criteria"] == {"all_cases_valid": False}
    assert result["invalid_cases"][0]["case_id"] == "ToothFairy3F_001"


def test_volume_bias_gate_is_casewise_absolute_delta():
    result = paired_identity_safety_gate(_four_rows(flow_volume_ratio=1.04))
    assert result["safe"] is False
    assert result["criteria"]["extra_abs_volume_bias"] is False


def test_missing_side_invalidates_case():
    cases = paired_case_rows(_four_rows()[:-1])
    bad = next(case for case in cases if case["case_id"] == "ToothFairy3F_002")
    assert bad["valid"] is False
    assert "requires_exactly_two" in bad["validity_reason"]


def test_pilot_panel_is_deterministic_stratified_and_matches_committed_file():
    splits = json.load(open("configs/splits.json"))
    first = select_pilot_cases(splits["folds"][0]["val"], seed=0)
    second = select_pilot_cases(list(reversed(splits["folds"][0]["val"])), seed=0)
    assert first == second
    assert len(first["quick_case_ids"]) == 4 and len(first["full_case_ids"]) == 10
    committed = json.load(open("configs/prompt3r_pilot_cases.json"))
    generated = build_pilot_case_manifest("configs/splits.json")
    assert committed == generated
