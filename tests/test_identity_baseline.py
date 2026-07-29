import pytest

import _pathsetup  # noqa: F401
from flow.validate import summarize_rows
from scripts.identity_baseline import (_ordered_validation_ids,
                                       write_config_from_report)


def test_summarize_rows_reports_both_aggregation_modes():
    rows = [
        {"case_id": "a", "side": 1, "dice": 0.5, "cldice": 0.7, "hd95": 2.0,
         "gap_mm": 1.0, "betti0": 0.0},
        {"case_id": "a", "side": 2, "dice": 0.9, "cldice": 0.9, "hd95": 4.0,
         "gap_mm": 3.0, "betti0": 2.0},
        {"case_id": "b", "side": 1, "dice": 0.7, "cldice": 0.8, "hd95": 1.0,
         "gap_mm": 2.0, "betti0": 1.0},
        {"case_id": "b", "side": 2, "dice": 0.3, "cldice": 0.6, "hd95": 3.0,
         "gap_mm": 4.0, "betti0": 1.0},
    ]
    side = summarize_rows(rows, "per_side")
    case = summarize_rows(rows, "per_case")
    assert side == pytest.approx(case)
    assert side["dice"] == pytest.approx(0.6)
    assert side["score"] == pytest.approx(0.675)


def test_validation_fold_union_must_match_development():
    splits = {"development": ["a", "b"],
              "folds": [{"val": ["a"]}, {"val": ["b"]}]}
    assert _ordered_validation_ids(splits) == ["a", "b"]

    duplicated = {"development": ["a"],
                  "folds": [{"val": ["a"]}, {"val": ["a"]}]}
    with pytest.raises(ValueError, match="not disjoint"):
        _ordered_validation_ids(duplicated)


def test_prior_config_requires_completed_three_path_report(tmp_path):
    config = tmp_path / "flow.yaml"
    config.write_text("prior_floor:\n  complete_cv: null\n  dice: null\nnoninferiority_margin: null\n")
    report = tmp_path / "identity.json"
    full = {
        "complete_cv": True, "evaluated_cases": 480, "cache_valid_cases": 480,
        "missing_cases": 0, "invalid_cases": 0,
        "direct_vs_sdf_voxel_difference": 0,
        "direct_vs_full_path_voxel_difference": 0,
        "overall": {"per_side": {"direct": {
            "dice": .8, "cldice": .9, "hd95": 2.0}}},
    }
    report.write_text(__import__("json").dumps(full))
    metrics = write_config_from_report(report, config)
    assert metrics["score"] == pytest.approx(.85)
    text = config.read_text()
    assert "complete_cv: true" in text
    assert "noninferiority_margin: null" in text

    full["evaluated_cases"] = 40
    report.write_text(__import__("json").dumps(full))
    with pytest.raises(ValueError, match="incomplete identity report"):
        write_config_from_report(report, config)
