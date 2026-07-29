import pytest

import _pathsetup  # noqa: F401
from flow.validate import summarize_rows
from scripts.identity_baseline import _ordered_validation_ids


def test_summarize_rows_reports_both_aggregation_modes():
    rows = [
        {"case_id": "a", "side": 1, "dice": 0.5, "cldice": 0.7, "hd95": 2.0},
        {"case_id": "a", "side": 2, "dice": 0.9, "cldice": 0.9, "hd95": 4.0},
        {"case_id": "b", "side": 1, "dice": 0.7, "cldice": 0.8, "hd95": 1.0},
        {"case_id": "b", "side": 2, "dice": 0.3, "cldice": 0.6, "hd95": 3.0},
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
