import json

import pytest

from flow.prompt3r_config import resolve_prompt3r_config, sha256_file


def _identity(path, *, complete=True):
    payload = {
        "complete_cv": complete, "evaluated_cases": 480,
        "cache_valid_cases": 480, "missing_cases": 0, "invalid_cases": 0,
        "direct_vs_sdf_voxel_difference": 0,
        "direct_vs_full_path_voxel_difference": 0,
        "overall": {"per_side": {"direct": {
            "dice": .910125, "cldice": .991210, "hd95": .829462,
            "score": .777123}}},
    }
    path.write_text(json.dumps(payload))


def test_resolved_prior_floor_is_copied_verbatim_including_source_score(tmp_path):
    identity = tmp_path / "identity.json"
    _identity(identity)
    resolved = resolve_prompt3r_config("configs/flow_prompt3r.yaml", identity)
    assert resolved["prior_floor"]["score"] == .777123
    assert resolved["prior_floor"]["dice"] == .910125
    assert resolved["prior_floor"]["source_sha256"] == sha256_file(identity)
    assert resolved["selection_policy"] == "paired_identity"


def test_incomplete_identity_source_is_rejected(tmp_path):
    identity = tmp_path / "identity.json"
    _identity(identity, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        resolve_prompt3r_config("configs/flow_prompt3r.yaml", identity)
