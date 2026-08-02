import inspect
import json

import numpy as np
import pytest
import torch

import _pathsetup  # noqa: F401
from flow.channel_contract import (ConditioningContractError,
                                   resolve_conditioning_spec,
                                   validate_checkpoint_contract)
from flow.conditioning import build_conditioning
from flow.model import ResidualVelocityUNet3D
from flow.sliding_window import predict_volume
from scripts.run_manifest import start_manifest


def _inputs(shape=(4, 5, 6)):
    cbct = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    hard_l = np.full(shape, 1, np.float32)
    hard_r = np.full(shape, 2, np.float32)
    sdf_l = np.full(shape, 3, np.float32)
    sdf_r = np.full(shape, 4, np.float32)
    coords = np.stack([np.full(shape, 5 + index, np.float32) for index in range(3)])
    return cbct, hard_l, hard_r, sdf_l, sdf_r, coords


@pytest.mark.parametrize("include,expected_names", [
    (True, ["cbct", "hard_left", "hard_right", "coarse_sdf_left",
            "coarse_sdf_right", "coord_x", "coord_y", "coord_z"]),
    (False, ["cbct", "hard_left", "hard_right", "coord_x", "coord_y", "coord_z"]),
])
def test_conditioning_contract_order_shape_and_cbct_channel(include, expected_names):
    spec = resolve_conditioning_spec({"cond_include_coarse_sdf": include})
    cond = build_conditioning(*_inputs(), spec=spec)
    assert list(spec.conditioning_channel_names) == expected_names
    assert spec.conditioning_channels == len(expected_names)
    assert cond.shape == (len(expected_names), 4, 5, 6)
    expected_cbct = (_inputs()[0] - _inputs()[0].mean()) / (_inputs()[0].std() + 1e-6)
    assert np.array_equal(cond[0], expected_cbct.astype(np.float32))
    assert np.array_equal(cond[1], _inputs()[1])
    assert np.array_equal(cond[2], _inputs()[2])


def test_ground_truth_is_not_a_conditioning_input():
    parameters = inspect.signature(build_conditioning).parameters
    assert "gt" not in parameters and "x1" not in parameters and "sdf_gt" not in parameters


def test_sliding_window_contract_mismatch_fails_explicitly():
    model = ResidualVelocityUNet3D(cond_ch=6, base=8, tdim=32)
    with pytest.raises(ConditioningContractError, match="model expects 6"):
        predict_volume(model, np.zeros((8, 8, 8, 8), np.float32),
                       np.ones((2, 8, 8, 8), np.float32), patch=8)


def test_checkpoint_contract_is_fail_closed_and_legacy_is_explicit():
    legacy = resolve_conditioning_spec({"cond_include_coarse_sdf": True})
    prompt3r = resolve_conditioning_spec({"cond_include_coarse_sdf": False})
    with pytest.raises(ConditioningContractError, match="missing channel_contract"):
        validate_checkpoint_contract({}, legacy)
    assert validate_checkpoint_contract({}, legacy, legacy_compatibility=True) == legacy
    with pytest.raises(ConditioningContractError, match="legacy 8-channel"):
        validate_checkpoint_contract({}, prompt3r, legacy_compatibility=True)
    checkpoint = {"channel_contract": legacy.to_dict()}
    with pytest.raises(ConditioningContractError, match="mismatch"):
        validate_checkpoint_contract(checkpoint, prompt3r)


def test_contract_channel_list_is_written_to_manifest(tmp_path):
    spec = resolve_conditioning_spec({"cond_include_coarse_sdf": False})
    start_manifest(tmp_path, {"cond_include_coarse_sdf": False}, 0, 7,
                   tmp_path, channel_contract=spec.to_dict())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["channel_contract"] == spec.to_dict()
    assert manifest["channel_contract"]["conditioning_channel_names"][0] == "cbct"
