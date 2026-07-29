from pathlib import Path

import numpy as np
import pytest

import _pathsetup  # noqa: F401
from nnunet.predict_oof import (cache_is_valid, resolve_checkpoint,
                                softmax_cache_is_valid)


def test_hard_and_true_softmax_have_separate_validators(tmp_path):
    hard = tmp_path / "case.npz"
    left = np.zeros((3, 4, 5), np.float16)
    left[1] = 1
    right = np.zeros_like(left)
    right[2] = 1
    np.savez_compressed(hard, prob_left=left, prob_right=right)
    assert cache_is_valid(hard)
    assert not softmax_cache_is_valid(hard)

    soft = tmp_path / "soft.npz"
    probs = np.full((3, 3, 4, 5), 1 / 3, np.float32)
    np.savez_compressed(soft, probabilities=probs)
    assert softmax_cache_is_valid(soft)
    assert not cache_is_valid(soft)


def test_softmax_validator_rejects_non_normalized_or_one_hot_alias(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, probabilities=np.full((3, 2, 2, 2), .5, np.float32))
    assert not softmax_cache_is_valid(bad)


def test_checkpoint_resolution_is_exact_and_unambiguous(tmp_path):
    checkpoint = (tmp_path / "Dataset801_IAC_LR" /
                  "nnUNetTrainerIAC_NoMirror__nnUNetPlans__3d_fullres" /
                  "fold_2" / "checkpoint_final.pth")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    assert Path(resolve_checkpoint(tmp_path, 801, "nnUNetTrainerIAC_NoMirror",
                                   "3d_fullres", 2)) == checkpoint
    duplicate = (tmp_path / "Dataset801_duplicate" /
                 "nnUNetTrainerIAC_NoMirror__OtherPlans__3d_fullres" /
                 "fold_2" / "checkpoint_final.pth")
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"model")
    with pytest.raises(FileNotFoundError, match="exactly one checkpoint"):
        resolve_checkpoint(tmp_path, 801, "nnUNetTrainerIAC_NoMirror", "3d_fullres", 2)
