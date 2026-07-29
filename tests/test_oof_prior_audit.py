import json

import nibabel as nib
import numpy as np

import _pathsetup  # noqa: F401
from analysis.oof_prior_audit import (audit_case, expected_fold_map, load_oof,
                                      probability_stats, write_reports)
from data.io_utils import mask_to_sdf_mm, normalize_sdf


def _case_tree(tmp_path, case_id="case_a"):
    paths = {name: tmp_path / name for name in
             ("images", "labels", "hard", "softmax", "coarse", "gt")}
    for path in paths.values():
        path.mkdir()
    affine = np.diag([.5, .5, .5, 1.0])
    shape = (12, 10, 8)
    label = np.zeros(shape, np.uint8)
    label[2:8, 2:4, 2:4] = 1
    label[3:9, 6:8, 4:6] = 2
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine),
             paths["images"] / f"{case_id}_0000.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), paths["labels"] / f"{case_id}.nii.gz")
    left = (label == 1).astype(np.float16); right = (label == 2).astype(np.float16)
    np.savez_compressed(paths["hard"] / f"{case_id}.npz", prob_left=left, prob_right=right)
    spacing = np.array([.5, .5, .5], np.float32)
    sdf = np.stack([normalize_sdf(mask_to_sdf_mm(label == side, spacing, 10), 10)
                    for side in (1, 2)]).astype(np.float16)
    np.savez_compressed(paths["coarse"] / f"{case_id}.npz", sdf=sdf, spacing=spacing,
                        prob_left=left, prob_right=right)
    np.savez_compressed(paths["gt"] / f"{case_id}.npz", sdf=sdf, spacing=spacing)
    return paths


def test_expected_fold_map_reconstructs_provenance():
    splits = {"development": ["a", "b"],
              "folds": [{"val": ["a"]}, {"val": ["b"]}]}
    assert expected_fold_map(splits) == {"a": 0, "b": 1}


def test_derived_one_hot_is_not_called_softmax(tmp_path):
    paths = _case_tree(tmp_path)
    probs, hard, artifact_type = load_oof(paths["hard"] / "case_a.npz")
    assert artifact_type == "derived_one_hot"
    assert set(np.unique(probs)) == {0.0, 1.0}
    assert probability_stats(probs)["entropy"]["max"] < 1e-6
    assert set(np.unique(hard)) == {0, 1, 2}


def test_true_softmax_remains_separate(tmp_path):
    path = tmp_path / "case.npz"
    probabilities = np.full((3, 4, 4, 4), .1, np.float32)
    probabilities[0] = .8
    np.savez_compressed(path, probabilities=probabilities)
    probs, _, artifact_type = load_oof(path)
    assert artifact_type == "true_softmax"
    assert probability_stats(probs)["entropy"]["mean"] > 0


def test_case_audit_checks_geometry_roundtrip_and_provenance(tmp_path):
    paths = _case_tree(tmp_path)
    provenance = {"case_a": {"prediction_fold": 2,
                              "source_checkpoint": "fold_2/checkpoint_final.pth"}}
    row = audit_case("case_a", 2, paths["images"], paths["labels"], paths["hard"],
                     paths["softmax"], paths["coarse"], paths["gt"], provenance)
    assert row["status"] == "valid"
    assert row["artifact_type"] == "derived_one_hot"
    assert row["hard_vs_sdf_sign_voxel_difference"] == 0
    assert row["conditioning_redundancy"]["hard_vs_sdf_sign_voxel_equality"] == 1.0


def test_reports_refuse_to_overwrite(tmp_path):
    case = {"case_id": "x", "status": "missing", "missing": ["hard_oof"]}
    prefix = tmp_path / "oof_prior_audit"
    write_reports(prefix, [case], legacy_adapter=True)
    assert json.loads(prefix.with_suffix(".json").read_text())["summary"]["total_cases"] == 1
    try:
        write_reports(prefix, [case], legacy_adapter=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError("audit report was overwritten")
