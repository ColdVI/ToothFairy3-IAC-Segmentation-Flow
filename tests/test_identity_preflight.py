import json

import nibabel as nib
import numpy as np

import _pathsetup  # noqa: F401
from data.io_utils import mask_to_sdf_mm, normalize_sdf
from scripts.identity_preflight import (evaluate_case_paths, preflight_decision,
                                        summarize, write_reports)


def _preflight_case(tmp_path):
    paths = {name: tmp_path / name for name in
             ("images", "labels", "hard", "coarse", "gt_sdf")}
    for path in paths.values():
        path.mkdir()
    case_id = "case_a"; shape = (16, 16, 16); affine = np.eye(4)
    label = np.zeros(shape, np.uint8)
    label[2:13, 4:7, 4:7] = 1; label[3:14, 10:13, 9:12] = 2
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine),
             paths["images"] / f"{case_id}_0000.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), paths["labels"] / f"{case_id}.nii.gz")
    left = (label == 1).astype(np.float16); right = (label == 2).astype(np.float16)
    np.savez_compressed(paths["hard"] / f"{case_id}.npz", prob_left=left, prob_right=right)
    spacing = np.ones(3, np.float32)
    sdf = np.stack([normalize_sdf(mask_to_sdf_mm(label == side, spacing, 10), 10)
                    for side in (1, 2)]).astype(np.float16)
    np.savez_compressed(paths["coarse"] / f"{case_id}.npz", sdf=sdf, spacing=spacing,
                        prob_left=left, prob_right=right)
    np.savez_compressed(paths["gt_sdf"] / f"{case_id}.npz", sdf=sdf, spacing=spacing)
    checkpoint = tmp_path / "fold_0_checkpoint_final.pth"; checkpoint.write_bytes(b"fixture")
    provenance = {case_id: {"prediction_fold": 0, "source_checkpoint": str(checkpoint),
                            "artifact_type": "derived_one_hot"}}
    return paths, provenance


def test_three_identity_paths_are_equal_on_known_case(tmp_path):
    paths, provenance = _preflight_case(tmp_path)
    result = evaluate_case_paths("case_a", 0, paths["images"], paths["labels"],
                                 paths["hard"], paths["coarse"], paths["gt_sdf"],
                                 provenance, patch=16, steps=2, device="cpu")
    assert result["provenance_valid"] and result["geometry"]["valid"]
    assert result["direct_vs_sdf_voxel_difference"] == 0
    assert result["direct_vs_full_path_voxel_difference"] == 0
    assert result["full_path_sdf_max_abs_error"] < 2e-6
    for side in result["sides"]:
        assert side["direct"]["dice"] == side["sdf_decode"]["dice"] == 1.0
        assert side["direct"]["dice"] == side["full_path"]["dice"]
    assert preflight_decision([result], 1) == "PASS"


def test_preflight_decision_stops_in_required_order():
    clean = {"provenance_valid": True, "geometry": {"valid": True},
             "direct_vs_sdf_voxel_difference": 0,
             "direct_vs_full_path_voxel_difference": 0,
             "full_path_sdf_max_abs_error": 0.0}
    assert preflight_decision([], 1) == "FAIL_CASE_COUNT"
    assert preflight_decision([{**clean, "provenance_valid": False}], 1) == "FAIL_FOLD_PROVENANCE"
    assert preflight_decision([{**clean, "geometry": {"valid": False}}], 1) == "FAIL_GEOMETRY"
    assert preflight_decision([{**clean, "direct_vs_sdf_voxel_difference": 1}], 1) == "FAIL_SDF_ROUNDTRIP"
    assert preflight_decision([{**clean, "direct_vs_full_path_voxel_difference": 1}], 1) == "FAIL_FULL_PATH_PLUMBING"
    assert preflight_decision([{**clean, "full_path_sdf_max_abs_error": 1e-3}], 1) == "FAIL_FULL_PATH_NUMERICS"


def test_preflight_reports_have_three_paths_and_refuse_overwrite(tmp_path):
    paths, provenance = _preflight_case(tmp_path)
    result = evaluate_case_paths("case_a", 0, paths["images"], paths["labels"],
                                 paths["hard"], paths["coarse"], paths["gt_sdf"],
                                 provenance, patch=16, steps=2, device="cpu")
    summary = summarize([result], 1)
    prefix = tmp_path / "identity_preflight_40"
    write_reports(prefix, [result], summary)
    saved = json.loads(prefix.with_suffix(".json").read_text())
    assert set(saved["summary"]["metrics_per_side"]) == {"direct", "sdf_decode", "full_path"}
    assert (tmp_path / "identity_preflight_40_cases.csv").is_file()
    try:
        write_reports(prefix, [result], summary)
    except FileExistsError:
        pass
    else:
        raise AssertionError("preflight report was overwritten")
