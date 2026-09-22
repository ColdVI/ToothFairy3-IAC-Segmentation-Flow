"""Verified ToothFairy3 project paths for the 532-case IAC-B experiment.

The refiner now reads the original 532-case ToothFairy3 images/labels directly:
  * full TF3 label 3 -> internal Left IAC
  * full TF3 label 4 -> internal Right IAC

Frozen nnU-Net probability channels still come from Dataset801_IAC_LR:
  * channel 1 -> Left IAC
  * channel 2 -> Right IAC

Probability provenance:
  * 480 development cases: true 5-fold OOF probabilities in canalmanifold_oof_softmax/fold_k
  * 52 ToothFairy3S cases: one unseen frozen fold per case, exported to
    canalmanifold_oof_softmax/external_singlefold
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import case_folds, resolve_oof_npz

DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/ToothFairy/ToothFairy3")


def resolve_project_paths(root=DEFAULT_DRIVE_ROOT):
    root = Path(root)
    iac_runs = root / "iac_runs"
    tf3_full = root / "ToothFairy3"
    iac_dataset_480 = iac_runs / "dataset_cache_colab_v2" / "Dataset801_IAC_LR"
    nnunet_model = (
        iac_runs / "nnUNet_results" / "Dataset801_IAC_LR"
        / "nnUNetTrainerIAC_NoMirror__nnUNetPlans__3d_fullres"
    )
    prob = iac_runs / "canalmanifold_oof_softmax"
    splits = iac_runs / "configs_cache" / "splits.json"
    persist = root / "iacb_runs_532"
    return dict(
        root=root,
        iac_runs=iac_runs,
        tf3_full=tf3_full,
        raw=tf3_full,
        iac_dataset_480=iac_dataset_480,
        nnunet_model=nnunet_model,
        prob=prob,
        external_prob=prob / "external_singlefold",
        splits=splits,
        persist=persist,
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def preflight(paths, require_external_priors=False, sample_cases=3):
    tf3_full = Path(paths["tf3_full"])
    iac_dataset = Path(paths["iac_dataset_480"])
    prob = Path(paths["prob"])
    splits = Path(paths["splits"])
    nnunet_model = Path(paths["nnunet_model"])

    required = [
        tf3_full / "imagesTr", tf3_full / "labelsTr", tf3_full / "dataset.json",
        iac_dataset / "dataset.json", prob, splits, nnunet_model,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing project inputs:\n  - " + "\n  - ".join(missing))

    full_meta = _read_json(tf3_full / "dataset.json")
    iac_meta = _read_json(iac_dataset / "dataset.json")
    full_labels = full_meta.get("labels", {})
    iac_labels = iac_meta.get("labels", {})
    if int(full_labels.get("Left Inferior Alveolar Canal", -999)) != 3:
        raise ValueError("Full ToothFairy3 Left IAC must be label 3")
    if int(full_labels.get("Right Inferior Alveolar Canal", -999)) != 4:
        raise ValueError("Full ToothFairy3 Right IAC must be label 4")
    if int(iac_labels.get("Left_IAC", -999)) != 1 or int(iac_labels.get("Right_IAC", -999)) != 2:
        raise ValueError("Frozen Dataset801 nnU-Net probability contract must be Left=1, Right=2")

    split_obj = _read_json(splits)
    development = list(split_obj.get("development", []))
    external = list(split_obj.get("external_test", []))
    folds = case_folds(splits, include_external=True)

    if len(development) != 480 or len(external) != 52 or len(folds) != 532:
        raise ValueError(
            f"Expected 480 development + 52 external = 532 cases, got "
            f"{len(development)} + {len(external)}; mapped={len(folds)}"
        )

    checkpoints = {}
    for fold in range(5):
        ckpt = nnunet_model / f"fold_{fold}" / "checkpoint_final.pth"
        checkpoints[f"fold_{fold}"] = ckpt.exists()
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing user's nnU-Net checkpoint: {ckpt}")

    for case in development[:sample_cases]:
        img = tf3_full / "imagesTr" / f"{case}_0000.nii.gz"
        lab = tf3_full / "labelsTr" / f"{case}.nii.gz"
        if not img.exists() or not lab.exists():
            raise FileNotFoundError(f"Full TF3 is missing image/label for {case}")
        resolve_oof_npz(prob, case, folds[case])

    ext_found = sum((prob / "external_singlefold" / f"{c}.npz").exists() for c in external)
    if require_external_priors and ext_found != len(external):
        raise FileNotFoundError(
            f"External priors incomplete: {ext_found}/{len(external)} in {prob / 'external_singlefold'}"
        )

    return dict(
        tf3_full_num_training=full_meta.get("numTraining"),
        full_tf3_label_ids={"left": 3, "right": 4},
        frozen_probability_channel_ids={"left": 1, "right": 2},
        n_development=len(development),
        n_external=len(external),
        n_total=len(folds),
        n_external_priors_ready=ext_found,
        user_nnunet_checkpoints=checkpoints,
        paths={k: str(v) for k, v in paths.items()},
    )
