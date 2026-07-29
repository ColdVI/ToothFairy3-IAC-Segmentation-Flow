#!/usr/bin/env python3
"""
predict_oof.py — leakage-free Out-Of-Fold nnU-Net predictions for the flow prior.

For each fold f, the model trained on the OTHER 4 folds predicts fold f's
validation cases. Concatenated over folds this yields a prediction for every
development case that was NEVER in that model's training set — the only correct
prior for training the residual flow. Training the flow on ordinary (in-sample)
nnU-Net outputs would let it learn from unrealistically clean masks.

Hard OOF and true-softmax artifacts are deliberately separate. The hard output
is stored as ``prob_left``/``prob_right`` derived one-hot arrays. With
``--save-probabilities``, nnU-Net's official probability export is preserved in
a separate ``--softmax-out`` directory; a hard mask is never relabelled as a
probability map.

    python nnunet/predict_oof.py --dataset 801 --config 3d_fullres \
        --trainer nnUNetTrainerIAC_NoMirror --splits configs/splits.json \
        --images $nnUNet_raw/Dataset801_IAC_LR/imagesTr --out outputs/oof_probs
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import nibabel as nib


def link_val_images(val_ids, images_dir, tmp):
    os.makedirs(tmp, exist_ok=True)
    for sid in val_ids:
        src = os.path.join(images_dir, f"{sid}_0000.nii.gz")
        dst = os.path.join(tmp, f"{sid}_0000.nii.gz")
        if os.path.lexists(dst):
            os.remove(dst)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"missing validation image: {src}")
        os.symlink(os.path.realpath(src), dst)
    return tmp


def cache_is_valid(path):
    """Only resume past a readable two-channel OOF cache."""
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path) as cached:
            left = cached["prob_left"]
            right = cached["prob_right"]
            return left.shape == right.shape and left.ndim == 3 and left.size > 0
    except (OSError, ValueError, KeyError):
        return False


def softmax_cache_is_valid(path):
    """Validate an official nnU-Net probability export without changing it."""
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path) as cached:
            key = next((name for name in ("probabilities", "softmax", "probs")
                        if name in cached), None)
            if key is None:
                return False
            probs = cached[key]
            if probs.ndim != 4 or not np.isfinite(probs).all():
                return False
            class_axis = 0 if probs.shape[0] in (2, 3) else (
                -1 if probs.shape[-1] in (2, 3) else None)
            if class_axis is None:
                return False
            sums = probs.sum(axis=class_axis, dtype=np.float32)
            return (float(probs.min()) >= -1e-4 and float(probs.max()) <= 1.0001
                    and bool(np.allclose(sums, 1.0, atol=2e-3, rtol=2e-3)))
    except (OSError, ValueError, KeyError):
        return False


def resolve_checkpoint(results_root, dataset, trainer, config, fold):
    """Resolve the exact Track-A fold checkpoint, refusing ambiguity."""
    root = os.path.realpath(results_root)
    patterns = [
        os.path.join(root, f"Dataset{int(dataset):03d}_*",
                     f"{trainer}__*__{config}", f"fold_{fold}", "checkpoint_final.pth"),
        os.path.join(root, f"Dataset{int(dataset):03d}_*",
                     f"{trainer}__*__{config}", str(fold), "checkpoint_final.pth"),
    ]
    import glob
    matches = sorted({os.path.realpath(path) for pattern in patterns
                      for path in glob.glob(pattern) if os.path.isfile(path)})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one checkpoint for dataset={dataset}, fold={fold}; "
            f"found {len(matches)} under {root}: {matches[:3]}")
    return matches[0]


def _atomic_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    partial = dst + ".partial"
    shutil.copy2(src, partial)
    os.replace(partial, dst)


def predict_fold(fold, val_ids, images_dir, dataset, config, trainer, out_dir,
                 device="cuda", step_size=0.5, npp=3, nps=3, not_on_device=False,
                 save_probabilities=False, softmax_out=None):
    if save_probabilities and not softmax_out:
        raise ValueError("--save-probabilities requires a separate --softmax-out directory")
    with tempfile.TemporaryDirectory() as tin, tempfile.TemporaryDirectory() as tout:
        link_val_images(val_ids, images_dir, tin)
        cmd = ["nnUNetv2_predict", "-i", tin, "-o", tout,
               "-d", str(dataset), "-c", config, "-tr", trainer,
               "-f", str(fold), "--disable_tta",
               "-device", device, "-step_size", str(step_size),
               "-npp", str(npp), "-nps", str(nps)]
        if save_probabilities:
            cmd.append("--save_probabilities")
        if device != "cuda" or not_on_device:
            cmd.append("--not_on_device")
        print("[oof]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        for sid in val_ids:
            seg_path = os.path.join(tout, f"{sid}.nii.gz")
            if not os.path.isfile(seg_path):
                print(f"[oof] WARNING missing prediction for {sid}")
                continue
            seg = np.asanyarray(nib.load(seg_path).dataobj)
            final_path = os.path.join(out_dir, f"{sid}.npz")
            partial_path = final_path + ".partial.npz"
            np.savez_compressed(partial_path,
                                prob_left=(seg == 1).astype(np.float16),
                                prob_right=(seg == 2).astype(np.float16))
            os.replace(partial_path, final_path)
            if save_probabilities:
                source_softmax = os.path.join(tout, f"{sid}.npz")
                if not softmax_cache_is_valid(source_softmax):
                    raise ValueError(f"invalid official softmax export for {sid}: {source_softmax}")
                _atomic_copy(source_softmax, os.path.join(softmax_out, f"{sid}.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=801)
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--trainer", default="nnUNetTrainerIAC_NoMirror")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/oof_probs")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"],
                    help="device passed to nnUNetv2_predict")
    ap.add_argument("--step-size", type=float, default=0.5,
                    help="nnU-Net sliding-window step size (larger is faster but changes output)")
    ap.add_argument("--npp", type=int, default=3, help="preprocessing worker count")
    ap.add_argument("--nps", type=int, default=3, help="segmentation export worker count")
    ap.add_argument("--not-on-device", action="store_true",
                    help="keep sliding-window accumulation in host RAM (useful for smaller CUDA GPUs)")
    ap.add_argument("--fold", type=int, default=None,
                    help="run only one fold; use with --resume for reset-safe persistent caching")
    ap.add_argument("--case-id", default=None,
                    help="run exactly one case; it must belong to the selected validation fold")
    ap.add_argument("--save-probabilities", action="store_true",
                    help="preserve nnU-Net's official softmax export separately")
    ap.add_argument("--softmax-out", default=None,
                    help="separate directory for official probabilities (required with export)")
    ap.add_argument("--resume", action="store_true",
                    help="skip readable, structurally valid case caches")
    a = ap.parse_args()

    # This project trains with custom trainer classes stored next to this script.
    # Recent nnU-Net versions discover out-of-package trainers through this env var.
    os.environ.setdefault("nnUNet_extTrainer", os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(a.out, exist_ok=True)
    if a.save_probabilities:
        if not a.softmax_out:
            ap.error("--save-probabilities requires --softmax-out")
        os.makedirs(a.softmax_out, exist_ok=True)
    with open(a.splits) as f:
        splits = json.load(f)
    if len(splits.get("folds", [])) != 5:
        raise ValueError(f"expected 5 folds in {a.splits}")
    if a.case_id:
        matching_folds = [index for index, item in enumerate(splits["folds"])
                          if a.case_id in item["val"]]
        if len(matching_folds) != 1:
            raise ValueError(f"case {a.case_id} belongs to {len(matching_folds)} validation folds")
        if a.fold is not None and a.fold != matching_folds[0]:
            raise ValueError(f"case {a.case_id} is validation fold {matching_folds[0]}, not {a.fold}")
        fold_indices = [matching_folds[0]]
    else:
        fold_indices = range(len(splits["folds"])) if a.fold is None else [a.fold]
    for f in fold_indices:
        if not 0 <= f < len(splits["folds"]):
            raise ValueError(f"fold must be in [0, {len(splits['folds']) - 1}], got {f}")
        fold = splits["folds"][f]
        val_ids = [a.case_id] if a.case_id else fold["val"]
        if a.resume:
            val_ids = [sid for sid in val_ids if not (
                cache_is_valid(os.path.join(a.out, f"{sid}.npz"))
                and (not a.save_probabilities or softmax_cache_is_valid(
                    os.path.join(a.softmax_out, f"{sid}.npz"))))]
            if not val_ids:
                print(f"[oof] fold {f}: all cases already cached, skipping")
                continue
        t0 = time.monotonic()
        print(f"[oof] fold {f}/{len(splits['folds']) - 1}: {len(val_ids)} cases starting "
              f"(cached total: {sum(cache_is_valid(os.path.join(a.out, f'{sid}.npz')) for sid in splits['development'])})",
              flush=True)
        predict_fold(f, val_ids, a.images, a.dataset, a.config, a.trainer, a.out,
                     a.device, a.step_size, a.npp, a.nps, a.not_on_device,
                     a.save_probabilities, a.softmax_out)
        cached = sum(cache_is_valid(os.path.join(a.out, f"{sid}.npz"))
                     for sid in splits["development"])
        print(f"[oof] fold {f} done in {(time.monotonic() - t0) / 60:.1f} min | "
              f"persistent cache: {cached}/{len(splits['development'])}", flush=True)
    print(f"[oof] done -> {a.out} (one npz per development case)")


if __name__ == "__main__":
    main()
