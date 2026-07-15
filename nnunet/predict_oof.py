#!/usr/bin/env python3
"""
predict_oof.py — leakage-free Out-Of-Fold nnU-Net predictions for the flow prior.

For each fold f, the model trained on the OTHER 4 folds predicts fold f's
validation cases. Concatenated over folds this yields a prediction for every
development case that was NEVER in that model's training set — the only correct
prior for training the residual flow. Training the flow on ordinary (in-sample)
nnU-Net outputs would let it learn from unrealistically clean masks.

Each OOF case is saved as outputs/oof_probs/<sid>.npz with prob_left/prob_right.
v1 derives these from the OOF segmentation (one-hot); swapping in true softmax
(nnUNetv2_predict --save_probabilities, resampled to label space) is a documented
later enhancement.

    python nnunet/predict_oof.py --dataset 801 --config 3d_fullres \
        --trainer nnUNetTrainerIAC_NoMirror --splits configs/splits.json \
        --images $nnUNet_raw/Dataset801_IAC_LR/imagesTr --out outputs/oof_probs
"""
import argparse
import json
import os
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


def predict_fold(fold, val_ids, images_dir, dataset, config, trainer, out_dir,
                 device="cuda", step_size=0.5, npp=3, nps=3, not_on_device=False):
    with tempfile.TemporaryDirectory() as tin, tempfile.TemporaryDirectory() as tout:
        link_val_images(val_ids, images_dir, tin)
        cmd = ["nnUNetv2_predict", "-i", tin, "-o", tout,
               "-d", str(dataset), "-c", config, "-tr", trainer,
               "-f", str(fold), "--disable_tta",
               "-device", device, "-step_size", str(step_size),
               "-npp", str(npp), "-nps", str(nps)]
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
    ap.add_argument("--resume", action="store_true",
                    help="skip readable, structurally valid case caches")
    a = ap.parse_args()

    # This project trains with custom trainer classes stored next to this script.
    # Recent nnU-Net versions discover out-of-package trainers through this env var.
    os.environ.setdefault("nnUNet_extTrainer", os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(a.out, exist_ok=True)
    with open(a.splits) as f:
        splits = json.load(f)
    if len(splits.get("folds", [])) != 5:
        raise ValueError(f"expected 5 folds in {a.splits}")
    fold_indices = range(len(splits["folds"])) if a.fold is None else [a.fold]
    for f in fold_indices:
        if not 0 <= f < len(splits["folds"]):
            raise ValueError(f"fold must be in [0, {len(splits['folds']) - 1}], got {f}")
        fold = splits["folds"][f]
        val_ids = fold["val"]
        if a.resume:
            val_ids = [sid for sid in val_ids if not cache_is_valid(
                os.path.join(a.out, f"{sid}.npz"))]
            if not val_ids:
                print(f"[oof] fold {f}: all cases already cached, skipping")
                continue
        t0 = time.monotonic()
        print(f"[oof] fold {f}/{len(splits['folds']) - 1}: {len(val_ids)} cases starting "
              f"(cached total: {sum(cache_is_valid(os.path.join(a.out, f'{sid}.npz')) for sid in splits['development'])})",
              flush=True)
        predict_fold(f, val_ids, a.images, a.dataset, a.config, a.trainer, a.out,
                     a.device, a.step_size, a.npp, a.nps, a.not_on_device)
        cached = sum(cache_is_valid(os.path.join(a.out, f"{sid}.npz"))
                     for sid in splits["development"])
        print(f"[oof] fold {f} done in {(time.monotonic() - t0) / 60:.1f} min | "
              f"persistent cache: {cached}/{len(splits['development'])}", flush=True)
    print(f"[oof] done -> {a.out} (one npz per development case)")


if __name__ == "__main__":
    main()
