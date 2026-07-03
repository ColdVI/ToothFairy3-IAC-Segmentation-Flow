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
import sys
import tempfile

import numpy as np
import nibabel as nib


def link_val_images(val_ids, images_dir, tmp):
    os.makedirs(tmp, exist_ok=True)
    for sid in val_ids:
        src = os.path.join(images_dir, f"{sid}_0000.nii.gz")
        dst = os.path.join(tmp, f"{sid}_0000.nii.gz")
        if os.path.lexists(dst):
            os.remove(dst)
        if os.path.isfile(src):
            os.symlink(os.path.realpath(src), dst)
    return tmp


def predict_fold(fold, val_ids, images_dir, dataset, config, trainer, out_dir):
    with tempfile.TemporaryDirectory() as tin, tempfile.TemporaryDirectory() as tout:
        link_val_images(val_ids, images_dir, tin)
        cmd = ["nnUNetv2_predict", "-i", tin, "-o", tout,
               "-d", str(dataset), "-c", config, "-tr", trainer,
               "-f", str(fold), "--disable_tta"]
        print("[oof]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        for sid in val_ids:
            seg_path = os.path.join(tout, f"{sid}.nii.gz")
            if not os.path.isfile(seg_path):
                print(f"[oof] WARNING missing prediction for {sid}")
                continue
            seg = np.asanyarray(nib.load(seg_path).dataobj)
            np.savez_compressed(os.path.join(out_dir, f"{sid}.npz"),
                                prob_left=(seg == 1).astype(np.float16),
                                prob_right=(seg == 2).astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=801)
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--trainer", default="nnUNetTrainerIAC_NoMirror")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/oof_probs")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    splits = json.load(open(a.splits))
    for f, fold in enumerate(splits["folds"]):
        predict_fold(f, fold["val"], a.images, a.dataset, a.config, a.trainer, a.out)
    print(f"[oof] done -> {a.out} (one npz per development case)")


if __name__ == "__main__":
    main()
