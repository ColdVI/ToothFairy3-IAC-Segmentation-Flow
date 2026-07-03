#!/usr/bin/env python3
"""
compute_coarse_sdf.py — cache coarse per-side SDFs from nnU-Net OOF predictions.

Consumes the Out-Of-Fold probability maps produced by nnunet/predict_oof.py
(one npz per case with prob_left/prob_right), thresholds them into a coarse
Left/Right mask, and computes the physical SDF exactly like the GT cache. These
coarse SDFs are the residual-flow starting point x0 = SDF(nnU-Net) — so they are
built ONLY from OOF (leakage-free) predictions, never from GT.

    python data/compute_coarse_sdf.py --oof outputs/oof_probs \
        --ref-labels $nnUNet_raw/Dataset801_IAC_LR/labelsTr \
        --out outputs/coarse_sdf --prob-thresh 0.5 --clip-mm 10
"""
import argparse
import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import voxel_spacing, mask_to_sdf_mm, normalize_sdf   # noqa: E402


def coarse_mask_from_probs(prob_left, prob_right, thresh):
    """Threshold + argmax into a {0,1,2} coarse mask (Left=1, Right=2)."""
    fg = (np.maximum(prob_left, prob_right) >= thresh)
    side_left = prob_left >= prob_right
    out = np.zeros(prob_left.shape, dtype=np.uint8)
    out[fg & side_left] = 1
    out[fg & ~side_left] = 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="dir of *.npz with prob_left/prob_right")
    ap.add_argument("--ref-labels", required=True, help="labelsTr for spacing reference")
    ap.add_argument("--out", default="outputs/coarse_sdf")
    ap.add_argument("--prob-thresh", type=float, default=0.5)
    ap.add_argument("--clip-mm", type=float, default=10.0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.oof) if f.endswith(".npz"))
    for i, f in enumerate(files):
        sid = f[: -len(".npz")]
        d = np.load(os.path.join(a.oof, f))
        pl, pr = d["prob_left"].astype(np.float32), d["prob_right"].astype(np.float32)
        ref = nib.load(os.path.join(a.ref_labels, f"{sid}.nii.gz"))
        sp = voxel_spacing(ref)
        coarse = coarse_mask_from_probs(pl, pr, a.prob_thresh)
        left = normalize_sdf(mask_to_sdf_mm(coarse == 1, sp, a.clip_mm), a.clip_mm)
        right = normalize_sdf(mask_to_sdf_mm(coarse == 2, sp, a.clip_mm), a.clip_mm)
        sdf = np.stack([left, right]).astype(np.float16)
        np.savez_compressed(os.path.join(a.out, f"{sid}.npz"), sdf=sdf,
                            prob_left=pl.astype(np.float16), prob_right=pr.astype(np.float16),
                            spacing=sp.astype(np.float32))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    print(f"[coarse-sdf] wrote {len(files)} caches -> {a.out}")


if __name__ == "__main__":
    main()
