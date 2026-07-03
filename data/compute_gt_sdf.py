#!/usr/bin/env python3
"""
compute_gt_sdf.py — cache ground-truth per-side physical SDFs.

For each converted label (Dataset801_IAC_LR/labelsTr/*.nii.gz) compute the
Left and Right signed distance fields in MILLIMETRES (anisotropic, using voxel
spacing), truncate at +-clip_mm and store normalised to [-1,1]. These are the
flow-matching x1 targets; caching once avoids recomputing EDTs every epoch.

Stored per case as compressed npz: sdf (2,D,H,W) float16, plus spacing.
    python data/compute_gt_sdf.py --labels $nnUNet_raw/Dataset801_IAC_LR/labelsTr \
        --out outputs/gt_sdf --clip-mm 10
"""
import argparse
import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import voxel_spacing, mask_to_sdf_mm, normalize_sdf   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labelsTr of Dataset801_IAC_LR")
    ap.add_argument("--out", default="outputs/gt_sdf")
    ap.add_argument("--clip-mm", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.labels) if f.endswith(".nii.gz"))
    if a.limit:
        files = files[: a.limit]

    for i, f in enumerate(files):
        sid = f[: -len(".nii.gz")]
        img = nib.load(os.path.join(a.labels, f))
        lab = np.asanyarray(img.dataobj)
        sp = voxel_spacing(img)
        left = normalize_sdf(mask_to_sdf_mm(lab == 1, sp, a.clip_mm), a.clip_mm)
        right = normalize_sdf(mask_to_sdf_mm(lab == 2, sp, a.clip_mm), a.clip_mm)
        sdf = np.stack([left, right]).astype(np.float16)
        np.savez_compressed(os.path.join(a.out, f"{sid}.npz"), sdf=sdf,
                            spacing=sp.astype(np.float32), clip_mm=np.float32(a.clip_mm))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    print(f"[gt-sdf] wrote {len(files)} caches -> {a.out}")


if __name__ == "__main__":
    main()
