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
import time

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import voxel_spacing, mask_to_sdf_mm, normalize_sdf   # noqa: E402


def cache_is_valid(path):
    try:
        with np.load(path) as item:
            return item["sdf"].shape[0] == 2 and item["sdf"].ndim == 4
    except (OSError, ValueError, KeyError, IndexError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labelsTr of Dataset801_IAC_LR")
    ap.add_argument("--out", default="outputs/gt_sdf")
    ap.add_argument("--clip-mm", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip readable existing caches")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.labels) if f.endswith(".nii.gz"))
    if a.limit:
        files = files[: a.limit]
    t0 = time.monotonic()
    completed = 0

    for i, f in enumerate(files):
        sid = f[: -len(".nii.gz")]
        final_path = os.path.join(a.out, f"{sid}.npz")
        if a.resume and cache_is_valid(final_path):
            completed += 1
            continue
        img = nib.load(os.path.join(a.labels, f))
        lab = np.asanyarray(img.dataobj)
        sp = voxel_spacing(img)
        left = normalize_sdf(mask_to_sdf_mm(lab == 1, sp, a.clip_mm), a.clip_mm)
        right = normalize_sdf(mask_to_sdf_mm(lab == 2, sp, a.clip_mm), a.clip_mm)
        sdf = np.stack([left, right]).astype(np.float16)
        partial_path = final_path + ".partial.npz"
        np.savez_compressed(partial_path, sdf=sdf,
                            spacing=sp.astype(np.float32), clip_mm=np.float32(a.clip_mm))
        os.replace(partial_path, final_path)
        completed += 1
        if completed % 25 == 0 or completed == len(files):
            elapsed = time.monotonic() - t0
            rate = completed / max(elapsed, 1e-6)
            eta = (len(files) - completed) / max(rate, 1e-6)
            print(f"  {completed}/{len(files)} | {elapsed / 60:.1f} min elapsed | "
                  f"ETA {eta / 60:.1f} min", flush=True)
    print(f"[gt-sdf] wrote {len(files)} caches -> {a.out}")


if __name__ == "__main__":
    main()
