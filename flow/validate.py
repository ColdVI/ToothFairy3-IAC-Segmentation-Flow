#!/usr/bin/env python3
"""
validate.py — real validation inference for checkpoint selection.

Runs the actual pipeline the model will be judged on (sliding-window ODE
integration -> SDF decode -> per-side metrics), NOT the training loss. The
checkpoint score is

    S_val = 0.5 * meanDice + 0.5 * meanClDice     (ties broken by lower HD95)

so `best.pt` is the model that segments best, not the one with the lowest MSE.
"""
import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from io_utils import physical_coord_grid, normalize_coords, voxel_spacing, sdf_stack_to_mask  # noqa: E402
from conditioning import build_conditioning                                                   # noqa: E402
from sliding_window import predict_volume                                                     # noqa: E402
from evaluation.metrics import dice, cldice, hd95                                              # noqa: E402


def _load_case(sid, images_dir, coarse_sdf_dir):
    img = nib.load(os.path.join(images_dir, f"{sid}_0000.nii.gz"))
    cbct = np.asanyarray(img.dataobj).astype(np.float32)
    sp = voxel_spacing(img)
    coords = normalize_coords(physical_coord_grid(cbct.shape, img.affine))
    co = np.load(os.path.join(coarse_sdf_dir, f"{sid}.npz"))
    coarse_sdf = co["sdf"].astype(np.float32)
    cond = build_conditioning(cbct, co["prob_left"].astype(np.float32),
                              co["prob_right"].astype(np.float32),
                              coarse_sdf[0], coarse_sdf[1], coords)
    return cond, coarse_sdf, sp


def validate(model, val_ids, images_dir, coarse_sdf_dir, gt_labels_dir,
             patch=96, steps=8, device="cpu", max_cases=None):
    model.eval()
    ids = val_ids if max_cases is None else val_ids[:max_cases]
    rows = []
    for sid in ids:
        cond, coarse_sdf, sp = _load_case(sid, images_dir, coarse_sdf_dir)
        endp = predict_volume(model, cond, coarse_sdf, patch=patch, steps=steps, device=device)
        pred = sdf_stack_to_mask(endp)
        gt = np.asanyarray(nib.load(os.path.join(gt_labels_dir, f"{sid}.nii.gz")).dataobj)
        for side in (1, 2):
            rows.append((dice(pred == side, gt == side),
                         cldice(pred == side, gt == side),
                         hd95(pred == side, gt == side, sp)))
    arr = np.array(rows, dtype=np.float64)
    mean_dice = float(np.nanmean(arr[:, 0]))
    mean_cldice = float(np.nanmean(arr[:, 1]))
    mean_hd95 = float(np.nanmean(arr[:, 2]))
    score = 0.5 * mean_dice + 0.5 * mean_cldice
    return {"dice": mean_dice, "cldice": mean_cldice, "hd95": mean_hd95, "score": score}
