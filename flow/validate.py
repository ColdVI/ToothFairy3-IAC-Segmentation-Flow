#!/usr/bin/env python3
"""
validate.py — real validation inference for checkpoint selection.

Runs the actual pipeline the model will be judged on (sliding-window ODE
integration -> SDF decode -> per-side metrics), NOT the training loss. It
returns overlap, boundary, and topology metrics used by the lexicographic
best-any / best-safe checkpoint policy in train.py.
"""
import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from io_utils import physical_coord_grid, normalize_coords, voxel_spacing, sdf_stack_to_mask  # noqa: E402
from conditioning import build_conditioning, resolve_conditioning_spec                        # noqa: E402
from sliding_window import predict_volume                                                     # noqa: E402
from evaluation.metrics import dice, cldice, hd95                                              # noqa: E402
from evaluation.topology_metrics import betti0_error, centerline_gap_length                    # noqa: E402


def _load_case(sid, images_dir, coarse_sdf_dir, conditioning_spec=None):
    img = nib.load(os.path.join(images_dir, f"{sid}_0000.nii.gz"))
    cbct = np.asanyarray(img.dataobj).astype(np.float32)
    sp = voxel_spacing(img)
    coords = normalize_coords(physical_coord_grid(cbct.shape, img.affine))
    co = np.load(os.path.join(coarse_sdf_dir, f"{sid}.npz"))
    coarse_sdf = co["sdf"].astype(np.float32)
    spec = conditioning_spec or resolve_conditioning_spec()
    cond = build_conditioning(cbct, co["prob_left"].astype(np.float32),
                              co["prob_right"].astype(np.float32),
                              coarse_sdf[0], coarse_sdf[1], coords, spec=spec)
    return cond, coarse_sdf, sp


def validation_rows(model, val_ids, images_dir, coarse_sdf_dir, gt_labels_dir,
                    patch=96, steps=8, device="cpu", max_cases=None,
                    progress=False, conditioning_spec=None):
    """Run inference and return one metric row per case and anatomical side."""
    model.eval()
    ids = val_ids if max_cases is None else val_ids[:max_cases]
    rows = []
    for case_index, sid in enumerate(ids, start=1):
        cond, coarse_sdf, sp = _load_case(
            sid, images_dir, coarse_sdf_dir, conditioning_spec)
        endp = predict_volume(model, cond, coarse_sdf, patch=patch, steps=steps, device=device)
        pred = sdf_stack_to_mask(endp)
        gt = np.asanyarray(nib.load(os.path.join(gt_labels_dir, f"{sid}.nii.gz")).dataobj)
        for side in (1, 2):
            rows.append({"case_id": sid, "side": side,
                         "dice": dice(pred == side, gt == side),
                         "cldice": cldice(pred == side, gt == side),
                         "hd95": hd95(pred == side, gt == side, sp),
                         "gap_mm": centerline_gap_length(pred == side, gt == side, sp),
                         "betti0": betti0_error(pred == side)})
        if progress:
            print(f"[validate] {case_index}/{len(ids)} {sid}", flush=True)
    return rows


def summarize_rows(rows, aggregation="per_side"):
    """Aggregate validation rows either directly or after bilateral case means."""
    if not rows:
        raise ValueError("cannot summarize an empty validation result")
    metrics = ("dice", "cldice", "hd95", "gap_mm", "betti0")
    if aggregation == "per_side":
        values = {metric: [row[metric] for row in rows] for metric in metrics}
    elif aggregation == "per_case":
        case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
        values = {metric: [] for metric in metrics}
        for sid in case_ids:
            case_rows = [row for row in rows if row["case_id"] == sid]
            for metric in metrics:
                values[metric].append(float(np.nanmean([row[metric] for row in case_rows])))
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")
    means = {metric: float(np.nanmean(values[metric])) for metric in metrics}
    mean_dice = means["dice"]
    mean_cldice = means["cldice"]
    mean_hd95 = means["hd95"]
    score = 0.5 * mean_dice + 0.5 * mean_cldice
    return {**means, "score": score}


def validate(model, val_ids, images_dir, coarse_sdf_dir, gt_labels_dir,
             patch=96, steps=8, device="cpu", max_cases=None,
             conditioning_spec=None):
    rows = validation_rows(model, val_ids, images_dir, coarse_sdf_dir, gt_labels_dir,
                           patch=patch, steps=steps, device=device, max_cases=max_cases,
                           conditioning_spec=conditioning_spec)
    return summarize_rows(rows, aggregation="per_side")
