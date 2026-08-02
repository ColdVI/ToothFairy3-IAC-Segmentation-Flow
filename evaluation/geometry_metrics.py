"""Shared physical geometry diagnostics for flow and shortcut evaluation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from evaluation.metrics import _skeletonize


def signed_surface_distance_summary(pred, gt, spacing):
    """Signed distance of the predicted surface to GT (negative inside GT)."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    surface = pred & ~binary_erosion(pred)
    if not surface.any():
        return {"mean_mm": None, "median_mm": None, "q05_mm": None,
                "q95_mm": None, "valid": False,
                "reason": "empty_prediction_surface"}
    if not gt.any():
        return {"mean_mm": None, "median_mm": None, "q05_mm": None,
                "q95_mm": None, "valid": False, "reason": "empty_ground_truth"}
    outside = distance_transform_edt(~gt, sampling=spacing)
    inside = distance_transform_edt(gt, sampling=spacing)
    signed = outside
    signed[gt] = -inside[gt]
    values = signed[surface]
    return {"mean_mm": float(values.mean()), "median_mm": float(np.median(values)),
            "q05_mm": float(np.percentile(values, 5)),
            "q95_mm": float(np.percentile(values, 95)), "valid": True,
            "reason": "ok"}


def radius_profile_summary(pred, gt, spacing):
    """Radius error sampled along the GT centreline in physical millimetres."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    skeleton = _skeletonize(gt)
    if not skeleton.any():
        return {"signed_mean_mm": None, "mae_mm": None, "valid": False,
                "reason": "empty_gt_skeleton"}
    gt_radius = distance_transform_edt(gt, sampling=spacing)[skeleton]
    pred_radius = distance_transform_edt(pred, sampling=spacing)[skeleton]
    difference = pred_radius - gt_radius
    return {"signed_mean_mm": float(difference.mean()),
            "mae_mm": float(np.abs(difference).mean()), "valid": True,
            "reason": "ok"}


def false_positive_component_summary(pred, gt, spacing):
    """Count predicted components with no GT overlap and their furthest reach."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    labelled, count = label(pred)
    fp_labels = [component for component in range(1, count + 1)
                 if not np.any(gt[labelled == component])]
    if not fp_labels:
        return {"fp_component_count": 0, "max_fp_distance_mm": 0.0,
                "valid": True, "reason": "ok"}
    if not gt.any():
        return {"fp_component_count": len(fp_labels), "max_fp_distance_mm": None,
                "valid": False, "reason": "empty_ground_truth"}
    distance = distance_transform_edt(~gt, sampling=spacing)
    max_distance = max(float(distance[labelled == component].max())
                       for component in fp_labels)
    return {"fp_component_count": len(fp_labels),
            "max_fp_distance_mm": max_distance, "valid": True, "reason": "ok"}


def side_geometry_metrics(pred, gt, spacing):
    """Return all Prompt-3R geometry fields for one anatomical side."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    surface = signed_surface_distance_summary(pred, gt, spacing)
    radius = radius_profile_summary(pred, gt, spacing)
    fp = false_positive_component_summary(pred, gt, spacing)
    _, components = label(pred)
    volume_ratio = float(pred.sum() / gt.sum()) if gt.any() else None
    valid = bool(surface["valid"] and radius["valid"] and fp["valid"])
    reasons = sorted({item["reason"] for item in (surface, radius, fp)
                      if not item["valid"]})
    return {
        "volume_ratio": volume_ratio,
        "signed_surface_mean_mm": surface["mean_mm"],
        "signed_surface_median_mm": surface["median_mm"],
        "signed_surface_q05_mm": surface["q05_mm"],
        "signed_surface_q95_mm": surface["q95_mm"],
        "radius_bias_mm": radius["signed_mean_mm"],
        "radius_mae_mm": radius["mae_mm"],
        "connected_component_count": int(components),
        "fp_component_count": fp["fp_component_count"],
        "max_fp_distance_mm": fp["max_fp_distance_mm"],
        "geometry_valid": valid,
        "geometry_reason": "ok" if valid else ";".join(reasons),
    }
