"""Small binary segmentation metric set in physical units."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def hd95_mm(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() or not target.any():
        return float("inf")
    pred_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    # The union bounding box contains every queried and reference surface
    # point, so cropping changes neither directed surface-distance set while
    # avoiding two full-CBCT EDT allocations.
    coordinates = np.argwhere(pred_surface | target_surface)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    pred_surface = pred_surface[crop]
    target_surface = target_surface[crop]
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate((distance_to_target[pred_surface], distance_to_pred[target_surface]))
    return float(np.percentile(distances, 95))


def connected_components(mask: np.ndarray) -> int:
    return int(label(np.asarray(mask, dtype=bool))[1])
