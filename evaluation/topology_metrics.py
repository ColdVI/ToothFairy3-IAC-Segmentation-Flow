#!/usr/bin/env python3
"""
topology_metrics.py — the metrics where a flow method should beat nnU-Net.

A per-voxel Dice can look fine while the canal is broken into pieces. These
capture that: component count, Betti-0 error (component count vs the ideal 1),
centerline gap length, spurious-branch length, left/right swap rate and empty
predictions.
"""
import numpy as np
from scipy.ndimage import label as cc_label


def n_components(mask):
    _, n = cc_label(mask.astype(bool))
    return int(n)


def betti0_error(mask, expected=1):
    """|#components - expected|. For a single canal side the ideal is 1."""
    if not mask.any():
        return expected
    return abs(n_components(mask) - expected)


def centerline_gap_length(pred, gt, spacing):
    """
    Total physical length of GT centerline that the prediction fails to cover
    (GT skeleton voxels whose nearest prediction voxel is > 1 voxel away),
    approximated in mm via the mean spacing.
    """
    from evaluation.metrics import _skeletonize
    from scipy.ndimage import distance_transform_edt
    g = gt.astype(bool)
    if not g.any():
        return 0.0
    sk = _skeletonize(g)
    if not pred.any():
        return float(sk.sum() * float(np.mean(spacing)))
    dt = distance_transform_edt(~pred.astype(bool), sampling=spacing)
    gap_vox = (dt[sk] > float(np.mean(spacing)))
    return float(gap_vox.sum() * float(np.mean(spacing)))


def false_branch_length(pred, gt, spacing):
    """Physical length of predicted centerline lying outside the GT (spurious)."""
    from evaluation.metrics import _skeletonize
    from scipy.ndimage import distance_transform_edt
    p = pred.astype(bool)
    if not p.any():
        return 0.0
    sk = _skeletonize(p)
    if not gt.any():
        return float(sk.sum() * float(np.mean(spacing)))
    dt = distance_transform_edt(~gt.astype(bool), sampling=spacing)
    false_vox = (dt[sk] > float(np.mean(spacing)))
    return float(false_vox.sum() * float(np.mean(spacing)))


def lr_swap_rate(pred_lr, gt_lr):
    """
    Fraction of foreground voxels assigned the wrong side. pred_lr / gt_lr are
    {0,1,2} maps. Measured over the union of GT foreground.
    """
    gt_fg = gt_lr > 0
    if not gt_fg.any():
        return 0.0
    both = gt_fg & (pred_lr > 0)
    if not both.any():
        return 0.0
    swapped = (pred_lr[both] != gt_lr[both])
    return float(swapped.sum() / both.sum())


def empty_prediction(pred):
    return int(not pred.astype(bool).any())
