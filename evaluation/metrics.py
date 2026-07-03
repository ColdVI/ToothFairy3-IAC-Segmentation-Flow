#!/usr/bin/env python3
"""
metrics.py — overlap + boundary + centerline metrics (physical, per side).

All spatial metrics take voxel `spacing` (mm) so results are physical, not
voxel-count artefacts. Each returns a scalar for a single (pred, gt) binary pair;
the CV/external evaluators aggregate over cases and sides.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return 2.0 * (p & g).sum() / denom


def _surface_distances(pred, gt, spacing):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return None
    # surface = voxels with at least one background neighbour (via EDT==0 border)
    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)
    from scipy.ndimage import binary_erosion
    surf_pred = pred & ~binary_erosion(pred)
    surf_gt = gt & ~binary_erosion(gt)
    d_pred_to_gt = dt_gt[surf_pred]
    d_gt_to_pred = dt_pred[surf_gt]
    return d_pred_to_gt, d_gt_to_pred


def hd95(pred, gt, spacing):
    sd = _surface_distances(pred, gt, spacing)
    if sd is None:
        return float("nan")
    d1, d2 = sd
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def nsd(pred, gt, spacing, tolerance_mm=1.0):
    """Normalised Surface Dice at a mm tolerance."""
    sd = _surface_distances(pred, gt, spacing)
    if sd is None:
        return float("nan")
    d1, d2 = sd
    n = (d1 <= tolerance_mm).sum() + (d2 <= tolerance_mm).sum()
    return float(n / (len(d1) + len(d2) + 1e-8))


def _skeletonize(mask):
    from skimage.morphology import skeletonize
    try:
        return skeletonize(mask.astype(bool))          # skimage >=0.21 supports 3D
    except TypeError:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(mask.astype(np.uint8)).astype(bool)


def cldice(pred, gt, eps=1e-6):
    """Hard clDice: topology-aware overlap of centerlines. 1.0 = best."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    sp, sg = _skeletonize(p), _skeletonize(g)
    tprec = (sp & g).sum() / (sp.sum() + eps)
    tsens = (sg & p).sum() / (sg.sum() + eps)
    return float(2 * tprec * tsens / (tprec + tsens + eps))
