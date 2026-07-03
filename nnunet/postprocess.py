#!/usr/bin/env python3
"""
postprocess.py — connected-component cleanup for L/R IAC masks.

NOT hard-coded to "keep the largest component". The rules (minimum physical
component volume, distance of a fragment to the main component, whether to keep
multiple nearby pieces) are parameters to be TUNED on the CV folds; both raw and
post-processed metrics get reported so the effect is visible, per the spec.
"""
import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import distance_transform_edt


def _clean_side(mask_side, spacing, min_volume_mm3, max_gap_mm, keep_multiple):
    """Clean one binary side. Keep the largest component, then optionally re-add
    smaller components that are large enough AND close enough to the main one."""
    lab, n = cc_label(mask_side)
    if n <= 1:
        return mask_side
    sizes = np.array([(lab == i).sum() for i in range(1, n + 1)])
    vox_mm3 = float(np.prod(spacing))
    main = int(sizes.argmax()) + 1
    out = (lab == main)
    if not keep_multiple:
        return out
    dt_main = distance_transform_edt(~out, sampling=spacing)
    for i in range(1, n + 1):
        if i == main:
            continue
        if sizes[i - 1] * vox_mm3 < min_volume_mm3:
            continue
        comp = lab == i
        if dt_main[comp].min() <= max_gap_mm:
            out = out | comp
    return out


def postprocess(mask, spacing, min_volume_mm3=2.0, max_gap_mm=3.0, keep_multiple=True):
    """mask: {0,1,2}. Returns cleaned {0,1,2}."""
    out = np.zeros_like(mask)
    for side in (1, 2):
        cleaned = _clean_side(mask == side, spacing, min_volume_mm3, max_gap_mm, keep_multiple)
        out[cleaned] = side
    return out
