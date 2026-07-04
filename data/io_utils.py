#!/usr/bin/env python3
"""
io_utils.py — NIfTI I/O + geometry + physical-space SDF helpers.

Reused by the data, flow and evaluation packages so that orientation handling,
physical-coordinate construction and signed-distance conversion are defined in
exactly one place (single source of truth). All distances are in MILLIMETRES,
never voxels — the IAC is thin and voxel-space distances are anisotropy-biased.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt


# --------------------------------------------------------------------------- #
# NIfTI loading / orientation
# --------------------------------------------------------------------------- #
def load_nii(path):
    """Return (array, nibabel_img). Array is the raw dataobj (no reorientation)."""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    return arr, img


def orientation_code(affine) -> str:
    """Anatomical axis codes, e.g. 'RPI', from a 4x4 affine (nibabel convention)."""
    return "".join(nib.aff2axcodes(affine))


def voxel_spacing(img) -> np.ndarray:
    """(sx, sy, sz) voxel size in mm from the NIfTI header."""
    return np.asarray(img.header.get_zooms()[:3], dtype=np.float64)


def affines_match(a, b, tol=1e-3) -> bool:
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


def reorient_to_code(arr, affine, target_code="RPI"):
    """
    Reorient a volume to `target_code` (e.g. 'RPI') using the affine's direction
    cosines — NOT a blind array flip. This reads which array axis/sign each
    anatomical direction actually corresponds to (from the affine) and permutes
    the array to match, updating the affine so every voxel keeps the same
    physical (world) location. Left stays left, right stays right, as long as
    the input affine is correct — unlike guessing from array index order.

    Returns (reoriented_array, reoriented_affine).
    """
    from nibabel.orientations import (io_orientation, axcodes2ornt,
                                       ornt_transform, apply_orientation, inv_ornt_aff)

    current = io_orientation(affine)
    target = axcodes2ornt(target_code)
    transform = ornt_transform(current, target)
    new_arr = apply_orientation(arr, transform)
    new_affine = affine @ inv_ornt_aff(transform, arr.shape)
    return new_arr, new_affine


# --------------------------------------------------------------------------- #
# Physical coordinates
# --------------------------------------------------------------------------- #
def physical_coord_grid(shape, affine) -> np.ndarray:
    """
    Physical (world) coordinates in mm for every voxel of a volume.

    Returns (3, D, H, W): channel 0/1/2 are the x/y/z world coordinate of each
    voxel, computed through the NIfTI affine (NOT array indices). This is what
    the flow model consumes as its laterality-aware positional conditioning.
    """
    d, h, w = shape
    ii, jj, kk = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=0).reshape(4, -1).astype(np.float64)
    world = affine @ idx                      # (4, N)
    return world[:3].reshape(3, d, h, w)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    Centre each physical axis at the volume centroid and scale to ~[-1, 1] by
    the per-axis half-extent. Keeps left/right sign meaningful while staying
    scale-stable across differing fields of view (relevant for the F/S subsets).
    """
    out = np.empty_like(coords, dtype=np.float32)
    for c in range(coords.shape[0]):
        v = coords[c]
        centre = 0.5 * (v.max() + v.min())
        half = 0.5 * (v.max() - v.min()) + 1e-6
        out[c] = ((v - centre) / half).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# Signed distance fields (millimetres)
# --------------------------------------------------------------------------- #
def mask_to_sdf_mm(mask: np.ndarray, spacing, clip_mm: float = 10.0) -> np.ndarray:
    """
    Signed distance transform of a binary mask, in millimetres, negative inside.

    Uses anisotropic `sampling=spacing` so the metric is physical. The field is
    clipped to +-clip_mm and returned in mm (NOT normalised — normalisation is a
    separate, tunable step so the truncation radius stays interpretable).
    """
    mask = mask.astype(bool)
    spacing = np.asarray(spacing, dtype=np.float64)
    if not mask.any():
        return np.full(mask.shape, clip_mm, dtype=np.float32)
    d_out = distance_transform_edt(~mask, sampling=spacing)
    d_in = distance_transform_edt(mask, sampling=spacing)
    sdf = d_out - d_in                                   # negative inside
    return np.clip(sdf, -clip_mm, clip_mm).astype(np.float32)


def normalize_sdf(sdf_mm: np.ndarray, clip_mm: float = 10.0) -> np.ndarray:
    """Map an mm SDF clipped at +-clip_mm to ~[-1, 1] for the network."""
    return (np.clip(sdf_mm, -clip_mm, clip_mm) / clip_mm).astype(np.float32)


def sdf_stack_to_mask(sdf_lr: np.ndarray, thresh: float = 0.0) -> np.ndarray:
    """
    Decode a 2-channel (Left, Right) SDF stack into a {0,1,2} label map.
    A voxel is foreground where either channel is below `thresh`; the side is the
    more-negative (deeper-inside) channel. Left=1, Right=2 (project convention).
    """
    left, right = sdf_lr[0], sdf_lr[1]
    inside = np.minimum(left, right) < thresh
    side_left = left <= right
    out = np.zeros(left.shape, dtype=np.uint8)
    out[inside & side_left] = 1
    out[inside & ~side_left] = 2
    return out


def save_like(reference_img, array, path, dtype=np.uint8):
    """Save `array` as NIfTI reusing a reference image's affine/header."""
    out = nib.Nifti1Image(array.astype(dtype), reference_img.affine, reference_img.header)
    out.set_data_dtype(dtype)
    nib.save(out, str(path))
