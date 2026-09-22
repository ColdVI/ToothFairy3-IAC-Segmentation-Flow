"""Identity-preserving normal-displacement decoding and fair post-processing."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
)
from scipy.ndimage import (
    label as connected_component_labels,
)
from scipy.spatial import cKDTree

from .geometry import TubeFrame
from .tube_state import angles_grid, radii_from_state


def signed_distance(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return (
        distance_transform_edt(~mask, sampling=spacing)
        - distance_transform_edt(mask, sampling=spacing)
    ).astype(np.float32)


def crop_to_mask(mask: np.ndarray, padding_voxels: np.ndarray) -> tuple[slice, ...]:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return tuple(slice(0, size) for size in mask.shape)
    padding_voxels = np.broadcast_to(
        np.asarray(padding_voxels, dtype=int), (mask.ndim,)
    )
    lower = np.maximum(coordinates.min(axis=0) - padding_voxels, 0)
    upper = np.minimum(
        coordinates.max(axis=0) + 1 + padding_voxels,
        np.asarray(mask.shape),
    )
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def tube_state_to_radial_displacement(
    q0_local: np.ndarray,
    q0_global: np.ndarray,
    predicted_local: np.ndarray,
    predicted_global: np.ndarray,
    *,
    n_angles: int = 32,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Convert an m<=8 state correction to first-order radial displacement.

    Radius and centre translation are combined in the fixed Bishop rays.  This
    is the small-deformation expression from the v2 design; unlike an SDF-field
    difference it produces one scalar displacement per surface coordinate.
    """
    angles = angles_grid(n_angles)
    q0_radius = radii_from_state(q0_local, q0_global, angles)
    predicted_radius = radii_from_state(predicted_local, predicted_global, angles)
    centre_delta = np.asarray(predicted_local[:, :2] - q0_local[:, :2])
    projected_centre = (
        centre_delta[:, 0, None] * np.cos(angles)[None]
        + centre_delta[:, 1, None] * np.sin(angles)[None]
    )
    h = predicted_radius - q0_radius + projected_centre
    return (
        h.astype(np.float32),
        (float(q0_global[1]), float(q0_global[2])),
        (float(predicted_global[1]), float(predicted_global[2])),
    )


def _periodic_angular_sample(
    table: np.ndarray,
    stations: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    n_angles = table.shape[1]
    coordinate = np.mod(theta, 2.0 * np.pi) * (n_angles / (2.0 * np.pi))
    lower = np.floor(coordinate).astype(np.int64) % n_angles
    upper = (lower + 1) % n_angles
    fraction = coordinate - np.floor(coordinate)
    return (
        (1.0 - fraction) * table[stations, lower]
        + fraction * table[stations, upper]
    )


def normal_displacement_decode(
    coarse_mask: np.ndarray,
    h_mm: np.ndarray,
    frame: TubeFrame,
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    q0_endpoints_mm: tuple[float, float] | None = None,
    predicted_endpoints_mm: tuple[float, float] | None = None,
    margin_mm: float = 12.0,
    max_abs_displacement_mm: float = 1.0,
    endpoint_band_mm: float = 0.75,
) -> np.ndarray:
    """Decode ``phi0-H`` with H extended along raw-boundary normals.

    ``h=0`` and unchanged endpoints return the raw nnU-Net mask bit-for-bit.
    The displacement is clipped by a reach-motivated trust limit. In the
    continuous normal-coordinate construction this is the diffeomorphic
    condition; the voxel implementation is an approximation and therefore
    still reports connected components explicitly.
    """
    coarse = np.asarray(coarse_mask, dtype=bool)
    h = np.asarray(h_mm, dtype=np.float32)
    if h.ndim != 2 or h.shape[0] != len(frame.arc_mm):
        raise ValueError(f"h_mm must be [S,A], got {h.shape}")
    if not np.isfinite(h).all():
        raise ValueError("h_mm contains NaN/Inf")
    q0_endpoints = q0_endpoints_mm or (
        float(frame.coarse_start_mm),
        float(frame.coarse_end_mm),
    )
    predicted_endpoints = predicted_endpoints_mm or q0_endpoints
    if (
        np.count_nonzero(h) == 0
        and tuple(map(float, predicted_endpoints)) == tuple(map(float, q0_endpoints))
    ):
        return coarse.astype(np.uint8, copy=True)
    if not coarse.any():
        return coarse.astype(np.uint8)

    limit = max(float(max_abs_displacement_mm), 0.0)
    h = np.clip(h, -limit, limit)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    padding = np.ceil(float(margin_mm) / np.maximum(spacing_array, 1e-6)).astype(int)
    crop = crop_to_mask(coarse, padding)
    coarse_crop = coarse[crop]
    boundary = coarse_crop & ~binary_erosion(coarse_crop)
    if not boundary.any():
        return coarse.astype(np.uint8, copy=True)

    crop_offset = np.asarray([axis.start for axis in crop], dtype=np.float64)
    boundary_voxel = np.argwhere(boundary).astype(np.float64) + crop_offset[None]
    boundary_world = boundary_voxel @ affine[:3, :3].T + affine[:3, 3]
    tree = cKDTree(frame.centerline_mm)
    _, station = tree.query(boundary_world, k=1, workers=-1)
    relative = boundary_world - frame.centerline_mm[station]
    x = np.einsum("ij,ij->i", relative, frame.normal1[station])
    y = np.einsum("ij,ij->i", relative, frame.normal2[station])
    theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    boundary_h = _periodic_angular_sample(h, station, theta)

    if q0_endpoints_mm is not None and predicted_endpoints_mm is not None:
        q0_start, q0_end = map(float, q0_endpoints)
        predicted_start, predicted_end = map(float, predicted_endpoints)
        arc = frame.arc_mm[station]
        axial = np.abs(np.einsum("ij,ij->i", relative, frame.tangent[station]))
        radial = np.sqrt(x * x + y * y)
        cap_like = axial >= 0.5 * radial
        start_cap = cap_like & (np.abs(arc - q0_start) <= float(endpoint_band_mm))
        end_cap = cap_like & (np.abs(arc - q0_end) <= float(endpoint_band_mm))
        boundary_h[start_cap] = np.clip(q0_start - predicted_start, -limit, limit)
        boundary_h[end_cap] = np.clip(predicted_end - q0_end, -limit, limit)

    boundary_field = np.zeros(coarse_crop.shape, dtype=np.float32)
    boundary_coordinates = np.argwhere(boundary)
    boundary_field[tuple(boundary_coordinates.T)] = boundary_h.astype(np.float32)
    _, nearest = distance_transform_edt(
        ~boundary,
        sampling=spacing,
        return_indices=True,
    )
    extended_h = boundary_field[tuple(nearest)]
    phi0 = signed_distance(coarse_crop, spacing)
    refined_crop = phi0 - extended_h <= 0.0
    output = coarse.astype(np.uint8, copy=True)
    output[crop] = refined_crop.astype(np.uint8)
    return output


def remove_small_components(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    minimum_volume_mm3: float = 0.27,
) -> np.ndarray:
    """Remove only components smaller than a physical-volume threshold."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any() or minimum_volume_mm3 <= 0.0:
        return binary.astype(np.uint8, copy=True)
    labels, count = connected_component_labels(binary)
    if count <= 1:
        return binary.astype(np.uint8, copy=True)
    voxel_volume = float(np.prod(np.asarray(spacing, dtype=np.float64)))
    minimum_voxels = max(1, math.ceil(float(minimum_volume_mm3) / voxel_volume))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_voxels
    keep[0] = False
    return keep[labels].astype(np.uint8)
