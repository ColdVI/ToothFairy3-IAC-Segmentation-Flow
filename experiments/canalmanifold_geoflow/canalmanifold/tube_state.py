"""Fit and decode the canonical m<=8 tube state and free radial surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .constants import GLOBAL_DIM, HARMONIC_START, HARMONICS, LOCAL_DIM, N_LOCAL
from .geometry import TubeFrame, polar_points, sample_volume, world_to_voxel

EPS = 1e-8


@dataclass(frozen=True)
class FittedTube:
    local: np.ndarray  # [S,9]
    global_state: np.ndarray  # [3] = g, axial start, axial end
    station_valid: np.ndarray  # [S]
    angle_valid: np.ndarray  # [S,A]
    radii_mm: np.ndarray  # [S,A]

    def validate(self) -> None:
        if self.local.shape != (N_LOCAL, LOCAL_DIM):
            raise ValueError(f"local state must be {(N_LOCAL, LOCAL_DIM)}, got {self.local.shape}")
        if self.global_state.shape != (GLOBAL_DIM,):
            raise ValueError(f"global state must be {(GLOBAL_DIM,)}, got {self.global_state.shape}")
        if not np.isfinite(self.local).all() or not np.isfinite(self.global_state).all():
            raise ValueError("Tube state contains NaN/Inf")
        active = self.station_valid.astype(bool)
        if active.any() and abs(float(self.local[active, 2].mean())) > 2e-4:
            raise ValueError("ell(s) gauge is not zero mean over valid stations")
        if not self.global_state[1] < self.global_state[2]:
            raise ValueError("endpoint_start must be smaller than endpoint_end")


def angles_grid(n_angles: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False, dtype=np.float64)


def harmonic_design(angles: np.ndarray, include_constant: bool = True) -> np.ndarray:
    columns: list[np.ndarray] = []
    if include_constant:
        columns.append(np.ones_like(angles))
    for harmonic in HARMONICS:
        columns.extend((np.cos(harmonic * angles), np.sin(harmonic * angles)))
    return np.column_stack(columns)


def radii_from_state(local: np.ndarray, global_state: np.ndarray, angles: np.ndarray) -> np.ndarray:
    basis = harmonic_design(angles, include_constant=False)
    log_radius = (
        global_state[0]
        + local[:, 2, None]
        + local[:, HARMONIC_START:LOCAL_DIM] @ basis.T
    )
    return np.exp(np.clip(log_radius, np.log(0.12), np.log(8.0)))


def boundary_from_profiles(
    profiles: np.ndarray,
    radii_grid_mm: np.ndarray,
    *,
    threshold: float = 0.5,
    minimum_radius_mm: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the outward 0.5 crossing of each centre-originating radial ray."""
    stations, angles, _ = profiles.shape
    output = np.full((stations, angles), np.nan, dtype=np.float64)
    valid = np.zeros((stations, angles), dtype=bool)
    for station in range(stations):
        for angle in range(angles):
            values = np.asarray(profiles[station, angle], dtype=np.float64)
            if values[0] < threshold:
                continue
            below = np.flatnonzero(values < threshold)
            stop = int(below[0]) if len(below) else len(values)
            last_inside = stop - 1
            if last_inside < 0:
                continue
            if stop < len(values):
                v0, v1 = values[last_inside], values[stop]
                fraction = (v0 - threshold) / max(v0 - v1, EPS)
                radius = radii_grid_mm[last_inside] + fraction * (
                    radii_grid_mm[stop] - radii_grid_mm[last_inside]
                )
            else:
                radius = radii_grid_mm[-1]
            if radius >= minimum_radius_mm and radius < radii_grid_mm[-1] - 1e-5:
                output[station, angle] = float(radius)
                valid[station, angle] = True
    return output, valid


def polar_centroid(
    profiles: np.ndarray,
    radii_grid_mm: np.ndarray,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Area centroid in each sampled cross-section, expressed in its local frame."""
    dr = float(np.mean(np.diff(radii_grid_mm)))
    dtheta = 2.0 * np.pi / len(angles)
    rho = radii_grid_mm[None, None, :]
    weights = np.clip(profiles, 0.0, 1.0) * rho * dr * dtheta
    area = weights.sum(axis=(1, 2))
    x = (weights * rho * np.cos(angles)[None, :, None]).sum(axis=(1, 2)) / np.maximum(area, EPS)
    y = (weights * rho * np.sin(angles)[None, :, None]).sum(axis=(1, 2)) / np.maximum(area, EPS)
    return np.column_stack((x, y)), area


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError("No active axial cross-section")
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks + 1, len(indices)]
    lengths = stops - starts
    best = int(np.argmax(lengths))
    run = indices[starts[best] : stops[best]]
    return int(run[0]), int(run[-1])


def axial_endpoints(areas_mm2: np.ndarray, arc_mm: np.ndarray, minimum_area_mm2: float = 0.18) -> tuple[float, float, np.ndarray]:
    active = areas_mm2 >= minimum_area_mm2
    start, end = _longest_true_run(active)
    if end <= start:
        raise ValueError("Tube has fewer than two active axial stations")
    run_mask = np.zeros_like(active)
    run_mask[start : end + 1] = True
    return float(arc_mm[start]), float(arc_mm[end]), run_mask


def _robust_harmonic_fit(radii: np.ndarray, valid: np.ndarray, angles: np.ndarray) -> np.ndarray:
    design = harmonic_design(angles, include_constant=True)
    selected = valid & np.isfinite(radii) & (radii > 0)
    if int(selected.sum()) < design.shape[1] + 2:
        raise ValueError(
            f"Insufficient valid rays for m=2..{max(HARMONICS)} harmonic fit"
        )
    x = design[selected]
    y = np.log(radii[selected])
    weights = np.ones_like(y)
    ridge_values = [1e-8]
    for harmonic in HARMONICS:
        penalty = 2e-4 * (float(harmonic) / 2.0) ** 2
        ridge_values.extend((penalty, penalty))
    ridge = np.diag(ridge_values)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(4):
        xtw = x.T * weights[None]
        coefficients = np.linalg.solve(xtw @ x + ridge, xtw @ y)
        residual = y - x @ coefficients
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-5
        ratio = np.abs(residual) / (1.5 * scale)
        weights = np.where(ratio <= 1.0, 1.0, 1.0 / np.maximum(ratio, EPS))
    return coefficients


def _interpolate_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    x = np.arange(len(output))
    good = np.flatnonzero(valid & np.isfinite(output))
    if len(good) == 0:
        raise ValueError("No valid station coefficient to interpolate")
    output[~np.isfinite(output)] = np.interp(x[~np.isfinite(output)], good, output[good])
    return output


def complete_boundary_radii(
    radii: np.ndarray,
    angle_valid: np.ndarray,
    *,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Periodically interpolate missing rays, then interpolate missing stations.

    The returned table is finite and positive and is used by the unrestricted
    ``h(s,theta)`` arm.  ``fallback`` should be a smooth analytic tube radius
    table when an entire station has no valid ray.
    """
    values = np.asarray(radii, dtype=np.float64).copy()
    valid = np.asarray(angle_valid, dtype=bool) & np.isfinite(values) & (values > 0)
    stations, n_angles = values.shape
    if valid.shape != values.shape:
        raise ValueError("angle_valid and radii must have the same shape")
    angle_index = np.arange(n_angles, dtype=np.float64)
    station_has_data = np.zeros(stations, dtype=bool)
    for station in range(stations):
        good = np.flatnonzero(valid[station])
        if len(good) < 2:
            continue
        extended_x = np.concatenate((good - n_angles, good, good + n_angles))
        extended_y = np.concatenate(
            (values[station, good], values[station, good], values[station, good])
        )
        values[station] = np.interp(angle_index, extended_x, extended_y)
        station_has_data[station] = True

    if not station_has_data.any():
        if fallback is None:
            raise ValueError("No valid boundary ray is available")
        return np.asarray(fallback, dtype=np.float32).copy()

    station_index = np.arange(stations, dtype=np.float64)
    good_stations = np.flatnonzero(station_has_data)
    for angle in range(n_angles):
        values[:, angle] = np.interp(
            station_index,
            good_stations,
            values[good_stations, angle],
        )
    if fallback is not None:
        fallback_values = np.asarray(fallback, dtype=np.float64)
        if fallback_values.shape != values.shape:
            raise ValueError("fallback and radii must have the same shape")
        invalid = ~np.isfinite(values) | (values <= 0)
        values[invalid] = fallback_values[invalid]
    return np.clip(values, 0.12, 8.0).astype(np.float32)


def fit_tube(
    field: np.ndarray,
    affine: np.ndarray,
    frame: TubeFrame,
    *,
    n_angles: int = 32,
    n_radii: int = 32,
    max_radius_mm: float = 5.0,
    threshold: float = 0.5,
    estimate_displacement: bool,
    maximum_displacement_mm: float = 3.0,
    minimum_valid_fraction: float = 0.60,
) -> FittedTube:
    """Fit q0 or q1 in the coarse centreline's common Bishop frame."""
    if len(frame.arc_mm) != N_LOCAL:
        raise ValueError(f"Canonical implementation requires {N_LOCAL} stations")
    angles = angles_grid(n_angles)
    radial_grid = np.linspace(0.0, max_radius_mm, n_radii, dtype=np.float64)
    displacement = np.zeros((N_LOCAL, 2), dtype=np.float64)

    points = polar_points(frame, angles, radial_grid)
    profiles = sample_volume(field, points, affine, order=1, cval=0.0)
    if estimate_displacement:
        # Two centroid corrections are enough at the sub-voxel scale while a
        # magnitude guard prevents a distant false component capturing a slice.
        for _ in range(2):
            residual, _ = polar_centroid(profiles, radial_grid, angles)
            residual_norm = np.linalg.norm(residual, axis=1, keepdims=True)
            residual *= np.minimum(1.0, maximum_displacement_mm / np.maximum(residual_norm, EPS))
            displacement += residual
            total_norm = np.linalg.norm(displacement, axis=1, keepdims=True)
            displacement *= np.minimum(1.0, maximum_displacement_mm / np.maximum(total_norm, EPS))
            points = polar_points(frame, angles, radial_grid, displacement)
            profiles = sample_volume(field, points, affine, order=1, cval=0.0)

    radii, angle_valid = boundary_from_profiles(profiles, radial_grid, threshold=threshold)
    _, areas = polar_centroid(profiles, radial_grid, angles)
    endpoint_start, endpoint_end, axial_active = axial_endpoints(areas, frame.arc_mm)
    station_valid = (angle_valid.mean(axis=1) >= minimum_valid_fraction) & axial_active

    coefficient_dim = 1 + 2 * len(HARMONICS)
    coefficients = np.full((N_LOCAL, coefficient_dim), np.nan, dtype=np.float64)
    for station in np.flatnonzero(station_valid):
        try:
            coefficients[station] = _robust_harmonic_fit(
                radii[station], angle_valid[station], angles
            )
        except ValueError:
            station_valid[station] = False
    if int(station_valid.sum()) < max(12, int(0.15 * N_LOCAL)):
        raise ValueError(f"Only {int(station_valid.sum())} stations produced a valid tube fit")

    for channel in range(coefficients.shape[1]):
        coefficients[:, channel] = _interpolate_invalid(coefficients[:, channel], station_valid)
    for channel in range(2):
        displacement[:, channel] = _interpolate_invalid(displacement[:, channel], axial_active)

    global_log_radius = float(coefficients[station_valid, 0].mean())
    ell = coefficients[:, 0] - global_log_radius
    # Exact gauge projection over authoritative valid stations.
    drift = float(ell[station_valid].mean())
    ell -= drift
    global_log_radius += drift

    local = np.zeros((N_LOCAL, LOCAL_DIM), dtype=np.float32)
    local[:, 0:2] = displacement.astype(np.float32)
    local[:, 2] = ell.astype(np.float32)
    local[:, HARMONIC_START:LOCAL_DIM] = coefficients[:, 1:].astype(np.float32)
    global_state = np.asarray([global_log_radius, endpoint_start, endpoint_end], dtype=np.float32)
    fitted = FittedTube(
        local=local,
        global_state=global_state,
        station_valid=station_valid.astype(bool),
        angle_valid=angle_valid.astype(bool),
        radii_mm=radii.astype(np.float32),
    )
    fitted.validate()
    return fitted


def sample_shell(
    frame: TubeFrame,
    affine: np.ndarray,
    channels: list[np.ndarray],
    *,
    n_angles: int = 32,
    n_radii: int = 24,
    max_radius_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = angles_grid(n_angles)
    radii = np.linspace(0.0, max_radius_mm, n_radii, dtype=np.float64)
    points = polar_points(frame, angles, radii)
    sampled = [sample_volume(channel, points, affine, order=1, cval=0.0) for channel in channels]
    shell = np.stack(sampled, axis=0).astype(np.float16)  # [C,S,A,R]
    return shell, angles.astype(np.float32), radii.astype(np.float32)


def surface_points(local: np.ndarray, global_state: np.ndarray, frame: TubeFrame, n_angles: int = 64) -> np.ndarray:
    angles = angles_grid(n_angles)
    radius = radii_from_state(local, global_state, angles)
    centre = (
        frame.centerline_mm
        + local[:, 0, None] * frame.normal1
        + local[:, 1, None] * frame.normal2
    )
    radial = (
        np.cos(angles)[None, :, None] * frame.normal1[:, None, :]
        + np.sin(angles)[None, :, None] * frame.normal2[:, None, :]
    )
    points = centre[:, None, :] + radius[:, :, None] * radial
    active = (frame.arc_mm >= global_state[1]) & (frame.arc_mm <= global_state[2])
    return points[active]


def decode_mask(
    local: np.ndarray,
    global_state: np.ndarray,
    frame: TubeFrame,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    *,
    extra_margin_mm: float = 1.5,
    chunk_voxels: int = 500_000,
) -> np.ndarray:
    """Hard-decode a connected positive-radius tube into the reference grid."""
    angles = angles_grid(64)
    radius_table = radii_from_state(local, global_state, angles)
    maximum_radius = float(np.nanmax(radius_table)) + float(np.linalg.norm(local[:, :2], axis=1).max())
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    centre_voxel = world_to_voxel(frame.centerline_mm, affine)
    pad = np.ceil((maximum_radius + extra_margin_mm) / np.maximum(spacing, EPS)).astype(int)
    lower = np.maximum(np.floor(centre_voxel.min(axis=0)).astype(int) - pad, 0)
    upper = np.minimum(np.ceil(centre_voxel.max(axis=0)).astype(int) + pad + 1, np.asarray(shape))
    crop_shape = tuple((upper - lower).astype(int))
    total = int(np.prod(crop_shape))
    tree = cKDTree(frame.centerline_mm)
    output_flat = np.zeros(total, dtype=np.uint8)

    yz = crop_shape[1] * crop_shape[2]
    for begin in range(0, total, chunk_voxels):
        end = min(total, begin + chunk_voxels)
        flat = np.arange(begin, end, dtype=np.int64)
        i = flat // yz
        remainder = flat % yz
        j = remainder // crop_shape[2]
        k = remainder % crop_shape[2]
        voxel = np.column_stack((i, j, k)).astype(np.float64) + lower[None]
        world = voxel @ affine[:3, :3].T + affine[:3, 3]
        _, station = tree.query(world, k=1, workers=-1)
        relative = world - frame.centerline_mm[station]
        x = np.einsum("ij,ij->i", relative, frame.normal1[station]) - local[station, 0]
        y = np.einsum("ij,ij->i", relative, frame.normal2[station]) - local[station, 1]
        rho = np.sqrt(x * x + y * y)
        theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        basis = harmonic_design(theta, include_constant=False)
        log_radius = (
            global_state[0]
            + local[station, 2]
            + np.sum(
                local[station, HARMONIC_START:LOCAL_DIM] * basis,
                axis=1,
            )
        )
        radius = np.exp(np.clip(log_radius, np.log(0.12), np.log(8.0)))
        axial = frame.arc_mm[station]
        output_flat[begin:end] = (
            (rho <= radius) & (axial >= global_state[1]) & (axial <= global_state[2])
        ).astype(np.uint8)

    output = np.zeros(shape, dtype=np.uint8)
    output[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = output_flat.reshape(crop_shape)
    return output


def decode_radial_mask(
    radii_mm: np.ndarray,
    endpoint_start_mm: float,
    endpoint_end_mm: float,
    frame: TubeFrame,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    *,
    extra_margin_mm: float = 1.5,
    chunk_voxels: int = 500_000,
) -> np.ndarray:
    """Decode an unrestricted star-shaped radial table in the Bishop chart."""
    radii_mm = np.asarray(radii_mm, dtype=np.float64)
    if radii_mm.ndim != 2 or radii_mm.shape[0] != len(frame.arc_mm):
        raise ValueError(f"Expected [S,A] radii table, got {radii_mm.shape}")
    n_angles = radii_mm.shape[1]
    if not np.isfinite(radii_mm).all() or np.any(radii_mm <= 0):
        raise ValueError("radii_mm must be finite and positive")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    centre_voxel = world_to_voxel(frame.centerline_mm, affine)
    pad = np.ceil((float(radii_mm.max()) + extra_margin_mm) / np.maximum(spacing, EPS)).astype(int)
    lower = np.maximum(np.floor(centre_voxel.min(axis=0)).astype(int) - pad, 0)
    upper = np.minimum(np.ceil(centre_voxel.max(axis=0)).astype(int) + pad + 1, np.asarray(shape))
    crop_shape = tuple((upper - lower).astype(int))
    total = int(np.prod(crop_shape))
    tree = cKDTree(frame.centerline_mm)
    output_flat = np.zeros(total, dtype=np.uint8)
    yz = crop_shape[1] * crop_shape[2]
    angular_scale = n_angles / (2.0 * np.pi)

    for begin in range(0, total, chunk_voxels):
        end = min(total, begin + chunk_voxels)
        flat = np.arange(begin, end, dtype=np.int64)
        i = flat // yz
        remainder = flat % yz
        j = remainder // crop_shape[2]
        k = remainder % crop_shape[2]
        voxel = np.column_stack((i, j, k)).astype(np.float64) + lower[None]
        world = voxel @ affine[:3, :3].T + affine[:3, 3]
        _, station = tree.query(world, k=1, workers=-1)
        relative = world - frame.centerline_mm[station]
        x = np.einsum("ij,ij->i", relative, frame.normal1[station])
        y = np.einsum("ij,ij->i", relative, frame.normal2[station])
        rho = np.sqrt(x * x + y * y)
        angle_float = np.mod(np.arctan2(y, x), 2.0 * np.pi) * angular_scale
        angle0 = np.floor(angle_float).astype(np.int64) % n_angles
        angle1 = (angle0 + 1) % n_angles
        fraction = angle_float - np.floor(angle_float)
        radius = (
            (1.0 - fraction) * radii_mm[station, angle0]
            + fraction * radii_mm[station, angle1]
        )
        axial = frame.arc_mm[station]
        output_flat[begin:end] = (
            (rho <= radius)
            & (axial >= float(endpoint_start_mm))
            & (axial <= float(endpoint_end_mm))
        ).astype(np.uint8)

    output = np.zeros(shape, dtype=np.uint8)
    output[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = output_flat.reshape(crop_shape)
    return output
