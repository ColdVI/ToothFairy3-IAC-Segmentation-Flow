"""Physical centreline and rotation-minimising tube geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import map_coordinates
from skimage.graph import MCP_Geometric

EPS = 1e-8


@dataclass(frozen=True)
class TubeFrame:
    centerline_mm: np.ndarray  # [S, 3]
    tangent: np.ndarray  # [S, 3]
    normal1: np.ndarray  # [S, 3]
    normal2: np.ndarray  # [S, 3]
    arc_mm: np.ndarray  # [S]
    coarse_start_mm: float
    coarse_end_mm: float

    def validate(self, atol: float = 3e-3) -> None:
        arrays = (self.centerline_mm, self.tangent, self.normal1, self.normal2)
        if any(a.ndim != 2 or a.shape[1] != 3 for a in arrays):
            raise ValueError("Frame arrays must have shape [S,3]")
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Frame contains NaN/Inf")
        basis = np.stack((self.tangent, self.normal1, self.normal2), axis=-1)
        gram = np.einsum("sji,sjk->sik", basis, basis)
        if not np.allclose(gram, np.eye(3)[None], atol=atol):
            raise ValueError(f"Bishop frame is not orthonormal; max error={np.abs(gram-np.eye(3)).max():.4g}")


def voxel_to_world(indices_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    indices_xyz = np.asarray(indices_xyz, dtype=np.float64)
    return indices_xyz @ affine[:3, :3].T + affine[:3, 3]


def world_to_voxel(points_mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points_mm = np.asarray(points_mm, dtype=np.float64)
    inverse = np.linalg.inv(affine)
    return points_mm @ inverse[:3, :3].T + inverse[:3, 3]


def sample_volume(
    volume: np.ndarray,
    points_mm: np.ndarray,
    affine: np.ndarray,
    *,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Sample an XYZ numpy volume at physical RAS points."""
    shape = points_mm.shape[:-1]
    voxels = world_to_voxel(points_mm.reshape(-1, 3), affine)
    sampled = map_coordinates(
        volume,
        voxels.T,
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=order > 1,
    )
    return sampled.reshape(shape)


def _normalise(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), EPS)


def _adaptive_support(probability: np.ndarray, minimum_voxels: int = 80) -> tuple[np.ndarray, float]:
    maximum = float(probability.max())
    if maximum <= 0:
        raise ValueError("The side probability map is empty")
    thresholds = (max(0.20, 0.45 * maximum), max(0.08, 0.25 * maximum), 0.03)
    for threshold in thresholds:
        support = probability >= threshold
        if int(support.sum()) >= minimum_voxels:
            return support, float(threshold)
    cutoff = np.partition(probability.reshape(-1), -minimum_voxels)[-minimum_voxels]
    return probability >= cutoff, float(cutoff)


def _weighted_pca_seeds(
    probability: np.ndarray,
    support: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    voxels = np.argwhere(support)
    if len(voxels) < 10:
        raise ValueError("Insufficient foreground support for centreline endpoints")
    world = voxel_to_world(voxels, affine)
    weights = probability[tuple(voxels.T)].astype(np.float64) + 1e-3
    centre = np.average(world, axis=0, weights=weights)
    centred = world - centre
    covariance = (centred * weights[:, None]).T @ centred / weights.sum()
    axis = np.linalg.eigh(covariance)[1][:, -1]
    projection = centred @ axis

    low_cut, high_cut = np.quantile(projection, (0.025, 0.975))
    low_ids = np.flatnonzero(projection <= low_cut)
    high_ids = np.flatnonzero(projection >= high_cut)
    low = voxels[low_ids[np.argmax(weights[low_ids])]]
    high = voxels[high_ids[np.argmax(weights[high_ids])]]

    # RAS +Y is anterior. Fix path orientation posterior -> anterior.
    low_world = voxel_to_world(low[None], affine)[0]
    high_world = voxel_to_world(high[None], affine)[0]
    if low_world[1] > high_world[1]:
        low, high = high, low
    return low.astype(int), high.astype(int)


def probability_geodesic(
    probability: np.ndarray,
    affine: np.ndarray,
    *,
    bbox_margin_mm: float = 5.0,
    probability_floor: float = 0.03,
) -> np.ndarray:
    """Extract one soft-probability geodesic without skeletonising a hard mask."""
    support, _ = _adaptive_support(probability)
    start, end = _weighted_pca_seeds(probability, support, affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    pad = np.ceil(bbox_margin_mm / np.maximum(spacing, EPS)).astype(int)
    support_points = np.argwhere(support)
    lower = np.maximum(support_points.min(axis=0) - pad, 0)
    upper = np.minimum(support_points.max(axis=0) + pad + 1, probability.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    local_probability = probability[slices]
    local_start = tuple((start - lower).tolist())
    local_end = tuple((end - lower).tolist())

    # The canonical cost is 1/(p+eps). A floor keeps short uncertain gaps
    # traversable while retaining a large penalty outside the coarse canal.
    cost = 1.0 / np.clip(local_probability.astype(np.float64), probability_floor, 1.0)
    mcp = MCP_Geometric(cost, fully_connected=True, sampling=tuple(float(v) for v in spacing))
    cumulative, _ = mcp.find_costs([local_start], [local_end])
    if not np.isfinite(cumulative[local_end]):
        raise RuntimeError("No finite soft-probability path between endpoint seeds")
    local_path = np.asarray(mcp.traceback(local_end), dtype=np.int64)
    path_voxel = local_path + lower[None]
    path_world = voxel_to_world(path_voxel, affine)
    if path_world[0, 1] > path_world[-1, 1]:
        path_world = path_world[::-1]
    return path_world


def _remove_near_duplicates(points: np.ndarray, tolerance_mm: float = 1e-4) -> np.ndarray:
    if len(points) < 2:
        return points
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, distance > tolerance_mm]
    return points[keep]


def resample_polyline(points: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    points = _remove_near_duplicates(np.asarray(points, dtype=np.float64))
    if len(points) < 2:
        raise ValueError("A curve requires at least two unique points")
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    target = np.linspace(0.0, cumulative[-1], int(n_points))
    output = np.column_stack([np.interp(target, cumulative, points[:, axis]) for axis in range(3)])
    return output, target


def smooth_resample_curve(
    points_mm: np.ndarray,
    *,
    n_points: int = 160,
    smoothing_mm: float = 0.45,
) -> np.ndarray:
    points = _remove_near_duplicates(points_mm)
    if len(points) < 4:
        return resample_polyline(points, n_points)[0]
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    u = cumulative / max(cumulative[-1], EPS)
    degree = min(3, len(points) - 1)
    try:
        spline, _ = splprep(
            points.T,
            u=u,
            k=degree,
            s=float(smoothing_mm**2 * len(points)),
        )
        dense = np.column_stack(splev(np.linspace(0.0, 1.0, max(8 * n_points, 512)), spline))
        return resample_polyline(dense, n_points)[0]
    except (TypeError, ValueError):
        return resample_polyline(points, n_points)[0]


def extend_curve(points_mm: np.ndarray, margin_mm: float, dense_points: int = 1024) -> tuple[np.ndarray, float, float]:
    core, core_arc = resample_polyline(points_mm, max(64, len(points_mm)))
    start_tangent = _normalise(core[min(4, len(core) - 1)] - core[0])
    end_tangent = _normalise(core[-1] - core[max(0, len(core) - 5)])
    step = max(core_arc[-1] / max(len(core) - 1, 1), 0.15)
    count = max(2, int(np.ceil(margin_mm / step)))
    offsets = np.linspace(margin_mm, step, count)
    prefix = core[0][None] - offsets[:, None] * start_tangent[None]
    suffix = core[-1][None] + offsets[::-1, None] * end_tangent[None]
    extended = np.concatenate((prefix, core, suffix), axis=0)
    dense, _arc = resample_polyline(extended, dense_points)
    coarse_start = margin_mm
    coarse_end = margin_mm + float(core_arc[-1])
    return dense, coarse_start, coarse_end


def _rodrigues(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalise(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def bishop_frame(centerline_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tangent = np.gradient(centerline_mm, axis=0)
    tangent = tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), EPS)

    superior = np.array([0.0, 0.0, 1.0])
    first = superior - np.dot(superior, tangent[0]) * tangent[0]
    if np.linalg.norm(first) < 0.2:
        fallback = np.array([1.0, 0.0, 0.0])
        first = fallback - np.dot(fallback, tangent[0]) * tangent[0]
    normal1 = np.zeros_like(tangent)
    normal2 = np.zeros_like(tangent)
    normal1[0] = _normalise(first)
    normal2[0] = _normalise(np.cross(tangent[0], normal1[0]))

    for index in range(1, len(centerline_mm)):
        cross = np.cross(tangent[index - 1], tangent[index])
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(np.dot(tangent[index - 1], tangent[index]), -1.0, 1.0))
        if sine < 1e-7:
            transported = normal1[index - 1]
        else:
            transported = _rodrigues(normal1[index - 1], cross / sine, np.arctan2(sine, cosine))
        transported -= np.dot(transported, tangent[index]) * tangent[index]
        normal1[index] = _normalise(transported)
        normal2[index] = _normalise(np.cross(tangent[index], normal1[index]))
    return tangent, normal1, normal2


def build_tube_frame(
    probability: np.ndarray,
    affine: np.ndarray,
    *,
    stations: int = 160,
    smoothing_mm: float = 0.45,
    endpoint_margin_mm: float = 4.0,
) -> TubeFrame:
    geodesic = probability_geodesic(probability, affine)
    core = smooth_resample_curve(geodesic, n_points=max(160, stations), smoothing_mm=smoothing_mm)
    extended, coarse_start, coarse_end = extend_curve(core, endpoint_margin_mm)
    centerline, arc = resample_polyline(extended, stations)
    tangent, normal1, normal2 = bishop_frame(centerline)
    frame = TubeFrame(
        centerline_mm=centerline.astype(np.float32),
        tangent=tangent.astype(np.float32),
        normal1=normal1.astype(np.float32),
        normal2=normal2.astype(np.float32),
        arc_mm=arc.astype(np.float32),
        coarse_start_mm=float(coarse_start),
        coarse_end_mm=float(coarse_end),
    )
    frame.validate()
    return frame


def polar_points(
    frame: TubeFrame,
    angles: np.ndarray,
    radii_mm: np.ndarray,
    displacement: np.ndarray | None = None,
) -> np.ndarray:
    """Return [S,A,R,3] physical sampling points."""
    angles = np.asarray(angles, dtype=np.float64)
    radii_mm = np.asarray(radii_mm, dtype=np.float64)
    radial = (
        np.cos(angles)[None, :, None] * frame.normal1[:, None, :]
        + np.sin(angles)[None, :, None] * frame.normal2[:, None, :]
    )
    centres = frame.centerline_mm.astype(np.float64)
    if displacement is not None:
        displacement = np.asarray(displacement, dtype=np.float64)
        centres = (
            centres
            + displacement[:, 0, None] * frame.normal1
            + displacement[:, 1, None] * frame.normal2
        )
    return centres[:, None, None, :] + radial[:, :, None, :] * radii_mm[None, None, :, None]
