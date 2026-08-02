"""Deterministic physical-space perturbations for the coarse-SDF start state.

The transforms operate only on the two-channel start-state/prior. Translation
is sampled in world millimetres and converted through the NIfTI affine; signed
offset and noise use the canonical SDF normalisation scale from ``io_utils``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import shift as nd_shift

try:
    from data.io_utils import sdf_mm_delta_to_normalized
except ImportError:  # direct execution from flow/
    from io_utils import sdf_mm_delta_to_normalized


@dataclass(frozen=True)
class PerturbationRecord:
    applied: bool
    translation_world_mm: tuple[float, float, float]
    translation_voxels: tuple[float, float, float]
    sdf_offset_mm: float
    sdf_noise_std_mm: float
    sdf_clip_mm: float

    def to_dict(self):
        return asdict(self)


def _settings(config):
    config = config or {}
    return config.get("prior_perturbation", config)


def _validate_inputs(coarse_sdf, affine, clip_mm):
    array = np.asarray(coarse_sdf, dtype=np.float32)
    affine = np.asarray(affine, dtype=np.float64)
    if array.ndim != 4 or array.shape[0] != 2:
        raise ValueError(f"coarse_sdf must have shape (2,D,H,W), got {array.shape}")
    if affine.shape != (4, 4):
        raise ValueError(f"affine must have shape (4,4), got {affine.shape}")
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("affine must contain an invertible voxel-to-world transform")
    if float(clip_mm) <= 0:
        raise ValueError("sdf_clip_mm must be positive")
    return array, affine


def apply_prior_transform(coarse_sdf, affine, *, translation_world_mm=(0, 0, 0),
                          sdf_offset_mm=0.0, sdf_noise_mm=None,
                          sdf_clip_mm=10.0):
    """Apply explicit physical transform components (the testable core)."""
    array, affine = _validate_inputs(coarse_sdf, affine, sdf_clip_mm)
    translation_world = np.asarray(translation_world_mm, dtype=np.float64)
    if translation_world.shape != (3,):
        raise ValueError("translation_world_mm must contain three values")
    translation_voxels = np.linalg.solve(affine[:3, :3], translation_world)
    shifted = np.stack([
        nd_shift(channel, shift=translation_voxels, order=1,
                 mode="constant", cval=1.0, prefilter=False)
        for channel in array
    ]).astype(np.float32)
    shifted += sdf_mm_delta_to_normalized(float(sdf_offset_mm), sdf_clip_mm)
    if sdf_noise_mm is not None:
        noise = np.asarray(sdf_noise_mm, dtype=np.float32)
        if noise.shape != shifted.shape:
            raise ValueError(f"sdf_noise_mm shape {noise.shape} != {shifted.shape}")
        shifted += sdf_mm_delta_to_normalized(noise, sdf_clip_mm)
    return np.clip(shifted, -1.0, 1.0).astype(np.float32), translation_voxels


def perturb_coarse_prior(coarse_sdf, affine, config, rng, *, sdf_clip_mm):
    """Apply one bilateral transform and return ``(perturbed, provenance)``.

    Both sides receive the same physical translation and signed offset. Noise
    is sampled independently per voxel/channel, then centred per channel so it
    cannot introduce a hidden global offset. No left/right swap is performed.
    Positive SDF offset erodes a negative-inside object; negative offset dilates.
    """
    array, affine = _validate_inputs(coarse_sdf, affine, sdf_clip_mm)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    cfg = _settings(config)
    enabled = bool(cfg.get("enabled", False))
    probability = float(cfg.get("probability", 1.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("prior_perturbation.probability must be in [0,1]")

    applied = enabled and bool(rng.random() < probability)
    if not applied:
        record = PerturbationRecord(False, (0.0,) * 3, (0.0,) * 3, 0.0, 0.0,
                                    float(sdf_clip_mm))
        return array.copy(), record.to_dict()

    max_translation = float(cfg.get("max_translation_mm", 0.0))
    max_offset = float(cfg.get("max_sdf_offset_mm", 0.0))
    noise_std = float(cfg.get("sdf_noise_std_mm", 0.0))
    if min(max_translation, max_offset, noise_std) < 0:
        raise ValueError("prior perturbation magnitudes must be non-negative")

    translation_world = rng.uniform(-max_translation, max_translation, size=3)
    offset_mm = float(rng.uniform(-max_offset, max_offset))
    noise = None
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=array.shape).astype(np.float32)
        noise -= noise.mean(axis=(1, 2, 3), keepdims=True)
    result, translation_voxels = apply_prior_transform(
        array, affine, translation_world_mm=translation_world,
        sdf_offset_mm=offset_mm, sdf_noise_mm=noise,
        sdf_clip_mm=sdf_clip_mm)
    record = PerturbationRecord(
        True, tuple(float(x) for x in translation_world),
        tuple(float(x) for x in translation_voxels), offset_mm, noise_std,
        float(sdf_clip_mm),
    )
    return result, record.to_dict()


def apply_perturbed_prior_to_conditioning(conditioning, perturbed_prior, spec):
    """Update only explicit coarse-SDF conditioning channels, if present."""
    cond = np.asarray(conditioning, dtype=np.float32).copy()
    prior = np.asarray(perturbed_prior, dtype=np.float32)
    names = list(spec.conditioning_channel_names)
    for side, name in enumerate(("coarse_sdf_left", "coarse_sdf_right")):
        if name in names:
            cond[names.index(name)] = prior[side]
    return cond


def perturb_training_sample(conditioning, coarse_sdf, target_sdf, affine, spec,
                            config, rng, *, sdf_clip_mm):
    """Perturb only the prior path; return the target byte-for-byte unchanged."""
    prior, record = perturb_coarse_prior(
        coarse_sdf, affine, config, rng, sdf_clip_mm=sdf_clip_mm)
    cond = apply_perturbed_prior_to_conditioning(conditioning, prior, spec)
    return cond, prior, target_sdf, record
