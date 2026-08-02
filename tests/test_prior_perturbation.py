import numpy as np
from scipy.ndimage import center_of_mass

from data.io_utils import sdf_stack_to_mask
from flow.channel_contract import resolve_conditioning_spec
from flow.prior_perturbation import (
    apply_perturbed_prior_to_conditioning,
    apply_prior_transform,
    perturb_coarse_prior,
    perturb_training_sample,
)


def _sphere(shape=(25, 25, 25), radius=5):
    zz, yy, xx = np.indices(shape)
    distance = np.sqrt((zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 12) ** 2)
    left = np.clip((distance - radius) / 10.0, -1, 1)
    right = np.ones(shape, dtype=np.float32)
    return np.stack([left, right]).astype(np.float32)


def test_signed_offset_erodes_and_dilates_negative_inside_prior():
    prior = _sphere()
    affine = np.eye(4)
    eroded, _ = apply_prior_transform(
        prior, affine, sdf_offset_mm=1.0, sdf_clip_mm=10.0)
    dilated, _ = apply_prior_transform(
        prior, affine, sdf_offset_mm=-1.0, sdf_clip_mm=10.0)
    base_n = np.count_nonzero(sdf_stack_to_mask(prior))
    assert np.count_nonzero(sdf_stack_to_mask(eroded)) < base_n
    assert np.count_nonzero(sdf_stack_to_mask(dilated)) > base_n


def test_world_translation_respects_anisotropic_affine_spacing():
    prior = _sphere()
    affine = np.diag([2.0, 1.0, 0.5, 1.0])
    translated, voxel_shift = apply_prior_transform(
        prior, affine, translation_world_mm=(2.0, -1.0, 1.0))
    assert np.allclose(voxel_shift, (1.0, -1.0, 2.0))
    before = np.asarray(center_of_mass(prior[0] < 0))
    after = np.asarray(center_of_mass(translated[0] < 0))
    measured_world = affine[:3, :3] @ (after - before)
    assert np.allclose(measured_world, (2.0, -1.0, 1.0), atol=0.26)


def test_conditioning_update_never_changes_cbct_or_nonprior_channels():
    spec = resolve_conditioning_spec({"cond_include_coarse_sdf": True})
    cond = np.arange(8 * 4 * 4 * 4, dtype=np.float32).reshape(8, 4, 4, 4)
    prior = np.stack([np.full((4, 4, 4), -0.2, dtype=np.float32),
                      np.full((4, 4, 4), 0.3, dtype=np.float32)])
    updated = apply_perturbed_prior_to_conditioning(cond, prior, spec)
    assert np.array_equal(updated[0], cond[0])
    assert np.array_equal(updated[1:3], cond[1:3])
    assert np.array_equal(updated[5:], cond[5:])
    assert np.array_equal(updated[3:5], prior)


def test_training_sample_perturbation_leaves_cbct_and_target_unchanged():
    spec = resolve_conditioning_spec({"cond_include_coarse_sdf": False})
    rng = np.random.default_rng(8)
    cond = rng.normal(size=(6, 7, 7, 7)).astype(np.float32)
    prior = rng.normal(size=(2, 7, 7, 7)).astype(np.float32)
    target = rng.normal(size=(2, 7, 7, 7)).astype(np.float32)
    changed_cond, _changed_prior, unchanged_target, _ = perturb_training_sample(
        cond, prior, target, np.eye(4), spec,
        {"enabled": True, "probability": 1, "max_translation_mm": 0.3,
         "max_sdf_offset_mm": 0.3, "sdf_noise_std_mm": 0.05},
        np.random.default_rng(9), sdf_clip_mm=10.0)
    assert np.array_equal(changed_cond[0], cond[0])
    assert unchanged_target is target
    assert np.array_equal(unchanged_target, target)


def test_prompt3r_six_channel_conditioning_is_unchanged():
    spec = resolve_conditioning_spec({"cond_include_coarse_sdf": False})
    cond = np.random.default_rng(1).normal(size=(6, 3, 3, 3)).astype(np.float32)
    prior = np.zeros((2, 3, 3, 3), dtype=np.float32)
    assert np.array_equal(apply_perturbed_prior_to_conditioning(cond, prior, spec), cond)


def test_disabled_mode_is_identity_and_does_not_alias_input():
    prior = _sphere()
    out, record = perturb_coarse_prior(
        prior, np.eye(4), {"enabled": False}, np.random.default_rng(7),
        sdf_clip_mm=10.0)
    assert np.array_equal(out, prior)
    assert out is not prior
    assert record["applied"] is False


def test_same_seed_produces_identical_transform_and_provenance():
    cfg = {"enabled": True, "probability": 1.0, "max_translation_mm": 0.3,
           "max_sdf_offset_mm": 0.3, "sdf_noise_std_mm": 0.05}
    a, rec_a = perturb_coarse_prior(
        _sphere(), np.eye(4), cfg, np.random.default_rng(42), sdf_clip_mm=10.0)
    b, rec_b = perturb_coarse_prior(
        _sphere(), np.eye(4), cfg, np.random.default_rng(42), sdf_clip_mm=10.0)
    assert np.array_equal(a, b)
    assert rec_a == rec_b


def test_transform_preserves_left_right_channel_identity():
    shape = (15, 15, 15)
    prior = np.ones((2,) + shape, dtype=np.float32)
    prior[0, 2:5, 2:5, 2:5] = -1
    prior[1, 10:13, 10:13, 10:13] = -1
    out, _ = perturb_coarse_prior(
        prior, np.eye(4), {"enabled": True, "probability": 1,
                           "max_translation_mm": 0, "max_sdf_offset_mm": 0,
                           "sdf_noise_std_mm": 0}, np.random.default_rng(3),
        sdf_clip_mm=10.0)
    assert np.asarray(center_of_mass(out[0] < 0))[0] < 7
    assert np.asarray(center_of_mass(out[1] < 0))[0] > 7
