import inspect

import numpy as np
import pytest
import torch
from scipy.ndimage import binary_dilation

import _pathsetup  # noqa: F401
from flow.losses import (compute_prompt3r_training_loss, predict_t0_endpoint,
                         sample_mixed_low_t, sdf_to_occupancy,
                         soft_cldice_loss, soft_dice_loss)
from flow.model import ResidualVelocityUNet3D


def _tube_sdf(shape=(20, 20, 20)):
    zz, yy, xx = np.ogrid[tuple(slice(size) for size in shape)]
    mask = ((yy - 10) ** 2 + (xx - 10) ** 2 <= 3 ** 2) & (zz >= 2) & (zz < 18)
    from scipy.ndimage import distance_transform_edt
    return (distance_transform_edt(~mask) - distance_transform_edt(mask)).astype(np.float32), mask


def test_soft_dice_perfect_and_dilation_behaviour():
    sdf, mask = _tube_sdf()
    perfect = torch.from_numpy(mask[None, None].astype(np.float32))
    dilated = torch.from_numpy(binary_dilation(mask, iterations=1)[None, None].astype(np.float32))
    assert soft_dice_loss(perfect, perfect).item() == pytest.approx(0.0, abs=1e-6)
    assert soft_dice_loss(dilated, perfect).item() > 0.05
    assert torch.isfinite(sdf_to_occupancy(torch.from_numpy(sdf))).all()


def test_broken_tube_increases_soft_cldice_loss():
    _, mask = _tube_sdf()
    intact = torch.from_numpy(mask[None, None].astype(np.float32))
    broken = intact.clone(); broken[:, :, 8:13] = 0
    assert soft_cldice_loss(broken, intact, iters=6) > soft_cldice_loss(intact, intact, iters=6)


def test_t0_predictor_does_not_accept_ground_truth_and_gradients_reach_model():
    assert "x1" not in inspect.signature(predict_t0_endpoint).parameters
    model = ResidualVelocityUNet3D(
        cond_ch=6, base=8, tdim=32, zero_init_head=True)
    state = torch.randn(1, 2, 12, 12, 12)
    target = torch.randn_like(state)
    cond = torch.randn(1, 6, 12, 12, 12)
    cfg = {"low_t_fraction": .5, "low_t_max": .25, "w_t0_endpoint": 1,
           "w_t0_sdf": 1, "w_t0_narrowband": 1, "w_t0_softdice": 1,
           "w_t0_cldice": .5, "narrowband_band": .2, "occ_tau": .05,
           "cldice_iters": 2, "random_t_aux_enabled": False}
    loss, components = compute_prompt3r_training_loss(model, cond, state, target, cfg)
    assert set(("fm_random_t", "t0_sdf", "t0_narrowband", "t0_softdice",
                "t0_cldice", "total")) <= set(components)
    assert all(np.isfinite(value) for value in components.values())
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_mixed_low_t_sampler_is_reproducible_and_partitioned():
    cfg = {"low_t_fraction": .5, "low_t_max": .25}
    g1 = torch.Generator().manual_seed(31)
    g2 = torch.Generator().manual_seed(31)
    t1, low1 = sample_mixed_low_t(20, cfg, "cpu", generator=g1)
    t2, low2 = sample_mixed_low_t(20, cfg, "cpu", generator=g2)
    assert torch.equal(t1, t2) and torch.equal(low1, low2)
    assert low1.sum().item() == 10
    assert torch.all(t1[low1] <= .25)


def test_zero_low_t_fraction_is_seeded_uniform():
    cfg = {"low_t_fraction": 0, "low_t_max": .25}
    t, low = sample_mixed_low_t(
        100, cfg, "cpu", generator=torch.Generator().manual_seed(2))
    assert not low.any()
    assert torch.all((t >= 0) & (t <= 1))
    assert torch.unique(t).numel() > 90


class _ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.value = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, state, time, cond):
        del time, cond
        return torch.zeros_like(state) + self.value


def test_zero_t0_weight_reduces_to_random_flow_matching():
    model = _ConstantVelocity()
    state = torch.randn(2, 2, 4, 4, 4)
    target = torch.randn_like(state)
    cond = torch.randn(2, 6, 4, 4, 4)
    cfg = {"low_t_fraction": .5, "low_t_max": .25,
           "w_t0_endpoint": 0, "random_t_aux_enabled": False}
    loss, components = compute_prompt3r_training_loss(
        model, cond, state, target, cfg,
        generator=torch.Generator().manual_seed(4))
    expected = torch.nn.functional.mse_loss(
        torch.zeros_like(state) + model.value, target - state)
    assert torch.equal(loss, expected)
    assert components["total"] == pytest.approx(components["fm_random_t"])
