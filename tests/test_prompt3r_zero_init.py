import numpy as np
import pytest
import torch

import _pathsetup  # noqa: F401
from data.io_utils import sdf_stack_to_mask
from flow.model import FLOW_STATE_CH, ResidualVelocityUNet3D
from flow.sampler import integrate
from flow.sliding_window import predict_volume


@pytest.mark.parametrize("cond_ch", [8, 6])
def test_zero_init_head_emits_exact_zero_velocity(cond_ch):
    model = ResidualVelocityUNet3D(
        cond_ch=cond_ch, base=8, tdim=32, zero_init_head=True).eval()
    state = torch.randn(2, FLOW_STATE_CH, 12, 12, 12)
    cond = torch.randn(2, cond_ch, 12, 12, 12)
    time = torch.rand(2)
    with torch.no_grad():
        velocity = model(state, time, cond)
    assert torch.count_nonzero(velocity) == 0


@pytest.mark.parametrize("cond_ch", [8, 6])
def test_zero_init_integrate_preserves_prior_exactly(cond_ch):
    model = ResidualVelocityUNet3D(
        cond_ch=cond_ch, base=8, tdim=32, zero_init_head=True).eval()
    state = torch.randn(1, FLOW_STATE_CH, 12, 12, 12)
    cond = torch.randn(1, cond_ch, 12, 12, 12)
    with torch.no_grad():
        endpoint = integrate(model, cond, state, steps=8)
    assert torch.equal(endpoint, state)


@pytest.mark.parametrize("cond_ch", [8, 6])
def test_zero_init_sliding_window_preserves_sdf_and_decoded_mask(cond_ch):
    model = ResidualVelocityUNet3D(
        cond_ch=cond_ch, base=8, tdim=32, zero_init_head=True).eval()
    rng = np.random.default_rng(19 + cond_ch)
    shape = (17, 18, 19)
    cond = rng.normal(size=(cond_ch, *shape)).astype(np.float32)
    coarse = rng.uniform(-1, 1, size=(FLOW_STATE_CH, *shape)).astype(np.float32)
    endpoint = predict_volume(
        model, cond, coarse, patch=12, overlap=0.5, steps=8, device="cpu")
    assert np.allclose(endpoint, coarse, atol=2e-6, rtol=2e-6)
    assert np.array_equal(sdf_stack_to_mask(endpoint), sdf_stack_to_mask(coarse))


def test_legacy_initialization_remains_nonzero_when_disabled():
    torch.manual_seed(4)
    model = ResidualVelocityUNet3D(
        cond_ch=8, base=8, tdim=32, zero_init_head=False)
    final = model.head[-1]
    assert torch.count_nonzero(final.weight) > 0
