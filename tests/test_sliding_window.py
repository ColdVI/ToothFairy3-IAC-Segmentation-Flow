import _pathsetup  # noqa: F401
import numpy as np
import torch

from model import ResidualVelocityUNet3D, COND_CH
from sliding_window import predict_volume, _tile_starts, _gaussian_weight


def test_tile_covers_volume():
    # every voxel must be covered by at least one patch start window
    size, patch, step = 40, 16, 8
    starts = _tile_starts(size, patch, step)
    covered = np.zeros(size, bool)
    for s in starts:
        covered[s:s + patch] = True
    assert covered.all()
    assert starts[-1] == size - patch


def test_gaussian_weight_positive_peaked():
    g = _gaussian_weight((8, 8, 8))
    assert g.shape == (8, 8, 8)
    assert g.min() > 0
    # centre weight is (near) the maximum
    assert g[4, 4, 4] >= g[0, 0, 0]


class _ZeroVelocity(torch.nn.Module):
    def forward(self, state, t, cond):
        del t, cond
        return torch.zeros_like(state)


def test_zero_velocity_full_path_preserves_identity_sdf_and_mask():
    shape = (35, 37, 39)
    rng = np.random.default_rng(7)
    cond = np.zeros((COND_CH, *shape), dtype=np.float32)
    coarse = rng.uniform(-1, 1, size=(2, *shape)).astype(np.float32)
    out = predict_volume(_ZeroVelocity(), cond, coarse, patch=16, overlap=.5,
                         steps=2, device="cpu")
    assert np.allclose(out, coarse, atol=2e-6, rtol=2e-6)
    assert np.array_equal(out < 0, coarse < 0)
    assert np.array_equal(np.argmin(out, axis=0), np.argmin(coarse, axis=0))


def test_predict_volume_shape_and_coverage():
    D = 24
    model = ResidualVelocityUNet3D(cond_ch=COND_CH, base=8, tdim=32).eval()
    cond = np.random.randn(COND_CH, D, D, D).astype(np.float32)
    coarse = np.random.randn(2, D, D, D).astype(np.float32)
    with torch.no_grad():
        out = predict_volume(model, cond, coarse, patch=16, overlap=0.5, steps=2, device="cpu")
    assert out.shape == (2, D, D, D)
    assert np.isfinite(out).all()


if __name__ == "__main__":
    import sys
    _pathsetup.run_module(sys.modules[__name__])
