import _pathsetup  # noqa: F401
import torch

from model import ResidualVelocityUNet3D, COND_CH, FLOW_STATE_CH
from losses import total_loss
from sampler import integrate


def test_forward_shapes():
    B, D = 2, 32
    model = ResidualVelocityUNet3D(cond_ch=COND_CH, base=8, tdim=32)
    xt = torch.randn(B, FLOW_STATE_CH, D, D, D)
    cond = torch.randn(B, COND_CH, D, D, D)
    t = torch.rand(B)
    v = model(xt, t, cond)
    assert v.shape == (B, FLOW_STATE_CH, D, D, D)


def test_total_loss_scalar_and_backward():
    B, D = 2, 16
    model = ResidualVelocityUNet3D(cond_ch=COND_CH, base=8, tdim=32)
    x0 = torch.randn(B, FLOW_STATE_CH, D, D, D)
    x1 = torch.randn(B, FLOW_STATE_CH, D, D, D)
    cond = torch.randn(B, COND_CH, D, D, D)
    t = torch.rand(B)
    tb = t.view(-1, 1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * x1
    v = model(xt, t, cond)
    cfg = {"w_narrowband": 1.0, "w_cldice": 0.5, "w_laterality": 0.5,
           "cldice_iters": 3, "occ_tau": 0.05, "narrowband_band": 0.2}
    loss, comp = total_loss(v, x0, x1, t, cfg)
    assert loss.ndim == 0
    for k in ("fm", "narrowband", "cldice", "laterality", "total"):
        assert k in comp
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_integrate_shape():
    B, D = 1, 16
    model = ResidualVelocityUNet3D(cond_ch=COND_CH, base=8, tdim=32).eval()
    cond = torch.randn(B, COND_CH, D, D, D)
    x0 = torch.randn(B, FLOW_STATE_CH, D, D, D)
    out = integrate(model, cond, x0, steps=4)
    assert out.shape == (B, FLOW_STATE_CH, D, D, D)


if __name__ == "__main__":
    import sys
    _pathsetup.run_module(sys.modules[__name__])
