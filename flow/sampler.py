#!/usr/bin/env python3
"""
sampler.py — ODE integration for the residual flow.

The residual flow starts NOT from pure noise but from the coarse nnU-Net SDF:

    x0 = coarse_sdf + sigma * eps          (eps optional; sigma=0 => deterministic)

and integrates dx/dt = v_theta(x_t, t, cond) from t=0 to t=1 with a Heun (2nd
order) solver. The deterministic sigma=0 run is the headline prediction; a
nonzero sigma with a fixed global noise field drives the uncertainty estimate.
"""
import torch


@torch.no_grad()
def integrate(model, cond, x0, steps=8, method="heun"):
    """
    model: ResidualVelocityUNet3D; cond: (B,cond_ch,D,H,W); x0: (B,2,D,H,W).
    Returns the endpoint SDF estimate x1_hat: (B,2,D,H,W).
    """
    x = x0
    B = x.shape[0]
    ts = torch.linspace(0, 1, steps + 1, device=x.device)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        tb0 = torch.full((B,), float(t0), device=x.device)
        v0 = model(x, tb0, cond)
        if method == "euler":
            x = x + dt * v0
        else:
            tb1 = torch.full((B,), float(t1), device=x.device)
            v1 = model(x + dt * v0, tb1, cond)
            x = x + dt * 0.5 * (v0 + v1)
    return x


def make_x0(coarse_sdf, sigma=0.0, noise=None):
    """x0 = coarse_sdf + sigma*noise. `noise` lets a caller supply a fixed field."""
    if sigma <= 0:
        return coarse_sdf
    if noise is None:
        noise = torch.randn_like(coarse_sdf)
    return coarse_sdf + sigma * noise
