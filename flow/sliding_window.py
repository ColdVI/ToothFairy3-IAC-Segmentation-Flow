#!/usr/bin/env python3
"""
sliding_window.py — coherent whole-volume residual-flow inference.

Tiles the volume into overlapping patches, integrates the ODE per patch from the
coarse-SDF start state, and blends the endpoint SDFs with a Gaussian window so
patch seams do not create spurious canal breaks. For uncertainty, a single
global noise field is cropped per patch (not re-sampled), keeping the noise
spatially coherent across seams.
"""
import numpy as np
import torch

from sampler import integrate, make_x0


def _gaussian_weight(patch, sigma_scale=0.125):
    coords = [np.linspace(-1, 1, p) for p in patch]
    g = np.ones(patch, np.float32)
    for ax, c in enumerate(coords):
        shape = [1, 1, 1]
        shape[ax] = len(c)
        g = g * np.exp(-(c ** 2) / (2 * sigma_scale ** 2)).reshape(shape).astype(np.float32)
    return g


def _tile_starts(size, patch, step):
    if size <= patch:
        return [0]
    xs = list(range(0, size - patch + 1, step))
    if xs[-1] != size - patch:
        xs.append(size - patch)
    return xs


@torch.no_grad()
def predict_volume(model, cond_vol, coarse_sdf_vol, patch=96, overlap=0.5,
                   steps=8, sigma=0.0, global_noise=None, device="cpu"):
    """
    cond_vol: (cond_ch,D,H,W) np; coarse_sdf_vol: (2,D,H,W) np (normalised SDF).
    Returns endpoint SDF (2,D,H,W) np. Set sigma>0 + a global_noise field for the
    stochastic uncertainty mode.
    """
    C, D, H, W = cond_vol.shape
    step = max(1, int(patch * (1 - overlap)))
    acc = np.zeros((2, D, H, W), np.float32)
    wsum = np.zeros((D, H, W), np.float32)
    gw = _gaussian_weight((patch, patch, patch))

    for z in _tile_starts(D, patch, step):
        for y in _tile_starts(H, patch, step):
            for x in _tile_starts(W, patch, step):
                sl = (slice(z, z + patch), slice(y, y + patch), slice(x, x + patch))
                c = cond_vol[:, sl[0], sl[1], sl[2]]
                x0 = coarse_sdf_vol[:, sl[0], sl[1], sl[2]]
                dz, dy, dx = c.shape[1:]
                # pad partial edge patches
                pad = ((0, 0), (0, patch - dz), (0, patch - dy), (0, patch - dx))
                cpad = np.pad(c, pad)
                x0pad = np.pad(x0, pad)
                cond_t = torch.tensor(cpad[None]).float().to(device)
                x0_t = torch.tensor(x0pad[None]).float().to(device)
                if sigma > 0:
                    n = None
                    if global_noise is not None:
                        n = torch.tensor(np.pad(global_noise[:, sl[0], sl[1], sl[2]], pad)[None]).float().to(device)
                    x0_t = make_x0(x0_t, sigma, n)
                endp = integrate(model, cond_t, x0_t, steps=steps)[0].cpu().numpy()
                gwp = gw[:dz, :dy, :dx]
                acc[:, sl[0], sl[1], sl[2]] += endp[:, :dz, :dy, :dx] * gwp
                wsum[sl[0], sl[1], sl[2]] += gwp
    acc /= np.maximum(wsum, 1e-6)[None]
    return acc
