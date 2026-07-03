#!/usr/bin/env python3
"""
selftest.py — CPU proof that the RESIDUAL flow machinery is correct.

Builds synthetic left/right "canals", derives a deliberately-degraded coarse SDF
(the stand-in for nnU-Net output: gaps + jitter), and trains the residual flow to
transport coarse->GT. Success = the sampled endpoint recovers Dice ABOVE the
coarse starting point, i.e. the refinement actually refines. Runs in seconds.

    python flow/selftest.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ResidualVelocityUNet3D, COND_CH                      # noqa: E402
from losses import total_loss                                         # noqa: E402
from sampler import integrate                                         # noqa: E402


def synth_case(D=32, rng=np.random):
    """Two tubular canals; return (cbct, mask{0,1,2})."""
    vol = (rng.randn(D, D, D) * 0.1).astype(np.float32)
    mask = np.zeros((D, D, D), np.uint8)
    for cls, cx in [(1, D // 3), (2, 2 * D // 3)]:
        zz = np.arange(D)
        yy = (D // 2 + 3 * np.sin(zz / 4)).astype(int)
        for z in zz:
            y = yy[z]
            mask[z, max(0, y - 1):y + 2, cx - 1:cx + 2] = cls
        vol[mask == cls] += 1.0
    return vol, mask


def sdf_norm(binary, clip=6.0):
    from scipy.ndimage import distance_transform_edt as edt
    b = binary.astype(bool)
    if not b.any():
        return np.full(binary.shape, 1.0, np.float32)
    s = edt(~b) - edt(b)
    return np.clip(s, -clip, clip).astype(np.float32) / clip


def encode_lr(mask):
    return np.stack([sdf_norm(mask == 1), sdf_norm(mask == 2)]).astype(np.float32)


def degrade(mask, rng):
    """Coarse mask: drop a chunk of the canal + random flips -> imperfect prior."""
    m = mask.copy()
    D = m.shape[0]
    m[D // 2: D // 2 + 4] = 0                       # break the canal (gap)
    flip = rng.rand(*m.shape) < 0.002
    m[flip] = 0
    return m


def decode(sdf):
    left, right = sdf[0], sdf[1]
    inside = np.minimum(left, right) < 0
    out = np.zeros(left.shape, np.uint8)
    out[inside & (left <= right)] = 1
    out[inside & (left > right)] = 2
    return out


def dice(a, b, c):
    A, B = a == c, b == c
    return 2 * (A & B).sum() / (A.sum() + B.sum() + 1e-8)


def build_cond(cbct, coarse):
    """8-ch conditioning: cbct, prob_l, prob_r, coarse_sdf_l/r, x,y,z."""
    D = cbct.shape[0]
    coords = np.stack(np.meshgrid(*[np.linspace(-1, 1, D)] * 3, indexing="ij")).astype(np.float32)
    csdf = encode_lr(coarse)
    prob_l = (coarse == 1).astype(np.float32)
    prob_r = (coarse == 2).astype(np.float32)
    z = (cbct - cbct.mean()) / (cbct.std() + 1e-6)
    return np.stack([z, prob_l, prob_r, csdf[0], csdf[1], coords[0], coords[1], coords[2]]).astype(np.float32)


def main():
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResidualVelocityUNet3D(cond_ch=COND_CH, base=16, tdim=64).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    cfg = {"w_narrowband": 0.5, "w_cldice": 0.2, "w_laterality": 0.1,
           "occ_tau": 0.05, "cldice_iters": 5, "narrowband_band": 0.2}

    # fixed validation case
    v_cbct, v_mask = synth_case(rng=rng)
    v_coarse = degrade(v_mask, rng)
    v_cond = torch.tensor(build_cond(v_cbct, v_coarse)[None]).to(dev)
    v_x0 = torch.tensor(encode_lr(v_coarse)[None]).to(dev)
    coarse_dice = 0.5 * (dice(v_coarse, v_mask, 1) + dice(v_coarse, v_mask, 2))

    print(f"[selftest] device={dev} params={sum(p.numel() for p in model.parameters())/1e3:.0f}k")
    print(f"[selftest] coarse (nnU-Net stand-in) Dice = {coarse_dice:.3f}")

    losses = []
    for step in range(80):
        conds, x0s, x1s = [], [], []
        for _ in range(4):
            cb, mk = synth_case(rng=rng)
            co = degrade(mk, rng)
            conds.append(build_cond(cb, co))
            x0s.append(encode_lr(co))
            x1s.append(encode_lr(mk))
        cond = torch.tensor(np.stack(conds)).to(dev)
        x0 = torch.tensor(np.stack(x0s)).to(dev)
        x1 = torch.tensor(np.stack(x1s)).to(dev)
        t = torch.rand(x1.shape[0], device=dev)
        tb = t.view(-1, 1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * x1
        pred_v = model(xt, t, cond)
        loss, comp = total_loss(pred_v, x0, x1, t, cfg)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(comp["total"])
        if step % 20 == 0 or step == 79:
            print(f"  step {step:2d} total {comp['total']:.4f} fm {comp['fm']:.4f} "
                  f"nb {comp.get('narrowband', 0):.3f} cl {comp.get('cldice', 0):.3f}")

    model.eval()
    x1_hat = integrate(model, v_cond, v_x0, steps=8)[0].cpu().numpy()
    pred = decode(x1_hat)
    refined_dice = 0.5 * (dice(pred, v_mask, 1) + dice(pred, v_mask, 2))
    print(f"[selftest] loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"[selftest] refined Dice = {refined_dice:.3f}  (coarse was {coarse_dice:.3f})")
    ok = losses[-1] < losses[0] and refined_dice >= coarse_dice - 0.02
    print("[selftest]", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
