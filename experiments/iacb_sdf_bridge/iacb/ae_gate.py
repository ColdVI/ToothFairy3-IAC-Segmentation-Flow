"""L0 measurement for the latent arm (the supervisor's "SDF -> latent encode/decode" hypothesis).

A latent bridge can never be better than its decoder. Before training any latent bridge, train a
frozen-able SDF autoencoder at downsampling factor f and measure the reconstruction CEILING on the
untouched eval fold:
  * GT SDF  -> AE -> mask : Dice, HD95, beta0 preserved?     (can the latent hold the true canal?)
  * prior SDF -> AE -> mask: beta0 preserved w.r.t. the prior (does the AE itself connect/cut?)
The AE is trained on training-fold SDFs only (both x0 and x1, since the latent bridge encodes both).

This is OPTIONAL and is not part of run_all: it never delays or blocks the pixel arm. Run it once,
at the single preregistered factor f = 2, after the pixel arm has a checkpoint. It is a measurement,
not a sweep.

Reference thresholds (for interpreting the number, not a gate):
  gt_dice_mean >= 0.98, gt_beta0_kept >= 0.99, gt_hd95_mean <= 0.35 mm, prior_beta0_kept >= 0.98

  python -m iacb.ae_gate --cache /content/iac/cache --out ae_f2 --factor 2 --zc 8 --iters 4000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridge import _gn, bridge_loss, decode_labels, gaussian_weight, pad_multiple, window_starts
from .common import side_metrics
from .train import PatchStream, splits


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv3d(c, c, 3, padding=1), _gn(c), nn.SiLU(),
                               nn.Conv3d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.b(x)


class SDFAE(nn.Module):
    def __init__(self, factor=2, zc=8, w=32):
        super().__init__()
        n = int(round(math.log2(factor)))
        assert 2 ** n == factor, "factor must be a power of two"
        self.factor = factor
        enc = [nn.Conv3d(2, w, 3, padding=1), Res(w)]
        for _ in range(n):
            enc += [nn.Conv3d(w, w, 3, stride=2, padding=1), Res(w)]
        enc += [_gn(w), nn.SiLU(), nn.Conv3d(w, zc, 1)]
        dec = [nn.Conv3d(zc, w, 3, padding=1), Res(w)]
        for _ in range(n):
            dec += [nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(w, w, 3, padding=1), Res(w)]
        dec += [_gn(w), nn.SiLU(), nn.Conv3d(w, 2, 1)]
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return torch.tanh(self.dec(self.enc(x)))


@torch.no_grad()
def reconstruct(model, x, patch, overlap=0.5):
    """x (1,2,Z,Y,X) normalised SDF on device -> reconstruction, sliding window."""
    _, C, Z, Y, X = x.shape
    pz, py, px = [min(p, s) for p, s in zip(patch, (Z, Y, X))]
    need = pad_multiple((pz, py, px), model.factor)
    w = gaussian_weight(need, x.device)
    acc = torch.zeros_like(x); norm = torch.zeros((1, 1, Z, Y, X), device=x.device)
    for z in window_starts(Z, pz, overlap):
        for y in window_starts(Y, py, overlap):
            for xx in window_starts(X, px, overlap):
                sl = (slice(None), slice(None), slice(z, z + pz), slice(y, y + py), slice(xx, xx + px))
                xs = F.pad(x[sl], [0, need[2] - px, 0, need[1] - py, 0, need[0] - pz], value=1.0)
                out = model(xs)[..., :pz, :py, :px]
                ww = w[:pz, :py, :px]
                acc[sl] += out * ww; norm[sl] += ww
    return acc / norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--zc", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--eval_fold", type=int, default=0)
    ap.add_argument("--patch", default="64,96,96")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit_eval", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    cache, out = Path(a.cache), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    patch = [int(v) for v in a.patch.split(",")]
    assert pad_multiple(patch, a.factor) == patch, "patch must be divisible by factor"
    tr, va, ev = splits(cache, a.eval_fold, 0, 0)
    assert not set(tr) & set(ev)
    clip = json.loads((cache / f"{tr[0]}_meta.json").read_text())["clip_mm"]
    model = SDFAE(a.factor, a.zc, a.width).to(dev)
    params = sum(p.numel() for p in model.parameters())
    ratio = a.zc / (2 * a.factor ** 3)
    print(f"AE factor {a.factor} zc {a.zc}: params {params/1e6:.2f}M, latent/state size ratio {ratio:.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(a.iters, 1), pct_start=0.05)
    it_tr = iter(torch.utils.data.DataLoader(PatchStream(cache, tr, patch, 0.3, 0.6, False, 0),
                                             batch_size=a.batch, num_workers=a.workers))
    t0 = time.time()
    for it in range(a.iters):
        _, x0, x1, _ = next(it_tr)
        pick = (torch.rand(len(x0)) < 0.5).view(-1, 1, 1, 1, 1)
        x = torch.where(pick, x0, x1).to(dev)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            xh = model(x)
        loss, parts = bridge_loss(xh, x, clip)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if it % 100 == 0 or it == a.iters - 1:
            print(dict(it=it, loss=float(loss), sec=round(time.time() - t0, 1), **parts), flush=True)
    torch.save(dict(model=model.state_dict(), factor=a.factor, zc=a.zc, width=a.width), out / "ae.pt")

    model.eval()
    rows = []
    for ci, case in enumerate(ev[: a.limit_eval or None]):
        meta = json.loads((cache / f"{case}_meta.json").read_text()); sp = meta["spacing"]
        gt = np.load(cache / f"{case}_gt.npy"); gt2 = np.stack([gt == 1, gt == 2])
        rec = {}
        for key in ("x1", "x0"):
            x = torch.from_numpy(np.load(cache / f"{case}_{key}.npy").astype(np.float32) / clip)[None].to(dev)
            rec[key] = decode_labels(reconstruct(model, x, patch))[0].cpu().numpy()
            if key == "x0":
                src = decode_labels(x)[0].cpu().numpy()
        for si, side in enumerate(("L", "R")):
            g = gt2[si]
            if not g.any():
                continue
            m1 = side_metrics(rec["x1"] == si + 1, g, sp, with_cldice=False)
            m0 = side_metrics(rec["x0"] == si + 1, src == si + 1, sp, with_cldice=False)
            rows.append(dict(case=case, side=side, gt_dice=m1["dice"], gt_hd95=m1["hd95"],
                             gt_beta0_kept=m1["beta0_err"] == 0, gt_vol_ratio=m1["vol_ratio"],
                             prior_dice=m0["dice"], prior_beta0_kept=m0["beta0_err"] == 0))
        print(f"[{ci+1}/{len(ev)}] {case}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "ae_per_side.csv", index=False)
    res = dict(factor=a.factor, zc=a.zc, params=params, latent_state_ratio=ratio, n_sides=len(df),
               gt_dice_mean=float(df.gt_dice.mean()), gt_dice_min=float(df.gt_dice.min()),
               gt_hd95_mean=float(df.gt_hd95.mean()), gt_beta0_kept=float(df.gt_beta0_kept.mean()),
               gt_vol_ratio_mean=float(df.gt_vol_ratio.mean()),
               prior_dice_mean=float(df.prior_dice.mean()), prior_beta0_kept=float(df.prior_beta0_kept.mean()))
    crit = dict(gt_dice_mean=res["gt_dice_mean"] >= 0.98, gt_beta0_kept=res["gt_beta0_kept"] >= 0.99,
                gt_hd95_mean=res["gt_hd95_mean"] <= 0.35, prior_beta0_kept=res["prior_beta0_kept"] >= 0.98)
    res["criteria"] = crit
    res["verdict"] = "latent representation preserves the canal at this factor" if all(crit.values()) else \
        "latent bridge at this factor would be capped below the pixel-space arm"
    (out / "ae_gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
