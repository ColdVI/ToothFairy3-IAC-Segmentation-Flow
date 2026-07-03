#!/usr/bin/env python3
"""
flow_matching_iac.py
====================
3D **conditional flow matching** for left/right inferior-alveolar-canal
segmentation — the research extension beyond a plain nnU-Net baseline.

Motivation
----------
The IAC is a thin, tubular, topologically-simple structure (one connected
curve per side, no branching, no holes). Discriminative voxel classifiers
(U-Net / nnU-Net) optimise per-voxel overlap (Dice / CE) and therefore leak,
break, or double the canal wherever local image evidence is weak — exactly the
failures that matter clinically (nerve-injury risk in implant planning).

Generative segmentation reframes the task: instead of predicting one label per
voxel, learn the *distribution* of plausible masks conditioned on the image,
and sample from it. **Flow matching** (Lipman et al. 2023; rectified flow,
Liu et al. 2023) is a simulation-free way to train such a generator: fit a
time-dependent velocity field that transports a simple prior to the mask
distribution along straight paths, then integrate an ODE at inference.

This is the same family SEAL-Flow sits in ("Continuous-time flow matching for
topologically consistent medical image segmentation"), but SEAL-Flow's public
release is 2D (PNG image/mask pairs, cell/nucleus datasets) with the trainer
and shape-regularisation bodies withheld. This module is an independent,
runnable **3D** formulation for CBCT.

Design
------
* Representation: the target mask is embedded as a **signed distance field
  (SDF)** per foreground class, not a one-hot indicator. SDFs vary smoothly in
  space, carry shape/topology information in their zero-level-set, and give the
  flow model an easier (continuous) regression target than a binary jump. At
  inference the sign of the integrated field recovers the mask. Set
  `--target onehot` to fall back to soft one-hot fields.
* Prior:  x0 ~ N(0, I), same shape as the target field.
* Path:   rectified (straight) flow   x_t = (1-t)*x0 + t*x1,   u = x1 - x0.
* Model:  a 3D U-Net velocity network v_theta(x_t, t, image). The CBCT image is
  the conditioning input (channel-concatenated); t is injected via a sinusoidal
  embedding added at every resolution.
* Loss:   E_{t,x0,x1} || v_theta(x_t,t,image) - (x1 - x0) ||^2
          + lambda_topo * boundary-consistency term (keeps the zero-level-set
          coherent — a lightweight stand-in for SEAL-Flow's shape regulariser).
* Sample: integrate dx/dt = v_theta from t=0..1 (Euler or Heun), then decode.

Usage
-----
    # smoke test on synthetic data (CPU, seconds) — proves the machinery:
    python flow_matching_iac.py selftest

    # train on the converted IAC dataset (GPU):
    python flow_matching_iac.py train \
        --data $nnUNet_raw/Dataset801_IAC_LR --patch 96 --epochs 500

    # sample a mask for one volume (GPU):
    python flow_matching_iac.py predict \
        --ckpt runs/fm_iac/best.pt --image case_0000.nii.gz --out pred.nii.gz
"""
import argparse, math, os, sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None


# =============================================================================
# 1. Time embedding
# =============================================================================
class SinusoidalTime(nn.Module):
    """Map scalar t in [0,1] to a `dim`-vector (transformer-style)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):                      # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        ang = t[:, None] * freqs[None, :]      # (B, half)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, dim)


# =============================================================================
# 2. 3D residual block with FiLM-style time conditioning
# =============================================================================
class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, cin), cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.temb  = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(min(8, cout), cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip  = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# =============================================================================
# 3. 3D U-Net velocity field  v_theta(x_t, t, image)
# =============================================================================
class VelocityUNet3D(nn.Module):
    """
    in_ch  = (#field channels) + (#image channels)   [x_t and image concatenated]
    out_ch = #field channels                          [predicted velocity]
    """
    def __init__(self, field_ch, img_ch=1, base=32, tdim=128):
        super().__init__()
        self.tmap = nn.Sequential(SinusoidalTime(tdim), nn.Linear(tdim, tdim),
                                  nn.SiLU(), nn.Linear(tdim, tdim))
        cin = field_ch + img_ch
        self.stem = nn.Conv3d(cin, base, 3, padding=1)
        # encoder (2 downsamples -> 3 resolution levels)
        self.e1 = ResBlock3D(base,     base,     tdim)   # full  res, base
        self.e2 = ResBlock3D(base,     base * 2, tdim)   # 1/2   res, base*2
        self.e3 = ResBlock3D(base * 2, base * 4, tdim)   # 1/4   res, base*4
        self.down = nn.AvgPool3d(2)
        # bottleneck
        self.mid = ResBlock3D(base * 4, base * 4, tdim)  # 1/4   res, base*4
        # decoder (symmetric; concat with same-res encoder skip)
        self.d2 = ResBlock3D(base * 4 + base * 2, base * 2, tdim)  # up->1/2: m(4b)+s2(2b)
        self.d1 = ResBlock3D(base * 2 + base,     base,     tdim)  # up->full: d2(2b)+s1(b)
        self.head = nn.Sequential(nn.GroupNorm(min(8, base), base), nn.SiLU(),
                                  nn.Conv3d(base, field_ch, 1))

    def forward(self, xt, t, image):
        temb = self.tmap(t)
        h = self.stem(torch.cat([xt, image], dim=1))
        s1 = self.e1(h, temb)                 # full res
        s2 = self.e2(self.down(s1), temb)     # 1/2 res
        s3 = self.e3(self.down(s2), temb)     # 1/4 res
        m  = self.mid(s3, temb)               # 1/4 res
        u = F.interpolate(m, size=s2.shape[2:], mode="trilinear", align_corners=False)
        u = self.d2(torch.cat([u, s2], 1), temb)
        u = F.interpolate(u, size=s1.shape[2:], mode="trilinear", align_corners=False)
        u = self.d1(torch.cat([u, s1], 1), temb)
        return self.head(u)


# =============================================================================
# 4. Flow-matching wrapper (rectified / straight-line conditional FM)
# =============================================================================
class RectifiedFlowSeg:
    def __init__(self, model, lambda_topo=0.1):
        self.model = model
        self.lambda_topo = lambda_topo

    def loss(self, x1, image):
        """x1: target field (B,C,D,H,W); image: conditioning (B,1,D,H,W)."""
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t  = torch.rand(B, device=x1.device)
        tb = t[:, None, None, None, None]
        xt = (1 - tb) * x0 + tb * x1               # straight path
        target_v = x1 - x0                          # constant velocity
        pred_v = self.model(xt, t, image)
        fm = F.mse_loss(pred_v, target_v)
        # --- lightweight topology / boundary-consistency term ---
        # push the predicted end-point field x1_hat = xt + (1-t)*pred_v toward a
        # spatially smooth zero-level-set (penalise gradient energy of the sign
        # transition region). This is a stand-in for SEAL-Flow's shape regulariser.
        x1_hat = xt + (1 - tb) * pred_v
        topo = self._boundary_energy(x1_hat)
        return fm + self.lambda_topo * topo, {"fm": fm.item(), "topo": topo.item()}

    @staticmethod
    def _boundary_energy(field):
        # total-variation of the tanh-squashed field ~ length of level sets;
        # discourages fragmented / holey boundaries.
        s = torch.tanh(field)
        dz = (s[:, :, 1:] - s[:, :, :-1]).abs().mean()
        dy = (s[:, :, :, 1:] - s[:, :, :, :-1]).abs().mean()
        dx = (s[:, :, :, :, 1:] - s[:, :, :, :, :-1]).abs().mean()
        return dz + dy + dx

    @torch.no_grad()
    def sample(self, image, field_ch, steps=50, method="heun"):
        """Integrate the ODE dx/dt = v(x,t,image) from t=0 (noise) to t=1."""
        B = image.shape[0]
        shape = (B, field_ch, *image.shape[2:])
        x = torch.randn(shape, device=image.device)
        ts = torch.linspace(0, 1, steps + 1, device=image.device)
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            dt = t1 - t0
            tb = torch.full((B,), t0, device=image.device)
            v0 = self.model(x, tb, image)
            if method == "euler":
                x = x + dt * v0
            else:  # Heun (2nd order) — straighter integration, fewer steps
                x_pred = x + dt * v0
                tb1 = torch.full((B,), t1, device=image.device)
                v1 = self.model(x_pred, tb1, image)
                x = x + dt * 0.5 * (v0 + v1)
        return x    # final field at t=1; decode with sign (SDF) or argmax (onehot)


# =============================================================================
# 5. Target-field encoders/decoders
# =============================================================================
def mask_to_sdf(mask, classes, clip=10.0):
    """mask:(D,H,W) int; return (C,D,H,W) signed distance per fg class (neg inside)."""
    from scipy.ndimage import distance_transform_edt as edt
    out = []
    for c in classes:
        m = (mask == c)
        if m.any():
            din  = edt(m)
            dout = edt(~m)
            sdf = dout - din          # negative inside, positive outside
        else:
            sdf = np.full(mask.shape, clip, np.float32)
        out.append(np.clip(sdf, -clip, clip) / clip)   # normalise to ~[-1,1]
    return np.stack(out).astype(np.float32)


def sdf_to_mask(field):
    """field:(C,D,H,W); voxel is class c+1 where that channel is most-negative & <0."""
    neg = -field                                   # larger => more inside
    best = neg.argmax(0)
    inside = (neg.max(0) > 0)
    out = np.where(inside, best + 1, 0).astype(np.uint8)
    return out


def mask_to_onehot(mask, classes):
    return np.stack([(mask == c).astype(np.float32) for c in classes])


def onehot_to_mask(field):
    fg = field.argmax(0) + 1
    return np.where(field.max(0) > 0.5, fg, 0).astype(np.uint8)


# =============================================================================
# 6. Self-test — synthetic 3D "canals", CPU, proves the machinery is correct
# =============================================================================
def selftest():
    assert torch is not None, "install torch first"
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = 32
    classes = [1, 2]
    field_ch = len(classes)

    def make_case():
        """A tiny CBCT-like volume with two tubular 'canals' (L & R)."""
        vol = np.random.randn(D, D, D).astype(np.float32) * 0.1
        mask = np.zeros((D, D, D), np.uint8)
        for cls, cx in [(1, D // 3), (2, 2 * D // 3)]:
            zz = np.arange(D)
            yy = (D // 2 + 3 * np.sin(zz / 4)).astype(int)
            for z in zz:
                y, x = yy[z], cx
                mask[z, max(0, y-1):y+2, max(0, x-1):x+2] = cls
            vol[mask == cls] += 1.0        # canal is brighter -> learnable cue
        return vol, mask

    def encode(mask):
        return mask_to_sdf(mask, classes)

    model = VelocityUNet3D(field_ch=field_ch, img_ch=1, base=16, tdim=64).to(dev)
    fm = RectifiedFlowSeg(model, lambda_topo=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    # one fixed validation case
    v_val, m_val = make_case()
    img_val = torch.tensor(v_val[None, None]).to(dev)

    print(f"[selftest] device={dev}  params={sum(p.numel() for p in model.parameters())/1e3:.0f}k")
    losses = []
    for step in range(60):
        vols, tgts = [], []
        for _ in range(4):                        # batch of 4 synthetic cases
            v, m = make_case()
            vols.append(v[None]); tgts.append(encode(m))
        image = torch.tensor(np.stack(vols)).to(dev)
        x1    = torch.tensor(np.stack(tgts)).to(dev)
        opt.zero_grad()
        loss, parts = fm.loss(x1, image)
        loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 15 == 0 or step == 59:
            print(f"  step {step:2d}  loss {loss.item():.4f}  fm {parts['fm']:.4f}  topo {parts['topo']:.4f}")

    # sample & score against ground truth
    field = fm.sample(img_val, field_ch, steps=40, method="heun")[0].cpu().numpy()
    pred = sdf_to_mask(field)
    def dice(a, b, c):
        A, B = (a == c), (b == c)
        return 2 * (A & B).sum() / (A.sum() + B.sum() + 1e-8)
    dL, dR = dice(pred, m_val, 1), dice(pred, m_val, 2)
    print(f"[selftest] loss {losses[0]:.3f} -> {losses[-1]:.3f}  (should decrease)")
    print(f"[selftest] sampled Dice  L={dL:.3f}  R={dR:.3f}  (should be >0 and rising with training)")
    ok = losses[-1] < losses[0] and (dL + dR) > 0
    print("[selftest]", "PASS ✓" if ok else "CHECK — loss/dice not as expected")
    return ok


# =============================================================================
# 7. CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True); tr.add_argument("--patch", type=int, default=96)
    tr.add_argument("--epochs", type=int, default=500); tr.add_argument("--bs", type=int, default=2)
    tr.add_argument("--target", choices=["sdf", "onehot"], default="sdf")
    tr.add_argument("--out", default="runs/fm_iac")
    pr = sub.add_parser("predict")
    pr.add_argument("--ckpt", required=True); pr.add_argument("--image", required=True)
    pr.add_argument("--out", default="pred.nii.gz"); pr.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()

    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if a.cmd == "train":
        from fm_train import train_on_dataset      # heavy path kept separate
        train_on_dataset(a)
    elif a.cmd == "predict":
        from fm_train import predict_volume
        predict_volume(a)


if __name__ == "__main__":
    main()
