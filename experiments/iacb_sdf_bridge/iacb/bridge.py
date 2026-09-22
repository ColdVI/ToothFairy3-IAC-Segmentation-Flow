"""OOF-coupled stochastic SDF bridge (torch).

State   x in R^{2 x Z x Y x X}: left/right SDF in units of clip_mm (so in [-1, 1]).
Coupling (x0, x1) = (SDF of the leakage-free OOF prior, SDF of GT), one pair per case.
Interpolant (Brownian bridge with spatially correlated innovations):
    x_t = (1 - t) x0 + t x1 + sigma * sqrt(t (1 - t)) * eps,   eps = normalised G_l * white
Network  f(x_t, t, image) -> x1_hat = x_t + Delta, Delta's last conv zero-initialised,
         so at initialisation f(x0, 0, I) == x0 exactly (identity contract).
Conditioning deliberately contains NO prior-derived channel (probabilities, logits, SDF):
with such a channel x0 is recoverable and x1 ~ (x_t - (1-t) x0)/t leaks through the
interpolant (the shortcut degeneracy seen before with cond_include_coarse_sdf).
Sampler  (exact Brownian-bridge posterior with x1 replaced by x1_hat, as in I2SB):
    x_s = x_t + (s-t)/(1-t) (x1_hat - x_t) + temp * sigma * sqrt((s-t)(1-s)/(1-t)) * eps
temp = 0 gives the deterministic path; NFE = 1 is the one-shot refiner and cannot be
stochastic (the last step has zero variance).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------- noise
class CorrelatedNoise:
    """White noise blurred by a separable Gaussian (std = corr_vox) and rescaled to unit
    pointwise variance. corr_vox = 0 gives white noise."""

    def __init__(self, corr_vox: float, device):
        self.corr = float(corr_vox)
        if self.corr > 0:
            r = int(math.ceil(3 * self.corr))
            x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
            k = torch.exp(-0.5 * (x / self.corr) ** 2)
            k = k / k.sum()
            self.k = k
            self.scale = 1.0 / float(torch.sqrt((k ** 2).sum()) ** 3)   # var of blurred white = (sum k^2)^3

    def __call__(self, shape, device, dtype=torch.float32):
        w = torch.randn(shape, device=device, dtype=torch.float32)
        if self.corr <= 0:
            return w.to(dtype)
        B, C = shape[:2]
        k = self.k
        r = (len(k) - 1) // 2
        x = w.reshape(B * C, 1, *shape[2:])
        x = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1, 1))
        x = F.conv3d(F.pad(x, (0, 0, r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1, 1))
        x = F.conv3d(F.pad(x, (r, r, 0, 0, 0, 0), mode="replicate"), k.view(1, 1, 1, 1, -1))
        return (x * self.scale).reshape(shape).to(dtype)


def interpolant(x0, x1, t, sigma, noise: CorrelatedNoise):
    tt = t.view(-1, 1, 1, 1, 1)
    eps = noise(x0.shape, x0.device)
    return (1 - tt) * x0 + tt * x1 + sigma * torch.sqrt(tt * (1 - tt)) * eps


# ------------------------------------------------------------------------- model
def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class FiLMRes(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n1 = _gn(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.n2 = _gn(cout)
        self.film = nn.Linear(tdim, 2 * cout)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = F.silu(self.n1(self.c1(x)))
        g, b = self.film(temb).chunk(2, dim=1)
        h = h * (1 + g.view(*g.shape, 1, 1, 1)) + b.view(*b.shape, 1, 1, 1)
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class StateUNet(nn.Module):
    """Input: [image (1), x_t (2)], time t. Output: x1_hat = x_t + Delta (2)."""

    def __init__(self, widths=(24, 48, 96, 160), tdim=64):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.stem = nn.Conv3d(3, widths[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, w in enumerate(widths):
            self.enc.append(FiLMRes(widths[max(i - 1, 0)] if i else widths[0], w, tdim))
            if i < len(widths) - 1:
                self.down.append(nn.Conv3d(w, w, 3, stride=2, padding=1))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(widths[i], widths[i - 1], 2, stride=2))
            self.dec.append(FiLMRes(2 * widths[i - 1], widths[i - 1], tdim))
        self.head = nn.Conv3d(widths[0], 2, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.levels = len(widths)

    def temb(self, t):
        half = self.tdim // 2
        f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * 1000.0 * f[None]
        return self.tmlp(torch.cat([a.sin(), a.cos()], 1))

    def forward(self, img, xt, t):
        te = self.temb(t)
        h = self.stem(torch.cat([img, xt], 1))
        skips = []
        for i, blk in enumerate(self.enc):
            h = blk(h, te)
            if i < self.levels - 1:
                skips.append(h)
                h = self.down[i](h)
        for up, blk in zip(self.up, self.dec):
            h = up(h)
            s = skips.pop()
            h = blk(torch.cat([h, s], 1), te)
        return xt + self.head(h)


def pad_multiple(shape, m):
    return [int(math.ceil(s / m) * m) for s in shape]


# -------------------------------------------------------------------------- loss
def bridge_loss(x1_hat, x1, clip_mm, tau_mm=0.3, band_mm=1.0, w_occ=0.5):
    """Band-weighted L1 on SDF (mm) + occupancy BCE + soft Dice. No clDice term: soft-clDice
    rewards thicker tubes (measured earlier: r 1.5 -> 2.1 mm lowered it 0.0448 -> 0.0297)."""
    xh, xg = x1_hat.float() * clip_mm, x1.float() * clip_mm
    w = 1.0 + 4.0 * torch.exp(-xg.abs() / band_mm)
    l_sdf = (w * (xh - xg).abs()).sum() / w.sum()
    occ_logit = -xh / tau_mm
    occ_gt = (xg < 0).float()
    l_bce = F.binary_cross_entropy_with_logits(occ_logit, occ_gt)
    p = torch.sigmoid(occ_logit)
    dims = (0, 2, 3, 4)
    inter = (p * occ_gt).sum(dims)
    l_dice = 1 - ((2 * inter + 1) / (p.sum(dims) + occ_gt.sum(dims) + 1)).mean()
    return l_sdf + w_occ * (l_bce + l_dice), dict(l_sdf=l_sdf.detach().item(), l_bce=l_bce.detach().item(), l_dice=l_dice.detach().item())


# ----------------------------------------------------------------- source augmentation
def sdf_cut_ball(x0n, clip_mm, spacing_mm, p, radius_mm=(1.0, 3.0), grow_prob=0.3):
    """Leakage-free coupling augmentation on normalised SDF: carve a ball out of the prior
    (SDF(A\\B) = max(SDF_A, -SDF_B)) or add a spurious ball (min(SDF_A, SDF_B)).
    Applied to x0 only, uses no GT. Marked as augmentation in every table."""
    B, C, Z, Y, X = x0n.shape
    out = x0n.clone()
    dev = x0n.device
    for b in range(B):
        if torch.rand(()) > p:
            continue
        c = int(torch.randint(0, C, ()))
        inside = (x0n[b, c] < 0).nonzero()
        grow = torch.rand(()) < grow_prob or len(inside) == 0
        if grow:
            ctr = torch.stack([torch.randint(0, s, ()) for s in (Z, Y, X)]).to(dev)
        else:
            ctr = inside[torch.randint(0, len(inside), ())]
        r = float(torch.empty(()).uniform_(*radius_mm))
        zz, yy, xx = torch.meshgrid(*(torch.arange(s, device=dev) for s in (Z, Y, X)), indexing="ij")
        d = torch.sqrt(((zz - ctr[0]) * spacing_mm) ** 2 + ((yy - ctr[1]) * spacing_mm) ** 2 +
                       ((xx - ctr[2]) * spacing_mm) ** 2) - r
        ball = (d / clip_mm).clamp(-1, 1)
        out[b, c] = torch.minimum(out[b, c], ball) if grow else torch.maximum(out[b, c], -ball)
    return out


# ---------------------------------------------------------------- sliding window
def gaussian_weight(patch, device):
    ws = []
    for s in patch:
        x = torch.arange(s, device=device, dtype=torch.float32) - (s - 1) / 2
        ws.append(torch.exp(-0.5 * (x / (s / 8)) ** 2))
    w = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    return (w / w.max()).clamp_min(1e-3)


def window_starts(size, patch, overlap):
    if size <= patch:
        return [0]
    step = max(1, int(patch * (1 - overlap)))
    st = list(range(0, size - patch + 1, step))
    if st[-1] != size - patch:
        st.append(size - patch)
    return st


@torch.no_grad()
def predict_x1(model, img, xt, t_scalar, patch, overlap=0.5, amp=True):
    """img (1,1,Z,Y,X), xt (K,2,Z,Y,X) on device -> x1_hat (K,2,Z,Y,X) float32, blended."""
    K, _, Z, Y, X = xt.shape
    pz, py, px = [min(p, s) for p, s in zip(patch, (Z, Y, X))]
    mult = 2 ** (model.levels - 1)
    need = pad_multiple((pz, py, px), mult)
    w = gaussian_weight(need, xt.device)
    acc = torch.zeros_like(xt, dtype=torch.float32)
    norm = torch.zeros((1, 1, Z, Y, X), device=xt.device)
    t = torch.full((K,), float(t_scalar), device=xt.device)
    for z in window_starts(Z, pz, overlap):
        for y in window_starts(Y, py, overlap):
            for x in window_starts(X, px, overlap):
                sl = (slice(z, z + pz), slice(y, y + py), slice(x, x + px))
                im = img[(slice(None), slice(None)) + sl].expand(K, -1, -1, -1, -1)
                xs = xt[(slice(None), slice(None)) + sl]
                padw = [0, need[2] - px, 0, need[1] - py, 0, need[0] - pz]
                im_p = F.pad(im, padw)
                xs_p = F.pad(xs, padw, value=1.0)
                with torch.autocast(device_type=xt.device.type, enabled=amp and xt.device.type == "cuda"):
                    out = model(im_p, xs_p, t).float()
                out = out[..., :pz, :py, :px]
                ww = w[:pz, :py, :px]
                acc[(slice(None), slice(None)) + sl] += out * ww
                norm[(slice(None), slice(None)) + sl] += ww
    return acc / norm


@torch.no_grad()
def sample_bridge(model, img, x0, nfe, sigma, temp, noise: CorrelatedNoise, patch, overlap=0.5, amp=True):
    """x0 (K,2,Z,Y,X). Returns final state x1 (K,2,Z,Y,X)."""
    x = x0.clone().float()
    grid = torch.linspace(0, 1, nfe + 1).tolist()
    for t, s in zip(grid[:-1], grid[1:]):
        xh = predict_x1(model, img, x, t, patch, overlap, amp).clamp(-1, 1)
        if s >= 1.0:
            x = xh
            break
        x = x + (s - t) / (1 - t) * (xh - x)
        if temp > 0 and sigma > 0:
            x = x + temp * sigma * math.sqrt((s - t) * (1 - s) / (1 - t)) * noise(x.shape, x.device)
    return x


def decode_labels(xn):
    """(K,2,Z,Y,X) normalised SDF -> (K,Z,Y,X) uint8 labels {0,1,2} (min-SDF side if < 0)."""
    m, a = xn.min(1)
    return torch.where(m < 0, a + 1, torch.zeros_like(a)).to(torch.uint8)
