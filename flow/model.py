#!/usr/bin/env python3
"""
model.py — 3D velocity U-Net for the residual IAC flow.

The network predicts a velocity field over the 2-channel (Left, Right) SDF
state. Its input is the flow state x_t concatenated with an 8-channel
conditioning tensor (CBCT, L/R probabilities, coarse L/R SDF, physical x/y/z):

    in_ch  = FLOW_STATE_CH (2) + COND_CH (8) = 10
    out_ch = FLOW_STATE_CH (2)

Time t is injected with a sinusoidal embedding and FiLM-style bias at every
residual block.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FLOW_STATE_CH = 2      # Left SDF, Right SDF
COND_CH = 8            # CBCT, prob_L, prob_R, coarse_SDF_L, coarse_SDF_R, x, y, z


class SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):                      # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(1, half - 1))
        ang = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, cin), cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(min(8, cout), cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ResidualVelocityUNet3D(nn.Module):
    def __init__(self, cond_ch=COND_CH, state_ch=FLOW_STATE_CH, base=32, tdim=128):
        super().__init__()
        self.state_ch = state_ch
        self.cond_ch = cond_ch
        self.tmap = nn.Sequential(SinusoidalTime(tdim), nn.Linear(tdim, tdim),
                                  nn.SiLU(), nn.Linear(tdim, tdim))
        cin = state_ch + cond_ch
        self.stem = nn.Conv3d(cin, base, 3, padding=1)
        self.e1 = ResBlock3D(base, base, tdim)
        self.e2 = ResBlock3D(base, base * 2, tdim)
        self.e3 = ResBlock3D(base * 2, base * 4, tdim)
        self.down = nn.AvgPool3d(2)
        self.mid = ResBlock3D(base * 4, base * 4, tdim)
        self.d2 = ResBlock3D(base * 4 + base * 2, base * 2, tdim)
        self.d1 = ResBlock3D(base * 2 + base, base, tdim)
        self.head = nn.Sequential(nn.GroupNorm(min(8, base), base), nn.SiLU(),
                                  nn.Conv3d(base, state_ch, 1))

    def forward(self, xt, t, cond):
        """xt: (B,state,D,H,W); t: (B,); cond: (B,cond_ch,D,H,W)."""
        temb = self.tmap(t)
        h = self.stem(torch.cat([xt, cond], dim=1))
        s1 = self.e1(h, temb)
        s2 = self.e2(self.down(s1), temb)
        s3 = self.e3(self.down(s2), temb)
        m = self.mid(s3, temb)
        u = F.interpolate(m, size=s2.shape[2:], mode="trilinear", align_corners=False)
        u = self.d2(torch.cat([u, s2], 1), temb)
        u = F.interpolate(u, size=s1.shape[2:], mode="trilinear", align_corners=False)
        u = self.d1(torch.cat([u, s1], 1), temb)
        return self.head(u)
