#!/usr/bin/env python3
"""
datasets.py — leakage-free patch dataset for the residual flow.

Each training sample provides:
    cond : (8,P,P,P)  conditioning (CBCT, OOF probs, coarse SDF, physical xyz)
    x0   : (2,P,P,P)  coarse SDF start state (from OOF predictions only)
    x1   : (2,P,P,P)  GT SDF target
Patches are sampled foreground-centred (IAC is <0.05% of the volume). Every
coarse input derives from OUT-OF-FOLD nnU-Net predictions, so a case the flow
trains on was never seen by the nnU-Net that produced its prior.
"""
import os
import sys

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from io_utils import physical_coord_grid, normalize_coords            # noqa: E402
from conditioning import build_conditioning                           # noqa: E402


class IACFlowDataset(Dataset):
    def __init__(self, case_ids, images_dir, gt_sdf_dir, coarse_sdf_dir,
                 patch=96, fg_prob=0.8, cache=8):
        self.ids = list(case_ids)
        self.images_dir = images_dir
        self.gt_sdf_dir = gt_sdf_dir
        self.coarse_sdf_dir = coarse_sdf_dir
        self.patch = patch
        self.fg_prob = fg_prob
        self.cache = cache
        self._mem = {}

    def __len__(self):
        return len(self.ids)

    def _load(self, sid):
        if sid in self._mem:
            return self._mem[sid]
        img = nib.load(os.path.join(self.images_dir, f"{sid}_0000.nii.gz"))
        cbct = np.asanyarray(img.dataobj).astype(np.float32)
        coords = normalize_coords(physical_coord_grid(cbct.shape, img.affine))
        gt = np.load(os.path.join(self.gt_sdf_dir, f"{sid}.npz"))["sdf"].astype(np.float32)
        co = np.load(os.path.join(self.coarse_sdf_dir, f"{sid}.npz"))
        coarse_sdf = co["sdf"].astype(np.float32)
        prob_l = co["prob_left"].astype(np.float32)
        prob_r = co["prob_right"].astype(np.float32)
        cond = build_conditioning(cbct, prob_l, prob_r, coarse_sdf[0], coarse_sdf[1], coords)
        item = (cond, coarse_sdf, gt)
        if len(self._mem) >= self.cache:
            self._mem.pop(next(iter(self._mem)))
        self._mem[sid] = item
        return item

    def _sample_patch(self, cond, coarse, gt):
        p = self.patch
        _, D, H, W = cond.shape
        fg = np.argwhere(np.minimum(gt[0], gt[1]) < 0)     # inside either canal
        if len(fg) and np.random.rand() < self.fg_prob:
            cz, cy, cx = fg[np.random.randint(len(fg))]
        else:
            cz, cy, cx = (np.random.randint(D), np.random.randint(H), np.random.randint(W))
        z0 = int(np.clip(cz - p // 2, 0, max(0, D - p)))
        y0 = int(np.clip(cy - p // 2, 0, max(0, H - p)))
        x0i = int(np.clip(cx - p // 2, 0, max(0, W - p)))
        sl = (slice(z0, z0 + p), slice(y0, y0 + p), slice(x0i, x0i + p))
        c = cond[:, sl[0], sl[1], sl[2]]
        x0 = coarse[:, sl[0], sl[1], sl[2]]
        x1 = gt[:, sl[0], sl[1], sl[2]]
        pad = ((0, 0), (0, p - c.shape[1]), (0, p - c.shape[2]), (0, p - c.shape[3]))
        if any(b for grp in pad for b in grp[1:]):
            c = np.pad(c, pad); x0 = np.pad(x0, pad); x1 = np.pad(x1, pad)
        return c, x0, x1

    def __getitem__(self, i):
        cond, coarse, gt = self._load(self.ids[i % len(self.ids)])
        c, x0, x1 = self._sample_patch(cond, coarse, gt)
        return (torch.from_numpy(c).float(),
                torch.from_numpy(x0).float(),
                torch.from_numpy(x1).float())
