#!/usr/bin/env python3
"""
fm_train.py  —  real-data training / prediction for flow_matching_iac.py
========================================================================
Kept separate from the model definition so the model file stays importable
for the (dependency-light) self-test. This module needs nibabel + scipy and
is meant to run on a GPU box against Dataset801_IAC_LR.

Patch strategy
--------------
IAC voxels are <0.05% of a CBCT volume, so uniform random patches almost never
contain canal. We sample patches **centred on a random foreground voxel** with
probability `fg_prob`, else uniformly — the standard nnU-Net oversampling trick.
"""
import glob, os, re, time
import numpy as np

import torch
import torch.nn.functional as F
import nibabel as nib

from flow_matching_iac import (
    VelocityUNet3D, RectifiedFlowSeg,
    mask_to_sdf, sdf_to_mask, mask_to_onehot, onehot_to_mask,
)

CASE_RE = re.compile(r"(ToothFairy3[FPS]_\d+)")
CLASSES = [1, 2]                      # Left_IAC, Right_IAC in Dataset801


# ----------------------------------------------------------------------------- data
def list_cases(data_dir):
    img_dir = os.path.join(data_dir, "imagesTr")
    lab_dir = os.path.join(data_dir, "labelsTr")
    labs = {CASE_RE.search(f).group(1): os.path.join(lab_dir, f)
            for f in os.listdir(lab_dir) if f.endswith(".nii.gz")}
    imgs = {CASE_RE.search(f).group(1): os.path.join(img_dir, f)
            for f in os.listdir(img_dir) if f.endswith("_0000.nii.gz")}
    ids = sorted(set(imgs) & set(labs))
    return [(imgs[i], labs[i]) for i in ids]


def normalize_cbct(vol):
    """z-score on the volume (nnU-Net CT-like); robust to intensity range."""
    v = vol.astype(np.float32)
    m, s = v.mean(), v.std() + 1e-6
    return (v - m) / s


def sample_patch(vol, mask, patch, fg_prob=0.7):
    D, H, W = mask.shape
    p = patch
    fg = np.argwhere(mask > 0)
    if len(fg) and np.random.rand() < fg_prob:
        cz, cy, cx = fg[np.random.randint(len(fg))]
    else:
        cz, cy, cx = (np.random.randint(0, max(1, D)),
                      np.random.randint(0, max(1, H)),
                      np.random.randint(0, max(1, W)))
    z0 = int(np.clip(cz - p // 2, 0, max(0, D - p)))
    y0 = int(np.clip(cy - p // 2, 0, max(0, H - p)))
    x0 = int(np.clip(cx - p // 2, 0, max(0, W - p)))
    vp = vol[z0:z0+p, y0:y0+p, x0:x0+p]
    mp = mask[z0:z0+p, y0:y0+p, x0:x0+p]
    # pad if the volume is smaller than the patch on some axis
    pad = [(0, p - s) for s in vp.shape]
    if any(b for _, b in pad):
        vp = np.pad(vp, pad); mp = np.pad(mp, pad)
    return vp, mp


class PatchStream:
    """In-RAM cache of a few volumes, streamed as random patches."""
    def __init__(self, cases, patch, target, cache=16):
        self.cases = cases; self.patch = patch; self.target = target
        self.cache = min(cache, len(cases))
        self._loaded = {}

    def _get(self, idx):
        if idx not in self._loaded:
            if len(self._loaded) >= self.cache:
                self._loaded.pop(next(iter(self._loaded)))
            ip, lp = self.cases[idx]
            vol = normalize_cbct(np.asanyarray(nib.load(ip).dataobj))
            mask = np.asanyarray(nib.load(lp).dataobj).astype(np.uint8)
            self._loaded[idx] = (vol, mask)
        return self._loaded[idx]

    def batch(self, bs):
        vols, tgts = [], []
        for _ in range(bs):
            vol, mask = self._get(np.random.randint(len(self.cases)))
            vp, mp = sample_patch(vol, mask, self.patch)
            enc = mask_to_sdf(mp, CLASSES) if self.target == "sdf" else mask_to_onehot(mp, CLASSES)
            vols.append(vp[None]); tgts.append(enc)
        return (torch.tensor(np.stack(vols)).float(),
                torch.tensor(np.stack(tgts)).float())


# ----------------------------------------------------------------------------- train
def train_on_dataset(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    cases = list_cases(a.data)
    n_val = max(1, int(0.1 * len(cases)))
    val, train = cases[:n_val], cases[n_val:]
    print(f"[train] {len(train)} train / {len(val)} val cases  device={dev}  target={a.target}")

    field_ch = len(CLASSES)
    model = VelocityUNet3D(field_ch=field_ch, img_ch=1, base=32).to(dev)
    fm = RectifiedFlowSeg(model, lambda_topo=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    stream = PatchStream(train, a.patch, a.target)

    steps_per_epoch = max(1, len(train) // a.bs)
    best = 1e9
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for _ in range(steps_per_epoch):
            image, x1 = stream.batch(a.bs)
            image, x1 = image.to(dev), x1.to(dev)
            opt.zero_grad()
            loss, parts = fm.loss(x1, image)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); run += loss.item()
        sched.step()
        avg = run / steps_per_epoch
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict(), "target": a.target,
                        "classes": CLASSES, "patch": a.patch}, os.path.join(a.out, "best.pt"))
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:4d}  loss {avg:.4f}  best {best:.4f}  {time.time()-t0:.1f}s/ep", flush=True)
    print("[train] done. best ckpt:", os.path.join(a.out, "best.pt"))


# ----------------------------------------------------------------------------- predict
@torch.no_grad()
def predict_volume(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    field_ch = len(ck["classes"]); patch = ck["patch"]; target = ck["target"]
    model = VelocityUNet3D(field_ch=field_ch, img_ch=1, base=32).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    fm = RectifiedFlowSeg(model)

    nii = nib.load(a.image)
    vol = normalize_cbct(np.asanyarray(nii.dataobj))
    D, H, W = vol.shape
    # sliding-window accumulation of the sampled end-point field
    acc = np.zeros((field_ch, D, H, W), np.float32)
    cnt = np.zeros((D, H, W), np.float32)
    step = patch // 2
    zs = list(range(0, max(1, D - patch + 1), step)) + [max(0, D - patch)]
    ys = list(range(0, max(1, H - patch + 1), step)) + [max(0, H - patch)]
    xs = list(range(0, max(1, W - patch + 1), step)) + [max(0, W - patch)]
    for z in sorted(set(zs)):
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                p = vol[z:z+patch, y:y+patch, x:x+patch]
                pad = [(0, patch - s) for s in p.shape]
                if any(b for _, b in pad): p = np.pad(p, pad)
                img = torch.tensor(p[None, None]).float().to(dev)
                field = fm.sample(img, field_ch, steps=a.steps, method="heun")[0].cpu().numpy()
                dz, dy, dx = min(patch, D-z), min(patch, H-y), min(patch, W-x)
                acc[:, z:z+dz, y:y+dy, x:x+dx] += field[:, :dz, :dy, :dx]
                cnt[z:z+dz, y:y+dy, x:x+dx] += 1
    acc /= np.maximum(cnt, 1)[None]
    mask = sdf_to_mask(acc) if target == "sdf" else onehot_to_mask(acc)
    out = nib.Nifti1Image(mask.astype(np.uint8), nii.affine, nii.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, a.out)
    print(f"[predict] wrote {a.out}  L={int((mask==1).sum())}  R={int((mask==2).sum())}")
