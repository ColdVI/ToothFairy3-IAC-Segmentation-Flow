"""Train the OOF-coupled SDF bridge.

Split contract (stacking hygiene):
  * eval fold (default 0) is never touched: no training, no checkpoint selection.
  * `--holdout_n` cases from the remaining folds are a bridge-val set (loss curves only).
  * every training case's x0 comes from the fold model that did NOT see it (OOF cache).
Checkpoint used for the final table: the LAST one (preregistered), not the best val.

  python -m iacb.train --cache /content/cache --out /content/run1 --iters 12000
  python -m iacb.train --cache /content/cache --out /content/bench --bench 30   # time + VRAM
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .bridge import CorrelatedNoise, StateUNet, bridge_loss, interpolant, pad_multiple, sdf_cut_ball


class Case:
    def __init__(self, cache: Path, case: str):
        self.meta = json.loads((cache / f"{case}_meta.json").read_text())
        self.img = np.load(cache / f"{case}_img.npy", mmap_mode="r")
        self.x0 = np.load(cache / f"{case}_x0.npy", mmap_mode="r")
        self.x1 = np.load(cache / f"{case}_x1.npy", mmap_mode="r")
        idx = np.load(cache / f"{case}_idx.npz")
        self.fg, self.dis = idx["fg"], idx["dis"]
        self.shape = self.img.shape


def sample_patch(c: Case, patch, p_dis, p_fg, rng):
    u = rng.random()
    if u < p_dis and len(c.dis):
        ctr = c.dis[rng.integers(len(c.dis))]
    elif u < p_dis + p_fg and len(c.fg):
        ctr = c.fg[rng.integers(len(c.fg))]
    else:
        ctr = np.array([rng.integers(s) for s in c.shape])
    sl, pad = [], []
    for k in range(3):
        lo = int(ctr[k]) - patch[k] // 2 + int(rng.integers(-patch[k] // 4, patch[k] // 4 + 1))
        lo = min(max(lo, -patch[k] // 4), c.shape[k] - 1)
        a, b = max(lo, 0), min(lo + patch[k], c.shape[k])
        sl.append(slice(a, b)); pad.append((a - lo, lo + patch[k] - b))
    sl = tuple(sl)
    C = c.meta["clip_mm"]
    img = np.pad(np.asarray(c.img[sl], np.float32), pad)
    x0 = np.pad(np.asarray(c.x0[(slice(None),) + sl], np.float32) / C, [(0, 0)] + pad, constant_values=1.0)
    x1 = np.pad(np.asarray(c.x1[(slice(None),) + sl], np.float32) / C, [(0, 0)] + pad, constant_values=1.0)
    return img[None], x0, x1


class PatchStream(torch.utils.data.IterableDataset):
    def __init__(self, cache, cases, patch, p_dis, p_fg, intensity_aug, seed):
        self.cache, self.cases, self.patch = Path(cache), cases, patch
        self.p_dis, self.p_fg, self.ia, self.seed = p_dis, p_fg, intensity_aug, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 7919 + int(time.time()))
        opened = {}
        while True:
            name = self.cases[rng.integers(len(self.cases))]
            if name not in opened:
                if len(opened) > 32:
                    opened.pop(next(iter(opened)))
                opened[name] = Case(self.cache, name)
            img, x0, x1 = sample_patch(opened[name], self.patch, self.p_dis, self.p_fg, rng)
            if self.ia:
                img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
            yield torch.from_numpy(img), torch.from_numpy(x0), torch.from_numpy(x1), \
                float(opened[name].meta["spacing"][0])


def splits(cache: Path, eval_fold: int, holdout_n: int, seed: int, train_all: bool = False):
    metas = [json.loads(p.read_text()) for p in sorted(cache.glob("*_meta.json"))]
    if train_all:
        pool = sorted(m["case"] for m in metas)
        ev = []
    else:
        # fold -1 denotes the 52 TF3S cases. They are unseen by every frozen nnU-Net
        # fold and are therefore always safe refiner-training cases.
        pool = sorted(m["case"] for m in metas if m["fold"] is not None and m["fold"] != eval_fold)
        ev = sorted(m["case"] for m in metas if m["fold"] == eval_fold)
    if holdout_n < 0 or holdout_n >= len(pool):
        if holdout_n != 0:
            raise ValueError(f"holdout_n={holdout_n} invalid for pool of {len(pool)} cases")
    rng = random.Random(seed); rng.shuffle(pool)
    return sorted(pool[holdout_n:]), sorted(pool[:holdout_n]), ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_fold", type=int, default=0)
    ap.add_argument("--holdout_n", type=int, default=0)
    ap.add_argument("--train_all", action="store_true",
                    help="final fit: train on all cached cases (532), with no untouched eval fold")
    ap.add_argument("--patch", default="96,128,128")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--widths", default="24,48,96,160")
    ap.add_argument("--sigma_mm", type=float, default=1.5, help="bridge diffusion scale; 0 = deterministic bridge")
    ap.add_argument("--noise_corr_vox", type=float, default=3.0)
    ap.add_argument("--t0_prob", type=float, default=0.15, help="fraction of samples at t=0 exactly")
    ap.add_argument("--src_aug_p", type=float, default=0.25)
    ap.add_argument("--p_dis", type=float, default=0.4)
    ap.add_argument("--p_fg", type=float, default=0.45)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    patch = [int(v) for v in a.patch.split(",")]
    widths = tuple(int(v) for v in a.widths.split(","))
    assert pad_multiple(patch, 2 ** (len(widths) - 1)) == patch, "patch must be divisible by 2^(levels-1)"
    tr, va, ev = splits(cache, a.eval_fold, a.holdout_n, a.seed, a.train_all)
    assert not (set(tr) & set(ev)) and not (set(va) & set(ev))
    clip = json.loads((cache / f"{tr[0]}_meta.json").read_text())["clip_mm"]
    (out / "config.json").write_text(json.dumps(dict(vars(a), clip_mm=clip, n_train=len(tr), n_val=len(va),
                                                     n_eval=len(ev), val_cases=va), indent=2))
    model = StateUNet(widths).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(a.iters, 1), pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    sigma = a.sigma_mm / clip
    noise = CorrelatedNoise(a.noise_corr_vox, dev)
    start = 0
    ck = out / "last.pt"
    if ck.exists() and not a.bench:
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start = s["it"]
        print(f"resumed at {start}")

    def loader(cases, seed, workers, aug=True):
        ds = PatchStream(cache, cases, patch, a.p_dis, a.p_fg, aug, seed)
        return iter(torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=workers,
                                                pin_memory=dev.type == "cuda", persistent_workers=workers > 0))

    it_tr = loader(tr, a.seed, a.workers)
    # ---- identity contract at initialisation (only meaningful before any update)
    if start == 0:
        img, x0, x1, sp = next(it_tr)
        with torch.no_grad():
            y = model(img.to(dev), x0.to(dev), torch.zeros(len(img), device=dev))
        err = float((y - x0.to(dev)).abs().max())
        print(f"[identity] max |f(x0,0,I) - x0| at init = {err:.3e}")
        assert err == 0.0, "zero-init identity contract violated"
    params = sum(p.numel() for p in model.parameters())
    print(f"train {len(tr)} | bridge-val {len(va)} | eval(untouched) {len(ev)} | params {params/1e6:.2f}M")
    log = open(out / "log.jsonl", "a")
    t_start = time.time()
    n_iters = a.bench if a.bench else a.iters
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for it in range(start, n_iters):
        model.train()
        img, x0, x1, sp = next(it_tr)
        img, x0, x1 = img.to(dev, non_blocking=True), x0.to(dev, non_blocking=True), x1.to(dev, non_blocking=True)
        if a.src_aug_p > 0:
            x0 = sdf_cut_ball(x0, clip, float(sp[0]), a.src_aug_p)
        B = len(img)
        t = torch.rand(B, device=dev)
        t = torch.where(torch.rand(B, device=dev) < a.t0_prob, torch.zeros_like(t), t)
        xt = interpolant(x0, x1, t, sigma, noise)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            xh = model(img, xt, t)
        loss, parts = bridge_loss(xh, x1, clip)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if it % 50 == 0 or it == n_iters - 1:
            rec = dict(it=it, loss=float(loss), gn=float(gn), lr=sched.get_last_lr()[0], sec=time.time() - t_start, **parts)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
        if a.bench:
            continue
        if (it + 1) % a.val_every == 0 and va:
            model.eval()
            it_va = loader(va, 10_000 + it, 0, aug=False)
            vl, vd = [], []
            with torch.no_grad():
                for _ in range(20):
                    img, x0, x1, sp = next(it_va)
                    img, x0, x1 = img.to(dev), x0.to(dev), x1.to(dev)
                    z = torch.zeros(len(img), device=dev)
                    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                        xh = model(img, x0, z)                    # one-shot from the prior
                    vl.append(bridge_loss(xh, x1, clip)[0].item())
                    vd.append(bridge_loss(x0, x1, clip)[0].item())   # prior's own loss, same patches
            rec = dict(it=it, val_loss_oneshot=float(np.mean(vl)), val_loss_prior=float(np.mean(vd)))
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if (it + 1) % a.save_every == 0 or it == n_iters - 1:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), it=it + 1,
                            widths=widths, clip_mm=clip, sigma_mm=a.sigma_mm, noise_corr_vox=a.noise_corr_vox,
                            patch=patch), out / "last.pt.tmp")
            (out / "last.pt.tmp").replace(out / "last.pt")
    if a.bench:
        el = time.time() - t_start
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
        print(json.dumps(dict(bench_iters=a.bench, sec_per_iter=el / a.bench, peak_gib=mem,
                              projected_hours_for_iters=el / a.bench * a.iters / 3600)))


if __name__ == "__main__":
    main()
