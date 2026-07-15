#!/usr/bin/env python3
"""
compute_coarse_sdf.py — cache coarse per-side SDFs from nnU-Net OOF predictions.

Consumes the Out-Of-Fold probability maps produced by nnunet/predict_oof.py
(one npz per case with prob_left/prob_right), thresholds them into a coarse
Left/Right mask, and computes the physical SDF exactly like the GT cache. These
coarse SDFs are the residual-flow starting point x0 = SDF(nnU-Net) — so they are
built ONLY from OOF (leakage-free) predictions, never from GT.

    python data/compute_coarse_sdf.py --oof outputs/oof_probs \
        --ref-labels $nnUNet_raw/Dataset801_IAC_LR/labelsTr \
        --out outputs/coarse_sdf --prob-thresh 0.5 --clip-mm 10
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import sys
import time

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import voxel_spacing, mask_to_sdf_mm, normalize_sdf   # noqa: E402


def cache_is_valid(path):
    try:
        with np.load(path) as item:
            sdf = item["sdf"]
            return (sdf.shape[0] == 2 and sdf.ndim == 4 and
                    item["prob_left"].shape == item["prob_right"].shape == sdf.shape[1:])
    except (OSError, ValueError, KeyError, IndexError):
        return False


def sync_caches(paths, sync_dir):
    """Atomically mirror completed cache files to persistent storage."""
    if not paths or not sync_dir:
        return 0
    os.makedirs(sync_dir, exist_ok=True)
    for src in paths:
        dst = os.path.join(sync_dir, os.path.basename(src))
        tmp = dst + ".partial"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    return len(paths)


def coarse_mask_from_probs(prob_left, prob_right, thresh):
    """Threshold + argmax into a {0,1,2} coarse mask (Left=1, Right=2)."""
    fg = (np.maximum(prob_left, prob_right) >= thresh)
    side_left = prob_left >= prob_right
    out = np.zeros(prob_left.shape, dtype=np.uint8)
    out[fg & side_left] = 1
    out[fg & ~side_left] = 2
    return out


def compute_one(job):
    """Compute one coarse cache in an isolated worker process."""
    oof, ref_labels, f, out, prob_thresh, clip_mm = job
    started = time.monotonic()
    sid = f[: -len(".npz")]
    final_path = os.path.join(out, f"{sid}.npz")
    d = np.load(os.path.join(oof, f))
    pl, pr = d["prob_left"].astype(np.float32), d["prob_right"].astype(np.float32)
    ref = nib.load(os.path.join(ref_labels, f"{sid}.nii.gz"))
    sp = voxel_spacing(ref)
    coarse = coarse_mask_from_probs(pl, pr, prob_thresh)
    left = normalize_sdf(mask_to_sdf_mm(coarse == 1, sp, clip_mm), clip_mm)
    right = normalize_sdf(mask_to_sdf_mm(coarse == 2, sp, clip_mm), clip_mm)
    sdf = np.stack([left, right]).astype(np.float16)
    partial_path = final_path + ".partial.npz"
    np.savez_compressed(partial_path, sdf=sdf,
                        prob_left=pl.astype(np.float16), prob_right=pr.astype(np.float16),
                        spacing=sp.astype(np.float32))
    os.replace(partial_path, final_path)
    return sid, final_path, time.monotonic() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="dir of *.npz with prob_left/prob_right")
    ap.add_argument("--ref-labels", required=True, help="labelsTr for spacing reference")
    ap.add_argument("--out", default="outputs/coarse_sdf")
    ap.add_argument("--prob-thresh", type=float, default=0.5)
    ap.add_argument("--clip-mm", type=float, default=10.0)
    ap.add_argument("--resume", action="store_true", help="skip readable existing caches")
    ap.add_argument("--sync-dir", default=None,
                    help="persistent directory to receive completed caches during this run")
    ap.add_argument("--sync-every", type=int, default=10,
                    help="copy completed caches to --sync-dir every N cases (default: 10)")
    ap.add_argument("--workers", type=int, default=1,
                    help="independent CPU worker processes (use 2 on a 12 GB runtime)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.oof) if f.endswith(".npz"))
    t0 = time.monotonic()
    completed = 0
    pending_sync = []
    def record(sid, final_path, case_seconds, cached=False):
        nonlocal completed
        completed += 1
        pending_sync.append(final_path)
        if a.sync_dir and len(pending_sync) >= a.sync_every:
            print(f"[coarse-sdf] Drive checkpoint: +{sync_caches(pending_sync, a.sync_dir)} "
                  f"-> {a.sync_dir}", flush=True)
            pending_sync.clear()
        elapsed = time.monotonic() - t0
        rate = completed / max(elapsed, 1e-6)
        eta = (len(files) - completed) / max(rate, 1e-6)
        print(f"[coarse-sdf] {completed}/{len(files)} ({100 * completed / len(files):.1f}%) | "
              f"{sid}{' cached' if cached else ''} | case {case_seconds:.1f}s | "
              f"elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)
    jobs = []
    for f in files:
        sid = f[: -len(".npz")]
        final_path = os.path.join(a.out, f"{sid}.npz")
        if a.resume and cache_is_valid(final_path):
            record(sid, final_path, 0.0, cached=True)
        else:
            jobs.append((a.oof, a.ref_labels, f, a.out, a.prob_thresh, a.clip_mm))
    if a.workers > 1:
        print(f"[coarse-sdf] CPU parallel workers={a.workers} | new cases={len(jobs)}", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for fut in as_completed([ex.submit(compute_one, job) for job in jobs]):
                record(*fut.result())
    else:
        for job in jobs:
            record(*compute_one(job))
    if a.sync_dir and pending_sync:
        print(f"[coarse-sdf] Drive checkpoint: +{sync_caches(pending_sync, a.sync_dir)} "
              f"-> {a.sync_dir}", flush=True)
    print(f"[coarse-sdf] wrote {len(files)} caches -> {a.out}")


if __name__ == "__main__":
    main()
