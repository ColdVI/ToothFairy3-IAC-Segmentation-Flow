#!/usr/bin/env python3
"""
compute_gt_sdf.py — cache ground-truth per-side physical SDFs.

For each converted label (Dataset801_IAC_LR/labelsTr/*.nii.gz) compute the
Left and Right signed distance fields in MILLIMETRES (anisotropic, using voxel
spacing), truncate at +-clip_mm and store normalised to [-1,1]. These are the
flow-matching x1 targets; caching once avoids recomputing EDTs every epoch.

Stored per case as compressed npz: sdf (2,D,H,W) float16, plus spacing.
    python data/compute_gt_sdf.py --labels $nnUNet_raw/Dataset801_IAC_LR/labelsTr \
        --out outputs/gt_sdf --clip-mm 10
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
            return item["sdf"].shape[0] == 2 and item["sdf"].ndim == 4
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


def compute_one(job):
    """Compute one GT cache in an isolated worker process."""
    labels, f, out, clip_mm = job
    started = time.monotonic()
    sid = f[: -len(".nii.gz")]
    final_path = os.path.join(out, f"{sid}.npz")
    img = nib.load(os.path.join(labels, f))
    lab = np.asanyarray(img.dataobj)
    sp = voxel_spacing(img)
    left = normalize_sdf(mask_to_sdf_mm(lab == 1, sp, clip_mm), clip_mm)
    right = normalize_sdf(mask_to_sdf_mm(lab == 2, sp, clip_mm), clip_mm)
    sdf = np.stack([left, right]).astype(np.float16)
    partial_path = final_path + ".partial.npz"
    np.savez_compressed(partial_path, sdf=sdf,
                        spacing=sp.astype(np.float32), clip_mm=np.float32(clip_mm))
    os.replace(partial_path, final_path)
    return sid, final_path, time.monotonic() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labelsTr of Dataset801_IAC_LR")
    ap.add_argument("--out", default="outputs/gt_sdf")
    ap.add_argument("--clip-mm", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip readable existing caches")
    ap.add_argument("--sync-dir", default=None,
                    help="persistent directory to receive completed caches during this run")
    ap.add_argument("--sync-every", type=int, default=10,
                    help="copy completed caches to --sync-dir every N cases (default: 10)")
    ap.add_argument("--workers", type=int, default=1,
                    help="independent CPU worker processes (use 2 on a 12 GB runtime)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.labels) if f.endswith(".nii.gz"))
    if a.limit:
        files = files[: a.limit]
    t0 = time.monotonic()
    completed = 0
    pending_sync = []

    def record(sid, final_path, case_seconds, cached=False):
        nonlocal completed
        completed += 1
        pending_sync.append(final_path)
        if a.sync_dir and len(pending_sync) >= a.sync_every:
            print(f"[gt-sdf] Drive checkpoint: +{sync_caches(pending_sync, a.sync_dir)} "
                  f"-> {a.sync_dir}", flush=True)
            pending_sync.clear()
        elapsed = time.monotonic() - t0
        rate = completed / max(elapsed, 1e-6)
        eta = (len(files) - completed) / max(rate, 1e-6)
        print(f"[gt-sdf] {completed}/{len(files)} ({100 * completed / len(files):.1f}%) | "
              f"{sid}{' cached' if cached else ''} | case {case_seconds:.1f}s | "
              f"elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)

    jobs = []
    for f in files:
        sid = f[: -len(".nii.gz")]
        final_path = os.path.join(a.out, f"{sid}.npz")
        if a.resume and cache_is_valid(final_path):
            record(sid, final_path, 0.0, cached=True)
        else:
            jobs.append((a.labels, f, a.out, a.clip_mm))
    if a.workers > 1:
        print(f"[gt-sdf] CPU parallel workers={a.workers} | new cases={len(jobs)}", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for fut in as_completed([ex.submit(compute_one, job) for job in jobs]):
                record(*fut.result())
    else:
        for job in jobs:
            record(*compute_one(job))
    if a.sync_dir and pending_sync:
        print(f"[gt-sdf] Drive checkpoint: +{sync_caches(pending_sync, a.sync_dir)} "
              f"-> {a.sync_dir}", flush=True)
    print(f"[gt-sdf] wrote {len(files)} caches -> {a.out}")


if __name__ == "__main__":
    main()
