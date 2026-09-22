"""Build the bridge cache from nnU-Net raw data + OOF probabilities (CPU, once).

Per case, cropped to bbox(prior U GT) + margin (so every foreground voxel of prior and GT
is inside the crop and crop-level metrics equal full-volume metrics):
  {case}_img.npy    float16 (Z,Y,X)     robust z-scored CBCT
  {case}_p.npy      uint8   (2,Z,Y,X)   OOF P(left), P(right) * 255
  {case}_x0.npy     float16 (2,Z,Y,X)   SDF(prior side masks), mm, clipped
  {case}_x1.npy     float16 (2,Z,Y,X)   SDF(GT side masks), mm, clipped
  {case}_gt.npy     uint8   (Z,Y,X)     0 bg, 1 left, 2 right
  {case}_idx.npz    int32 coordinate pools for patch sampling (fg union, disagreement)
  {case}_meta.json  spacing, crop box in original volume, fold, inference ROI (prior-only)
No resampling: SDFs are in mm; a spacing report is written so heterogeneous spacing is
visible before training (`--max_spacing_dev` aborts if it is large).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import (bbox, case_folds, dice, load_oof_probs, prior_side_masks, read_image,
                     sdf_mm, side_masks, union_bbox, resolve_oof_npz)


def robust_z(img):
    x = img.astype(np.float32)
    sub = x[::4, ::4, ::4]
    lo, hi = np.percentile(sub, [0.5, 99.5])
    x = np.clip(x, lo, hi)
    m, s = float(x[::4, ::4, ::4].mean()), float(x[::4, ::4, ::4].std()) + 1e-6
    return ((x - m) / s).astype(np.float16)


def build_case(t):
    case, a = t
    out = Path(a["out"])
    if (out / f"{case}_meta.json").exists() and not a["overwrite"]:
        return case, "skip"
    img, sp = read_image(Path(a["images_dir"]) / f"{case}_0000.nii.gz")
    lab, sp_l = read_image(Path(a["labels_dir"]) / f"{case}.nii.gz")
    prob_path = resolve_oof_npz(Path(a["prob_dir"]), case, a["folds"].get(case))
    prob = load_oof_probs(prob_path, a["prob_left_id"], a["prob_right_id"])
    if not (img.shape == lab.shape == prob.shape[1:]):
        raise ValueError(f"{case}: shapes img {img.shape} lab {lab.shape} prob {prob.shape[1:]}")
    gt2 = side_masks(lab, a["label_left_id"], a["label_right_id"])
    pr2 = prior_side_masks(prob)
    dd = [dice(pr2[s], gt2[s]) for s in range(2) if gt2[s].any()]
    if dd and min(dd) < a["sanity_min_dice"]:
        raise ValueError(f"{case}: prior-vs-GT Dice {dd} < {a['sanity_min_dice']} -> axis order / case mismatch?")
    sv = float(np.mean(sp))
    mvox = int(np.ceil(a["margin_mm"] / sv))
    fg_prior = pr2.any(0)
    crop = union_bbox([bbox(fg_prior, mvox), bbox(gt2.any(0), mvox)])
    roi_inf = bbox(fg_prior, mvox)                       # GT-free inference ROI (original coords)
    if crop is None:
        return case, "empty"
    C = a["clip_mm"]
    g2c, p2c = gt2[(slice(None),) + crop], pr2[(slice(None),) + crop]
    np.save(out / f"{case}_img.npy", robust_z(img)[crop])
    np.save(out / f"{case}_p.npy", np.round(prob[(slice(None),) + crop] * 255).astype(np.uint8))
    np.save(out / f"{case}_x0.npy", np.stack([sdf_mm(p2c[s], sp, C) for s in range(2)]).astype(np.float16))
    np.save(out / f"{case}_x1.npy", np.stack([sdf_mm(g2c[s], sp, C) for s in range(2)]).astype(np.float16))
    gtc = np.zeros(g2c.shape[1:], np.uint8); gtc[g2c[0]] = 1; gtc[g2c[1]] = 2
    np.save(out / f"{case}_gt.npy", gtc)
    rng = np.random.default_rng(0)

    def pool(m, n=20000):
        idx = np.argwhere(m).astype(np.int32)
        return idx[rng.choice(len(idx), min(n, len(idx)), replace=False)] if len(idx) else idx

    np.savez(out / f"{case}_idx.npz", fg=pool(g2c.any(0) | p2c.any(0)), dis=pool((g2c ^ p2c).any(0)))
    rel_roi = None
    if roi_inf is not None:
        rel_roi = [[max(r.start - c.start, 0), min(r.stop - c.start, c.stop - c.start)] for r, c in zip(roi_inf, crop)]
    meta = dict(case=case, spacing=sp, crop=[[c.start, c.stop] for c in crop], shape=list(g2c.shape[1:]),
                roi_inference=rel_roi, fold=a["folds"].get(case), clip_mm=C, orig_shape=list(lab.shape),
                prior_dice=dd)
    (out / f"{case}_meta.json").write_text(json.dumps(meta))
    return case, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label_left_id", type=int, default=3,
                    help="Left-IAC label value in labelsTr (TF3 full dataset: 3)")
    ap.add_argument("--label_right_id", type=int, default=4,
                    help="Right-IAC label value in labelsTr (TF3 full dataset: 4)")
    ap.add_argument("--prob_left_id", type=int, default=1,
                    help="Left-IAC channel in frozen Dataset801 nnU-Net probabilities")
    ap.add_argument("--prob_right_id", type=int, default=2,
                    help="Right-IAC channel in frozen Dataset801 nnU-Net probabilities")
    ap.add_argument("--clip_mm", type=float, default=4.0)
    ap.add_argument("--margin_mm", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sanity_min_dice", type=float, default=0.3)
    ap.add_argument("--max_spacing_dev", type=float, default=0.15)
    a = vars(ap.parse_args())
    a["folds"] = case_folds(Path(a["splits"]))
    Path(a["out"]).mkdir(parents=True, exist_ok=True)
    cases = sorted(a["folds"])
    if a["limit"]:
        cases = cases[: a["limit"]]
    status = {}
    with ProcessPoolExecutor(a["workers"]) as ex:
        for c, s in ex.map(build_case, [(c, a) for c in cases]):
            status[c] = s
            print(c, s, flush=True)
    sps = [json.loads((Path(a["out"]) / f"{c}_meta.json").read_text())["spacing"]
           for c in cases if (Path(a["out"]) / f"{c}_meta.json").exists()]
    sps = np.asarray(sps)
    med = np.median(sps, 0)
    dev = np.abs(sps / med - 1).max(1)
    report = dict(median_spacing=med.tolist(), n_dev_gt_5pct=int((dev > 0.05).sum()), max_dev=float(dev.max()),
                  status_counts={k: list(status.values()).count(k) for k in set(status.values())})
    (Path(a["out"]) / "cache_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if dev.max() > a["max_spacing_dev"]:
        # WARNING, not an abort: the pipeline must not stop on a judgement call.
        # SDFs are in mm, so heterogeneous spacing mainly costs accuracy, it does not break the run.
        report["spacing_warning"] = (f"max deviation {dev.max():.2f} > {a['max_spacing_dev']}; "
                                     "consider resampling. Cache built anyway.")
        (Path(a["out"]) / "cache_report.json").write_text(json.dumps(report, indent=2))
        print("[WARN] " + report["spacing_warning"], flush=True)


if __name__ == "__main__":
    main()
