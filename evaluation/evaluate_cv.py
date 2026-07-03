#!/usr/bin/env python3
"""
evaluate_cv.py — aggregate per-case metrics over a prediction folder vs GT.

Computes, per side then bilateral-averaged: Dice, HD95, clDice, NSD, component
count, Betti-0 error, centerline gap / false-branch length, L/R swap and empty
rates. Reports fold mean +- std and writes a per-case CSV. Paired bootstrap CIs
for model-vs-model comparison live in compare_bootstrap().

    python evaluation/evaluate_cv.py --pred outputs/preds --gt .../labelsTr --out outputs/cv_metrics.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluation.metrics import dice, hd95, cldice, nsd                          # noqa: E402
from evaluation.topology_metrics import (betti0_error, centerline_gap_length,   # noqa: E402
                                         false_branch_length, lr_swap_rate,
                                         empty_prediction, n_components)


def voxel_spacing(img):
    return np.asarray(img.header.get_zooms()[:3], dtype=np.float64)


def evaluate_case(pred, gt, spacing):
    row = {}
    for side, name in ((1, "L"), (2, "R")):
        p, g = pred == side, gt == side
        row[f"dice_{name}"] = dice(p, g)
        row[f"hd95_{name}"] = hd95(p, g, spacing)
        row[f"cldice_{name}"] = cldice(p, g)
        row[f"nsd_{name}"] = nsd(p, g, spacing)
        row[f"betti0_{name}"] = betti0_error(p)
        row[f"components_{name}"] = n_components(p)
        row[f"gap_mm_{name}"] = centerline_gap_length(p, g, spacing)
        row[f"falsebranch_mm_{name}"] = false_branch_length(p, g, spacing)
        row[f"empty_{name}"] = empty_prediction(p)
    row["swap_rate"] = lr_swap_rate(pred, gt)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="outputs/cv_metrics.csv")
    a = ap.parse_args()

    rows = []
    for pf in sorted(f for f in os.listdir(a.pred) if f.endswith(".nii.gz")):
        gpath = os.path.join(a.gt, pf)
        if not os.path.isfile(gpath):
            continue
        pimg = nib.load(os.path.join(a.pred, pf))
        gimg = nib.load(gpath)
        pred = np.asanyarray(pimg.dataobj)
        gt = np.asanyarray(gimg.dataobj)
        r = evaluate_case(pred, gt, voxel_spacing(gimg))
        r["case_id"] = pf[:-len(".nii.gz")]
        rows.append(r)

    if not rows:
        print("[eval] no matched cases."); return
    keys = ["case_id"] + [k for k in rows[0] if k != "case_id"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    def agg(metric):
        vals = []
        for r in rows:
            vals += [r.get(f"{metric}_L"), r.get(f"{metric}_R")]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), 0)

    print(f"[eval] {len(rows)} cases -> {a.out}")
    for m in ("dice", "cldice", "hd95", "nsd", "betti0", "gap_mm", "falsebranch_mm"):
        mu, sd = agg(m)
        print(f"  {m:14s} {mu:.4f} +- {sd:.4f}")
    print(f"  swap_rate      {np.mean([r['swap_rate'] for r in rows]):.4f}")


def compare_bootstrap(scores_a, scores_b, n_boot=10000, seed=0):
    """Paired bootstrap 95% CI of mean(a-b) for case-wise scores of two models."""
    rng = np.random.RandomState(seed)
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    diff = a - b
    boots = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci95": (float(lo), float(hi))}


if __name__ == "__main__":
    main()
