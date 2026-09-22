"""Evaluate external predictions (e.g. the supervisor's trainer) with the SAME metric code and on
the SAME cases as our pipeline, then optionally pair them with an infer.py per_side.csv.

Why: "teacher 0.9019 (107 binary cases) vs ours 0.9050 (97 three-class cases)" is not a
comparison. This script makes one.

Predictions: a folder of {case}.nii.gz in the label geometry.
  * three-class preds (values in {0, left_id, right_id}) are used as-is.
  * binary preds (values in {0,1}) are split into sides by --side_assign:
      pred_midline   : GT-free voxel assignment. Split at the midpoint of the predicted foreground
                       along --lr_axis (default: inferred from GT side centroids, reported per case).
      gt_nearest     : each predicted voxel goes to the nearer GT side. Uses GT -> optimistic
                       for the binary model; report it only as an upper bound.
Always also writes whole-structure binary metrics (no side split), which is how the
supervisor's number was reported.

  python -m iacb.compare_external --pred_dir teacher_fold0 --labels_dir $RAW/labelsTr \
      --splits $PRE/splits_final.json --folds 0 --name teacher --out cmp_teacher \
      --merge_per_side eval_T1/per_side.csv
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .common import bbox, case_folds, read_image, side_masks, side_metrics, union_bbox
from .stats import mcnemar_topology, paired


def lr_axis_from_gt(gt2: np.ndarray) -> int:
    """Axis along which the two GT side centroids differ most (dataset orientation convention)."""
    if not gt2[0].any() or not gt2[1].any():
        return -1
    c0 = np.array([np.nonzero(gt2[0])[k].mean() for k in range(3)])
    c1 = np.array([np.nonzero(gt2[1])[k].mean() for k in range(3)])
    return int(np.argmax(np.abs(c0 - c1)))


def split_binary(pred: np.ndarray, gt2: np.ndarray, spacing, how: str, axis: int) -> np.ndarray:
    fg = pred > 0
    out = np.zeros((2,) + pred.shape, bool)
    if not fg.any():
        return out
    if how == "pred_midline":
        # GT-free voxel assignment: split at the midpoint of the predicted foreground along `axis`.
        # GT is used only to NAME the two halves (which half is 'left'), never to move voxels.
        co = np.nonzero(fg)[axis]
        mid = 0.5 * (co.min() + co.max())
        grid = np.arange(pred.shape[axis]).reshape([-1 if k == axis else 1 for k in range(3)])
        low, high = fg & (grid <= mid), fg & (grid > mid)
        gl = np.nonzero(gt2[0])[axis].mean() if gt2[0].any() else (
            np.inf if gt2[1].any() and np.nonzero(gt2[1])[axis].mean() <= mid else -np.inf)
        out[0], out[1] = (low, high) if gl <= mid else (high, low)
    elif how == "gt_nearest":
        if not gt2[0].any() or not gt2[1].any():
            out[0 if gt2[0].any() else 1] = fg
            return out
        d0 = ndi.distance_transform_edt(~gt2[0], sampling=spacing)
        d1 = ndi.distance_transform_edt(~gt2[1], sampling=spacing)
        out[0], out[1] = fg & (d0 <= d1), fg & (d0 > d1)
    else:
        raise ValueError(how)
    return out


def run_case(t):
    case, a = t
    lab, sp = read_image(Path(a["labels_dir"]) / f"{case}.nii.gz")
    pred, sp_p = read_image(Path(a["pred_dir"]) / f"{case}.nii.gz")
    if pred.shape != lab.shape:
        raise ValueError(f"{case}: pred {pred.shape} vs label {lab.shape}")
    if not np.allclose(sp, sp_p, rtol=1e-3):
        raise ValueError(f"{case}: spacing pred {sp_p} vs label {sp} (resampled prediction?)")
    gt2 = side_masks(lab, a["left_id"], a["right_id"])
    vals = set(np.unique(pred).tolist())
    if a["pred_kind"] != "auto":
        kind = a["pred_kind"]
    else:   # auto: any label other than {0,1} means three-class; set --pred_kind explicitly to be safe
        kind = "three_class" if (vals - {0, 1}) else "binary"
    if kind == "three_class":
        pr2 = side_masks(pred, a["left_id"], a["right_id"])
    else:
        axis = a["lr_axis"] if a["lr_axis"] >= 0 else lr_axis_from_gt(gt2)
        pr2 = split_binary(pred, gt2, sp, a["side_assign"], max(axis, 0))
    rows = []
    axis_used = (a["lr_axis"] if a["lr_axis"] >= 0 else lr_axis_from_gt(gt2)) if kind == "binary" else None
    for s, side in enumerate(("L", "R")):
        box = union_bbox([bbox(gt2[s], 3), bbox(pr2[s], 3)])
        if box is None:
            continue
        r = dict(case=case, side=side, method=a["name"], pred_kind=kind, lr_axis=axis_used,
                 **side_metrics(pr2[s][box], gt2[s][box], sp, not a["no_cldice"]))
        rows.append(r)
    # whole-structure binary metrics (the supervisor's reporting)
    g, p = gt2.any(0), pred > 0
    box = union_bbox([bbox(g, 3), bbox(p, 3)])
    whole = dict(case=case, method=a["name"], pred_kind=kind,
                 **(side_metrics(p[box], g[box], sp, not a["no_cldice"]) if box is not None else {}))
    return rows, whole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--cases_txt", default=None, help="one case id per line; overrides --splits")
    ap.add_argument("--left_id", type=int, default=1)
    ap.add_argument("--right_id", type=int, default=2)
    ap.add_argument("--pred_kind", default="auto", choices=["auto", "binary", "three_class"])
    ap.add_argument("--side_assign", default="pred_midline", choices=["pred_midline", "gt_nearest"])
    ap.add_argument("--lr_axis", type=int, default=-1,
                    help="array axis (Z,Y,X order) separating left/right; -1 = infer per case from GT centroids")
    ap.add_argument("--merge_per_side", default=None, help="infer.py per_side.csv to pair with")
    ap.add_argument("--no_cldice", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = vars(ap.parse_args())
    if a["cases_txt"]:
        wanted = [l.strip() for l in Path(a["cases_txt"]).read_text().splitlines() if l.strip()]
    else:
        folds = {int(f) for f in a["folds"].split(",")}
        wanted = sorted(c for c, f in case_folds(Path(a["splits"])).items() if f in folds)
    have = {p.name[:-7] for p in Path(a["pred_dir"]).glob("*.nii.gz")}
    cases = sorted(set(wanted) & have)
    out = Path(a["out"]); out.mkdir(parents=True, exist_ok=True)
    overlap = dict(wanted=len(wanted), with_prediction=len(cases),
                   missing_prediction=sorted(set(wanted) - have)[:50],
                   predictions_outside_case_list=len(have - set(wanted)))
    rows, wholes = [], []
    with ProcessPoolExecutor(a["workers"]) as ex:
        for r, w in ex.map(run_case, [(c, a) for c in cases]):
            rows.extend(r); wholes.append(w)
    df, dw = pd.DataFrame(rows), pd.DataFrame(wholes)
    df.to_csv(out / "per_side.csv", index=False); dw.to_csv(out / "whole_binary.csv", index=False)
    rep = dict(config=a, overlap=overlap,
               per_side=dict(dice=float(df.dice.mean()), hd95=float(df.hd95.mean()),
                             hd95_median=float(df.hd95.median()),
                             sides_topo_ok=float((df.beta0_err == 0).mean()), n_sides=len(df),
                             lr_axes_used=sorted({int(v) for v in df.lr_axis.dropna()})),
               whole_binary=dict(dice=float(dw.dice.mean()), hd95=float(dw.hd95.mean()),
                                 hd95_median=float(dw.hd95.median()), n_cases=len(dw)))
    if a["merge_per_side"]:
        ours = pd.read_csv(a["merge_per_side"])
        common = set(map(tuple, ours[["case", "side"]].values)) & set(map(tuple, df[["case", "side"]].values))
        both = pd.concat([ours, df], ignore_index=True)
        both = both[[(c, s) in common for c, s in zip(both.case, both.side)]]
        both.to_csv(out / "merged_per_side.csv", index=False)
        rep["merged"] = dict(n_common_sides=len(common), n_common_cases=len({c for c, _ in common}))
        rep["paired_vs_external"] = {
            m: dict(dice=paired(both, a["name"], m, "dice"), hd95=paired(both, a["name"], m, "hd95"),
                    topology=mcnemar_topology(both, a["name"], m))
            for m in sorted(set(ours.method))}
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "paired_vs_external"}, indent=2))
    if len(cases) < len(wanted):
        print(f"[WARN] {len(wanted) - len(cases)} cases without a prediction -> comparison restricted")


if __name__ == "__main__":
    main()
