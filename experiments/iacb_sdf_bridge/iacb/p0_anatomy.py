"""Phase 0 (CPU only, no training). Decides whether the bridge is worth GPU time.

For each OOF case and side it measures
  * metrics of prior / prior+LCC / prior+MCP(soft) / prior+MCP(hard)
  * where the prior's error voxels are: boundary band vs gap FN vs spurious FP
  * every GT-centreline voxel the prior misses, classified by the prior probability
    around it:  displaced (max p >= .5 nearby), smeared (max p < .5 but local
    probability mass >= .5 of GT mass), absent (mass < .5).
    The bridge + TC-MBR hypothesis needs 'smeared' gaps: mass present, marginal < .5.

Usage
  python -m iacb.p0_anatomy --images_dir .../imagesTr --labels_dir .../labelsTr \
      --prob_dir .../oof_npz --splits .../splits_final.json --folds 0 \
      --left_id 1 --right_id 2 --out p0_out --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .common import (bbox, case_folds, components, dice, largest_component, load_oof_probs,
                     prior_side_masks, read_image, side_masks, side_metrics, skeleton, surface_stats,
                     union_bbox, resolve_oof_npz)
from .decode import mcp_connect


def gap_anatomy(prior, prob, gt, spacing, boundary_mm=0.6, slab_mult=3.0):
    sv = float(np.mean(spacing))
    out = {}
    # ---- error voxel decomposition
    fn, fp = gt & ~prior, prior & ~gt
    d_to_prior = ndi.distance_transform_edt(~prior, sampling=spacing) if prior.any() else np.full(gt.shape, 1e9)
    d_to_gt = ndi.distance_transform_edt(~gt, sampling=spacing) if gt.any() else np.full(gt.shape, 1e9)
    fn_boundary = fn & (d_to_prior <= boundary_mm)
    fp_boundary = fp & (d_to_gt <= boundary_mm)
    lab, n, _ = components(prior)
    touching = set(np.unique(lab[ndi.binary_dilation(gt, iterations=1)])) - {0}
    spurious = fp & (lab > 0) & ~np.isin(lab, list(touching))
    n_err = max(int(fn.sum() + fp.sum()), 1)
    out.update(
        err_vox=int(fn.sum() + fp.sum()),
        frac_boundary=float((fn_boundary.sum() + fp_boundary.sum()) / n_err),
        frac_fn_nonboundary=float((fn & ~fn_boundary).sum() / n_err),
        frac_fp_spurious=float(spurious.sum() / n_err),
        frac_fp_other=float((fp & ~fp_boundary & ~spurious).sum() / n_err),
    )
    # oracle Dice if only boundary errors were fixed / if only non-boundary errors were fixed
    fix_b = (prior | fn_boundary) & ~fp_boundary
    fix_nb = (prior | (fn & ~fn_boundary)) & ~(fp & ~fp_boundary)
    out.update(oracle_dice_fix_boundary=dice(fix_b, gt), oracle_dice_fix_nonboundary=dice(fix_nb, gt))
    # oracle HD95: where does the HD95 tail come from? (component-level vs boundary errors)
    #   drop_spurious : remove predicted components that do not touch GT
    #   fill_gaps     : add the non-boundary FN (missed canal segments)
    #   component     : both (what a perfect component-level decoder could reach)
    #   fix_boundary  : only boundary-band errors fixed (what a perfect radius/boundary refiner could reach)
    drop = prior & ~spurious
    fill = prior | (fn & ~fn_boundary)
    comp = drop | (fn & ~fn_boundary)
    out.update(
        hd95_prior=surface_stats(prior, gt, spacing)["hd95"],
        oracle_hd95_drop_spurious=surface_stats(drop, gt, spacing)["hd95"],
        oracle_hd95_fill_gaps=surface_stats(fill, gt, spacing)["hd95"],
        oracle_hd95_component=surface_stats(comp, gt, spacing)["hd95"],
        oracle_hd95_fix_boundary=surface_stats(fix_b, gt, spacing)["hd95"],
        oracle_dice_component=dice(comp, gt),
    )

    # ---- centreline gap classification (per missed run, cross-sectional neighbourhood)
    sk = skeleton(gt)
    covered = ndi.binary_dilation(prior, iterations=1)
    miss = sk & ~covered
    lens = dict(displaced=0.0, smeared=0.0, absent=0.0)
    r_med = 1.3
    nruns = 0
    if sk.any():
        r_gt = ndi.distance_transform_edt(gt, sampling=spacing)
        r_med = float(np.median(r_gt[sk]))
    if miss.any():
        runs, nruns = ndi.label(miss, structure=np.ones((3, 3, 3), bool))
        dist, inds = ndi.distance_transform_edt(~sk, sampling=spacing, return_indices=True)
        owner = runs[tuple(inds)]                       # run id of the nearest centreline voxel
        owner[dist > slab_mult * r_med] = 0             # cross-sectional slab of each missed run
        ids = np.arange(1, nruns + 1)
        maxp = np.asarray(ndi.maximum(prob, owner, ids))
        mass_p = np.bincount(owner.ravel(), weights=prob.ravel(), minlength=nruns + 1)[1:]
        mass_g = np.bincount(owner.ravel(), weights=gt.ravel().astype(np.float64), minlength=nruns + 1)[1:]
        run_len = np.bincount(runs.ravel(), minlength=nruns + 1)[1:] * sv
        ratio = mass_p / np.maximum(mass_g, 1e-6)
        cls = np.where(maxp >= 0.5, "displaced", np.where(ratio >= 0.5, "smeared", "absent"))
        for c in lens:
            lens[c] = float(run_len[cls == c].sum())
    out.update(cl_len_mm=float(sk.sum() * sv), miss_len_mm=float(miss.sum() * sv), n_gap_runs=int(nruns),
               r_med_mm=r_med, **{f"miss_{k}_mm": v for k, v in lens.items()})
    return out


def run_case(args_tuple):
    case, a = args_tuple
    img_path = Path(a["labels_dir"]) / f"{case}.nii.gz"
    lab, spacing = read_image(img_path)
    fold = a.get("case_folds", {}).get(case)
    prob_path = resolve_oof_npz(Path(a["prob_dir"]), case, fold)
    prob = load_oof_probs(prob_path, a["prob_left_id"], a["prob_right_id"])
    if prob.shape[1:] != lab.shape:
        raise ValueError(f"{case}: prob {prob.shape[1:]} vs label {lab.shape} (axis order?)")
    gt2 = side_masks(lab, a["label_left_id"], a["label_right_id"])
    pr2 = prior_side_masks(prob)
    rows = []
    for s, side in enumerate(("L", "R")):
        box = union_bbox([bbox(gt2[s], 60), bbox(pr2[s], 60)])
        if box is None:
            continue
        g, p, pb = gt2[s][box], pr2[s][box], prob[s][box]
        base = dict(case=case, side=side)
        if dice(p, g) < a["sanity_min_dice"] and g.sum() > 0:
            base["sanity_flag"] = True
        methods = {"prior": p,
                   "prior_lcc": None, "prior_mcp_soft": None, "prior_mcp_hard": None}
        methods["prior_lcc"] = largest_component(p)
        methods["prior_mcp_soft"], info_s = mcp_connect(p, pb, spacing, hard=False)
        methods["prior_mcp_hard"], info_h = mcp_connect(p, pb, spacing, hard=True)
        for name, m in methods.items():
            r = dict(base, method=name, **side_metrics(m, g, spacing))
            if name == "prior":
                r.update(gap_anatomy(p, pb, g, spacing))
            if name == "prior_mcp_soft":
                r.update({f"mcp_{k}": v for k, v in info_s.items()})
            rows.append(r)
    return rows


def gates(df, a):
    pr = df[df.method == "prior"]
    miss = pr[["miss_displaced_mm", "miss_smeared_mm", "miss_absent_mm"]].sum()
    tot = max(miss.sum(), 1e-9)
    by_m = df.groupby("method").agg(dice=("dice", "mean"), hd95=("hd95", "mean"),
                                    hd95_p90=("hd95", lambda x: np.percentile(x, 90)),
                                    beta0_err=("beta0_err", "mean"), cldice=("cldice", "mean"),
                                    sides_topo_ok=("beta0_err", lambda x: float((x == 0).mean())))
    res = dict(
        n_cases=int(pr.case.nunique()), n_sides=int(len(pr)),
        pooled_miss_fraction=dict((k, float(v / tot)) for k, v in miss.items()),
        mean_frac_boundary=float(pr.frac_boundary.mean()),
        mean_oracle_dice_fix_boundary=float(pr.oracle_dice_fix_boundary.mean()),
        mean_oracle_dice_fix_nonboundary=float(pr.oracle_dice_fix_nonboundary.mean()),
        prior_sides_with_topology_error=int((pr.beta0_err > 0).sum()),
        mean_oracle_dice_component=float(pr.oracle_dice_component.mean()),
        hd95_headroom=dict(
            prior_mean=float(pr.hd95_prior.mean()), prior_median=float(pr.hd95_prior.median()),
            oracle_component_mean=float(pr.oracle_hd95_component.mean()),
            oracle_drop_spurious_mean=float(pr.oracle_hd95_drop_spurious.mean()),
            oracle_fill_gaps_mean=float(pr.oracle_hd95_fill_gaps.mean()),
            oracle_fix_boundary_mean=float(pr.oracle_hd95_fix_boundary.mean()),
            # share of the mean-HD95 excess over the median that a perfect component-level decoder removes
            component_share_of_tail=float((pr.hd95_prior.mean() - pr.oracle_hd95_component.mean())
                                          / max(pr.hd95_prior.mean() - pr.hd95_prior.median(), 1e-9)),
        ),
        methods=by_m.round(5).to_dict(orient="index"),
    )
    smeared = res["pooled_miss_fraction"]["miss_smeared_mm"]
    n_topo = res["prior_sides_with_topology_error"]
    mcp_ok = res["methods"].get("prior_mcp_hard", {}).get("sides_topo_ok", 0)
    prior_ok = res["methods"]["prior"]["sides_topo_ok"]
    remaining = (1 - mcp_ok) * res["n_sides"]
    res["gates"] = {
        "G1_enough_topology_errors (>= min_topo_sides)": n_topo >= a["min_topo_sides"],
        "G2_smeared_share (>= min_smeared)": smeared >= a["min_smeared"],
        "G3_mcp_leaves_headroom (sides still wrong after MCP-hard >= min_remaining)": remaining >= a["min_remaining"],
    }
    res["verdict"] = "GO: train the bridge" if all(res["gates"].values()) else \
        "NO-GO for the topology claim: see failed gate(s) before spending GPU"
    res["notes"] = dict(prior_sides_topo_ok=prior_ok, mcp_hard_sides_topo_ok=mcp_ok)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--label_left_id", type=int, default=3)
    ap.add_argument("--label_right_id", type=int, default=4)
    ap.add_argument("--prob_left_id", type=int, default=1)
    ap.add_argument("--prob_right_id", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sanity_min_dice", type=float, default=0.5)
    # preregistered gate thresholds (change only BEFORE looking at the output)
    ap.add_argument("--min_topo_sides", type=int, default=10)
    ap.add_argument("--min_smeared", type=float, default=0.30)
    ap.add_argument("--min_remaining", type=int, default=6)
    ap.add_argument("--images_dir", default=None, help="unused, kept for a uniform CLI")
    a = vars(ap.parse_args())
    a["case_folds"] = case_folds(Path(a["splits"]))
    folds = {int(f) for f in a["folds"].split(",")}
    cases = sorted(c for c, f in a["case_folds"].items() if f in folds)
    if a["limit"]:
        cases = cases[: a["limit"]]
    out = Path(a["out"]); out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(a["workers"]) as ex:
        for r in ex.map(run_case, [(c, a) for c in cases]):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "p0_per_side.csv", index=False)
    if "sanity_flag" in df and df.sanity_flag.fillna(False).any():
        bad = sorted(df[df.sanity_flag.fillna(False)].case.unique())
        print(f"[WARN] prior-vs-GT Dice < sanity threshold for {bad}: check axis order before trusting anything")
    res = gates(df, a)
    (out / "p0_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
