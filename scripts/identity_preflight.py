#!/usr/bin/env python3
"""Compare direct-hard, SDF-sign, and full-path identity on cached OOF cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "data", ROOT / "flow"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.oof_prior_audit import (atomic_json, expected_fold_map,              # noqa: E402
                                      find_case_artifact, load_oof)
from data.io_utils import sdf_stack_to_mask, voxel_spacing                         # noqa: E402
from evaluation.metrics import cldice, dice, hd95                                  # noqa: E402
from evaluation.topology_metrics import n_components                              # noqa: E402
from flow.sliding_window import predict_volume                                     # noqa: E402
from flow.validate import _load_case                                                # noqa: E402
from scripts.identity_baseline import ZeroVelocity                                 # noqa: E402

FULL_PATH_SDF_ATOL = 2e-6


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _side_metrics(mask, gt, side, spacing):
    pred_side, gt_side = mask == side, gt == side
    gt_volume = int(gt_side.sum())
    return {
        "dice": _finite(dice(pred_side, gt_side)),
        "cldice": _finite(cldice(pred_side, gt_side)),
        "hd95": _finite(hd95(pred_side, gt_side, spacing)),
        "volume_ratio": _finite(pred_side.sum() / gt_volume) if gt_volume else None,
        "components": n_components(pred_side),
    }


def _cache_geometry(image_path, label_path, hard_shape, coarse_item, gt_item):
    image = nib.load(str(image_path)); label = nib.load(str(label_path))
    image_spacing = voxel_spacing(image); label_spacing = voxel_spacing(label)
    coarse_sdf = coarse_item["sdf"]
    gt_sdf = gt_item["sdf"]
    coarse_spacing = coarse_item.get("spacing")
    gt_spacing = gt_item.get("spacing")
    checks = {
        "image_label_shape": image.shape == label.shape,
        "image_label_affine": bool(np.allclose(image.affine, label.affine, atol=1e-3)),
        "image_label_spacing": bool(np.allclose(image_spacing, label_spacing, atol=1e-3)),
        "hard_shape": tuple(hard_shape) == tuple(label.shape),
        "coarse_shape": tuple(coarse_sdf.shape[1:]) == tuple(label.shape),
        "gt_sdf_shape": tuple(gt_sdf.shape[1:]) == tuple(label.shape),
        "coarse_spacing": coarse_spacing is not None
                          and bool(np.allclose(coarse_spacing, label_spacing, atol=1e-3)),
        "gt_sdf_spacing": gt_spacing is not None
                          and bool(np.allclose(gt_spacing, label_spacing, atol=1e-3)),
    }
    return {
        "shape": list(label.shape), "affine": np.asarray(label.affine).tolist(),
        "spacing": label_spacing.tolist(), "checks": checks,
        "valid": all(checks.values()),
    }


def evaluate_case_paths(case_id, expected_fold, images, labels, hard_dir,
                        coarse_dir, gt_sdf_dir, provenance, patch=96, steps=8,
                        device="cpu"):
    hard_path = find_case_artifact(hard_dir, case_id)
    if hard_path is None:
        raise FileNotFoundError(f"missing hard OOF artifact for {case_id}")
    probs, direct_mask, artifact_type = load_oof(hard_path)
    del probs
    if artifact_type not in ("hard_segmentation", "derived_one_hot"):
        raise ValueError(f"identity preflight requires hard OOF, got {artifact_type}")

    image_path = Path(images) / f"{case_id}_0000.nii.gz"
    label_path = Path(labels) / f"{case_id}.nii.gz"
    coarse_path = Path(coarse_dir) / f"{case_id}.npz"
    gt_sdf_path = Path(gt_sdf_dir) / f"{case_id}.npz"
    with np.load(coarse_path) as item:
        coarse = {key: item[key] for key in item.files}
    with np.load(gt_sdf_path) as item:
        gt_cache = {key: item[key] for key in item.files}
    coarse_sdf = coarse["sdf"].astype(np.float32)
    sdf_mask = sdf_stack_to_mask(coarse_sdf)

    cond, loaded_coarse, spacing = _load_case(case_id, images, coarse_dir)
    endpoint = predict_volume(ZeroVelocity().to(device).eval(), cond, loaded_coarse,
                              patch=patch, steps=steps, device=device)
    full_mask = sdf_stack_to_mask(endpoint)
    gt = np.asanyarray(nib.load(str(label_path)).dataobj).astype(np.uint8)
    geometry = _cache_geometry(image_path, label_path, direct_mask.shape, coarse, gt_cache)

    case_provenance = provenance.get(case_id, {})
    source_fold = case_provenance.get("prediction_fold")
    source_checkpoint = case_provenance.get("source_checkpoint")
    provenance_checks = {
        "fold": source_fold == expected_fold,
        "checkpoint_recorded": bool(source_checkpoint),
        "checkpoint_exists": bool(source_checkpoint) and Path(source_checkpoint).is_file(),
        "artifact_type": case_provenance.get("artifact_type") in
                         (None, artifact_type),
    }
    result = {
        "case_id": case_id, "expected_fold": expected_fold,
        "prediction_source_fold": source_fold, "source_checkpoint": source_checkpoint,
        "artifact_type": artifact_type,
        "provenance_checks": provenance_checks,
        "provenance_valid": all(provenance_checks.values()),
        "geometry": geometry,
        "direct_vs_sdf_voxel_difference": int(np.count_nonzero(direct_mask != sdf_mask)),
        "direct_vs_full_path_voxel_difference": int(np.count_nonzero(direct_mask != full_mask)),
        "sdf_vs_full_path_voxel_difference": int(np.count_nonzero(sdf_mask != full_mask)),
        "full_path_sdf_max_abs_error": float(np.max(np.abs(endpoint - coarse_sdf))),
        "full_path_sdf_mean_abs_error": float(np.mean(np.abs(endpoint - coarse_sdf))),
        "sides": [],
    }
    for side, side_name in ((1, "left"), (2, "right")):
        result["sides"].append({
            "side": side_name,
            "direct": _side_metrics(direct_mask, gt, side, spacing),
            "sdf_decode": _side_metrics(sdf_mask, gt, side, spacing),
            "full_path": _side_metrics(full_mask, gt, side, spacing),
            "direct_vs_sdf_voxel_difference": int(np.count_nonzero(
                (direct_mask == side) != (sdf_mask == side))),
            "direct_vs_full_path_voxel_difference": int(np.count_nonzero(
                (direct_mask == side) != (full_mask == side))),
        })
    return result


def preflight_decision(cases, expected_count):
    if len(cases) != expected_count:
        return "FAIL_CASE_COUNT"
    if any(not case.get("provenance_valid", False) for case in cases):
        return "FAIL_FOLD_PROVENANCE"
    if any(not case.get("geometry", {}).get("valid", False) for case in cases):
        return "FAIL_GEOMETRY"
    if any(case.get("direct_vs_sdf_voxel_difference", 1) != 0 for case in cases):
        return "FAIL_SDF_ROUNDTRIP"
    if any(case.get("direct_vs_full_path_voxel_difference", 1) != 0 for case in cases):
        return "FAIL_FULL_PATH_PLUMBING"
    if any(case.get("full_path_sdf_max_abs_error", math.inf) > FULL_PATH_SDF_ATOL
           for case in cases):
        return "FAIL_FULL_PATH_NUMERICS"
    return "PASS"


def _mean(rows, path, metric):
    values = [row[path][metric] for row in rows if row[path][metric] is not None]
    return None if not values else float(np.mean(values))


def summarize(cases, expected_count):
    sides = [{**side, "case_id": case["case_id"], "fold": case["expected_fold"]}
             for case in cases for side in case.get("sides", [])]
    metrics = {}
    for path in ("direct", "sdf_decode", "full_path"):
        metrics[path] = {metric: _mean(sides, path, metric)
                         for metric in ("dice", "cldice", "hd95", "volume_ratio", "components")}
    return {
        "decision": preflight_decision(cases, expected_count),
        "evaluated_cases": len(cases), "expected_cases": expected_count,
        "metrics_per_side": metrics,
        "provenance_errors": sum(not case.get("provenance_valid", False) for case in cases),
        "geometry_errors": sum(not case.get("geometry", {}).get("valid", False) for case in cases),
        "direct_vs_sdf_voxel_difference": int(sum(
            case.get("direct_vs_sdf_voxel_difference", 0) for case in cases)),
        "direct_vs_full_path_voxel_difference": int(sum(
            case.get("direct_vs_full_path_voxel_difference", 0) for case in cases)),
        "max_full_path_sdf_abs_error": max(
            (case.get("full_path_sdf_max_abs_error", 0.0) for case in cases), default=None),
    }


def _csv_rows(cases):
    for case in cases:
        for side in case.get("sides", []):
            row = {
                "case_id": case["case_id"], "side": side["side"],
                "expected_fold": case["expected_fold"],
                "prediction_source_fold": case["prediction_source_fold"],
                "source_checkpoint": case["source_checkpoint"],
                "artifact_type": case["artifact_type"],
                "provenance_valid": case["provenance_valid"],
                "geometry_valid": case["geometry"]["valid"],
                "shape": json.dumps(case["geometry"]["shape"]),
                "affine": json.dumps(case["geometry"]["affine"]),
                "spacing": json.dumps(case["geometry"]["spacing"]),
                "direct_vs_sdf_voxel_difference": side["direct_vs_sdf_voxel_difference"],
                "direct_vs_full_path_voxel_difference": side["direct_vs_full_path_voxel_difference"],
                "full_path_sdf_max_abs_error": case["full_path_sdf_max_abs_error"],
                "full_path_sdf_mean_abs_error": case["full_path_sdf_mean_abs_error"],
            }
            for path in ("direct", "sdf_decode", "full_path"):
                for metric, value in side[path].items():
                    row[f"{path}_{metric}"] = value
            yield row


def write_reports(prefix, cases, summary):
    prefix = Path(prefix)
    csv_path = prefix.parent / f"{prefix.name}_cases.csv"
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    for path in (csv_path, json_path, md_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite preflight report: {path}")
    rows = list(_csv_rows(cases))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    partial = csv_path.with_suffix(csv_path.suffix + ".partial")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(partial, csv_path)
    atomic_json(json_path, {"summary": summary, "cases": cases})
    def fmt(value):
        return "NA" if value is None else f"{value:.4f}"

    metric_lines = []
    for name, values in summary["metrics_per_side"].items():
        metric_lines.append(
            f"| {name} | {fmt(values['dice'])} | {fmt(values['cldice'])} | "
            f"{fmt(values['hd95'])} | {fmt(values['volume_ratio'])} | "
            f"{fmt(values['components'])} |\n")
    markdown = ["# Identity preflight\n\n", f"**Decision: `{summary['decision']}`**\n\n",
                f"Cases: {summary['evaluated_cases']}/{summary['expected_cases']}\n\n",
                "| path | Dice | clDice | HD95 | volume ratio | components |\n",
                "|---|---:|---:|---:|---:|---:|\n", *metric_lines, "\n",
                f"- Provenance errors: {summary['provenance_errors']}\n",
                f"- Geometry errors: {summary['geometry_errors']}\n",
                f"- Direct vs SDF changed voxels: "
                f"{summary['direct_vs_sdf_voxel_difference']}\n",
                f"- Direct vs full-path changed voxels: "
                f"{summary['direct_vs_full_path_voxel_difference']}\n",
                f"- Max full-path SDF absolute error: "
                f"{summary['max_full_path_sdf_abs_error']} "
                f"(tolerance {FULL_PATH_SDF_ATOL})\n"]
    partial = md_path.with_suffix(md_path.suffix + ".partial")
    partial.write_text("".join(markdown)); os.replace(partial, md_path)


def _complete_case_ids(development, directories):
    complete = []
    for case_id in development:
        paths = [find_case_artifact(directories["hard"], case_id),
                 Path(directories["images"]) / f"{case_id}_0000.nii.gz",
                 Path(directories["labels"]) / f"{case_id}.nii.gz",
                 Path(directories["coarse"]) / f"{case_id}.npz",
                 Path(directories["gt_sdf"]) / f"{case_id}.npz"]
        if all(path is not None and Path(path).is_file() for path in paths):
            complete.append(case_id)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="configs/splits.json")
    parser.add_argument("--images", required=True); parser.add_argument("--labels", required=True)
    hard = parser.add_mutually_exclusive_group(required=True)
    hard.add_argument("--oof-hard"); hard.add_argument("--legacy-oof")
    parser.add_argument("--coarse-sdf", required=True); parser.add_argument("--gt-sdf", required=True)
    parser.add_argument("--provenance-manifest", required=True)
    parser.add_argument("--case-ids", default=None,
                        help="optional JSON list; otherwise use the complete-cache intersection")
    parser.add_argument("--expected-cases", type=int, default=40)
    parser.add_argument("--patch", type=int, default=96); parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-prefix", default="outputs/baselines/identity_preflight_40")
    parser.add_argument("--state", default=None); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    folds = expected_fold_map(splits)
    provenance_raw = json.loads(Path(args.provenance_manifest).read_text())
    provenance = provenance_raw.get("cases", provenance_raw)
    directories = {"images": args.images, "labels": args.labels,
                   "hard": args.oof_hard or args.legacy_oof,
                   "coarse": args.coarse_sdf, "gt_sdf": args.gt_sdf}
    case_ids = (json.loads(Path(args.case_ids).read_text()) if args.case_ids
                else _complete_case_ids(splits["development"], directories))
    if len(case_ids) != args.expected_cases:
        raise SystemExit(f"preflight requires exactly {args.expected_cases} complete cases; "
                         f"found {len(case_ids)}")

    prefix = Path(args.out_prefix)
    state_path = Path(args.state) if args.state else prefix.parent / f"{prefix.name}_state.json"
    state = {"cases": {}}
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text())
    for index, case_id in enumerate(case_ids, start=1):
        if args.resume and case_id in state["cases"]:
            continue
        state["cases"][case_id] = evaluate_case_paths(
            case_id, folds[case_id], args.images, args.labels, directories["hard"],
            args.coarse_sdf, args.gt_sdf, provenance, args.patch, args.steps, args.device)
        atomic_json(state_path, state)
        print(f"[identity-preflight] {index}/{len(case_ids)} {case_id}", flush=True)
    cases = [state["cases"][case_id] for case_id in case_ids]
    summary = summarize(cases, args.expected_cases)
    write_reports(prefix, cases, summary)
    print(json.dumps(summary, indent=2))
    if summary["decision"] != "PASS":
        raise SystemExit(f"identity preflight failed: {summary['decision']}")


if __name__ == "__main__":
    main()
