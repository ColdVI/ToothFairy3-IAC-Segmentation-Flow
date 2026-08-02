#!/usr/bin/env python3
"""
validate.py — real validation inference for checkpoint selection.

Runs the actual pipeline the model will be judged on (sliding-window ODE
integration -> SDF decode -> per-side metrics), NOT the training loss. It
returns overlap, boundary, and topology metrics used by the lexicographic
best-any / best-safe checkpoint policy in train.py.
"""
import os
import sys
import math

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from io_utils import physical_coord_grid, normalize_coords, voxel_spacing, sdf_stack_to_mask  # noqa: E402
from conditioning import build_conditioning, resolve_conditioning_spec                        # noqa: E402
from sliding_window import predict_volume                                                     # noqa: E402
from evaluation.metrics import dice, cldice, hd95                                              # noqa: E402
from evaluation.geometry_metrics import side_geometry_metrics                                 # noqa: E402
from evaluation.topology_metrics import (betti0_error, centerline_gap_length,                  # noqa: E402
                                         lr_swap_rate)


def _load_case(sid, images_dir, coarse_sdf_dir, conditioning_spec=None):
    img = nib.load(os.path.join(images_dir, f"{sid}_0000.nii.gz"))
    cbct = np.asanyarray(img.dataobj).astype(np.float32)
    sp = voxel_spacing(img)
    coords = normalize_coords(physical_coord_grid(cbct.shape, img.affine))
    co = np.load(os.path.join(coarse_sdf_dir, f"{sid}.npz"))
    coarse_sdf = co["sdf"].astype(np.float32)
    spec = conditioning_spec or resolve_conditioning_spec()
    cond = build_conditioning(cbct, co["prob_left"].astype(np.float32),
                              co["prob_right"].astype(np.float32),
                              coarse_sdf[0], coarse_sdf[1], coords, spec=spec)
    return cond, coarse_sdf, sp


def validation_rows(model, val_ids, images_dir, coarse_sdf_dir, gt_labels_dir,
                    patch=96, steps=8, device="cpu", max_cases=None,
                    progress=False, conditioning_spec=None):
    """Run inference and return one metric row per case and anatomical side."""
    model.eval()
    ids = val_ids if max_cases is None else val_ids[:max_cases]
    rows = []
    for case_index, sid in enumerate(ids, start=1):
        cond, coarse_sdf, sp = _load_case(
            sid, images_dir, coarse_sdf_dir, conditioning_spec)
        endp = predict_volume(model, cond, coarse_sdf, patch=patch, steps=steps, device=device)
        pred = sdf_stack_to_mask(endp)
        gt = np.asanyarray(nib.load(os.path.join(gt_labels_dir, f"{sid}.nii.gz")).dataobj)
        for side in (1, 2):
            rows.append({"case_id": sid, "side": side,
                         "dice": dice(pred == side, gt == side),
                         "cldice": cldice(pred == side, gt == side),
                         "hd95": hd95(pred == side, gt == side, sp),
                         "gap_mm": centerline_gap_length(pred == side, gt == side, sp),
                         "betti0": betti0_error(pred == side)})
        if progress:
            print(f"[validate] {case_index}/{len(ids)} {sid}", flush=True)
    return rows


CORE_METRICS = ("dice", "cldice", "hd95_mm", "gap_mm", "betti0")
GEOMETRY_METRICS = (
    "volume_ratio", "signed_surface_mean_mm", "radius_bias_mm",
    "connected_components", "false_positive_component_count",
    "max_false_positive_distance_mm",
)


def scanner_group(case_id):
    if case_id.startswith("ToothFairy3P_"):
        return "P"
    if case_id.startswith("ToothFairy3F_"):
        return "F"
    raise ValueError(f"unrecognized scanner group for case: {case_id}")


def _side_metrics(pred, gt, spacing):
    values = {
        "dice": dice(pred, gt), "cldice": cldice(pred, gt),
        "hd95_mm": hd95(pred, gt, spacing),
        "gap_mm": centerline_gap_length(pred, gt, spacing),
        "betti0": betti0_error(pred),
    }
    geometry = side_geometry_metrics(pred, gt, spacing)
    values.update({
        "volume_ratio": geometry["volume_ratio"],
        "signed_surface_mean_mm": geometry["signed_surface_mean_mm"],
        "radius_bias_mm": geometry["radius_bias_mm"],
        "connected_components": geometry["connected_component_count"],
        "false_positive_component_count": geometry["fp_component_count"],
        "max_false_positive_distance_mm": geometry["max_fp_distance_mm"],
        "geometry_valid": geometry["geometry_valid"],
        "geometry_reason": geometry["geometry_reason"],
    })
    return values


def paired_validation_rows(model, val_ids, images_dir, coarse_sdf_dir,
                           gt_labels_dir, patch=96, steps=8, device="cpu",
                           progress=False, conditioning_spec=None, *, epoch=None,
                           validation_tier=None):
    """Evaluate identity and flow on exactly the same cases and full-volume path."""
    if not val_ids:
        raise ValueError("paired validation requires at least one case")
    model.eval()
    rows = []
    for case_index, sid in enumerate(val_ids, start=1):
        cond, coarse_sdf, spacing = _load_case(
            sid, images_dir, coarse_sdf_dir, conditioning_spec)
        flow_sdf = predict_volume(
            model, cond, coarse_sdf, patch=patch, steps=steps, device=device)
        identity_label = sdf_stack_to_mask(coarse_sdf)
        flow_label = sdf_stack_to_mask(flow_sdf)
        gt_label = np.asanyarray(
            nib.load(os.path.join(gt_labels_dir, f"{sid}.nii.gz")).dataobj)
        identity_swap = lr_swap_rate(identity_label, gt_label)
        flow_swap = lr_swap_rate(flow_label, gt_label)
        for side in (1, 2):
            identity = _side_metrics(identity_label == side, gt_label == side, spacing)
            flow = _side_metrics(flow_label == side, gt_label == side, spacing)
            row = {"case_id": sid, "scanner_group": scanner_group(sid), "side": side,
                   "epoch": epoch, "validation_tier": validation_tier,
                   "identity_lr_swap": identity_swap,
                   "flow_lr_swap": flow_swap,
                   "delta_lr_swap": flow_swap - identity_swap}
            for name in CORE_METRICS + GEOMETRY_METRICS:
                row[f"identity_{name}"] = identity[name]
                row[f"flow_{name}"] = flow[name]
                left, right = identity[name], flow[name]
                row[f"delta_{name}"] = (None if left is None or right is None
                                         else float(right) - float(left))
            row["delta_abs_volume_bias"] = (
                None if identity["volume_ratio"] is None or flow["volume_ratio"] is None
                else abs(float(flow["volume_ratio"]) - 1.0)
                - abs(float(identity["volume_ratio"]) - 1.0))
            reasons = []
            if not identity["geometry_valid"]:
                reasons.append(f"identity:{identity['geometry_reason']}")
            if not flow["geometry_valid"]:
                reasons.append(f"flow:{flow['geometry_reason']}")
            numeric = [row[f"{prefix}_{name}"] for prefix in ("identity", "flow")
                       for name in CORE_METRICS + GEOMETRY_METRICS]
            if any(value is None or not math.isfinite(float(value)) for value in numeric):
                reasons.append("nonfinite_metric")
            row["metric_valid"] = not reasons
            row["exclusion_reason"] = "" if not reasons else ";".join(sorted(set(reasons)))
            rows.append(row)
        if progress:
            print(f"[paired-validate] {case_index}/{len(val_ids)} {sid}", flush=True)
    return rows


def paired_case_rows(rows, *, catastrophic_hd95_mm=10.0):
    """Aggregate bilaterally before selection; invalid sides invalidate the case."""
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    result = []
    for case_id, side_rows in by_case.items():
        reasons = []
        if sorted(row["side"] for row in side_rows) != [1, 2]:
            reasons.append("requires_exactly_two_anatomical_sides")
        reasons.extend(row["exclusion_reason"] for row in side_rows if not row["metric_valid"])
        case = {"case_id": case_id, "scanner_group": side_rows[0]["scanner_group"]}
        for prefix in ("identity", "flow", "delta"):
            for name in CORE_METRICS + GEOMETRY_METRICS:
                values = [row[f"{prefix}_{name}"] for row in side_rows]
                if any(value is None or not math.isfinite(float(value)) for value in values):
                    case[f"{prefix}_{name}"] = None
                else:
                    case[f"{prefix}_{name}"] = float(sum(values) / len(values))
        for prefix in ("identity", "flow", "delta"):
            key = f"{prefix}_lr_swap"
            values = {float(row[key]) for row in side_rows}
            case[key] = values.pop() if len(values) == 1 else None
            if case[key] is None:
                reasons.append("inconsistent_case_lr_swap_rate")
        identity_catastrophic = any(
            float(row["identity_hd95_mm"]) > catastrophic_hd95_mm for row in side_rows
            if row["identity_hd95_mm"] is not None)
        flow_catastrophic = any(
            float(row["flow_hd95_mm"]) > catastrophic_hd95_mm for row in side_rows
            if row["flow_hd95_mm"] is not None)
        case["identity_catastrophic_hd95"] = int(identity_catastrophic)
        case["flow_catastrophic_hd95"] = int(flow_catastrophic)
        case["extra_catastrophic_hd95"] = int(flow_catastrophic) - int(identity_catastrophic)
        case["identity_lr_swap_case"] = int((case["identity_lr_swap"] or 0) > 0)
        case["flow_lr_swap_case"] = int((case["flow_lr_swap"] or 0) > 0)
        case["extra_lr_swap_case"] = case["flow_lr_swap_case"] - case["identity_lr_swap_case"]
        case["valid"] = not reasons
        case["validity_reason"] = "ok" if not reasons else ";".join(sorted(set(reasons)))
        result.append(case)
    return result


def paired_identity_safety_gate(rows, config=None):
    """Apply the predeclared Prompt-3R non-inferiority/safety contract."""
    cfg = config or {}
    threshold = float(cfg.get("catastrophic_hd95_threshold_mm", 10.0))
    cases = paired_case_rows(rows, catastrophic_hd95_mm=threshold)
    invalid = [case for case in cases if not case["valid"]]
    criteria = {"all_cases_valid": not invalid}
    if invalid or not cases:
        return {"safe": False, "criteria": criteria, "case_rows": cases,
                "invalid_cases": [{"case_id": case["case_id"],
                                   "reason": case["validity_reason"]} for case in invalid]}

    def mean(name):
        values = [float(case[name]) for case in cases]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite paired gate field: {name}")
        return float(sum(values) / len(values))

    aggregate = {
        "mean_delta_dice": mean("delta_dice"),
        "mean_delta_cldice": mean("delta_cldice"),
        "mean_delta_hd95_mm": mean("delta_hd95_mm"),
        "mean_delta_gap_mm": mean("delta_gap_mm"),
        "extra_abs_volume_bias": float(sum(
            abs(float(case["flow_volume_ratio"]) - 1.0)
            - abs(float(case["identity_volume_ratio"]) - 1.0)
            for case in cases) / len(cases)),
        "extra_abs_radius_bias_mm": float(sum(
            abs(float(case["flow_radius_bias_mm"]))
            - abs(float(case["identity_radius_bias_mm"]))
            for case in cases) / len(cases)),
        "extra_catastrophic_hd95_cases": sum(
            case["extra_catastrophic_hd95"] for case in cases),
        "extra_lr_swap_cases": sum(case["extra_lr_swap_case"] for case in cases),
    }
    aggregate["identity_abs_volume_bias"] = float(sum(
        abs(float(case["identity_volume_ratio"]) - 1.0) for case in cases) / len(cases))
    aggregate["flow_abs_volume_bias"] = float(sum(
        abs(float(case["flow_volume_ratio"]) - 1.0) for case in cases) / len(cases))
    aggregate["identity_abs_radius_bias_mm"] = float(sum(
        abs(float(case["identity_radius_bias_mm"])) for case in cases) / len(cases))
    aggregate["flow_abs_radius_bias_mm"] = float(sum(
        abs(float(case["flow_radius_bias_mm"])) for case in cases) / len(cases))

    criteria.update({
        "mean_delta_dice": aggregate["mean_delta_dice"] >= float(
            cfg.get("min_mean_delta_dice", -0.005)),
        "mean_delta_cldice": aggregate["mean_delta_cldice"] >= float(
            cfg.get("min_mean_delta_cldice", -0.002)),
        "hd95_or_gap_improves": (aggregate["mean_delta_hd95_mm"] < 0 or
                                 aggregate["mean_delta_gap_mm"] < 0),
        "extra_abs_volume_bias": aggregate["extra_abs_volume_bias"] <= float(
            cfg.get("max_extra_abs_volume_bias", 0.03)),
        "extra_abs_radius_bias": aggregate["extra_abs_radius_bias_mm"] <= float(
            cfg.get("max_extra_abs_radius_bias_mm", 0.10)),
        "extra_catastrophic_hd95": aggregate["extra_catastrophic_hd95_cases"] <= int(
            cfg.get("max_extra_catastrophic_hd95_cases", 0)),
        "extra_lr_swaps": aggregate["extra_lr_swap_cases"] <= int(
            cfg.get("max_extra_lr_swaps", 0)),
    })
    return {"safe": all(criteria.values()), "criteria": criteria,
            "aggregate": aggregate, "case_rows": cases, "invalid_cases": []}


def summarize_rows(rows, aggregation="per_side"):
    """Aggregate validation rows either directly or after bilateral case means."""
    if not rows:
        raise ValueError("cannot summarize an empty validation result")
    metrics = ("dice", "cldice", "hd95", "gap_mm", "betti0")
    if aggregation == "per_side":
        values = {metric: [row[metric] for row in rows] for metric in metrics}
    elif aggregation == "per_case":
        case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
        values = {metric: [] for metric in metrics}
        for sid in case_ids:
            case_rows = [row for row in rows if row["case_id"] == sid]
            for metric in metrics:
                values[metric].append(float(np.nanmean([row[metric] for row in case_rows])))
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")
    means = {metric: float(np.nanmean(values[metric])) for metric in metrics}
    mean_dice = means["dice"]
    mean_cldice = means["cldice"]
    mean_hd95 = means["hd95"]
    score = 0.5 * mean_dice + 0.5 * mean_cldice
    return {**means, "score": score}


def validate(model, val_ids, images_dir, coarse_sdf_dir, gt_labels_dir,
             patch=96, steps=8, device="cpu", max_cases=None,
             conditioning_spec=None):
    rows = validation_rows(model, val_ids, images_dir, coarse_sdf_dir, gt_labels_dir,
                           patch=patch, steps=steps, device=device, max_cases=max_cases,
                           conditioning_spec=conditioning_spec)
    return summarize_rows(rows, aggregation="per_side")
