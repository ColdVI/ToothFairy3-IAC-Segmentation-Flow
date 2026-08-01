#!/usr/bin/env python3
"""Read-only audit of Track B OOF prior semantics, provenance, and geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.io_utils import sdf_stack_to_mask, voxel_spacing  # noqa: E402

ARTIFACT_TYPES = {"hard_segmentation", "derived_one_hot", "true_softmax"}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(partial, path)


def expected_fold_map(splits):
    mapping = {}
    for fold_index, fold in enumerate(splits["folds"]):
        for case_id in fold["val"]:
            if case_id in mapping:
                raise ValueError(f"case appears in multiple validation folds: {case_id}")
            mapping[case_id] = fold_index
    development = splits.get("development", list(mapping))
    if set(mapping) != set(development):
        raise ValueError("fold validation union does not equal the development cohort")
    return mapping


def _binary(array):
    return bool(np.all((array == 0) | (array == 1)))


def _channel_first(array):
    array = np.asarray(array)
    if array.ndim != 4:
        raise ValueError(f"probability array must be 4-D, got {array.shape}")
    if array.shape[0] in (2, 3):
        return array
    if array.shape[-1] in (2, 3):
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"cannot identify class axis in {array.shape}")


def load_oof(path):
    """Return (p_bg,p_left,p_right), decoded hard mask, and semantic type."""
    path = Path(path)
    if path.name.endswith(".nii.gz"):
        seg = np.asanyarray(nib.load(str(path)).dataobj).astype(np.uint8)
        probs = np.stack([seg == 0, seg == 1, seg == 2]).astype(np.float32)
        return probs, seg, "hard_segmentation"

    with np.load(path) as item:
        if "prob_left" in item and "prob_right" in item:
            left = item["prob_left"].astype(np.float32)
            right = item["prob_right"].astype(np.float32)
            background = np.clip(1.0 - left - right, 0.0, 1.0)
            probs = np.stack([background, left, right])
            derived = _binary(left) and _binary(right) and not np.any((left > 0) & (right > 0))
            artifact_type = "derived_one_hot" if derived else "true_softmax"
        else:
            key = next((key for key in ("probabilities", "softmax", "probs") if key in item), None)
            if key is None:
                raise KeyError(f"no supported OOF arrays in {path}; found {item.files}")
            raw = _channel_first(item[key]).astype(np.float32)
            if raw.shape[0] == 2:
                background = np.clip(1.0 - raw.sum(axis=0), 0.0, 1.0)
                probs = np.concatenate([background[None], raw], axis=0)
            else:
                probs = raw
            artifact_type = "derived_one_hot" if all(_binary(ch) for ch in probs) else "true_softmax"
    hard = np.argmax(probs, axis=0).astype(np.uint8)
    return probs, hard, artifact_type


def find_case_artifact(directory, case_id):
    if not directory:
        return None
    directory = Path(directory)
    for suffix in (".npz", ".nii.gz"):
        candidate = directory / f"{case_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _unique_summary(array, max_sample=200_000):
    flat = np.asarray(array).reshape(-1)
    if flat.size > max_sample and not _binary(array):
        indices = np.linspace(0, flat.size - 1, max_sample, dtype=np.int64)
        values = np.unique(flat[indices])
        sampled = True
    else:
        values = np.unique(flat)
        sampled = False
    if len(values) <= 32:
        displayed = [float(value) for value in values]
    else:
        displayed = [float(value) for value in np.quantile(values, [0, .01, .25, .5, .75, .99, 1])]
    return {"values": displayed, "count": int(len(values)), "sampled": sampled}


def _safe_corr(a, b, max_sample=250_000):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size != b.size:
        return None
    if a.size > max_sample:
        indices = np.linspace(0, a.size - 1, max_sample, dtype=np.int64)
        a, b = a[indices], b[indices]
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return None if denominator == 0 else float(np.dot(a, b) / denominator)


def probability_stats(probs):
    eps = 1e-12
    total = probs.sum(axis=0, dtype=np.float32)
    entropy = -(np.clip(probs, eps, 1.0) * np.log(np.clip(probs, eps, 1.0))).sum(axis=0)
    channels = {}
    for index, name in enumerate(("background", "left", "right")):
        channel = probs[index]
        channels[name] = {
            "unique": _unique_summary(channel),
            "min": float(channel.min()), "max": float(channel.max()),
            "mean": float(channel.mean()), "sum": float(channel.sum(dtype=np.float64)),
        }
    return {
        "channels": channels,
        "probability_sum": {"min": float(total.min()), "max": float(total.max()),
                            "mean": float(total.mean())},
        "entropy": {"min": float(entropy.min()), "max": float(entropy.max()),
                    "mean": float(entropy.mean()),
                    "p50": float(np.quantile(entropy, .5)),
                    "p95": float(np.quantile(entropy, .95))},
    }


def _geometry(image_path, label_path):
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    image_spacing = voxel_spacing(image)
    label_spacing = voxel_spacing(label)
    return {
        "shape": list(label.shape),
        "image_shape": list(image.shape),
        "image_label_shape_match": image.shape == label.shape,
        "image_affine": np.asarray(image.affine).tolist(),
        "label_affine": np.asarray(label.affine).tolist(),
        "affine_match": bool(np.allclose(image.affine, label.affine, atol=1e-3)),
        "image_spacing": image_spacing.tolist(),
        "label_spacing": label_spacing.tolist(),
        "spacing_match": bool(np.allclose(image_spacing, label_spacing, atol=1e-3)),
    }


def audit_case(case_id, expected_fold, images, labels, hard_dir, softmax_dir,
               coarse_dir, gt_sdf_dir, provenance):
    image_path = Path(images) / f"{case_id}_0000.nii.gz"
    label_path = Path(labels) / f"{case_id}.nii.gz"
    hard_path = find_case_artifact(hard_dir, case_id)
    softmax_path = find_case_artifact(softmax_dir, case_id)
    coarse_path = Path(coarse_dir) / f"{case_id}.npz" if coarse_dir else None
    gt_sdf_path = Path(gt_sdf_dir) / f"{case_id}.npz" if gt_sdf_dir else None
    required = {"image": image_path, "label": label_path, "hard_oof": hard_path,
                "coarse_sdf": coarse_path, "gt_sdf": gt_sdf_path}
    missing = [name for name, path in required.items() if path is None or not Path(path).is_file()]
    if missing:
        return {"case_id": case_id, "expected_fold": expected_fold, "status": "missing",
                "missing": missing}

    probs, hard_mask, artifact_type = load_oof(hard_path)
    if artifact_type not in ARTIFACT_TYPES:
        raise AssertionError(artifact_type)
    geometry = _geometry(image_path, label_path)
    with np.load(coarse_path) as coarse_item:
        coarse_sdf = coarse_item["sdf"].astype(np.float32)
        coarse_spacing = coarse_item.get("spacing")
    with np.load(gt_sdf_path) as gt_item:
        gt_sdf = gt_item["sdf"]
        gt_spacing = gt_item.get("spacing")
    sdf_mask = sdf_stack_to_mask(coarse_sdf)
    direct_vs_sdf = int(np.count_nonzero(hard_mask != sdf_mask))
    shape = tuple(geometry["shape"])
    shape_status = {
        "oof": tuple(hard_mask.shape) == shape,
        "coarse_sdf": tuple(coarse_sdf.shape[1:]) == shape,
        "gt_sdf": tuple(gt_sdf.shape[1:]) == shape,
    }
    label_spacing = np.asarray(geometry["label_spacing"])
    cache_spacing_status = {
        "coarse": coarse_spacing is not None and bool(np.allclose(coarse_spacing, label_spacing, atol=1e-3)),
        "gt": gt_spacing is not None and bool(np.allclose(gt_spacing, label_spacing, atol=1e-3)),
    }
    case_provenance = provenance.get(case_id, {})
    source_fold = case_provenance.get("expected_fold", case_provenance.get("prediction_fold"))
    source_checkpoint = case_provenance.get("source_checkpoint")
    provenance_source = case_provenance.get("provenance_source")
    provenance_ok = (source_fold == expected_fold and bool(source_checkpoint)
                     and provenance_source in (None, "legacy_import", "runtime_prediction"))

    softmax_type = None
    softmax_stats = None
    if softmax_path is not None:
        softmax_probs, _, softmax_type = load_oof(softmax_path)
        softmax_stats = probability_stats(softmax_probs)

    stats = probability_stats(probs)
    row = {
        "case_id": case_id,
        "status": "valid" if (all(shape_status.values()) and all(cache_spacing_status.values())
                                and geometry["image_label_shape_match"] and geometry["affine_match"]
                                and geometry["spacing_match"] and provenance_ok
                                and direct_vs_sdf == 0) else "invalid",
        "expected_fold": expected_fold,
        "prediction_source_fold": source_fold,
        "source_checkpoint": source_checkpoint,
        "provenance_source": provenance_source,
        "exact_reproduction_claim": case_provenance.get("exact_reproduction_claim"),
        "cross_runtime_validation": case_provenance.get("cross_runtime_validation"),
        "provenance_status": "valid" if provenance_ok else "missing_or_mismatch",
        "artifact_type": artifact_type,
        "hard_artifact_path": str(hard_path),
        "softmax_artifact_type": softmax_type,
        "softmax_artifact_path": str(softmax_path) if softmax_path else None,
        "probability_stats": stats,
        "softmax_stats": softmax_stats,
        "shape": list(hard_mask.shape),
        "geometry": geometry,
        "cache_shape_match": shape_status,
        "cache_spacing_match": cache_spacing_status,
        "hard_vs_sdf_sign_voxel_difference": direct_vs_sdf,
        "conditioning_redundancy": {
            "hard_left_vs_coarse_sdf_corr": _safe_corr(hard_mask == 1, coarse_sdf[0]),
            "hard_right_vs_coarse_sdf_corr": _safe_corr(hard_mask == 2, coarse_sdf[1]),
            "hard_vs_sdf_sign_voxel_equality": float(np.mean(hard_mask == sdf_mask)),
        },
    }
    return row


def _flatten_for_csv(row):
    stats = row.get("probability_stats", {})
    channels = stats.get("channels", {})
    geometry = row.get("geometry", {})
    return {
        "case_id": row["case_id"], "status": row["status"],
        "expected_fold": row.get("expected_fold"),
        "prediction_source_fold": row.get("prediction_source_fold"),
        "source_checkpoint": row.get("source_checkpoint"),
        "provenance_status": row.get("provenance_status"),
        "artifact_type": row.get("artifact_type"),
        "softmax_artifact_type": row.get("softmax_artifact_type"),
        "unique_values": json.dumps({name: value.get("unique") for name, value in channels.items()}),
        "channel_min": json.dumps({name: value.get("min") for name, value in channels.items()}),
        "channel_max": json.dumps({name: value.get("max") for name, value in channels.items()}),
        "channel_mean": json.dumps({name: value.get("mean") for name, value in channels.items()}),
        "channel_sums": json.dumps({name: value.get("sum") for name, value in channels.items()}),
        "probability_sum": json.dumps(stats.get("probability_sum")),
        "entropy": json.dumps(stats.get("entropy")),
        "shape": json.dumps(row.get("shape")),
        "affine": json.dumps(geometry.get("label_affine")),
        "spacing": json.dumps(geometry.get("label_spacing")),
        "geometry_match": all(bool(geometry.get(key)) for key in
                              ("image_label_shape_match", "affine_match", "spacing_match")),
        "cache_shape_match": all(row.get("cache_shape_match", {}).values()),
        "cache_spacing_match": all(row.get("cache_spacing_match", {}).values()),
        "hard_vs_sdf_sign_voxel_difference": row.get("hard_vs_sdf_sign_voxel_difference"),
        "conditioning_redundancy": json.dumps(row.get("conditioning_redundancy")),
        "missing": json.dumps(row.get("missing", [])),
        "error": row.get("error"),
    }


def write_reports(prefix, cases, legacy_adapter, replace=False):
    prefix = Path(prefix)
    csv_path = prefix.parent / f"{prefix.name}_cases.csv"
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    for path in (csv_path, json_path, md_path):
        if path.exists() and not replace:
            raise FileExistsError(f"refusing to overwrite existing audit report: {path}")
    counts = Counter(row["status"] for row in cases)
    types = Counter(row.get("artifact_type", "missing") for row in cases)
    provenance_errors = sum(row.get("provenance_status") != "valid" for row in cases)
    split_leakage_errors = sum(
        row.get("prediction_source_fold") != row.get("expected_fold") for row in cases)
    sdf_differences = sum((row.get("hard_vs_sdf_sign_voxel_difference") or 0) for row in cases)
    summary = {
        "total_cases": len(cases), "status_counts": dict(counts),
        "artifact_type_counts": dict(types), "provenance_errors": provenance_errors,
        "split_leakage_errors": split_leakage_errors,
        "hard_vs_sdf_sign_voxel_difference": int(sdf_differences),
        "legacy_adapter": legacy_adapter,
        "contract": {"hard_directory": "oof_hard", "softmax_directory": "oof_softmax",
                     "hard_is_not_calibrated_probability": True},
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_partial = csv_path.with_suffix(csv_path.suffix + ".partial")
    csv_rows = [_flatten_for_csv(row) for row in cases]
    with csv_partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    os.replace(csv_partial, csv_path)
    atomic_json(json_path, {"summary": summary, "cases": cases})
    invalid = [row["case_id"] for row in cases if row["status"] != "valid"]
    markdown = ["# OOF prior audit\n\n",
                f"- Cases: {len(cases)}\n",
                f"- Status: `{dict(counts)}`\n",
                f"- Artifact types: `{dict(types)}`\n",
                f"- Provenance errors: {provenance_errors}\n",
                f"- Split leakage/fold assignment errors: {split_leakage_errors}\n",
                f"- Hard vs SDF-sign changed voxels: {sdf_differences}\n",
                f"- Legacy adapter: `{legacy_adapter}`\n\n",
                "Hard/derived-one-hot artifacts are not calibrated probabilities.\n\n",
                "## Invalid or missing cases\n\n",
                (", ".join(invalid) if invalid else "None"), "\n"]
    md_partial = md_path.with_suffix(md_path.suffix + ".partial")
    md_partial.write_text("".join(markdown))
    os.replace(md_partial, md_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="configs/splits.json")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    hard = parser.add_mutually_exclusive_group(required=True)
    hard.add_argument("--oof-hard", help="preferred oof_hard directory")
    hard.add_argument("--legacy-oof", help="legacy oof_probs directory; audited without relabelling")
    parser.add_argument("--oof-softmax", default=None)
    parser.add_argument("--coarse-sdf", required=True)
    parser.add_argument("--gt-sdf", required=True)
    parser.add_argument("--provenance-manifest", default=None)
    parser.add_argument("--out-prefix", default="outputs/analysis/oof_prior_audit")
    parser.add_argument("--state", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-ids", default=None,
                        help="optional JSON list selecting an explicit audited cohort")
    parser.add_argument("--heartbeat-every", type=int, default=10)
    parser.add_argument("--progress-prefix", default=None)
    parser.add_argument("--inventory", default=None,
                        help="optional inventory used to invalidate changed resumed cases")
    parser.add_argument("--replace-reports", action="store_true")
    args = parser.parse_args()

    with open(args.splits) as handle:
        splits = json.load(handle)
    fold_map = expected_fold_map(splits)
    case_ids = (json.loads(Path(args.case_ids).read_text()) if args.case_ids
                else list(splits["development"]))
    unknown = sorted(set(case_ids) - set(splits["development"]))
    if unknown:
        raise ValueError(f"audit case list contains non-development IDs: {unknown[:5]}")
    if args.max_cases is not None:
        case_ids = case_ids[:args.max_cases]
    provenance = {}
    if args.provenance_manifest:
        with open(args.provenance_manifest) as handle:
            raw = json.load(handle)
        provenance = raw.get("cases", raw)

    prefix = Path(args.out_prefix)
    state_path = Path(args.state) if args.state else prefix.parent / f"{prefix.name}_state.json"
    state = {"cases": {}}
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text())
    hard_dir = args.oof_hard or args.legacy_oof
    inventory_cases = {}
    inventory_git_sha = None
    if args.inventory:
        inventory = json.loads(Path(args.inventory).read_text())
        inventory_cases = inventory.get("cases", {})
        inventory_git_sha = inventory.get("pinned_git_sha")
    state.setdefault("signatures", {})
    for index, case_id in enumerate(case_ids, start=1):
        signature_payload = {
            "inventory": inventory_cases.get(case_id),
            "provenance": provenance.get(case_id),
            "expected_fold": fold_map[case_id],
            "pinned_git_sha": inventory_git_sha,
        }
        signature = hashlib.sha256(json.dumps(
            signature_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (args.resume and case_id in state["cases"]
                and state["signatures"].get(case_id) == signature):
            if (index % args.heartbeat_every == 0 or index == len(case_ids)):
                progress = args.progress_prefix or "[oof-audit]"
                print(f"{progress} {index}/{len(case_ids)}", flush=True)
            continue
        try:
            row = audit_case(case_id, fold_map[case_id], args.images, args.labels,
                             hard_dir, args.oof_softmax, args.coarse_sdf, args.gt_sdf,
                             provenance)
        except Exception as error:  # one corrupt case must not hide the rest of the audit
            row = {"case_id": case_id, "expected_fold": fold_map[case_id],
                   "status": "invalid", "error": f"{type(error).__name__}: {error}"}
        state["cases"][case_id] = row
        state["signatures"][case_id] = signature
        atomic_json(state_path, state)
        if index % args.heartbeat_every == 0 or index == len(case_ids):
            progress = args.progress_prefix or "[oof-audit]"
            print(f"{progress} {index}/{len(case_ids)}", flush=True)
    cases = [state["cases"][case_id] for case_id in case_ids]
    summary = write_reports(prefix, cases, legacy_adapter=bool(args.legacy_oof),
                            replace=args.replace_reports)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
