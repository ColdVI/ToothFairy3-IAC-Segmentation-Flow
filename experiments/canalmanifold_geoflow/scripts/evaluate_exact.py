#!/usr/bin/env python3
"""Exact voxel-domain evaluation for raw, tube, and prior-anchored outputs.

Primary evaluation is unconditional: no representation-fidelity fallback is
allowed.  Right-side model-chart states are mapped back to the native physical
Bishop frame before rasterisation.  Confidence intervals resample patients,
not the two correlated side shards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import distance_transform_edt
from scipy.stats import trim_mean
from skimage.morphology import skeletonize

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from canalmanifold.chart import native_local_numpy
from canalmanifold.data import TubeCacheDataset
from canalmanifold.evaluate import _frame_from_archive, _predict_state
from canalmanifold.io import load_label_on_reference, load_probability_channels
from canalmanifold.manifest import load_splits
from canalmanifold.metrics import connected_components, dice_score, hd95_mm
from canalmanifold.train import load_trained_model
from canalmanifold.tube_state import decode_mask

METHODS = ("raw", "q0_tube", "flow_tube", "flow_anchor")


def crop_to_union(*masks: np.ndarray, pad: np.ndarray | int = 0) -> tuple[slice, ...]:
    union = np.logical_or.reduce([np.asarray(mask, dtype=bool) for mask in masks])
    coordinates = np.argwhere(union)
    if not len(coordinates):
        return tuple(slice(0, size) for size in union.shape)
    pad_array = np.broadcast_to(np.asarray(pad, dtype=int), (union.ndim,))
    lower = np.maximum(coordinates.min(axis=0) - pad_array, 0)
    upper = np.minimum(coordinates.max(axis=0) + 1 + pad_array, union.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def signed_distance(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return (
        distance_transform_edt(~mask, sampling=spacing)
        - distance_transform_edt(mask, sampling=spacing)
    ).astype(np.float32)


def prior_anchored_decode(
    coarse: np.ndarray,
    q0_tube: np.ndarray,
    predicted_tube: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    margin_mm: float = 12.0,
) -> np.ndarray:
    """phi0 + psi(qhat) - psi(q0); identity is exactly the raw prior."""
    padding = np.ceil(float(margin_mm) / np.maximum(spacing, 1e-6)).astype(int)
    crop = crop_to_union(coarse, q0_tube, predicted_tube, pad=padding)
    coarse_crop = np.asarray(coarse[crop], dtype=bool)
    q0_crop = np.asarray(q0_tube[crop], dtype=bool)
    predicted_crop = np.asarray(predicted_tube[crop], dtype=bool)
    anchored = (
        signed_distance(coarse_crop, spacing)
        + signed_distance(predicted_crop, spacing)
        - signed_distance(q0_crop, spacing)
    )
    output = np.asarray(coarse, dtype=np.uint8).copy()
    # Zero is fixed a priori; no validation- or test-tuned threshold is used.
    output[crop] = (anchored <= 0.0).astype(np.uint8)
    return output


def cldice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    crop = crop_to_union(prediction, target, pad=2)
    prediction = prediction[crop]
    target = target[crop]
    skeleton_prediction = np.asarray(
        skeletonize(prediction, method="lee"), dtype=bool
    )
    skeleton_target = np.asarray(skeletonize(target, method="lee"), dtype=bool)
    precision_denominator = int(skeleton_prediction.sum())
    sensitivity_denominator = int(skeleton_target.sum())
    topology_precision = (
        float(target[skeleton_prediction].mean()) if precision_denominator else 0.0
    )
    topology_sensitivity = (
        float(prediction[skeleton_target].mean()) if sensitivity_denominator else 0.0
    )
    denominator = topology_precision + topology_sensitivity
    return 0.0 if denominator == 0 else 2.0 * topology_precision * topology_sensitivity / denominator


def add_metrics(row: dict, prefix: str, prediction: np.ndarray, target: np.ndarray, spacing) -> None:
    row[f"{prefix}_dice"] = dice_score(prediction, target)
    row[f"{prefix}_hd95_mm"] = hd95_mm(prediction, target, spacing)
    row[f"{prefix}_cldice"] = cldice_score(prediction, target)
    row[f"{prefix}_components"] = connected_components(prediction)
    row[f"{prefix}_volume_ratio"] = float(prediction.sum() / max(int(target.sum()), 1))


def bootstrap_patient_delta(
    frame: pd.DataFrame,
    metric: str,
    *,
    baseline: str = "raw",
    method: str = "flow_anchor",
    samples: int = 5000,
    seed: int = 20260830,
) -> dict[str, float | int]:
    columns = ["case_id", f"{baseline}_{metric}", f"{method}_{metric}"]
    patient = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    patient = patient.groupby("case_id", as_index=False).mean(numeric_only=True)
    delta = (patient[f"{method}_{metric}"] - patient[f"{baseline}_{metric}"]).to_numpy()
    if not len(delta):
        return {"patients": 0, "mean_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(int(samples), len(delta)))
    bootstrapped = delta[draws].mean(axis=1)
    low, high = np.quantile(bootstrapped, (0.025, 0.975))
    return {
        "patients": len(delta),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def summarize_subset(name: str, frame: pd.DataFrame, bootstrap_samples: int) -> dict:
    summary: dict[str, object] = {
        "name": name,
        "patients": int(frame.case_id.nunique()),
        "sides": len(frame),
    }
    for method in METHODS:
        for metric in ("dice", "hd95_mm", "cldice", "components", "volume_ratio"):
            values = frame[f"{method}_{metric}"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summary[f"{method}_{metric}_mean"] = float(finite.mean()) if len(finite) else float("inf")
            summary[f"{method}_{metric}_median"] = float(np.median(finite)) if len(finite) else float("inf")
            summary[f"{method}_{metric}_trimmed10"] = (
                float(trim_mean(finite, 0.10)) if len(finite) else float("inf")
            )
            summary[f"{method}_{metric}_finite_fraction"] = float(len(finite) / max(len(values), 1))
        summary[f"{method}_component_error_mean"] = float(
            np.abs(frame[f"{method}_components"] - frame["target_components"]).mean()
        )
    summary["flow_anchor_dice_improved_sides"] = int(
        (frame.flow_anchor_dice > frame.raw_dice).sum()
    )
    summary["flow_anchor_hd95_improved_sides"] = int(
        (frame.flow_anchor_hd95_mm < frame.raw_hd95_mm).sum()
    )
    summary["patient_bootstrap_delta_dice"] = bootstrap_patient_delta(
        frame, "dice", samples=bootstrap_samples
    )
    summary["patient_bootstrap_delta_hd95_mm"] = bootstrap_patient_delta(
        frame, "hd95_mm", samples=bootstrap_samples
    )
    summary["patient_bootstrap_delta_cldice"] = bootstrap_patient_delta(
        frame, "cldice", samples=bootstrap_samples
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-manifest-cases", action="store_true")
    parser.add_argument("--cohort", default="internal_fold0")
    parser.add_argument("--sensitivity-exclude-case")
    parser.add_argument("--sensitivity-exclude-key")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--anchor-margin-mm", type=float, default=12.0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["paths"]["manifest_file"]).read_text(encoding="utf-8"))
    items = {item["case_id"]: item for item in manifest["cases"]}
    if args.all_manifest_cases:
        case_ids = sorted(items)
    else:
        case_ids = load_splits(config["paths"]["splits_file"])[args.fold]["val"]
    dataset = TubeCacheDataset(
        config["paths"]["cache_dir"], case_ids, exclude_fallback=False, cache_in_ram=False
    )
    expected = 2 * len(case_ids)
    if len(dataset) != expected:
        raise RuntimeError(f"Expected {expected} L/R shards, found {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(args.checkpoint, device)
    heun_steps = int(config.get("training", {}).get("heun_steps", 4))
    threshold = float(config.get("geometry", {}).get("probability_threshold", 0.5))
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "metrics.partial.csv"
    completed_metrics = output / "metrics.csv"
    if partial.exists():
        rows = pd.read_csv(partial).to_dict("records")
    elif completed_metrics.exists():
        rows = pd.read_csv(completed_metrics).to_dict("records")
    else:
        rows = []
    expected_epoch = int(checkpoint["epoch"]) + 1
    if rows:
        existing_epochs = {int(row["checkpoint_epoch"]) for row in rows}
        existing_modes = {str(row["mode"]) for row in rows}
        if existing_epochs != {expected_epoch} or existing_modes != {checkpoint["mode"]}:
            raise RuntimeError(
                "Existing metrics belong to another checkpoint: "
                f"epochs={existing_epochs}, modes={existing_modes}; "
                f"requested epoch={expected_epoch}, mode={checkpoint['mode']}"
            )
    completed = {(str(row["case_id"]), str(row["side"])) for row in rows}

    cached_case = None
    cached_label = cached_left = cached_right = None
    for index in range(len(dataset)):
        sample = dataset[index]
        case_id, side = str(sample["case_id"]), str(sample["side"])
        if (case_id, side) in completed:
            continue
        item = items[case_id]
        with np.load(sample["path"], allow_pickle=False) as archive:
            shape = tuple(map(int, archive["volume_shape"]))
            affine = archive["affine"].astype(np.float64)
            spacing = tuple(map(float, archive["spacing_mm"]))
            frame = _frame_from_archive(archive)
            q1_ceiling = float(archive["q1_ceiling_dice"])
            identity_dice = float(archive["q0_identity_dice"])

        if cached_case != case_id:
            reference = SimpleNamespace(shape=shape, affine=affine)
            cached_label = load_label_on_reference(item["label"], reference)
            cached_left, cached_right, _ = load_probability_channels(
                item["probability"],
                shape,
                int(manifest["left_probability_channel"]),
                int(manifest["right_probability_channel"]),
            )
            cached_case = case_id
        label_id = int(
            manifest["left_label_id"] if side == "L" else manifest["right_label_id"]
        )
        target = cached_label == label_id
        coarse = (cached_left if side == "L" else cached_right) >= threshold

        q0_local = native_local_numpy(sample["q0_local"].numpy(), side)
        q0_global = sample["q0_global"].numpy()
        q0_tube = decode_mask(q0_local, q0_global, frame, shape, affine)

        predicted_local, predicted_global = _predict_state(
            model, checkpoint, sample, device, heun_steps
        )
        predicted_local = native_local_numpy(predicted_local, side)
        flow_tube = decode_mask(predicted_local, predicted_global, frame, shape, affine)
        flow_anchor = prior_anchored_decode(
            coarse,
            q0_tube,
            flow_tube,
            spacing,
            margin_mm=args.anchor_margin_mm,
        )

        row = {
            "case_id": case_id,
            "side": side,
            "key": f"{case_id}_{side}",
            "cohort": args.cohort,
            "mode": checkpoint["mode"],
            "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
            "nfe": 1 if checkpoint["mode"] == "direct" else 2 * heun_steps,
            "q0_identity_dice": identity_dice,
            "q1_representation_ceiling_dice": q1_ceiling,
            "target_components": connected_components(target),
        }
        add_metrics(row, "raw", coarse, target, spacing)
        add_metrics(row, "q0_tube", q0_tube, target, spacing)
        add_metrics(row, "flow_tube", flow_tube, target, spacing)
        add_metrics(row, "flow_anchor", flow_anchor, target, spacing)
        rows.append(row)
        temporary = output / "metrics.partial.tmp.csv"
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(partial)
        print(
            f"[EXACT] {len(rows)}/{len(dataset)} {case_id}_{side} "
            f"raw={row['raw_dice']:.5f} anchor={row['flow_anchor_dice']:.5f}",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values(["case_id", "side"]).reset_index(drop=True)
    frame.to_csv(output / "metrics.csv", index=False)
    subsets = {"all": frame}
    if args.sensitivity_exclude_case:
        subsets[f"without_case_{args.sensitivity_exclude_case}"] = frame[
            frame.case_id != args.sensitivity_exclude_case
        ]
    if args.sensitivity_exclude_key:
        subsets[f"without_key_{args.sensitivity_exclude_key}"] = frame[
            frame.key != args.sensitivity_exclude_key
        ]
    summaries = {
        name: summarize_subset(name, subset, args.bootstrap_samples)
        for name, subset in subsets.items()
    }
    payload = {
        "cohort": args.cohort,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mode": checkpoint["mode"],
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "unconditional": True,
        "prior_anchor_threshold": 0.0,
        "subsets": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
