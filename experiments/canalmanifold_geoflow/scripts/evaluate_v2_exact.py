#!/usr/bin/env python3
"""Exact fair evaluation of raw, R1/R2, and corrected GeoFlow outputs."""

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
from scipy.stats import binomtest
from skimage.morphology import skeletonize

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from canalmanifold.chart import native_local_numpy, reflect_surface_numpy
from canalmanifold.data import TubeCacheDataset
from canalmanifold.displacement import (
    normal_displacement_decode,
    remove_small_components,
    tube_state_to_radial_displacement,
)
from canalmanifold.evaluate import _frame_from_archive, _predict_state
from canalmanifold.geoflow_train import geoflow_heun_rollout, load_geoflow_model
from canalmanifold.io import load_label_on_reference, load_probability_channels
from canalmanifold.manifest import load_splits
from canalmanifold.metrics import connected_components, dice_score, hd95_mm
from canalmanifold.surface_train import (
    load_surface_model,
    surface_heun_rollout,
)
from canalmanifold.train import load_trained_model
from canalmanifold.tube_state import decode_mask


def crop_to_union(*masks: np.ndarray, pad: int = 2) -> tuple[slice, ...]:
    union = np.logical_or.reduce([np.asarray(mask, dtype=bool) for mask in masks])
    coordinates = np.argwhere(union)
    if not len(coordinates):
        return tuple(slice(0, size) for size in union.shape)
    lower = np.maximum(coordinates.min(axis=0) - int(pad), 0)
    upper = np.minimum(coordinates.max(axis=0) + 1 + int(pad), union.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def cldice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    crop = crop_to_union(prediction, target)
    prediction, target = prediction[crop], target[crop]
    pred_skeleton = np.asarray(skeletonize(prediction, method="lee"), dtype=bool)
    target_skeleton = np.asarray(skeletonize(target, method="lee"), dtype=bool)
    precision = float(target[pred_skeleton].mean()) if pred_skeleton.any() else 0.0
    sensitivity = (
        float(prediction[target_skeleton].mean()) if target_skeleton.any() else 0.0
    )
    denominator = precision + sensitivity
    return 0.0 if denominator == 0.0 else 2.0 * precision * sensitivity / denominator


def add_metrics(
    row: dict[str, object],
    prefix: str,
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> None:
    row[f"{prefix}_dice"] = dice_score(prediction, target)
    row[f"{prefix}_hd95_mm"] = hd95_mm(prediction, target, spacing)
    row[f"{prefix}_cldice"] = cldice_score(prediction, target)
    row[f"{prefix}_components"] = connected_components(prediction)
    row[f"{prefix}_volume_ratio"] = float(
        prediction.sum() / max(int(target.sum()), 1)
    )


def patient_bootstrap(
    frame: pd.DataFrame,
    method: str,
    metric: str,
    *,
    baseline: str = "raw_pp",
    samples: int = 5000,
    seed: int = 20260831,
) -> dict[str, float | int]:
    columns = ["case_id", f"{baseline}_{metric}", f"{method}_{metric}"]
    patient = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    patient = patient.groupby("case_id", as_index=False).mean(numeric_only=True)
    delta = (
        patient[f"{method}_{metric}"] - patient[f"{baseline}_{metric}"]
    ).to_numpy()
    if not len(delta):
        return {
            "patients": 0,
            "mean_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(int(samples), len(delta)))
    means = delta[draws].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "patients": len(delta),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def summarize(
    frame: pd.DataFrame,
    methods: list[str],
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "patients": int(frame.case_id.nunique()),
        "sides": len(frame),
        "primary_baseline": "raw_pp",
    }
    for method in methods:
        for metric in ("dice", "hd95_mm", "cldice", "components", "volume_ratio"):
            values = frame[f"{method}_{metric}"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            result[f"{method}_{metric}_mean"] = (
                float(finite.mean()) if len(finite) else float("inf")
            )
        result[f"{method}_component_error_mean"] = float(
            np.abs(frame[f"{method}_components"] - frame["target_components"]).mean()
        )
        if method not in {"raw", "raw_pp", "q0_tube"}:
            improved = int((frame[f"{method}_dice"] > frame.raw_pp_dice).sum())
            tied = int((frame[f"{method}_dice"] == frame.raw_pp_dice).sum())
            trials = len(frame) - tied
            result[f"{method}_dice_improved_sides"] = improved
            result[f"{method}_dice_tied_sides"] = tied
            result[f"{method}_paired_sign_p_two_sided"] = (
                float(binomtest(improved, trials, 0.5).pvalue) if trials else 1.0
            )
            result[f"{method}_patient_bootstrap_delta_dice"] = patient_bootstrap(
                frame,
                method,
                "dice",
                samples=bootstrap_samples,
            )
            result[f"{method}_patient_bootstrap_delta_hd95_mm"] = patient_bootstrap(
                frame,
                method,
                "hd95_mm",
                samples=bootstrap_samples,
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--tube-checkpoint")
    parser.add_argument("--surface-checkpoint")
    parser.add_argument("--geoflow-checkpoint")
    parser.add_argument("--all-manifest-cases", action="store_true")
    parser.add_argument("--cohort", default="internal")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--heun-steps",
        type=int,
        help="Override training.heun_steps for an NFE ablation",
    )
    args = parser.parse_args()
    if not any(
        (args.tube_checkpoint, args.surface_checkpoint, args.geoflow_checkpoint)
    ):
        raise ValueError(
            "Provide at least one of --tube-checkpoint / --surface-checkpoint / "
            "--geoflow-checkpoint"
        )

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["paths"]["manifest_file"]).read_text(encoding="utf-8"))
    items = {item["case_id"]: item for item in manifest["cases"]}
    case_ids = (
        sorted(items)
        if args.all_manifest_cases
        else load_splits(config["paths"]["splits_file"])[args.fold]["val"]
    )
    dataset = TubeCacheDataset(
        config["paths"]["cache_dir"],
        case_ids,
        exclude_fallback=False,
        cache_in_ram=False,
    )
    if len(dataset) != 2 * len(case_ids):
        raise RuntimeError(f"Expected {2 * len(case_ids)} L/R shards, found {len(dataset)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tube_model = tube_checkpoint = None
    if args.tube_checkpoint:
        tube_model, tube_checkpoint = load_trained_model(args.tube_checkpoint, device)
    surface_model = surface_checkpoint = None
    if args.surface_checkpoint:
        surface_model, surface_checkpoint = load_surface_model(
            args.surface_checkpoint, device
        )
    geoflow_model = geoflow_checkpoint = None
    if args.geoflow_checkpoint:
        geoflow_model, geoflow_checkpoint = load_geoflow_model(
            args.geoflow_checkpoint, device
        )
    heun_steps = int(
        args.heun_steps
        if args.heun_steps is not None
        else config.get("training", {}).get("heun_steps", 4)
    )
    if heun_steps < 1:
        raise ValueError("--heun-steps must be >= 1")
    threshold = float(config.get("geometry", {}).get("probability_threshold", 0.5))
    evaluation = config.get("evaluation", {})
    minimum_component_volume = float(
        evaluation.get("minimum_component_volume_mm3", 0.27)
    )
    maximum_displacement = float(
        evaluation.get("normal_displacement_limit_mm", 1.0)
    )
    rows: list[dict[str, object]] = []
    cached_case = None
    cached_label = cached_left = cached_right = None
    methods = ["raw", "raw_pp", "q0_tube"]
    if tube_model is not None:
        methods.extend(("r1_tube", "r1_normal", "r1_normal_pp"))
    if surface_model is not None:
        methods.extend(("r2_normal", "r2_normal_pp"))
    if geoflow_model is not None:
        methods.extend(("gbf_newton_normal", "gbf_newton_normal_pp"))

    for index in range(len(dataset)):
        sample = dataset[index]
        case_id, side = str(sample["case_id"]), str(sample["side"])
        item = items[case_id]
        with np.load(sample["path"], allow_pickle=False) as archive:
            shape = tuple(map(int, archive["volume_shape"]))
            affine = archive["affine"].astype(np.float64)
            spacing = tuple(map(float, archive["spacing_mm"]))
            frame = _frame_from_archive(archive)
            q1_ceiling = float(archive["q1_ceiling_dice"])
            q1_free_ceiling = float(archive["q1_free_ceiling_dice"])
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
        raw_pp = remove_small_components(
            coarse, spacing, minimum_volume_mm3=minimum_component_volume
        )
        q0_local_model = sample["q0_local"].numpy()
        q0_global = sample["q0_global"].numpy()
        q0_local_native = native_local_numpy(q0_local_model, side)
        q0_tube = decode_mask(q0_local_native, q0_global, frame, shape, affine)
        predictions: dict[str, np.ndarray] = {
            "raw": coarse.astype(np.uint8),
            "raw_pp": raw_pp,
            "q0_tube": q0_tube,
        }

        if tube_model is not None and tube_checkpoint is not None:
            predicted_local_model, predicted_global = _predict_state(
                tube_model, tube_checkpoint, sample, device, heun_steps
            )
            predicted_local_native = native_local_numpy(predicted_local_model, side)
            predictions["r1_tube"] = decode_mask(
                predicted_local_native, predicted_global, frame, shape, affine
            )
            h_native, q0_endpoints, predicted_endpoints = tube_state_to_radial_displacement(
                q0_local_native,
                q0_global,
                predicted_local_native,
                predicted_global,
                n_angles=int(config.get("geometry", {}).get("angles", 32)),
            )
            r1_normal = normal_displacement_decode(
                coarse,
                h_native,
                frame,
                affine,
                spacing,
                q0_endpoints_mm=q0_endpoints,
                predicted_endpoints_mm=predicted_endpoints,
                max_abs_displacement_mm=maximum_displacement,
            )
            predictions["r1_normal"] = r1_normal
            predictions["r1_normal_pp"] = remove_small_components(
                r1_normal,
                spacing,
                minimum_volume_mm3=minimum_component_volume,
            )

        if surface_model is not None and surface_checkpoint is not None:
            tensors = {
                key: value[None].to(device) if torch.is_tensor(value) else value
                for key, value in sample.items()
            }
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                h_model, predicted_endpoint = surface_heun_rollout(
                    surface_model,
                    tensors["q0_ray_radii_mm"],
                    tensors["q0_global"][:, 1:3],
                    tensors["shell"],
                    tensors["side_id"],
                    tensors["station_mask"],
                    steps=heun_steps,
                )
            h_native = h_model[0].float().cpu().numpy()
            if side == "R":
                h_native = reflect_surface_numpy(h_native)
            predicted_endpoints = tuple(
                map(float, predicted_endpoint[0].float().cpu().numpy())
            )
            r2_normal = normal_displacement_decode(
                coarse,
                h_native,
                frame,
                affine,
                spacing,
                q0_endpoints_mm=(float(q0_global[1]), float(q0_global[2])),
                predicted_endpoints_mm=predicted_endpoints,
                max_abs_displacement_mm=maximum_displacement,
            )
            predictions["r2_normal"] = r2_normal
            predictions["r2_normal_pp"] = remove_small_components(
                r2_normal,
                spacing,
                minimum_volume_mm3=minimum_component_volume,
            )

        geoflow_diagnostics: dict[str, float] = {}
        if geoflow_model is not None and geoflow_checkpoint is not None:
            tensors = {
                key: value[None].to(device) if torch.is_tensor(value) else value
                for key, value in sample.items()
            }
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                h_model, predicted_endpoint, rollout = geoflow_heun_rollout(
                    geoflow_model,
                    tensors["q0_ray_radii_mm"],
                    tensors["q0_global"][:, 1:3],
                    tensors["shell"],
                    tensors["side_id"],
                    tensors["station_mask"],
                    tensors.get("arc_mm"),
                    steps=heun_steps,
                    return_diagnostics=True,
                )
            geoflow_diagnostics = {
                f"gbf_{key}": float(value)
                for key, value in rollout.items()
            }
            h_native = h_model[0].float().cpu().numpy()
            if side == "R":
                h_native = reflect_surface_numpy(h_native)
            predicted_endpoints = tuple(
                map(float, predicted_endpoint[0].float().cpu().numpy())
            )
            gbf_normal = normal_displacement_decode(
                coarse,
                h_native,
                frame,
                affine,
                spacing,
                q0_endpoints_mm=(float(q0_global[1]), float(q0_global[2])),
                predicted_endpoints_mm=predicted_endpoints,
                max_abs_displacement_mm=maximum_displacement,
            )
            predictions["gbf_newton_normal"] = gbf_normal
            predictions["gbf_newton_normal_pp"] = remove_small_components(
                gbf_normal,
                spacing,
                minimum_volume_mm3=minimum_component_volume,
            )

        row: dict[str, object] = {
            "case_id": case_id,
            "side": side,
            "key": f"{case_id}_{side}",
            "cohort": args.cohort,
            "q0_identity_dice": identity_dice,
            "q1_m8_ceiling_dice": q1_ceiling,
            "q1_free_surface_ceiling_dice": q1_free_ceiling,
            "target_components": connected_components(target),
            "tube_checkpoint_epoch": (
                int(tube_checkpoint["epoch"]) + 1
                if tube_checkpoint is not None
                else -1
            ),
            "surface_checkpoint_epoch": (
                int(surface_checkpoint["epoch"]) + 1
                if surface_checkpoint is not None
                else -1
            ),
            "geoflow_checkpoint_epoch": (
                int(geoflow_checkpoint["epoch"]) + 1
                if geoflow_checkpoint is not None
                else -1
            ),
            **geoflow_diagnostics,
        }
        for method, prediction in predictions.items():
            add_metrics(row, method, prediction, target, spacing)
        rows.append(row)
        print(
            f"[V2 EXACT] {len(rows)}/{len(dataset)} {case_id}_{side} "
            f"raw_pp={row['raw_pp_dice']:.5f} "
            + " ".join(
                f"{method}={row[f'{method}_dice']:.5f}"
                for method in (
                    "r1_normal_pp",
                    "r2_normal_pp",
                    "gbf_newton_normal_pp",
                )
                if f"{method}_dice" in row
            ),
            flush=True,
        )

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["case_id", "side"]).reset_index(drop=True)
    frame.to_csv(output / "metrics.csv", index=False)
    payload = {
        "cohort": args.cohort,
        "fold": args.fold,
        "unconditional": True,
        "component_filter_applied_symmetrically": True,
        "minimum_component_volume_mm3": minimum_component_volume,
        "normal_displacement_limit_mm": maximum_displacement,
        "heun_steps": heun_steps,
        "methods": methods,
        "tube_checkpoint_epoch": (
            int(tube_checkpoint["epoch"]) + 1 if tube_checkpoint is not None else None
        ),
        "surface_checkpoint_epoch": (
            int(surface_checkpoint["epoch"]) + 1
            if surface_checkpoint is not None
            else None
        ),
        "geoflow_checkpoint_epoch": (
            int(geoflow_checkpoint["epoch"]) + 1
            if geoflow_checkpoint is not None
            else None
        ),
        "geoflow_representation": (
            geoflow_checkpoint.get("representation")
            if geoflow_checkpoint is not None
            else None
        ),
        "geoflow_evidence_version": (
            geoflow_checkpoint.get("training_contract", {}).get("evidence_version")
            if geoflow_checkpoint is not None
            else None
        ),
        "summary": summarize(
            frame,
            methods,
            bootstrap_samples=args.bootstrap_samples,
        ),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
