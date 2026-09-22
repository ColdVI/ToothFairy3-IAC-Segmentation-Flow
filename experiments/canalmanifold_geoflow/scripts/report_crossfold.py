#!/usr/bin/env python3
"""Aggregate only folds that did not participate in epoch selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument(
        "--metrics",
        action="append",
        required=True,
        help="Repeat as FOLD=/absolute/path/to/metrics.csv",
    )
    parser.add_argument("--method", default="r2_normal_pp")
    parser.add_argument("--baseline", default="raw_pp")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    expected_folds = set(map(int, lock["report_folds"]))
    selected_fold = int(lock["selection_fold"])
    locked_epoch = int(lock["locked_epoch"])
    frames = []
    supplied_folds = set()
    for specification in args.metrics:
        fold_text, path_text = specification.split("=", 1)
        fold = int(fold_text)
        if fold == selected_fold:
            raise ValueError("Selection-fold metrics cannot enter the primary report")
        if fold not in expected_folds:
            raise ValueError(f"Fold {fold} is not in locked report_folds={sorted(expected_folds)}")
        frame = pd.read_csv(Path(path_text).expanduser())
        epoch_columns = {
            "linear": "tube_checkpoint_epoch",
            "surface_h": "surface_checkpoint_epoch",
            "geoflow_newton": "geoflow_checkpoint_epoch",
        }
        try:
            epoch_column = epoch_columns[lock["representation"]]
        except KeyError as error:
            raise ValueError(
                f"Unsupported representation in lock: {lock['representation']}"
            ) from error
        observed = set(frame[epoch_column].astype(int).unique())
        if observed != {locked_epoch}:
            raise RuntimeError(
                f"Fold {fold} uses epochs {sorted(observed)}, locked epoch is {locked_epoch}"
            )
        frame["refinement_fold"] = fold
        frames.append(frame)
        supplied_folds.add(fold)
    if supplied_folds != expected_folds:
        raise RuntimeError(
            f"Need all locked report folds {sorted(expected_folds)}, got {sorted(supplied_folds)}"
        )
    frame = pd.concat(frames, ignore_index=True)
    method_column = f"{args.method}_dice"
    baseline_column = f"{args.baseline}_dice"
    if method_column not in frame or baseline_column not in frame:
        raise KeyError(f"Missing {baseline_column} or {method_column}")
    side_delta = (frame[method_column] - frame[baseline_column]).to_numpy(dtype=float)
    patient = frame.groupby("case_id", as_index=False)[
        [method_column, baseline_column]
    ].mean()
    patient_delta = (
        patient[method_column] - patient[baseline_column]
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(20260831)
    draws = rng.integers(
        0,
        len(patient_delta),
        size=(int(args.bootstrap_samples), len(patient_delta)),
    )
    bootstrap = patient_delta[draws].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    non_ties = side_delta[side_delta != 0.0]
    improved = int((non_ties > 0.0).sum())
    fold_deltas = {
        str(fold): float(
            (subset[method_column] - subset[baseline_column]).mean()
        )
        for fold, subset in frame.groupby("refinement_fold")
    }
    metric_deltas = {}
    for metric in ("dice", "hd95_mm", "cldice", "components", "volume_ratio"):
        method_metric = f"{args.method}_{metric}"
        baseline_metric = f"{args.baseline}_{metric}"
        if method_metric not in frame or baseline_metric not in frame:
            continue
        usable = frame[
            ["case_id", "refinement_fold", method_metric, baseline_metric]
        ].replace([np.inf, -np.inf], np.nan).dropna()
        if usable.empty:
            continue
        side_metric_delta = (
            usable[method_metric] - usable[baseline_metric]
        ).to_numpy(dtype=float)
        patient_metric = usable.groupby("case_id", as_index=False)[
            [method_metric, baseline_metric]
        ].mean()
        patient_metric_delta = (
            patient_metric[method_metric] - patient_metric[baseline_metric]
        ).to_numpy(dtype=float)
        metric_draws = rng.integers(
            0,
            len(patient_metric_delta),
            size=(int(args.bootstrap_samples), len(patient_metric_delta)),
        )
        metric_bootstrap = patient_metric_delta[metric_draws].mean(axis=1)
        metric_low, metric_high = np.quantile(metric_bootstrap, (0.025, 0.975))
        metric_deltas[metric] = {
            "patients": int(len(patient_metric_delta)),
            "sides": int(len(side_metric_delta)),
            "mean_side_delta": float(side_metric_delta.mean()),
            "mean_patient_delta": float(patient_metric_delta.mean()),
            "patient_bootstrap_ci95": [float(metric_low), float(metric_high)],
            "fold_mean_delta": {
                str(fold): float(
                    (subset[method_metric] - subset[baseline_metric]).mean()
                )
                for fold, subset in usable.groupby("refinement_fold")
            },
            "improvement_direction": {
                "dice": "positive",
                "hd95_mm": "negative",
                "cldice": "positive",
                "components": "toward_target_components",
                "volume_ratio": "toward_1",
            }[metric],
        }
    payload = {
        "protocol": lock["protocol"],
        "selection_fold": selected_fold,
        "reported_folds": sorted(supplied_folds),
        "locked_epoch": locked_epoch,
        "method": args.method,
        "baseline": args.baseline,
        "patients": int(frame.case_id.nunique()),
        "sides": len(frame),
        "mean_side_delta_dice": float(side_delta.mean()),
        "mean_patient_delta_dice": float(patient_delta.mean()),
        "patient_bootstrap_ci95": [float(low), float(high)],
        "side_sign_improved": improved,
        "side_sign_trials_excluding_ties": len(non_ties),
        "side_sign_p_two_sided": (
            float(binomtest(improved, len(non_ties), 0.5).pvalue)
            if len(non_ties)
            else 1.0
        ),
        "fold_mean_delta_dice": fold_deltas,
        "folds_with_positive_delta": int(sum(value > 0.0 for value in fold_deltas.values())),
        "metric_deltas": metric_deltas,
        "checkpoint_selection_optimism_removed": True,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
