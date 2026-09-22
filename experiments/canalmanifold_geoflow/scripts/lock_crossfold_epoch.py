#!/usr/bin/env python3
"""Lock one epoch on a selection fold before evaluating the other folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--selection-fold", type=int, default=0)
    parser.add_argument(
        "--representation",
        choices=("linear", "surface_h", "geoflow_newton"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_dir = (
        Path(args.run_root).expanduser().resolve()
        / f"fold_{args.selection_fold}"
        / args.representation
    )
    log_path = run_dir / "training_log.csv"
    frame = pd.read_csv(log_path)
    if args.representation == "surface_h":
        primary, tie_breaker = "val_surface_soft_dice", "val_surface_rmse_mm"
    elif args.representation == "geoflow_newton":
        primary, tie_breaker = "val_soft_dice", "val_rmse_mm"
    else:
        primary, tie_breaker = "val_tube_dice", "val_tube_hd95_mm"
    usable = frame[
        np.isfinite(frame[primary].to_numpy(dtype=float))
        & np.isfinite(frame[tie_breaker].to_numpy(dtype=float))
        & (frame.epoch >= 0)
    ].copy()
    if usable.empty:
        raise RuntimeError(f"No selectable rows in {log_path}")
    usable = usable.sort_values(
        [primary, tie_breaker, "epoch"],
        ascending=[False, True, True],
    )
    selected = usable.iloc[0]
    epoch = int(selected.epoch) + 1
    checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Locked checkpoint is absent: {checkpoint}. Set "
            "archive_checkpoint_every_epochs <= selection_every_epochs."
        )
    payload = {
        "protocol": "single-fold epoch selection; report only non-selection folds",
        "selection_fold": int(args.selection_fold),
        "report_folds": [fold for fold in range(5) if fold != args.selection_fold],
        "representation": args.representation,
        "locked_epoch": epoch,
        "selection_primary_metric": primary,
        "selection_primary_value": float(selected[primary]),
        "selection_tie_breaker": tie_breaker,
        "selection_tie_breaker_value": float(selected[tie_breaker]),
        "selection_checkpoint": str(checkpoint),
        "selection_fold_is_not_primary_evidence": True,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
