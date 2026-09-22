#!/usr/bin/env python3
"""Estimate train-only geometric trust limits for a refinement fold."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from canalmanifold.constants import HARMONIC_START, HARMONICS, LOCAL_DIM
from canalmanifold.data import split_datasets


def surface_rmse(q0_local, q0_global, q1_local, q1_global, mask, n_angles=64):
    if not np.any(mask):
        raise ValueError("surface_rmse is undefined for an empty station mask")
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    basis = np.column_stack(
        [
            function(harmonic * angles)
            for harmonic in HARMONICS
            for function in (np.cos, np.sin)
        ]
    )

    def points(local, global_state):
        log_radius = (
            global_state[0]
            + local[:, 2, None]
            + local[:, HARMONIC_START:LOCAL_DIM] @ basis.T
        )
        radius = np.exp(np.clip(log_radius, math.log(0.12), math.log(8.0)))
        x = local[:, 0, None] + radius * np.cos(angles)[None]
        y = local[:, 1, None] + radius * np.sin(angles)[None]
        return x, y

    x0, y0 = points(q0_local, q0_global)
    x1, y1 = points(q1_local, q1_global)
    squared = (x1 - x0) ** 2 + (y1 - y0) ** 2
    return float(np.sqrt(squared[mask].mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_dataset, _ = split_datasets(config, args.fold)

    surface_values = []
    endpoint_values = []
    centre_values = []
    empty_loss_mask_shards = 0
    for index in range(len(train_dataset)):
        sample = train_dataset[index]
        q0_local = sample["q0_local"].numpy()
        q1_local = sample["q1_local"].numpy()
        q0_global = sample["q0_global"].numpy()
        q1_global = sample["q1_global"].numpy()
        mask = sample["loss_mask"].numpy().astype(bool)
        # The physical surface displacement is undefined when q0 and q1 have
        # no common valid stations. Such a shard remains part of training and
        # evaluation, but it cannot contribute a value to this train-only
        # empirical trust statistic. Endpoint displacement remains defined.
        if np.any(mask):
            surface_values.append(
                surface_rmse(q0_local, q0_global, q1_local, q1_global, mask)
            )
            centre_values.extend(
                np.linalg.norm(q1_local[mask, 0:2], axis=1).tolist()
            )
        else:
            empty_loss_mask_shards += 1
        endpoint_values.extend(np.abs(q1_global[1:3] - q0_global[1:3]).tolist())

    surface_values = np.asarray(surface_values, dtype=np.float64)
    endpoint_values = np.asarray(endpoint_values, dtype=np.float64)
    centre_values = np.asarray(centre_values, dtype=np.float64)
    surface_values = surface_values[np.isfinite(surface_values)]
    endpoint_values = endpoint_values[np.isfinite(endpoint_values)]
    centre_values = centre_values[np.isfinite(centre_values)]
    if surface_values.size == 0:
        raise RuntimeError("No finite non-empty surface displacement values were found")
    if endpoint_values.size == 0:
        raise RuntimeError("No finite endpoint displacement values were found")
    if centre_values.size == 0:
        raise RuntimeError("No finite centre displacement values were found")

    quantile = float(args.quantile)
    result = {
        "fold": int(args.fold),
        "train_shards": len(train_dataset),
        "surface_shards_used": int(surface_values.size),
        "empty_loss_mask_shards": int(empty_loss_mask_shards),
        "quantile": quantile,
        "trust_surface_rmse_mm": float(np.quantile(surface_values, quantile)),
        "trust_endpoint_mm": float(np.quantile(endpoint_values, quantile)),
        "observed_q1_center_norm_mm": float(np.quantile(centre_values, quantile)),
        "surface_rmse_max_mm": float(np.max(surface_values)),
        "endpoint_max_mm": float(np.max(endpoint_values)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
