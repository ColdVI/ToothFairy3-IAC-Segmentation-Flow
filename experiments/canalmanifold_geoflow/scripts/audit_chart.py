#!/usr/bin/env python3
"""Zero-training chirality, residual-correlation, and frame-fallback audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from canalmanifold.chart import reflect_local_numpy
from canalmanifold.manifest import load_splits

NAMES = ("d1", "d2", "ell", "a2", "b2", "a3", "b3", "a4", "b4")


def safe_correlation(left: list[float], right: list[float]) -> float:
    left_array, right_array = np.asarray(left), np.asarray(right)
    if len(left_array) < 3 or left_array.std() < 1e-8 or right_array.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split = load_splits(config["paths"]["splits_file"])[args.fold]
    train_cases = set(split["train"])
    cache = Path(config["paths"]["cache_dir"])
    native = {name: ([], []) for name in NAMES}
    canonical = {name: ([], []) for name in NAMES}
    fallback_like = []
    case_rows = []

    for case_id in sorted(train_cases):
        side_data = {}
        for side in ("L", "R"):
            matches = list(cache.glob(f"fold_*/{case_id}_{side}.npz"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one shard for {case_id}_{side}, found {matches}")
            with np.load(matches[0], allow_pickle=False) as archive:
                q0 = archive["q0_local"].astype(np.float32)
                q1 = archive["q1_local"].astype(np.float32)
                q0_valid = archive.get("q0_station_valid", archive["station_mask"]).astype(bool)
                q1_valid = archive.get("q1_station_valid", archive["station_mask"]).astype(bool)
                tangent0 = archive["tangent"].astype(np.float64)[0]
            delta = q1 - q0
            side_data[side] = (delta, q0_valid & q1_valid)
            superior_projection = np.linalg.norm(
                np.asarray([0.0, 0.0, 1.0])
                - tangent0[2] * tangent0
            )
            fallback_like.append(superior_projection < 0.2)

        shared = side_data["L"][1] & side_data["R"][1]
        left_delta = side_data["L"][0][shared]
        right_native = side_data["R"][0][shared]
        right_canonical = reflect_local_numpy(right_native)
        row = {"case_id": case_id, "shared_stations": int(shared.sum())}
        for channel, name in enumerate(NAMES):
            left_values = left_delta[:, channel].tolist()
            native_right_values = right_native[:, channel].tolist()
            canonical_right_values = right_canonical[:, channel].tolist()
            native[name][0].extend(left_values)
            native[name][1].extend(native_right_values)
            canonical[name][0].extend(left_values)
            canonical[name][1].extend(canonical_right_values)
            row[f"native_corr_{name}"] = safe_correlation(left_values, native_right_values)
            row[f"canonical_corr_{name}"] = safe_correlation(left_values, canonical_right_values)
        case_rows.append(row)

    correlations = []
    for name in NAMES:
        correlations.append(
            {
                "component": name,
                "native_lr_correlation": safe_correlation(*native[name]),
                "canonical_lr_correlation": safe_correlation(*canonical[name]),
                "reflection_sign": -1 if name in {"d2", "b2", "b3", "b4"} else 1,
            }
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(correlations).to_csv(output / "component_correlations.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output / "case_correlations.csv", index=False)
    summary = {
        "fold": args.fold,
        "train_cases": len(train_cases),
        "frame_initial_superior_fallback_like_fraction": float(np.mean(fallback_like)),
        "note": (
            "The fallback-like statistic is reconstructed from cached initial tangents; "
            "it tests the same norm<0.2 condition without recomputing centerlines."
        ),
        "component_correlations": correlations,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
