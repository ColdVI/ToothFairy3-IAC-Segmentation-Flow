#!/usr/bin/env python3
"""A0/A1/A2 audit: raw prior, q0 tube, and train-mean transport."""

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

from canalmanifold.chart import native_local_numpy
from canalmanifold.constants import LOCAL_DIM
from canalmanifold.data import TubeCacheDataset
from canalmanifold.evaluate import _frame_from_archive
from canalmanifold.io import load_label_on_reference, load_nifti
from canalmanifold.manifest import load_splits
from canalmanifold.metrics import dice_score, hd95_mm
from canalmanifold.tube_state import decode_mask


def project_gauge(local, global_state, mask):
    local = np.asarray(local, dtype=np.float32).copy()
    global_state = np.asarray(global_state, dtype=np.float32).copy()
    mean = float(local[mask, 2].mean()) if np.any(mask) else 0.0
    local[:, 2] -= mean
    global_state[0] += mean
    return local, global_state


def train_mean_transport(dataset):
    local_sum = np.zeros((160, LOCAL_DIM), dtype=np.float64)
    local_count = np.zeros((160, 1), dtype=np.float64)
    global_values = []
    for index in range(len(dataset)):
        sample = dataset[index]
        delta = (sample["q1_local"] - sample["q0_local"]).numpy()
        mask = sample["loss_mask"].numpy().astype(bool)
        local_sum[mask] += delta[mask]
        local_count[mask] += 1.0
        global_values.append((sample["q1_global"] - sample["q0_global"]).numpy())
    mean_local = local_sum / np.maximum(local_count, 1.0)
    mean_global = np.mean(global_values, axis=0)
    return mean_local.astype(np.float32), mean_global.astype(np.float32)


def summarize(frame):
    return {
        "samples": len(frame),
        "A0_raw_dice": float(frame.A0_raw_dice.mean()),
        "A1_q0_tube_dice": float(frame.A1_q0_tube_dice.mean()),
        "A2_mean_transport_dice": float(frame.A2_mean_transport_dice.mean()),
        "A0_raw_hd95_mm": float(frame.A0_raw_hd95_mm.mean()),
        "A1_q0_tube_hd95_mm": float(frame.A1_q0_tube_hd95_mm.mean()),
        "A2_mean_transport_hd95_mm": float(frame.A2_mean_transport_hd95_mm.mean()),
        "representation_delta_dice": float(
            (frame.A1_q0_tube_dice - frame.A0_raw_dice).mean()
        ),
        "mean_transport_delta_over_q0_dice": float(
            (frame.A2_mean_transport_dice - frame.A1_q0_tube_dice).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    splits = load_splits(config["paths"]["splits_file"])
    fold = splits[args.fold]
    train_dataset = TubeCacheDataset(
        config["paths"]["cache_dir"], fold["train"], exclude_fallback=False
    )
    val_dataset = TubeCacheDataset(
        config["paths"]["cache_dir"], fold["val"], exclude_fallback=False
    )
    mean_local, mean_global = train_mean_transport(train_dataset)

    manifest = json.loads(
        Path(config["paths"]["manifest_file"]).read_text(encoding="utf-8")
    )
    items = {item["case_id"]: item for item in manifest["cases"]}
    rows = []
    cached_case, cached_image, cached_label = None, None, None

    for index in range(len(val_dataset)):
        sample = val_dataset[index]
        case_id, side = str(sample["case_id"]), str(sample["side"])
        item = items[case_id]
        if cached_case != case_id:
            cached_image = load_nifti(item["image"], dtype=np.float32)
            cached_label = load_label_on_reference(item["label"], cached_image)
            cached_case = case_id
        label_id = int(
            manifest["left_label_id"] if side == "L" else manifest["right_label_id"]
        )
        target = cached_label == label_id

        with np.load(sample["path"], allow_pickle=False) as archive:
            frame = _frame_from_archive(archive)
            shape = tuple(map(int, archive["volume_shape"]))
            affine = archive["affine"].astype(np.float64)
            spacing = tuple(map(float, archive["spacing_mm"]))
            a0_dice = float(archive["base_raw_dice"])
            a0_hd = float(archive["base_raw_hd95_mm"])

        q0_local = sample["q0_local"].numpy()
        q0_global = sample["q0_global"].numpy()
        q0_native = native_local_numpy(q0_local, side)
        q0_mask = decode_mask(q0_native, q0_global, frame, shape, affine)

        a2_local = q0_local + mean_local
        a2_global = q0_global + mean_global
        a2_local, a2_global = project_gauge(
            a2_local, a2_global, sample["station_mask"].numpy().astype(bool)
        )
        a2_native = native_local_numpy(a2_local, side)
        a2_mask = decode_mask(a2_native, a2_global, frame, shape, affine)

        rows.append(
            {
                "case_id": case_id,
                "side": side,
                "A0_raw_dice": a0_dice,
                "A1_q0_tube_dice": dice_score(q0_mask, target),
                "A2_mean_transport_dice": dice_score(a2_mask, target),
                "A0_raw_hd95_mm": a0_hd,
                "A1_q0_tube_hd95_mm": hd95_mm(q0_mask, target, spacing),
                "A2_mean_transport_hd95_mm": hd95_mm(a2_mask, target, spacing),
            }
        )
        print(f"[A0/A1/A2] {index + 1}/{len(val_dataset)} {case_id}_{side}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    summary = summarize(frame)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
