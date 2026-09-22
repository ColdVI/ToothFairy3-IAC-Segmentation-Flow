#!/usr/bin/env python3
"""Sample an interpretable station-wise R2 safety envelope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from canalmanifold.data import TubeCacheDataset
from canalmanifold.manifest import load_splits
from canalmanifold.surface_train import load_surface_model, surface_heun_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--initial-jitter-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if not 0.0 < args.coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    if args.samples < 8:
        raise ValueError("Use at least eight samples")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    case_ids = load_splits(config["paths"]["splits_file"])[args.fold]["val"]
    dataset = TubeCacheDataset(
        config["paths"]["cache_dir"],
        case_ids,
        exclude_fallback=False,
        cache_in_ram=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_surface_model(args.checkpoint, device)
    steps = int(config.get("training", {}).get("heun_steps", 4))
    lower_q = 0.5 * (1.0 - float(args.coverage))
    upper_q = 1.0 - lower_q
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    output = Path(args.output).expanduser().resolve()
    envelope_dir = output / "envelopes"
    envelope_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        q0_radius = sample["q0_ray_radii_mm"][None].to(device)
        endpoint0 = sample["q0_global"][None, 1:3].to(device)
        shell = sample["shell"][None].to(device)
        side = sample["side_id"][None].to(device)
        station_mask = sample["station_mask"][None].to(device)
        radius_samples, endpoint_samples = [], []
        with torch.no_grad():
            for _ in range(int(args.samples)):
                initial_h = (
                    torch.randn(
                        q0_radius.shape,
                        generator=generator,
                        device=device,
                        dtype=q0_radius.dtype,
                    )
                    * model.h_output_scale[None, None]
                    * float(args.initial_jitter_fraction)
                )
                initial_h = initial_h * station_mask[:, :, None].to(initial_h.dtype)
                sampled_endpoint = endpoint0 + (
                    torch.randn(
                        endpoint0.shape,
                        generator=generator,
                        device=device,
                        dtype=endpoint0.dtype,
                    )
                    * model.endpoint_output_scale[None]
                    * float(args.initial_jitter_fraction)
                )
                h, endpoints = surface_heun_rollout(
                    model,
                    q0_radius,
                    sampled_endpoint,
                    shell,
                    side,
                    station_mask,
                    steps=steps,
                    initial_h=initial_h,
                )
                radius_samples.append((q0_radius + h)[0].float().cpu().numpy())
                endpoint_samples.append(endpoints[0].float().cpu().numpy())
        radius_stack = np.stack(radius_samples)
        endpoint_stack = np.stack(endpoint_samples)
        lower_radius, median_radius, upper_radius = np.quantile(
            radius_stack, (lower_q, 0.5, upper_q), axis=0
        )
        lower_endpoint, median_endpoint, upper_endpoint = np.quantile(
            endpoint_stack, (lower_q, 0.5, upper_q), axis=0
        )
        target_radius = sample["q1_ray_radii_mm"].numpy()
        ray_mask = sample["ray_loss_mask"].numpy().astype(bool)
        coverage = (
            (target_radius >= lower_radius) & (target_radius <= upper_radius)
        )
        key = f"{sample['case_id']}_{sample['side']}"
        np.savez_compressed(
            envelope_dir / f"{key}.npz",
            lower_radius_mm=lower_radius.astype(np.float32),
            median_radius_mm=median_radius.astype(np.float32),
            upper_radius_mm=upper_radius.astype(np.float32),
            lower_endpoint_mm=lower_endpoint.astype(np.float32),
            median_endpoint_mm=median_endpoint.astype(np.float32),
            upper_endpoint_mm=upper_endpoint.astype(np.float32),
            nominal_coverage=np.asarray(args.coverage, dtype=np.float32),
            samples=np.asarray(args.samples, dtype=np.int32),
        )
        rows.append(
            {
                "case_id": sample["case_id"],
                "side": sample["side"],
                "empirical_radius_coverage": float(coverage[ray_mask].mean()),
                "mean_envelope_width_mm": float((upper_radius - lower_radius)[ray_mask].mean()),
                "start_endpoint_covered": int(
                    lower_endpoint[0]
                    <= float(sample["q1_global"][1])
                    <= upper_endpoint[0]
                ),
                "end_endpoint_covered": int(
                    lower_endpoint[1]
                    <= float(sample["q1_global"][2])
                    <= upper_endpoint[1]
                ),
            }
        )
        print(f"[ENVELOPE] {index + 1}/{len(dataset)} {key}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "coverage_metrics.csv", index=False)
    payload = {
        "fold": args.fold,
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "samples_per_side": int(args.samples),
        "nominal_coverage": float(args.coverage),
        "initial_jitter_fraction": float(args.initial_jitter_fraction),
        "mean_empirical_radius_coverage": float(frame.empirical_radius_coverage.mean()),
        "mean_envelope_width_mm": float(frame.mean_envelope_width_mm.mean()),
        "endpoint_coverage": float(
            frame[["start_endpoint_covered", "end_endpoint_covered"]].to_numpy().mean()
        ),
        "calibration_guard": (
            "Initial-state jitter is an explicit experimental sampling law. "
            "Tune it only on the selection fold and report calibration on locked folds."
        ),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
