#!/usr/bin/env python3
"""Measure the untouched OOF prior through the complete Track B inference path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "data", ROOT / "flow"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.io_utils import mask_to_sdf_mm, normalize_sdf, sdf_stack_to_mask, voxel_spacing  # noqa: E402
from flow.validate import summarize_rows, validation_rows  # noqa: E402


class ZeroVelocity(torch.nn.Module):
    """Identity-flow model: every ODE evaluation returns exactly zero."""

    def forward(self, xt, t, cond):
        del t, cond
        return torch.zeros_like(xt)


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(partial, path)


def _write_prior_floor(config_path: Path, metrics):
    """Replace the YAML prior_floor mapping without discarding comments/layout."""
    lines = config_path.read_text().splitlines(keepends=True)
    start = next((index for index, line in enumerate(lines)
                  if line.rstrip() == "prior_floor:"), None)
    if start is None:
        raise ValueError(f"prior_floor block not found in {config_path}")
    end = start + 1
    while end < len(lines) and (lines[end].startswith(" ") or not lines[end].strip()):
        end += 1
    block = ["prior_floor:\n"] + [f"  {key}: {metrics[key]:.10g}\n"
                                    for key in ("dice", "cldice", "hd95", "score")]
    partial = config_path.with_suffix(config_path.suffix + ".partial")
    partial.write_text("".join(lines[:start] + block + lines[end:]))
    os.replace(partial, config_path)


def _ordered_validation_ids(splits):
    ids = [sid for fold in splits["folds"] for sid in fold["val"]]
    duplicates = sorted(sid for sid, count in Counter(ids).items() if count != 1)
    if duplicates:
        raise ValueError(f"validation folds are not disjoint: {duplicates[:5]}")
    development = splits.get("development")
    if development is not None and set(ids) != set(development):
        raise ValueError("union of fold validation IDs does not match development set")
    return ids


def _available_ids(directory: Path, suffix: str):
    return {path.name[:-len(suffix)] for path in directory.glob(f"*{suffix}")}


def _roundtrip_check(case_id, coarse_sdf_dir, labels_dir, clip_mm):
    coarse = np.load(Path(coarse_sdf_dir) / f"{case_id}.npz")["sdf"].astype(np.float32)
    mask = sdf_stack_to_mask(coarse)
    gt_img = nib.load(str(Path(labels_dir) / f"{case_id}.nii.gz"))
    spacing = voxel_spacing(gt_img)
    rebuilt = np.stack([
        normalize_sdf(mask_to_sdf_mm(mask == side, spacing, clip_mm), clip_mm)
        for side in (1, 2)
    ])
    decoded = sdf_stack_to_mask(rebuilt)
    changed = int(np.count_nonzero(decoded != mask))
    return {"case_id": case_id, "changed_voxels": changed,
            "identical": changed == 0, "clip_mm": float(clip_mm)}


def _subset_rows(rows, ids):
    wanted = set(ids)
    return [row for row in rows if row["case_id"] in wanted]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="configs/splits.json")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--coarse-sdf", required=True)
    parser.add_argument("--out", default="outputs/baselines/identity_prior.json")
    parser.add_argument("--config", default="configs/flow.yaml")
    parser.add_argument("--write-config", action="store_true",
                        help="write the complete-CV per-side metrics to config prior_floor")
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clip-mm", type=float, default=10.0)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="diagnostic only: evaluate the cache intersection and mark output incomplete")
    args = parser.parse_args()

    with open(args.splits) as handle:
        splits = json.load(handle)
    expected_ids = _ordered_validation_ids(splits)
    available = (_available_ids(Path(args.images), "_0000.nii.gz")
                 & _available_ids(Path(args.labels), ".nii.gz")
                 & _available_ids(Path(args.coarse_sdf), ".npz"))
    missing = [sid for sid in expected_ids if sid not in available]
    if missing and not args.allow_incomplete:
        raise SystemExit(
            f"identity baseline requires all {len(expected_ids)} CV cases; "
            f"{len(missing)} are missing (first: {missing[:5]}). "
            "Use --allow-incomplete only for a clearly labelled diagnostic output."
        )
    ids = [sid for sid in expected_ids if sid in available]
    if not ids:
        raise SystemExit("no cases have matching image, label, and coarse-SDF files")

    model = ZeroVelocity().to(args.device).eval()
    rows = validation_rows(model, ids, args.images, args.coarse_sdf, args.labels,
                           patch=args.patch, steps=args.steps, device=args.device,
                           progress=True)
    per_side = summarize_rows(rows, "per_side")
    per_case = summarize_rows(rows, "per_case")

    first_20_ids = []
    for fold in splits["folds"]:
        first_20_ids.extend(sid for sid in fold["val"][:20] if sid in available)
    first_20_rows = _subset_rows(rows, first_20_ids)
    h1 = summarize_rows(first_20_rows, "per_side") if first_20_rows else None
    h3 = _roundtrip_check(ids[0], args.coarse_sdf, args.labels, args.clip_mm)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete_cv": not missing,
        "coverage": {"evaluated": len(ids), "expected": len(expected_ids),
                     "missing": missing},
        "inference": {"model": "zero_velocity", "patch": args.patch,
                      "ode_steps": args.steps, "device": args.device},
        "per_side": per_side,
        "per_case": per_case,
        "hypotheses": {
            "H1_val_max_cases_20": {"first_20_per_fold": h1,
                                    "full_available_cv": per_side},
            "H2_metric_aggregation": {"per_side": per_side, "per_case": per_case},
            "H3_sdf_roundtrip": h3,
        },
        "track_a_reference_dice": 0.9101,
        "delta_dice_vs_track_a": per_case["dice"] - 0.9101,
    }
    _atomic_json(Path(args.out), payload)
    if args.write_config:
        if missing:
            raise SystemExit("refusing --write-config: identity baseline is not complete CV")
        _write_prior_floor(Path(args.config), per_side)
        print(f"[identity] wrote prior_floor -> {args.config}")
    print(f"[identity] {len(ids)}/{len(expected_ids)} cases -> {args.out}")
    print(f"[identity] per-side Dice={per_side['dice']:.4f}, "
          f"per-case Dice={per_case['dice']:.4f}, score={per_case['score']:.4f}")
    print(f"[identity] round-trip changed voxels={h3['changed_voxels']}")


if __name__ == "__main__":
    main()
