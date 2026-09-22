#!/usr/bin/env python3
"""Upgrade the 960-shard Corrected cache to m<=8 without reading CBCT volumes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from canalmanifold.constants import MAX_HARMONIC
from canalmanifold.evaluate import _frame_from_archive
from canalmanifold.io import load_label_on_reference, load_probability_channels
from canalmanifold.metrics import dice_score, hd95_mm
from canalmanifold.tube_state import (
    boundary_from_profiles,
    complete_boundary_radii,
    decode_mask,
    decode_radial_mask,
    fit_tube,
    radii_from_state,
    sample_shell,
)


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["paths"]["manifest_file"]).read_text(encoding="utf-8"))
    items = {item["case_id"]: item for item in manifest["cases"]}
    geometry = config.get("geometry", {})
    source_paths = sorted(Path(args.source_cache).expanduser().glob("fold_*/*.npz"))
    if args.limit:
        source_paths = source_paths[: int(args.limit)]
    if not source_paths:
        raise FileNotFoundError(args.source_cache)
    output_root = Path(args.output_cache).expanduser().resolve()
    rows = []
    cached_case = None
    probability_left = probability_right = label = None
    for index, source in enumerate(source_paths):
        fold_name = source.parent.name
        destination = output_root / fold_name / source.name
        if destination.exists() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as archive:
                if (
                    "representation_version" in archive
                    and int(archive["representation_version"]) >= 2
                    and int(archive["max_harmonic"]) == MAX_HARMONIC
                ):
                    rows.append({"key": source.stem, "status": "cached", "path": str(destination)})
                    continue
            raise RuntimeError(f"Incompatible destination exists: {destination}")

        case_id, side = source.stem.rsplit("_", 1)
        item = items[case_id]
        with np.load(source, allow_pickle=False) as old:
            arrays = {key: np.asarray(old[key]) for key in old.files}
            shape = tuple(map(int, old["volume_shape"]))
            affine = old["affine"].astype(np.float64)
            spacing = tuple(map(float, old["spacing_mm"]))
            frame = _frame_from_archive(old)
            shell = old["shell"].astype(np.float16)
            shell_radii_mm = old["shell_radii_mm"].astype(np.float32)
            shell_angles = old["shell_angles"].astype(np.float32)
        if cached_case != case_id:
            reference = SimpleNamespace(shape=shape, affine=affine)
            label = load_label_on_reference(item["label"], reference)
            probability_left, probability_right, _ = load_probability_channels(
                item["probability"],
                shape,
                int(manifest["left_probability_channel"]),
                int(manifest["right_probability_channel"]),
            )
            cached_case = case_id
        probability = probability_left if side == "L" else probability_right
        label_id = int(
            manifest["left_label_id"] if side == "L" else manifest["right_label_id"]
        )
        target = (label == label_id).astype(np.float32)
        n_angles = int(geometry.get("angles", shell.shape[2]))
        fit_radii = int(geometry.get("fit_radii", 40))
        free_fit_radii = int(geometry.get("free_fit_radii", max(64, fit_radii)))
        maximum_radius = float(geometry.get("max_radius_mm", 5.0))
        threshold = float(geometry.get("probability_threshold", 0.5))
        q0 = fit_tube(
            probability,
            affine,
            frame,
            n_angles=n_angles,
            n_radii=fit_radii,
            max_radius_mm=maximum_radius,
            threshold=threshold,
            estimate_displacement=True,
            maximum_displacement_mm=float(geometry.get("max_center_shift_mm", 3.0)),
        )
        q1 = fit_tube(
            target,
            affine,
            frame,
            n_angles=n_angles,
            n_radii=fit_radii,
            max_radius_mm=maximum_radius,
            threshold=0.5,
            estimate_displacement=True,
            maximum_displacement_mm=float(geometry.get("max_center_shift_mm", 3.0)),
        )
        target_shell, _, _ = sample_shell(
            frame,
            affine,
            [target],
            n_angles=n_angles,
            n_radii=len(shell_radii_mm),
            max_radius_mm=maximum_radius,
        )
        free_profiles, free_angles, free_grid = sample_shell(
            frame,
            affine,
            [probability, target],
            n_angles=n_angles,
            n_radii=free_fit_radii,
            max_radius_mm=maximum_radius,
        )
        q0_raw, q0_valid = boundary_from_profiles(
            free_profiles[0].astype(np.float32), free_grid, threshold=threshold
        )
        q1_raw, q1_valid = boundary_from_profiles(
            free_profiles[1].astype(np.float32), free_grid, threshold=0.5
        )
        q0_ray = complete_boundary_radii(
            q0_raw,
            q0_valid,
            fallback=radii_from_state(q0.local, q0.global_state, free_angles),
        )
        q1_ray = complete_boundary_radii(
            q1_raw,
            q1_valid,
            fallback=radii_from_state(q1.local, q1.global_state, free_angles),
        )
        coarse = probability >= threshold
        target_mask = target > 0.5
        q0_mask = decode_mask(q0.local, q0.global_state, frame, shape, affine)
        q1_mask = decode_mask(q1.local, q1.global_state, frame, shape, affine)
        free_mask = decode_radial_mask(
            q1_ray,
            float(q1.global_state[1]),
            float(q1.global_state[2]),
            frame,
            shape,
            affine,
        )
        q0_dice = dice_score(q0_mask, coarse)
        q1_dice = dice_score(q1_mask, target_mask)
        q1_hd95 = hd95_mm(q1_mask, target_mask, spacing)
        q1_free_dice = dice_score(free_mask, target_mask)
        q1_free_hd95 = hd95_mm(free_mask, target_mask, spacing)
        identity_threshold = float(geometry.get("minimum_identity_dice", 0.80))
        ceiling_threshold = float(geometry.get("minimum_ceiling_dice", 0.85))
        training_excluded = bool(
            q0_dice < identity_threshold or q1_dice < ceiling_threshold
        )
        inference_fallback = bool(q0_dice < identity_threshold)

        arrays.update(
            {
                "q0_local": q0.local,
                "q0_global": q0.global_state,
                "q1_local": q1.local,
                "q1_global": q1.global_state,
                "station_mask": (q0.station_valid & q1.station_valid).astype(np.uint8),
                "metric_mask": (q0.station_valid | q1.station_valid).astype(np.uint8),
                "q0_station_valid": q0.station_valid.astype(np.uint8),
                "q1_station_valid": q1.station_valid.astype(np.uint8),
                "target_occupancy_shell": target_shell[0].astype(np.float16),
                "shell_angles": shell_angles,
                "shell_radii_mm": shell_radii_mm,
                "ray_angles": free_angles.astype(np.float32),
                "ray_sample_radii_mm": free_grid.astype(np.float32),
                "q0_ray_radii_mm": q0_ray.astype(np.float32),
                "q1_ray_radii_mm": q1_ray.astype(np.float32),
                "q0_ray_valid": q0_valid.astype(np.uint8),
                "q1_ray_valid": q1_valid.astype(np.uint8),
                "free_h_target_mm": (q1_ray - q0_ray).astype(np.float32),
                "q0_identity_dice": np.asarray(q0_dice, dtype=np.float32),
                "q1_ceiling_dice": np.asarray(q1_dice, dtype=np.float32),
                "q1_ceiling_hd95_mm": np.asarray(q1_hd95, dtype=np.float32),
                "q1_free_ceiling_dice": np.asarray(q1_free_dice, dtype=np.float32),
                "q1_free_ceiling_hd95_mm": np.asarray(q1_free_hd95, dtype=np.float32),
                "training_excluded": np.asarray(training_excluded, dtype=np.uint8),
                "inference_fallback": np.asarray(inference_fallback, dtype=np.uint8),
                "fallback": np.asarray(training_excluded, dtype=np.uint8),
                "representation_version": np.asarray(2, dtype=np.int16),
                "max_harmonic": np.asarray(MAX_HARMONIC, dtype=np.int16),
            }
        )
        if "provenance_json" in arrays:
            provenance = json.loads(str(arrays["provenance_json"]))
        else:
            provenance = {}
        provenance.update(
            {
                "schema_version": 2,
                "max_harmonic": MAX_HARMONIC,
                "upgraded_from": str(source),
                "upgrade_reads_cbct": False,
                "target_occupancy_is_supervision_only": True,
            }
        )
        arrays["provenance_json"] = np.asarray(json.dumps(provenance))
        atomic_savez(destination, arrays)
        rows.append(
            {
                "key": source.stem,
                "status": "ok",
                "q0_identity_dice": q0_dice,
                "q1_m8_ceiling_dice": q1_dice,
                "q1_free_ceiling_dice": q1_free_dice,
                "path": str(destination),
            }
        )
        print(
            f"[CACHE V2] {index + 1}/{len(source_paths)} {source.stem} "
            f"m8={q1_dice:.5f} free={q1_free_dice:.5f}",
            flush=True,
        )

    report = output_root / "upgrade_report.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(report)


if __name__ == "__main__":
    main()
