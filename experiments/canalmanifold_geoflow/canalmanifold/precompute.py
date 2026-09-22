"""Turn raw CBCT/OOF pairs into compact physical tube-shell shards."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .config import section
from .constants import MAX_HARMONIC
from .geometry import build_tube_frame
from .io import (
    load_feature_channels,
    load_label_on_reference,
    load_nifti,
    load_probability_channels,
    robust_image_normalize,
)
from .metrics import dice_score, hd95_mm
from .tube_state import (
    boundary_from_profiles,
    complete_boundary_radii,
    decode_mask,
    decode_radial_mask,
    fit_tube,
    radii_from_state,
    sample_shell,
)


def _gradient_magnitude(volume: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    gradients = np.gradient(volume.astype(np.float32), *spacing, edge_order=1)
    return np.sqrt(sum(component * component for component in gradients)).astype(np.float32)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _process_side(
    *,
    item: dict[str, Any],
    side: str,
    label_id: int,
    probability: np.ndarray,
    entropy: np.ndarray,
    image,
    image_normalized: np.ndarray,
    label: np.ndarray,
    decoder_features: np.ndarray | None,
    output_dir: Path,
    geometry_options: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    case_id = item["case_id"]
    fold = int(item["oof_fold"])
    output = output_dir / f"fold_{fold}" / f"{case_id}_{side}.npz"
    if output.exists() and not overwrite:
        with np.load(output, allow_pickle=False) as archive:
            schema = int(archive["representation_version"]) if "representation_version" in archive else 1
            harmonic = int(archive["max_harmonic"]) if "max_harmonic" in archive else 4
            if schema < 2 or harmonic != MAX_HARMONIC or "target_occupancy_shell" not in archive:
                raise RuntimeError(
                    f"Legacy cache shard {output} is incompatible with v2 "
                    f"(schema={schema}, m<={harmonic}). Use a new cache_dir or --overwrite."
                )
            return {
                "case_id": case_id,
                "side": side,
                "fold": fold,
                "status": "cached",
                "q0_identity_dice": float(archive["q0_identity_dice"]),
                "q1_ceiling_dice": float(archive["q1_ceiling_dice"]),
                "q1_ceiling_hd95_mm": float(archive["q1_ceiling_hd95_mm"]),
                "q1_free_ceiling_dice": float(archive["q1_free_ceiling_dice"]),
                "q1_free_ceiling_hd95_mm": float(archive["q1_free_ceiling_hd95_mm"]),
                "training_excluded": bool(
                    archive["training_excluded"] if "training_excluded" in archive else archive["fallback"]
                ),
                "inference_fallback": bool(
                    archive["inference_fallback"] if "inference_fallback" in archive else archive["fallback"]
                ),
                "path": str(output),
            }

    stations = int(geometry_options.get("stations", 160))
    if stations != 160:
        raise ValueError("The locked state is 160x9+3; geometry.stations must equal 160")
    n_angles = int(geometry_options.get("angles", 32))
    shell_radii = int(geometry_options.get("shell_radii", 24))
    fit_radii = int(geometry_options.get("fit_radii", 40))
    maximum_radius = float(geometry_options.get("max_radius_mm", 5.0))

    frame = build_tube_frame(
        probability,
        image.affine,
        stations=stations,
        smoothing_mm=float(geometry_options.get("centerline_smoothing_mm", 0.45)),
        endpoint_margin_mm=float(geometry_options.get("endpoint_margin_mm", 4.0)),
    )
    q0 = fit_tube(
        probability,
        image.affine,
        frame,
        n_angles=n_angles,
        n_radii=fit_radii,
        max_radius_mm=maximum_radius,
        threshold=float(geometry_options.get("probability_threshold", 0.5)),
        estimate_displacement=True,
        maximum_displacement_mm=float(
            geometry_options.get("max_center_shift_mm", 3.0)
        ),
    )
    target = (label == label_id).astype(np.float32)
    q1 = fit_tube(
        target,
        image.affine,
        frame,
        n_angles=n_angles,
        n_radii=fit_radii,
        max_radius_mm=maximum_radius,
        threshold=0.5,
        estimate_displacement=True,
        maximum_displacement_mm=float(geometry_options.get("max_center_shift_mm", 3.0)),
    )

    probability_gradient = _gradient_magnitude(probability, image.spacing)
    image_gradient = _gradient_magnitude(image_normalized, image.spacing)
    shell_channels = [image_normalized, probability, entropy, probability_gradient, image_gradient]
    channel_names = ["cbct", "oof_probability", "entropy", "probability_gradient", "image_gradient"]
    if decoder_features is not None:
        shell_channels.extend(list(decoder_features))
        channel_names.extend([f"decoder_feature_{index}" for index in range(len(decoder_features))])
    shell, shell_angles, shell_radius_grid = sample_shell(
        frame,
        image.affine,
        shell_channels,
        n_angles=n_angles,
        n_radii=shell_radii,
        max_radius_mm=maximum_radius,
    )
    target_shell, _, _ = sample_shell(
        frame,
        image.affine,
        [target],
        n_angles=n_angles,
        n_radii=shell_radii,
        max_radius_mm=maximum_radius,
    )
    target_occupancy_shell = target_shell[0]

    # R2 uses a full station/angle boundary table rather than a truncated
    # harmonic output.  It is fit directly in the common coarse Bishop chart.
    free_fit_radii = int(geometry_options.get("free_fit_radii", max(64, fit_radii)))
    free_profiles, free_angles, free_radius_grid = sample_shell(
        frame,
        image.affine,
        [probability, target],
        n_angles=n_angles,
        n_radii=free_fit_radii,
        max_radius_mm=maximum_radius,
    )
    q0_ray_raw, q0_ray_valid = boundary_from_profiles(
        free_profiles[0].astype(np.float32),
        free_radius_grid,
        threshold=float(geometry_options.get("probability_threshold", 0.5)),
    )
    q1_ray_raw, q1_ray_valid = boundary_from_profiles(
        free_profiles[1].astype(np.float32),
        free_radius_grid,
        threshold=0.5,
    )
    q0_ray_fallback = radii_from_state(q0.local, q0.global_state, free_angles)
    q1_ray_fallback = radii_from_state(q1.local, q1.global_state, free_angles)
    q0_ray_radii = complete_boundary_radii(
        q0_ray_raw, q0_ray_valid, fallback=q0_ray_fallback
    )
    q1_ray_radii = complete_boundary_radii(
        q1_ray_raw, q1_ray_valid, fallback=q1_ray_fallback
    )

    q0_mask = decode_mask(q0.local, q0.global_state, frame, image.shape, image.affine)
    q1_mask = decode_mask(q1.local, q1.global_state, frame, image.shape, image.affine)
    q1_free_mask = decode_radial_mask(
        q1_ray_radii,
        float(q1.global_state[1]),
        float(q1.global_state[2]),
        frame,
        image.shape,
        image.affine,
    )
    coarse_mask = probability >= float(geometry_options.get("probability_threshold", 0.5))
    target_mask = target > 0.5
    q0_dice = dice_score(q0_mask, coarse_mask)
    q1_dice = dice_score(q1_mask, target_mask)
    q1_hd95 = hd95_mm(q1_mask, target_mask, image.spacing)
    q1_free_dice = dice_score(q1_free_mask, target_mask)
    q1_free_hd95 = hd95_mm(q1_free_mask, target_mask, image.spacing)
    base_raw_dice = dice_score(coarse_mask, target_mask)
    base_raw_hd95 = hd95_mm(coarse_mask, target_mask, image.spacing)
    fallback_threshold = float(geometry_options.get("minimum_ceiling_dice", 0.85))
    identity_threshold = float(geometry_options.get("minimum_identity_dice", 0.80))
    training_excluded = bool(q1_dice < fallback_threshold or q0_dice < identity_threshold)
    # This is the only fallback available at test time: it compares the fitted
    # q0 against the coarse prediction, never against GT.
    inference_fallback = bool(q0_dice < identity_threshold)
    station_mask = q0.station_valid & q1.station_valid
    metric_mask = q0.station_valid | q1.station_valid

    provenance = {
        "schema_version": 2,
        "case_id": case_id,
        "side": side,
        "oof_fold": fold,
        "image": item["image"],
        "label": item["label"],
        "probability": item["probability"],
        "decoder_features": item.get("decoder_features"),
        "channel_names": channel_names,
        "all_distances_are_mm": True,
        "max_harmonic": MAX_HARMONIC,
        "target_occupancy_is_supervision_only": True,
    }
    _atomic_savez(
        output,
        q0_local=q0.local,
        q0_global=q0.global_state,
        q1_local=q1.local,
        q1_global=q1.global_state,
        station_mask=station_mask.astype(np.uint8),
        metric_mask=metric_mask.astype(np.uint8),
        q0_station_valid=q0.station_valid.astype(np.uint8),
        q1_station_valid=q1.station_valid.astype(np.uint8),
        shell=shell,
        target_occupancy_shell=target_occupancy_shell.astype(np.float16),
        shell_angles=shell_angles,
        shell_radii_mm=shell_radius_grid,
        ray_angles=free_angles.astype(np.float32),
        ray_sample_radii_mm=free_radius_grid.astype(np.float32),
        q0_ray_radii_mm=q0_ray_radii.astype(np.float32),
        q1_ray_radii_mm=q1_ray_radii.astype(np.float32),
        q0_ray_valid=q0_ray_valid.astype(np.uint8),
        q1_ray_valid=q1_ray_valid.astype(np.uint8),
        free_h_target_mm=(q1_ray_radii - q0_ray_radii).astype(np.float32),
        centerline_mm=frame.centerline_mm,
        tangent=frame.tangent,
        normal1=frame.normal1,
        normal2=frame.normal2,
        arc_mm=frame.arc_mm,
        coarse_start_mm=np.asarray(frame.coarse_start_mm, dtype=np.float32),
        coarse_end_mm=np.asarray(frame.coarse_end_mm, dtype=np.float32),
        affine=image.affine.astype(np.float64),
        volume_shape=np.asarray(image.shape, dtype=np.int32),
        spacing_mm=np.asarray(image.spacing, dtype=np.float32),
        side_id=np.asarray(0 if side == "L" else 1, dtype=np.int64),
        q0_identity_dice=np.asarray(q0_dice, dtype=np.float32),
        q1_ceiling_dice=np.asarray(q1_dice, dtype=np.float32),
        q1_ceiling_hd95_mm=np.asarray(q1_hd95, dtype=np.float32),
        q1_free_ceiling_dice=np.asarray(q1_free_dice, dtype=np.float32),
        q1_free_ceiling_hd95_mm=np.asarray(q1_free_hd95, dtype=np.float32),
        base_raw_dice=np.asarray(base_raw_dice, dtype=np.float32),
        base_raw_hd95_mm=np.asarray(base_raw_hd95, dtype=np.float32),
        training_excluded=np.asarray(training_excluded, dtype=np.uint8),
        inference_fallback=np.asarray(inference_fallback, dtype=np.uint8),
        fallback=np.asarray(training_excluded, dtype=np.uint8),
        representation_version=np.asarray(2, dtype=np.int16),
        max_harmonic=np.asarray(MAX_HARMONIC, dtype=np.int16),
        provenance_json=np.asarray(json.dumps(provenance)),
    )
    return {
        "case_id": case_id,
        "side": side,
        "fold": fold,
        "status": "ok",
        "q0_identity_dice": q0_dice,
        "q1_ceiling_dice": q1_dice,
        "q1_ceiling_hd95_mm": q1_hd95,
        "q1_free_ceiling_dice": q1_free_dice,
        "q1_free_ceiling_hd95_mm": q1_free_hd95,
        "training_excluded": training_excluded,
        "inference_fallback": inference_fallback,
        "path": str(output),
    }


def _process_case(
    item: dict[str, Any],
    manifest_meta: dict[str, Any],
    output_dir: str,
    geometry_options: dict[str, Any],
    overwrite: bool,
) -> list[dict[str, Any]]:
    try:
        image = load_nifti(item["image"], dtype=np.float32)
        label = load_label_on_reference(item["label"], image)
        p_left, p_right, entropy = load_probability_channels(
            item["probability"],
            image.shape,
            int(manifest_meta["left_probability_channel"]),
            int(manifest_meta["right_probability_channel"]),
        )
        stable_seed = int.from_bytes(
            hashlib.sha256(item["case_id"].encode("utf-8")).digest()[:4], "little"
        )
        image_normalized = robust_image_normalize(image.data, seed=stable_seed)
        decoder_features = None
        if item.get("decoder_features"):
            decoder_features = load_feature_channels(item["decoder_features"], image.shape)
        results = []
        for side, label_id, probability in (
            ("L", int(manifest_meta["left_label_id"]), p_left),
            ("R", int(manifest_meta["right_label_id"]), p_right),
        ):
            results.append(
                _process_side(
                    item=item,
                    side=side,
                    label_id=label_id,
                    probability=probability,
                    entropy=entropy,
                    image=image,
                    image_normalized=image_normalized,
                    label=label,
                    decoder_features=decoder_features,
                    output_dir=Path(output_dir),
                    geometry_options=geometry_options,
                    overwrite=overwrite,
                )
            )
        return results
    except Exception as error:  # noqa: BLE001 - isolate one bad case in batch precompute
        return [
            {
                "case_id": item["case_id"],
                "side": "case",
                "fold": int(item["oof_fold"]),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        ]


def precompute(config: dict[str, Any], *, limit: int | None = None, overwrite: bool = False) -> Path:
    paths = section(config, "paths")
    with Path(paths["manifest_file"]).expanduser().open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    output_dir = Path(paths["cache_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    options = section(config, "geometry")
    runtime = section(config, "runtime")
    items = manifest["cases"][:limit] if limit else manifest["cases"]
    workers = int(runtime.get("precompute_workers", max(1, min(4, (os.cpu_count() or 2) // 2))))
    rows: list[dict[str, Any]] = []
    if workers == 1:
        for item in items:
            rows.extend(_process_case(item, manifest, str(output_dir), options, overwrite))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_case, item, manifest, str(output_dir), options, overwrite): item
                for item in items
            }
            for future in as_completed(futures):
                rows.extend(future.result())

    report = output_dir / "precompute_report.csv"
    fields = [
        "case_id",
        "side",
        "fold",
        "status",
        "q0_identity_dice",
        "q1_ceiling_dice",
        "q1_ceiling_hd95_mm",
        "q1_free_ceiling_dice",
        "q1_free_ceiling_hd95_mm",
        "training_excluded",
        "inference_fallback",
        "path",
        "error",
    ]
    temporary = report.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(report)
    errors = [row for row in rows if row.get("status") == "error"]
    if errors:
        error_log = output_dir / "precompute_errors.json"
        error_log.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    return report
