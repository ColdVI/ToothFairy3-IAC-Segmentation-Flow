"""Voxel-domain evaluation with automatic base-prior fallback."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from .chart import native_local_numpy
from .data import TubeCacheDataset
from .geometry import TubeFrame
from .io import load_label_on_reference, load_nifti, load_probability_channels
from .manifest import load_splits
from .metrics import connected_components, dice_score, hd95_mm
from .paths import heun_rollout, project_ell_gauge
from .train import load_trained_model
from .tube_state import decode_mask


def _frame_from_archive(archive) -> TubeFrame:
    frame = TubeFrame(
        centerline_mm=archive["centerline_mm"].astype(np.float32),
        tangent=archive["tangent"].astype(np.float32),
        normal1=archive["normal1"].astype(np.float32),
        normal2=archive["normal2"].astype(np.float32),
        arc_mm=archive["arc_mm"].astype(np.float32),
        coarse_start_mm=float(archive["coarse_start_mm"]),
        coarse_end_mm=float(archive["coarse_end_mm"]),
    )
    frame.validate()
    return frame


def _predict_state(model, checkpoint, sample: dict[str, Any], device: torch.device, heun_steps: int):
    tensors = {
        key: value[None].to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        if checkpoint["mode"] == "direct":
            time = torch.zeros(1, device=device)
            delta_local, delta_global = model(
                tensors["q0_local"],
                tensors["q0_global"],
                tensors["shell"],
                tensors["side_id"],
                tensors["station_mask"],
                time,
            )
            local = tensors["q0_local"] + delta_local
            global_state = tensors["q0_global"] + delta_global
            local, global_state = project_ell_gauge(local, global_state, tensors["station_mask"])
        else:
            local, global_state = heun_rollout(
                model,
                tensors["q0_local"],
                tensors["q0_global"],
                tensors["shell"],
                tensors["side_id"],
                tensors["station_mask"],
                steps=heun_steps,
                mode=checkpoint["mode"],
            )
    return local[0].float().cpu().numpy(), global_state[0].float().cpu().numpy()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate(
    config: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    refinement_fold: int,
    save_predictions: bool = False,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(checkpoint_path, device)
    splits = load_splits(config["paths"]["splits_file"])
    case_ids = splits[int(refinement_fold)]["val"]
    dataset = TubeCacheDataset(
        config["paths"]["cache_dir"],
        case_ids,
        exclude_fallback=False,
        cache_in_ram=False,
    )
    manifest = json.loads(Path(config["paths"]["manifest_file"]).read_text(encoding="utf-8"))
    manifest_items = {item["case_id"]: item for item in manifest["cases"]}
    output_root = Path(config["paths"]["run_dir"]).expanduser().resolve()
    output_dir = output_root / f"fold_{refinement_fold}" / checkpoint["mode"] / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    if save_predictions:
        prediction_dir.mkdir(exist_ok=True)
    heun_steps = int(config.get("training", {}).get("heun_steps", 4))
    rows = []

    for index in range(len(dataset)):
        sample = dataset[index]
        item = manifest_items[sample["case_id"]]
        side = sample["side"]
        label_id = int(manifest["left_label_id"] if side == "L" else manifest["right_label_id"])
        image = load_nifti(item["image"], dtype=np.float32)
        target = load_label_on_reference(item["label"], image) == label_id
        p_left, p_right, _ = load_probability_channels(
            item["probability"],
            image.shape,
            int(manifest["left_probability_channel"]),
            int(manifest["right_probability_channel"]),
        )
        coarse = (p_left if side == "L" else p_right) >= float(
            config.get("geometry", {}).get("probability_threshold", 0.5)
        )
        with np.load(sample["path"], allow_pickle=False) as archive:
            fallback_key = "inference_fallback" if "inference_fallback" in archive else "fallback"
            fallback = bool(archive[fallback_key])
            ceiling_dice = float(archive["q1_ceiling_dice"])
            frame = _frame_from_archive(archive)
        if fallback:
            prediction = coarse.astype(np.uint8)
        else:
            local, global_state = _predict_state(model, checkpoint, sample, device, heun_steps)
            local = native_local_numpy(local, side)
            prediction = decode_mask(local, global_state, frame, image.shape, image.affine)
        row = {
            "case_id": sample["case_id"],
            "side": side,
            "fold": refinement_fold,
            "mode": checkpoint["mode"],
            "used_base_fallback": int(fallback),
            "base_dice": dice_score(coarse, target),
            "refined_dice": dice_score(prediction, target),
            "base_hd95_mm": hd95_mm(coarse, target, image.spacing),
            "refined_hd95_mm": hd95_mm(prediction, target, image.spacing),
            "base_components": connected_components(coarse),
            "refined_components": connected_components(prediction),
            "representation_ceiling_dice": ceiling_dice,
            "nfe": 1 if checkpoint["mode"] == "direct" else 2 * heun_steps,
        }
        rows.append(row)
        if save_predictions:
            output = prediction_dir / f"{sample['case_id']}_{side}.nii.gz"
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), image.affine), output)

    result_csv = output_dir / "metrics.csv"
    _write_csv(result_csv, rows)
    finite_base_hd = [row["base_hd95_mm"] for row in rows if np.isfinite(row["base_hd95_mm"])]
    finite_refined_hd = [row["refined_hd95_mm"] for row in rows if np.isfinite(row["refined_hd95_mm"])]
    summary = {
        "fold": refinement_fold,
        "mode": checkpoint["mode"],
        "samples": len(rows),
        "base_dice_mean": float(np.mean([row["base_dice"] for row in rows])),
        "refined_dice_mean": float(np.mean([row["refined_dice"] for row in rows])),
        "base_hd95_mm_mean": float(np.mean(finite_base_hd)) if finite_base_hd else float("inf"),
        "refined_hd95_mm_mean": float(np.mean(finite_refined_hd)) if finite_refined_hd else float("inf"),
        "fallback_fraction": float(np.mean([row["used_base_fallback"] for row in rows])),
        "dice_non_inferior": bool(
            np.mean([row["refined_dice"] for row in rows])
            >= np.mean([row["base_dice"] for row in rows]) - float(config.get("evaluation", {}).get("dice_margin", 0.002))
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result_csv
