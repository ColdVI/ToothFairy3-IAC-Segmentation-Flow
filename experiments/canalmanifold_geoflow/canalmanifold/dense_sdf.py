"""R3 comparator: prior-coupled, band-weighted dense SDF flow."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .displacement import remove_small_components, signed_distance
from .io import (
    load_label_on_reference,
    load_nifti,
    load_probability_channels,
    robust_image_normalize,
)
from .manifest import load_splits
from .metrics import connected_components, dice_score, hd95_mm
from .train import _autocast, _torch_load, seed_everything


def _resize(volume: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    factors = np.asarray(shape, dtype=np.float64) / np.asarray(volume.shape, dtype=np.float64)
    return zoom(volume, factors, order=order, mode="nearest", prefilter=order > 1)


def _coarse_crop(
    probability: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    threshold: float,
    margin_mm: float,
) -> tuple[slice, slice, slice]:
    support = probability >= threshold
    if not support.any():
        support = probability >= max(float(np.quantile(probability, 0.9995)), 0.05)
    coordinates = np.argwhere(support)
    if not len(coordinates):
        maximum = np.unravel_index(int(np.argmax(probability)), probability.shape)
        coordinates = np.asarray(maximum, dtype=int)[None]
    padding = np.ceil(float(margin_mm) / np.maximum(spacing, 1e-6)).astype(int)
    lower = np.maximum(coordinates.min(axis=0) - padding, 0)
    upper = np.minimum(coordinates.max(axis=0) + 1 + padding, probability.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def _patch_spacing(
    crop_shape: tuple[int, int, int],
    source_spacing: tuple[float, float, float],
    patch_shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    physical = (np.asarray(crop_shape) - 1).clip(min=1) * np.asarray(source_spacing)
    denominator = (np.asarray(patch_shape) - 1).clip(min=1)
    return tuple(map(float, physical / denominator))


def precompute_dense_sdf(
    config: dict[str, Any], *, overwrite: bool = False, limit: int | None = None
) -> Path:
    manifest = json.loads(Path(config["paths"]["manifest_file"]).read_text(encoding="utf-8"))
    output_root = Path(config["paths"]["dense_cache_dir"]).expanduser().resolve()
    options = config.get("dense", {})
    patch_shape = tuple(map(int, options.get("patch_shape", [96, 96, 128])))
    margin_mm = float(options.get("crop_margin_mm", 6.0))
    max_sdf_mm = float(options.get("max_sdf_mm", 8.0))
    threshold = float(config.get("geometry", {}).get("probability_threshold", 0.5))
    items = manifest["cases"][:limit] if limit else manifest["cases"]
    rows = []
    for case_index, item in enumerate(items):
        image = load_nifti(item["image"], dtype=np.float32)
        label = load_label_on_reference(item["label"], image)
        left, right, _ = load_probability_channels(
            item["probability"],
            image.shape,
            int(manifest["left_probability_channel"]),
            int(manifest["right_probability_channel"]),
        )
        normalized = robust_image_normalize(image.data, seed=case_index + 20260831)
        for side, probability, label_id in (
            ("L", left, int(manifest["left_label_id"])),
            ("R", right, int(manifest["right_label_id"])),
        ):
            fold = int(item["oof_fold"])
            output = output_root / f"fold_{fold}" / f"{item['case_id']}_{side}.npz"
            if output.exists() and not overwrite:
                rows.append({"key": output.stem, "status": "cached", "path": str(output)})
                continue
            crop = _coarse_crop(
                probability,
                image.spacing,
                threshold=threshold,
                margin_mm=margin_mm,
            )
            crop_shape = tuple(axis.stop - axis.start for axis in crop)
            patch_spacing = _patch_spacing(crop_shape, image.spacing, patch_shape)
            coarse_source = probability[crop] >= threshold
            target_source = (label[crop] == label_id)
            coarse_patch = _resize(coarse_source.astype(np.float32), patch_shape, 0) >= 0.5
            target_patch = _resize(target_source.astype(np.float32), patch_shape, 0) >= 0.5
            coarse_sdf = np.clip(
                signed_distance(coarse_patch, patch_spacing), -max_sdf_mm, max_sdf_mm
            )
            target_sdf = np.clip(
                signed_distance(target_patch, patch_spacing), -max_sdf_mm, max_sdf_mm
            )
            conditions = np.stack(
                (
                    _resize(normalized[crop], patch_shape, 1),
                    _resize(probability[crop], patch_shape, 1),
                ),
                axis=0,
            ).astype(np.float16)
            lower = np.asarray([axis.start for axis in crop], dtype=np.int32)
            upper = np.asarray([axis.stop for axis in crop], dtype=np.int32)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp.npz")
            np.savez_compressed(
                temporary,
                condition=conditions,
                coarse_sdf=coarse_sdf.astype(np.float16),
                target_sdf=target_sdf.astype(np.float16),
                target_mask=target_patch.astype(np.uint8),
                patch_spacing_mm=np.asarray(patch_spacing, dtype=np.float32),
                source_spacing_mm=np.asarray(image.spacing, dtype=np.float32),
                source_shape=np.asarray(image.shape, dtype=np.int32),
                crop_lower=lower,
                crop_upper=upper,
                side_id=np.asarray(0 if side == "L" else 1, dtype=np.int64),
                case_id=np.asarray(item["case_id"]),
                side=np.asarray(side),
                schema_version=np.asarray(1, dtype=np.int16),
                prior_coupled=np.asarray(1, dtype=np.uint8),
            )
            temporary.replace(output)
            rows.append(
                {
                    "key": output.stem,
                    "status": "ok",
                    "coarse_patch_dice": dice_score(coarse_patch, target_patch),
                    "path": str(output),
                }
            )
        print(f"[DENSE CACHE] {case_index + 1}/{len(items)} {item['case_id']}", flush=True)
    report = output_root / "precompute_report.csv"
    with report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return report


class DenseSDFDataset(Dataset):
    def __init__(self, cache_dir: str | Path, case_ids: list[str] | set[str]) -> None:
        wanted = set(map(str, case_ids))
        self.paths = [
            path
            for path in sorted(Path(cache_dir).glob("fold_*/*.npz"))
            if path.stem.rsplit("_", 1)[0] in wanted
        ]
        if not self.paths:
            raise FileNotFoundError(cache_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as archive:
            return {
                "condition": torch.from_numpy(archive["condition"].astype(np.float32)),
                "coarse_sdf": torch.from_numpy(archive["coarse_sdf"].astype(np.float32))[None],
                "target_sdf": torch.from_numpy(archive["target_sdf"].astype(np.float32))[None],
                "target_mask": torch.from_numpy(archive["target_mask"].astype(np.float32))[None],
                "side_id": torch.as_tensor(int(archive["side_id"]), dtype=torch.long),
                "path": str(path),
                "case_id": path.stem.rsplit("_", 1)[0],
                "side": path.stem.rsplit("_", 1)[1],
            }


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        self.block = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv3d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class DenseSDFVelocityUNet(nn.Module):
    def __init__(self, base_channels: int = 24, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(base_channels)
        # image, probability, frozen coarse SDF, current SDF, t, sin(pi*t)
        self.enc1 = ConvBlock(6, c)
        self.down1 = nn.Conv3d(c, 2 * c, 2, stride=2)
        self.enc2 = ConvBlock(2 * c, 2 * c)
        self.down2 = nn.Conv3d(2 * c, 4 * c, 2, stride=2)
        self.mid = nn.Sequential(ConvBlock(4 * c, 4 * c), nn.Dropout3d(dropout))
        self.up2 = nn.Conv3d(4 * c, 2 * c, 1)
        self.dec2 = ConvBlock(4 * c, 2 * c)
        self.up1 = nn.Conv3d(2 * c, c, 1)
        self.dec1 = ConvBlock(2 * c, c)
        self.output = nn.Conv3d(c, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.model_config = {"base_channels": c, "dropout": float(dropout)}

    def forward(
        self,
        condition: torch.Tensor,
        coarse_sdf: torch.Tensor,
        current_sdf: torch.Tensor,
        time_value: torch.Tensor,
    ) -> torch.Tensor:
        spatial = current_sdf.shape[2:]
        time_channel = time_value[:, None, None, None, None].expand(-1, 1, *spatial)
        sine_channel = torch.sin(math.pi * time_channel)
        x = torch.cat((condition, coarse_sdf, current_sdf, time_channel, sine_channel), dim=1)
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        middle = self.mid(self.down2(x2))
        up2 = F.interpolate(middle, size=x2.shape[2:], mode="trilinear", align_corners=False)
        up2 = self.dec2(torch.cat((self.up2(up2), x2), dim=1))
        up1 = F.interpolate(up2, size=x1.shape[2:], mode="trilinear", align_corners=False)
        return self.output(self.dec1(torch.cat((self.up1(up1), x1), dim=1)))


def dense_heun_rollout(
    model: DenseSDFVelocityUNet,
    condition: torch.Tensor,
    coarse_sdf: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    current = coarse_sdf.clone()
    batch = len(current)
    for index in range(int(steps)):
        t0 = torch.full((batch,), index / steps, device=current.device, dtype=current.dtype)
        t1 = torch.full((batch,), (index + 1) / steps, device=current.device, dtype=current.dtype)
        velocity0 = model(condition, coarse_sdf, current, t0)
        euler = current + velocity0 / steps
        velocity1 = model(condition, coarse_sdf, euler, t1)
        current = current + 0.5 * (velocity0 + velocity1) / steps
    return current


def _dense_loss(
    model: DenseSDFVelocityUNet,
    batch: dict[str, Any],
    *,
    band_sigma_mm: float,
    decoded_dice_weight: float,
    decoded_temperature_mm: float,
    path_noise_fraction: float,
    q0_jitter_fraction: float,
    augment: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q0, q1 = batch["coarse_sdf"], batch["target_sdf"]
    delta = q1 - q0
    scale = delta.flatten(1).std(dim=1).clamp_min(0.05)
    q0_train = q0
    if augment and q0_jitter_fraction > 0.0:
        q0_train = q0 + torch.randn_like(q0) * scale[:, None, None, None, None] * float(q0_jitter_fraction)
    noise = torch.zeros_like(q0)
    if augment and path_noise_fraction > 0.0:
        noise = torch.randn_like(q0) * scale[:, None, None, None, None] * float(path_noise_fraction)
    time_value = torch.rand(len(q0), device=q0.device, dtype=q0.dtype) * 0.9998 + 0.0001
    beta = time_value * (1.0 - time_value)
    current = (
        q0_train
        + time_value[:, None, None, None, None] * (q1 - q0_train)
        + beta[:, None, None, None, None] * noise
    )
    target_velocity = (q1 - q0_train) + (1.0 - 2.0 * time_value)[:, None, None, None, None] * noise
    predicted = model(batch["condition"], q0, current, time_value)
    weight = torch.exp(-q1.abs() / max(float(band_sigma_mm), 1e-4))
    physical = ((predicted - target_velocity).square() * weight).sum() / weight.sum().clamp_min(1.0)
    remaining = 1.0 - time_value
    terminal = (
        current
        + remaining[:, None, None, None, None] * predicted
        - remaining.square()[:, None, None, None, None] * noise
    )
    occupancy = torch.sigmoid(-terminal / max(float(decoded_temperature_mm), 1e-4))
    target = batch["target_mask"]
    intersection = (occupancy * target).sum(dim=(1, 2, 3, 4))
    denominator = occupancy.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    decoded = 1.0 - dice.mean()
    return physical + float(decoded_dice_weight) * decoded, {
        "physical": physical.detach(),
        "decoded": decoded.detach(),
        "soft_dice": dice.mean().detach(),
    }


def _dense_splits(config: dict[str, Any], fold: int) -> tuple[DenseSDFDataset, DenseSDFDataset]:
    split = load_splits(config["paths"]["splits_file"])[fold]
    return (
        DenseSDFDataset(config["paths"]["dense_cache_dir"], split["train"]),
        DenseSDFDataset(config["paths"]["dense_cache_dir"], split["val"]),
    )


def train_dense_sdf(
    config: dict[str, Any], *, refinement_fold: int, resume: bool = True
) -> Path:
    options = config.get("dense", {})
    seed_everything(int(options.get("seed", 20260831)) + refinement_fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not bool(options.get("allow_cpu_training", False)):
        raise RuntimeError("Dense SDF comparator requires CUDA")
    train_dataset, val_dataset = _dense_splits(config, refinement_fold)
    batch_size = int(options.get("batch_size", 1))
    workers = int(options.get("dataloader_workers", 2))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = DenseSDFVelocityUNet(
        base_channels=int(options.get("base_channels", 24)),
        dropout=float(options.get("dropout", 0.0)),
    ).to(device)
    epochs = int(options.get("epochs", 60))
    optimizer = AdamW(
        model.parameters(),
        lr=float(options.get("learning_rate", 2e-4)),
        weight_decay=float(options.get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    amp = bool(options.get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    run_dir = (
        Path(config["paths"]["run_dir"]).expanduser().resolve()
        / f"fold_{refinement_fold}"
        / "dense_sdf"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    archive = run_dir / "checkpoints"
    archive.mkdir(exist_ok=True)
    last, best = run_dir / "last.pt", run_dir / "best.pt"
    contract = {
        "band_sigma_mm": float(options.get("band_sigma_mm", 1.0)),
        "decoded_dice_weight": float(options.get("decoded_dice_weight", 0.30)),
        "decoded_temperature_mm": float(options.get("decoded_temperature_mm", 0.10)),
        "path_noise_fraction": float(options.get("path_noise_fraction", 0.10)),
        "q0_jitter_fraction": float(options.get("q0_jitter_fraction", 0.20)),
    }
    start_epoch, best_rank, rows = 0, (float("inf"), float("inf")), []
    if resume and last.exists():
        checkpoint = _torch_load(last, device)
        if checkpoint["training_contract"] != contract:
            raise ValueError("Dense SDF checkpoint contract differs from config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rank = tuple(checkpoint["best_rank"])
        if (run_dir / "training_log.csv").exists():
            rows = pd.read_csv(run_dir / "training_log.csv").to_dict("records")
    heun_steps = int(options.get("heun_steps", 4))
    archive_every = max(1, int(options.get("archive_checkpoint_every_epochs", 2)))
    for epoch in range(start_epoch, epochs):
        model.train()
        started = time.time()
        train_losses = []
        for raw in train_loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in raw.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss, _ = _dense_loss(model, batch, augment=True, **contract)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(options.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))
        scheduler.step()
        model.eval()
        val_dice, val_rmse, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for raw in val_loader:
                batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in raw.items()
                }
                with _autocast(device, amp):
                    final = dense_heun_rollout(
                        model,
                        batch["condition"],
                        batch["coarse_sdf"],
                        steps=heun_steps,
                    )
                prediction = final <= 0.0
                target = batch["target_mask"] > 0.5
                intersection = (prediction & target).sum(dim=(1, 2, 3, 4)).float()
                denominator = prediction.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
                dice = torch.where(denominator > 0, 2.0 * intersection / denominator, torch.ones_like(intersection))
                rmse = torch.sqrt((final - batch["target_sdf"]).square().mean(dim=(1, 2, 3, 4)))
                val_dice += float(dice.sum())
                val_rmse += float(rmse.sum())
                val_count += len(final)
        validation_dice = val_dice / max(val_count, 1)
        validation_rmse = val_rmse / max(val_count, 1)
        row = {
            "epoch": epoch,
            "fold": refinement_fold,
            "train_loss": float(np.mean(train_losses)),
            "val_dense_dice": validation_dice,
            "val_dense_sdf_rmse_mm": validation_rmse,
            "seconds": time.time() - started,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "training_log.csv", index=False)
        rank = (-validation_dice, validation_rmse)
        is_best = rank < best_rank
        state = {
            "schema_version": 1,
            "representation": "dense_sdf",
            "mode": "linear",
            "epoch": epoch,
            "refinement_fold": refinement_fold,
            "best_rank": rank if is_best else best_rank,
            "model": model.state_dict(),
            "model_config": model.model_config,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "training_contract": contract,
        }
        temporary = last.with_suffix(".tmp.pt")
        torch.save(state, temporary)
        temporary.replace(last)
        if (epoch + 1) % archive_every == 0 or epoch + 1 == epochs:
            archived = archive / f"epoch_{epoch + 1:04d}.pt"
            temporary = archived.with_suffix(".tmp.pt")
            torch.save(state, temporary)
            temporary.replace(archived)
        if is_best:
            best_rank = rank
            state["best_rank"] = best_rank
            temporary = best.with_suffix(".tmp.pt")
            torch.save(state, temporary)
            temporary.replace(best)
        print(
            f"[DENSE R3 {epoch + 1:03d}/{epochs:03d}] train={row['train_loss']:.6f} "
            f"dice={validation_dice:.6f} rmse={validation_rmse:.4f}mm "
            f"best={'yes' if is_best else 'no'}",
            flush=True,
        )
    return best


def load_dense_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[DenseSDFVelocityUNet, dict[str, Any]]:
    checkpoint = _torch_load(checkpoint_path, device)
    model = DenseSDFVelocityUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def evaluate_dense_sdf(
    config: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    refinement_fold: int,
    output_dir: str | Path,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_dense_model(checkpoint_path, device)
    split = load_splits(config["paths"]["splits_file"])[refinement_fold]
    dataset = DenseSDFDataset(config["paths"]["dense_cache_dir"], split["val"])
    manifest = json.loads(Path(config["paths"]["manifest_file"]).read_text(encoding="utf-8"))
    items = {item["case_id"]: item for item in manifest["cases"]}
    threshold = float(config.get("geometry", {}).get("probability_threshold", 0.5))
    minimum_volume = float(config.get("evaluation", {}).get("minimum_component_volume_mm3", 0.27))
    heun_steps = int(config.get("dense", {}).get("heun_steps", 4))
    rows = []
    cached_case = None
    cached_image = cached_label = cached_left = cached_right = None
    for index in range(len(dataset)):
        sample = dataset[index]
        case_id, side = sample["case_id"], sample["side"]
        if cached_case != case_id:
            cached_image = load_nifti(items[case_id]["image"], dtype=np.float32)
            cached_label = load_label_on_reference(items[case_id]["label"], cached_image)
            cached_left, cached_right, _ = load_probability_channels(
                items[case_id]["probability"],
                cached_image.shape,
                int(manifest["left_probability_channel"]),
                int(manifest["right_probability_channel"]),
            )
            cached_case = case_id
        probability = cached_left if side == "L" else cached_right
        label_id = int(manifest["left_label_id"] if side == "L" else manifest["right_label_id"])
        target = cached_label == label_id
        coarse = probability >= threshold
        with np.load(sample["path"], allow_pickle=False) as archive:
            lower = archive["crop_lower"].astype(int)
            upper = archive["crop_upper"].astype(int)
        condition = sample["condition"][None].to(device)
        coarse_sdf_patch = sample["coarse_sdf"][None].to(device)
        with torch.no_grad(), _autocast(device, device.type == "cuda"):
            final = dense_heun_rollout(model, condition, coarse_sdf_patch, steps=heun_steps)
        delta_patch = (final - coarse_sdf_patch)[0, 0].float().cpu().numpy()
        crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        crop_shape = tuple(int(hi - lo) for lo, hi in zip(lower, upper))
        delta_source = _resize(delta_patch, crop_shape, 1)
        phi0 = signed_distance(coarse[crop], cached_image.spacing)
        refined = coarse.astype(np.uint8, copy=True)
        refined[crop] = (phi0 + delta_source <= 0.0).astype(np.uint8)
        raw_pp = remove_small_components(coarse, cached_image.spacing, minimum_volume_mm3=minimum_volume)
        refined_pp = remove_small_components(refined, cached_image.spacing, minimum_volume_mm3=minimum_volume)
        rows.append(
            {
                "case_id": case_id,
                "side": side,
                "fold": refinement_fold,
                "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
                "raw_pp_dice": dice_score(raw_pp, target),
                "dense_sdf_pp_dice": dice_score(refined_pp, target),
                "raw_pp_hd95_mm": hd95_mm(raw_pp, target, cached_image.spacing),
                "dense_sdf_pp_hd95_mm": hd95_mm(refined_pp, target, cached_image.spacing),
                "raw_pp_components": connected_components(raw_pp),
                "dense_sdf_pp_components": connected_components(refined_pp),
            }
        )
        print(f"[DENSE EXACT] {index + 1}/{len(dataset)} {case_id}_{side}", flush=True)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    path = output / "metrics.csv"
    frame.to_csv(path, index=False)
    delta = frame.dense_sdf_pp_dice - frame.raw_pp_dice
    summary = {
        "fold": refinement_fold,
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "raw_pp_dice_mean": float(frame.raw_pp_dice.mean()),
        "dense_sdf_pp_dice_mean": float(frame.dense_sdf_pp_dice.mean()),
        "mean_delta_dice": float(delta.mean()),
        "identity_anchor": "phi0_original + resize(predicted_sdf-coarse_sdf)",
        "component_filter_applied_symmetrically": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
