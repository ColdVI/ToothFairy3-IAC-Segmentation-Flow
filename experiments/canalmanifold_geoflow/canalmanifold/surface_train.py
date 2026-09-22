"""Training and rollout for the unrestricted R2 h(s,theta) flow arm."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import TubeCacheDataset, split_datasets
from .surface_model import SurfaceDisplacementTransformer, SurfaceStats
from .train import _autocast, _to_device, _torch_load, seed_everything


def compute_surface_stats(dataset: TubeCacheDataset) -> SurfaceStats:
    h_values: list[np.ndarray] = []
    q0_values: list[np.ndarray] = []
    endpoint_values: list[np.ndarray] = []
    endpoint_delta: list[np.ndarray] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        ray_mask = sample["ray_loss_mask"].numpy().astype(bool)
        station_mask = sample["loss_mask"].numpy().astype(bool)
        h = sample["free_h_target_mm"].numpy()
        q0 = sample["q0_ray_radii_mm"].numpy()
        # Keep an angle axis so output scales remain phase aware.
        masked_h = np.where(ray_mask, h, np.nan)
        h_values.append(masked_h)
        q0_values.append(q0[station_mask])
        q0_endpoint = sample["q0_global"].numpy()[1:3]
        q1_endpoint = sample["q1_global"].numpy()[1:3]
        endpoint_values.extend((q0_endpoint[None], q1_endpoint[None]))
        endpoint_delta.append((q1_endpoint - q0_endpoint)[None])
    h_stack = np.stack(h_values, axis=0)
    h_scale = np.nanstd(h_stack, axis=(0, 1))
    h_scale = np.where(np.isfinite(h_scale), h_scale, 0.0)
    h_scale = np.maximum(h_scale, 0.02)
    q0_flat = np.concatenate(q0_values) if q0_values else np.asarray([1.5])
    endpoint = np.concatenate(endpoint_values, axis=0)
    endpoint_change = np.concatenate(endpoint_delta, axis=0)
    return SurfaceStats(
        h_output_scale=h_scale.astype(float).tolist(),
        q0_radius_mean=float(np.mean(q0_flat)),
        q0_radius_std=float(max(np.std(q0_flat), 0.05)),
        endpoint_mean=endpoint.mean(axis=0).astype(float).tolist(),
        endpoint_std=np.maximum(endpoint.std(axis=0), 0.10).astype(float).tolist(),
        endpoint_output_scale=np.maximum(
            endpoint_change.std(axis=0), 0.10
        ).astype(float).tolist(),
    )


def surface_physical_loss(
    predicted_h: torch.Tensor,
    predicted_endpoints: torch.Tensor,
    target_h: torch.Tensor,
    target_endpoints: torch.Tensor,
    ray_mask: torch.Tensor,
    *,
    endpoint_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = ray_mask.to(predicted_h.dtype)
    squared = (predicted_h - target_h).square()
    surface = (squared * weights).sum() / weights.sum().clamp_min(1.0)
    endpoint = (predicted_endpoints - target_endpoints).square().mean()
    loss = surface + float(endpoint_weight) * endpoint
    return loss, {
        "surface_rmse_mm": torch.sqrt(surface.detach().clamp_min(0.0)),
        "endpoint_rmse_mm": torch.sqrt(endpoint.detach().clamp_min(0.0)),
    }


def surface_shell_soft_dice_loss(
    q0_radius: torch.Tensor,
    h: torch.Tensor,
    endpoints: torch.Tensor,
    target_occupancy: torch.Tensor,
    shell_radii_mm: torch.Tensor,
    arc_mm: torch.Tensor,
    station_mask: torch.Tensor,
    *,
    temperature_mm: float,
    endpoint_temperature_mm: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = target_occupancy.to(h.dtype)
    radial_grid = shell_radii_mm.to(h.dtype)
    if radial_grid.ndim == 1:
        radial_grid = radial_grid[None].expand(len(h), -1)
    boundary = (q0_radius + h).clamp(0.12, 8.0)
    prediction = torch.sigmoid(
        (boundary[..., None] - radial_grid[:, None, None, :])
        / max(float(temperature_mm), 1e-4)
    )
    arc = arc_mm.to(h.dtype)
    axial = torch.sigmoid(
        (arc - endpoints[:, None, 0]) / max(float(endpoint_temperature_mm), 1e-4)
    ) * torch.sigmoid(
        (endpoints[:, None, 1] - arc) / max(float(endpoint_temperature_mm), 1e-4)
    )
    weights = station_mask.to(h.dtype)[:, :, None, None]
    prediction = prediction * axial[:, :, None, None] * weights
    target = target * weights
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean(), dice.mean().detach()


def _jitter_and_noise(
    model: SurfaceDisplacementTransformer,
    h0: torch.Tensor,
    endpoints0: torch.Tensor,
    station_mask: torch.Tensor,
    *,
    q0_jitter_fraction: float,
    path_noise_fraction: float,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h_origin = h0
    endpoint_origin = endpoints0
    if augment and q0_jitter_fraction > 0.0:
        jitter = (
            torch.randn_like(h0)
            * model.h_output_scale[None, None]
            * float(q0_jitter_fraction)
        )
        h_origin = h_origin + jitter * station_mask[:, :, None].to(h0.dtype)
        endpoint_origin = endpoint_origin + (
            torch.randn_like(endpoints0)
            * model.endpoint_output_scale[None]
            * float(q0_jitter_fraction)
        )
    noise_h = torch.zeros_like(h0)
    noise_endpoint = torch.zeros_like(endpoints0)
    if augment and path_noise_fraction > 0.0:
        noise_h = (
            torch.randn_like(h0)
            * model.h_output_scale[None, None]
            * float(path_noise_fraction)
        )
        noise_h = noise_h * station_mask[:, :, None].to(h0.dtype)
        noise_endpoint = (
            torch.randn_like(endpoints0)
            * model.endpoint_output_scale[None]
            * float(path_noise_fraction)
        )
    return h_origin, endpoint_origin, noise_h, noise_endpoint


def _surface_batch(
    model: SurfaceDisplacementTransformer,
    batch: dict[str, Any],
    *,
    endpoint_weight: float,
    decoded_dice_weight: float,
    decoded_temperature_mm: float,
    path_noise_fraction: float,
    q0_jitter_fraction: float,
    augment: bool,
    time_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_h = batch["free_h_target_mm"]
    h0 = torch.zeros_like(target_h)
    endpoint0 = batch["q0_global"][:, 1:3]
    endpoint1 = batch["q1_global"][:, 1:3]
    h0, endpoint0, noise_h, noise_endpoint = _jitter_and_noise(
        model,
        h0,
        endpoint0,
        batch["station_mask"],
        q0_jitter_fraction=q0_jitter_fraction,
        path_noise_fraction=path_noise_fraction,
        augment=augment,
    )
    if time_value is None:
        batch_size = len(h0)
        time_value = (
            torch.arange(batch_size, device=h0.device, dtype=h0.dtype)
            + torch.rand(batch_size, device=h0.device, dtype=h0.dtype)
        ) / max(batch_size, 1)
        time_value = time_value[torch.randperm(batch_size, device=h0.device)]
        time_value = time_value * 0.9998 + 0.0001
    else:
        time_value = time_value.to(device=h0.device, dtype=h0.dtype)
    beta = time_value * (1.0 - time_value)
    derivative = 1.0 - 2.0 * time_value
    delta_h = target_h - h0
    delta_endpoint = endpoint1 - endpoint0
    current_h = (
        h0
        + time_value[:, None, None] * delta_h
        + beta[:, None, None] * noise_h
    )
    current_endpoint = (
        endpoint0
        + time_value[:, None] * delta_endpoint
        + beta[:, None] * noise_endpoint
    )
    target_velocity_h = delta_h + derivative[:, None, None] * noise_h
    target_velocity_endpoint = delta_endpoint + derivative[:, None] * noise_endpoint
    predicted_h, predicted_endpoint = model(
        batch["q0_ray_radii_mm"],
        current_h,
        current_endpoint,
        batch["shell"],
        batch["side_id"],
        batch["station_mask"],
        time_value,
    )
    physical, diagnostics = surface_physical_loss(
        predicted_h,
        predicted_endpoint,
        target_velocity_h,
        target_velocity_endpoint,
        batch["ray_loss_mask"],
        endpoint_weight=endpoint_weight,
    )
    remaining = 1.0 - time_value
    terminal_h = (
        current_h
        + remaining[:, None, None] * predicted_h
        - remaining.square()[:, None, None] * noise_h
    )
    terminal_endpoint = (
        current_endpoint
        + remaining[:, None] * predicted_endpoint
        - remaining.square()[:, None] * noise_endpoint
    )
    decoded, decoded_dice = surface_shell_soft_dice_loss(
        batch["q0_ray_radii_mm"],
        terminal_h,
        terminal_endpoint,
        batch["target_occupancy_shell"],
        batch["shell_radii_mm"],
        batch["arc_mm"],
        batch["metric_mask"],
        temperature_mm=decoded_temperature_mm,
    )
    total = physical + float(decoded_dice_weight) * decoded
    diagnostics.update(
        {
            "physical_loss": physical.detach(),
            "decoded_loss": decoded.detach(),
            "decoded_soft_dice": decoded_dice,
        }
    )
    return total, diagnostics


@torch.no_grad()
def surface_heun_rollout(
    model: SurfaceDisplacementTransformer,
    q0_radius: torch.Tensor,
    endpoints0: torch.Tensor,
    shell: torch.Tensor,
    side_id: torch.Tensor,
    station_mask: torch.Tensor,
    *,
    steps: int,
    initial_h: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.zeros_like(q0_radius) if initial_h is None else initial_h.clone()
    endpoints = endpoints0.clone()
    origin_h = h.clone()
    origin_endpoints = endpoints.clone()
    batch = len(h)
    for index in range(int(steps)):
        t0 = torch.full((batch,), index / steps, device=h.device, dtype=h.dtype)
        t1 = torch.full((batch,), (index + 1) / steps, device=h.device, dtype=h.dtype)
        velocity0_h, velocity0_endpoint = model(
            q0_radius, h, endpoints, shell, side_id, station_mask, t0
        )
        dt = 1.0 / steps
        euler_h = h + dt * velocity0_h
        euler_endpoint = endpoints + dt * velocity0_endpoint
        velocity1_h, velocity1_endpoint = model(
            q0_radius, euler_h, euler_endpoint, shell, side_id, station_mask, t1
        )
        h = h + 0.5 * dt * (velocity0_h + velocity1_h)
        endpoints = endpoints + 0.5 * dt * (
            velocity0_endpoint + velocity1_endpoint
        )
        difference = h - origin_h
        weights = station_mask.to(h.dtype)[:, :, None]
        rmse = torch.sqrt(
            (difference.square() * weights).sum(dim=(1, 2))
            / (weights.sum(dim=(1, 2)) * h.shape[2]).clamp_min(1.0)
        )
        scale = torch.minimum(
            torch.ones_like(rmse),
            torch.as_tensor(model.trust_surface_rmse_mm, device=h.device, dtype=h.dtype)
            / rmse.clamp_min(1e-6),
        )
        h = origin_h + scale[:, None, None] * difference
        endpoint_delta = (endpoints - origin_endpoints).clamp(
            -model.trust_endpoint_mm, model.trust_endpoint_mm
        )
        endpoints = origin_endpoints + endpoint_delta
    return h, endpoints


@torch.no_grad()
def _validate_surface(
    model: SurfaceDisplacementTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    endpoint_weight: float,
    decoded_dice_weight: float,
    decoded_temperature_mm: float,
    heun_steps: int,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    total = loss_sum = dice_sum = rmse_sum = 0.0
    validation_times = (0.125, 0.375, 0.625, 0.875)
    for raw in loader:
        batch = _to_device(raw, device)
        batch_size = len(batch["q0_local"])
        losses = []
        with _autocast(device, amp):
            for value in validation_times:
                fixed_time = torch.full(
                    (batch_size,), value, device=device, dtype=batch["q0_local"].dtype
                )
                loss, _ = _surface_batch(
                    model,
                    batch,
                    endpoint_weight=endpoint_weight,
                    decoded_dice_weight=decoded_dice_weight,
                    decoded_temperature_mm=decoded_temperature_mm,
                    path_noise_fraction=0.0,
                    q0_jitter_fraction=0.0,
                    augment=False,
                    time_value=fixed_time,
                )
                losses.append(loss)
            final_h, final_endpoint = surface_heun_rollout(
                model,
                batch["q0_ray_radii_mm"],
                batch["q0_global"][:, 1:3],
                batch["shell"],
                batch["side_id"],
                batch["station_mask"],
                steps=heun_steps,
            )
            _, dice = surface_shell_soft_dice_loss(
                batch["q0_ray_radii_mm"],
                final_h,
                final_endpoint,
                batch["target_occupancy_shell"],
                batch["shell_radii_mm"],
                batch["arc_mm"],
                batch["metric_mask"],
                temperature_mm=decoded_temperature_mm,
            )
            error = final_h - batch["free_h_target_mm"]
            weights = batch["ray_loss_mask"].to(error.dtype)
            rmse = torch.sqrt(
                (error.square() * weights).sum() / weights.sum().clamp_min(1.0)
            )
        loss_sum += float(torch.stack(losses).mean()) * batch_size
        dice_sum += float(dice) * batch_size
        rmse_sum += float(rmse) * batch_size
        total += batch_size
    total = max(total, 1.0)
    return {
        "val_loss": loss_sum / total,
        "val_surface_soft_dice": dice_sum / total,
        "val_surface_rmse_mm": rmse_sum / total,
    }


def _write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train_surface(
    config: dict[str, Any], *, refinement_fold: int, resume: bool = True
) -> Path:
    training = {**config.get("training", {}), **config.get("surface_training", {})}
    seed = int(training.get("seed", 20260826)) + 100 + int(refinement_fold)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not bool(training.get("allow_cpu_training", False)):
        raise RuntimeError("GPU not detected; set allow_cpu_training=true only for a smoke run")
    train_dataset, val_dataset = split_datasets(config, refinement_fold)
    sample = train_dataset[0]
    run_dir = (
        Path(config["paths"]["run_dir"]).expanduser().resolve()
        / f"fold_{refinement_fold}"
        / "surface_h"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / "surface_stats.json"
    if stats_path.exists():
        stats = SurfaceStats.from_dict(json.loads(stats_path.read_text(encoding="utf-8")))
    else:
        stats = compute_surface_stats(train_dataset)
        stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    geometry = config.get("geometry", {})
    options = config.get("surface_model", {})
    model = SurfaceDisplacementTransformer(
        shell_channels=int(sample["shell"].shape[0]),
        stats=stats,
        stations=int(geometry.get("stations", 160)),
        n_angles=int(geometry.get("angles", 32)),
        shell_radii=int(geometry.get("shell_radii", 24)),
        max_radius_mm=float(geometry.get("max_radius_mm", 5.0)),
        profile_offsets_mm=list(
            options.get("profile_offsets_mm", [-0.6, -0.3, 0.0, 0.3, 0.6])
        ),
        d_model=int(options.get("d_model", 192)),
        n_heads=int(options.get("n_heads", 6)),
        n_layers=int(options.get("n_layers", 4)),
        angle_hidden=int(options.get("angle_hidden", 48)),
        angular_modes=int(options.get("angular_modes", 8)),
        dropout=float(options.get("dropout", 0.10)),
        probability_channel=int(options.get("probability_channel", 1)),
        evidence_threshold=float(options.get("evidence_threshold", 0.5)),
        trust_surface_rmse_mm=float(options.get("trust_surface_rmse_mm", 1.0)),
        trust_endpoint_mm=float(options.get("trust_endpoint_mm", 2.0)),
    ).to(device)
    with torch.no_grad():
        probe = _to_device(
            {
                key: value[None] if torch.is_tensor(value) else value
                for key, value in sample.items()
            },
            device,
        )
        zero_h, zero_endpoint = model(
            probe["q0_ray_radii_mm"],
            torch.zeros_like(probe["q0_ray_radii_mm"]),
            probe["q0_global"][:, 1:3],
            probe["shell"],
            probe["side_id"],
            probe["station_mask"],
            torch.zeros(1, device=device),
        )
        if not (
            torch.equal(zero_h, torch.zeros_like(zero_h))
            and torch.equal(zero_endpoint, torch.zeros_like(zero_endpoint))
        ):
            raise AssertionError("Surface-flow zero initialization is not identity")

    batch_size = int(training.get("batch_size", 8))
    workers = int(training.get("dataloader_workers", 2))
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    epochs = int(training.get("epochs_surface", 60))
    optimizer = AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    endpoint_weight = float(training.get("endpoint_weight", 1.0))
    decoded_weight = float(training.get("decoded_dice_weight", 0.30))
    decoded_temperature = float(training.get("decoded_temperature_mm", 0.10))
    path_noise = float(training.get("path_noise_fraction", 0.10))
    q0_jitter = float(training.get("q0_jitter_fraction", 0.20))
    heun_steps = int(training.get("heun_steps", 4))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    archive_every = max(1, int(training.get("archive_checkpoint_every_epochs", 2)))
    contract = {
        "decoded_dice_weight": decoded_weight,
        "decoded_temperature_mm": decoded_temperature,
        "path_noise_fraction": path_noise,
        "q0_jitter_fraction": q0_jitter,
    }
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    archive_dir = run_dir / "checkpoints"
    archive_dir.mkdir(exist_ok=True)
    start_epoch, best_rank, rows = 0, (float("inf"), float("inf")), []
    if resume and last_path.exists():
        checkpoint = _torch_load(last_path, device)
        if checkpoint.get("training_contract") != contract:
            raise ValueError("Surface checkpoint contract differs from current config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_rank = tuple(checkpoint["best_rank"])
        log_path = run_dir / "training_log.csv"
        if log_path.exists():
            import pandas as pd

            rows = pd.read_csv(log_path).to_dict("records")

    for epoch in range(start_epoch, epochs):
        model.train()
        started = time.time()
        losses = []
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss, _ = _surface_batch(
                    model,
                    batch,
                    endpoint_weight=endpoint_weight,
                    decoded_dice_weight=decoded_weight,
                    decoded_temperature_mm=decoded_temperature,
                    path_noise_fraction=path_noise,
                    q0_jitter_fraction=q0_jitter,
                    augment=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation = _validate_surface(
            model,
            val_loader,
            device,
            endpoint_weight=endpoint_weight,
            decoded_dice_weight=decoded_weight,
            decoded_temperature_mm=decoded_temperature,
            heun_steps=heun_steps,
            amp=amp,
        )
        row = {
            "epoch": epoch,
            "fold": refinement_fold,
            "train_loss": float(np.mean(losses)),
            **validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        rows.append(row)
        _write_log(run_dir / "training_log.csv", rows)
        rank = (-validation["val_surface_soft_dice"], validation["val_surface_rmse_mm"])
        is_best = rank < best_rank
        state = {
            "schema_version": 3,
            "representation": "surface_h",
            "mode": "linear",
            "epoch": epoch,
            "refinement_fold": refinement_fold,
            "best_rank": rank if is_best else best_rank,
            "model": model.state_dict(),
            "model_config": model.model_config,
            "surface_stats": stats.to_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "training_contract": contract,
            "seed": seed,
        }
        temporary = last_path.with_suffix(".tmp.pt")
        torch.save(state, temporary)
        temporary.replace(last_path)
        if (epoch + 1) % archive_every == 0 or epoch + 1 == epochs:
            archived = archive_dir / f"epoch_{epoch + 1:04d}.pt"
            temporary = archived.with_suffix(".tmp.pt")
            torch.save(state, temporary)
            temporary.replace(archived)
        if is_best:
            best_rank = rank
            state["best_rank"] = best_rank
            temporary = best_path.with_suffix(".tmp.pt")
            torch.save(state, temporary)
            temporary.replace(best_path)
        print(
            f"[SURFACE R2 {epoch + 1:03d}/{epochs:03d}] "
            f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"softDice={row['val_surface_soft_dice']:.6f} "
            f"rmse={row['val_surface_rmse_mm']:.4f}mm "
            f"best={'yes' if is_best else 'no'}",
            flush=True,
        )
    return best_path


def load_surface_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[SurfaceDisplacementTransformer, dict[str, Any]]:
    checkpoint = _torch_load(checkpoint_path, device)
    stats = SurfaceStats.from_dict(checkpoint["surface_stats"])
    model = SurfaceDisplacementTransformer(
        stats=stats, **checkpoint["model_config"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
