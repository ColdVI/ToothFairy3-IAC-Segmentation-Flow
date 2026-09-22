"""Resume-safe mixed-precision training for direct and flow arms."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data import StateStats, compute_state_stats, save_stats, split_datasets
from .losses import (
    decoded_shell_soft_dice_loss,
    decoded_tube_metrics,
    physical_velocity_loss,
    state_surface_error,
)
from .model import CanalVelocityTransformer
from .paths import (
    heun_rollout,
    interpolate_noisy_linear_path,
    interpolate_path,
    project_ell_gauge,
    transport_increment_from_displacement,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _build_model(config: dict[str, Any], stats: StateStats, shell_channels: int) -> CanalVelocityTransformer:
    geometry = config.get("geometry", {})
    options = config.get("model", {})
    return CanalVelocityTransformer(
        shell_channels=shell_channels,
        stats=stats,
        stations=int(geometry.get("stations", 160)),
        n_angles=int(geometry.get("angles", 32)),
        max_radius_mm=float(geometry.get("max_radius_mm", 5.0)),
        profile_offsets_mm=list(options.get("profile_offsets_mm", [-0.6, -0.3, 0.0, 0.3, 0.6])),
        d_model=int(options.get("d_model", 192)),
        n_heads=int(options.get("n_heads", 6)),
        n_layers=int(options.get("n_layers", 4)),
        angular_hidden=int(options.get("angular_hidden", 64)),
        angular_modes=int(options.get("angular_modes", 8)),
        dropout=float(options.get("dropout", 0.10)),
        trust_surface_rmse_mm=float(options.get("trust_surface_rmse_mm", 1.0e6)),
        trust_endpoint_mm=float(options.get("trust_endpoint_mm", 1.0e6)),
        max_center_shift_mm=float(
            options.get(
                "max_center_shift_mm",
                geometry.get("max_center_shift_mm", 3.0),
            )
        ),
        probability_channel=int(options.get("probability_channel", 1)),
        evidence_threshold=float(options.get("evidence_threshold", 0.5)),
        evidence_radii=int(options.get("evidence_radii", 32)),
    )


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def _torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location=device)


def _train_batch(
    model: CanalVelocityTransformer,
    batch: dict[str, Any],
    mode: str,
    *,
    endpoint_weight: float,
    decoded_dice_weight: float,
    decoded_temperature_mm: float,
    path_noise_fraction: float,
    q0_jitter_fraction: float,
    augment: bool,
    time_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q0_local, q0_global = batch["q0_local"], batch["q0_global"]
    q1_local, q1_global = batch["q1_local"], batch["q1_global"]
    if augment and q0_jitter_fraction > 0.0:
        jitter_local = (
            torch.randn_like(q0_local)
            * model.output_local_scale[None, None]
            * float(q0_jitter_fraction)
        )
        jitter_local = jitter_local * batch["station_mask"][:, :, None].to(
            jitter_local.dtype
        )
        jitter_global = (
            torch.randn_like(q0_global)
            * model.output_global_scale[None]
            * float(q0_jitter_fraction)
        )
        q0_local = q0_local + jitter_local
        q0_global = q0_global + jitter_global
        q0_local, q0_global = project_ell_gauge(
            q0_local, q0_global, batch["station_mask"]
        )
        midpoint = q0_global[:, 1:3].mean(dim=1)
        half_length = (
            0.5 * (q0_global[:, 2] - q0_global[:, 1])
        ).clamp_min(0.125)
        q0_global = q0_global.clone()
        q0_global[:, 1] = midpoint - half_length
        q0_global[:, 2] = midpoint + half_length

    noise_local = torch.zeros_like(q0_local)
    noise_global = torch.zeros_like(q0_global)
    if augment and path_noise_fraction > 0.0:
        noise_local = (
            torch.randn_like(q0_local)
            * model.output_local_scale[None, None]
            * float(path_noise_fraction)
        )
        noise_local = noise_local * batch["station_mask"][:, :, None].to(
            noise_local.dtype
        )
        noise_global = (
            torch.randn_like(q0_global)
            * model.output_global_scale[None]
            * float(path_noise_fraction)
        )
        noise_local, noise_global = project_ell_gauge(
            noise_local, noise_global, batch["station_mask"]
        )
    if mode == "direct":
        time_value = torch.zeros(len(q0_local), device=q0_local.device, dtype=q0_local.dtype)
        reference_local, reference_global = q0_local, q0_global
        target_local, target_global = q1_local - q0_local, q1_global - q0_global
    else:
        if time_value is None:
            # One jittered stratum per sample keeps every mini-batch spread over
            # the complete path instead of occasionally starving G or H.
            batch_size = len(q0_local)
            time_value = (
                torch.arange(batch_size, device=q0_local.device, dtype=q0_local.dtype)
                + torch.rand(batch_size, device=q0_local.device, dtype=q0_local.dtype)
            ) / max(batch_size, 1)
            time_value = time_value[torch.randperm(batch_size, device=q0_local.device)]
            time_value = time_value * 0.9998 + 0.0001
        else:
            time_value = time_value.to(device=q0_local.device, dtype=q0_local.dtype)
        if mode == "linear":
            reference_local, reference_global, path_velocity_local, path_velocity_global = (
                interpolate_noisy_linear_path(
                    q0_local,
                    q0_global,
                    q1_local,
                    q1_global,
                    time_value,
                    noise_local,
                    noise_global,
                )
            )
        else:
            if augment and path_noise_fraction > 0.0:
                raise ValueError(
                    "Exact t(1-t) path noise is defined for the linear arm only; "
                    "staged-vs-linear is no longer a v2 primary experiment."
                )
            reference_local, reference_global, path_velocity_local, path_velocity_global = interpolate_path(
                q0_local, q0_global, q1_local, q1_global, time_value, mode
            )
        if mode == "staged":
            # Schedule-normalized parameterization: the network estimates the
            # underlying displacement and paths.py applies alpha'_G/L/H during
            # ODE integration. This removes the staged path's 3x velocity
            # pulses (9x squared-loss variance) without changing the path.
            target_local = q1_local - q0_local
            target_global = q1_global - q0_global
        else:
            target_local, target_global = path_velocity_local, path_velocity_global
    predicted_local, predicted_global = model(
        reference_local,
        reference_global,
        batch["shell"],
        batch["side_id"],
        batch["station_mask"],
        time_value,
    )
    physical_loss, diagnostics = physical_velocity_loss(
        predicted_local,
        predicted_global,
        target_local,
        target_global,
        reference_local,
        reference_global,
        batch["loss_mask"],
        endpoint_weight=endpoint_weight,
    )
    if mode == "direct":
        terminal_local = reference_local + predicted_local
        terminal_global = reference_global + predicted_global
    elif mode == "linear":
        remaining = 1.0 - time_value
        terminal_local = (
            reference_local
            + remaining[:, None, None] * predicted_local
            - remaining.square()[:, None, None] * noise_local
        )
        terminal_global = (
            reference_global
            + remaining[:, None] * predicted_global
            - remaining.square()[:, None] * noise_global
        )
    else:
        one = torch.ones_like(time_value)
        increment_local, increment_global = transport_increment_from_displacement(
            predicted_local, predicted_global, time_value, one, mode
        )
        terminal_local = reference_local + increment_local
        terminal_global = reference_global + increment_global
    terminal_local, terminal_global = project_ell_gauge(
        terminal_local, terminal_global, batch["station_mask"]
    )
    if decoded_dice_weight > 0.0:
        decoded_loss, decoded_dice = decoded_shell_soft_dice_loss(
            terminal_local,
            terminal_global,
            batch["target_occupancy_shell"],
            batch["shell_radii_mm"],
            batch["arc_mm"],
            batch["metric_mask"],
            temperature_mm=decoded_temperature_mm,
        )
    else:
        decoded_loss = physical_loss.new_zeros(())
        decoded_dice = physical_loss.new_ones(())
    total = physical_loss + float(decoded_dice_weight) * decoded_loss
    diagnostics.update(
        {
            "physical_loss": physical_loss.detach(),
            "decoded_loss": decoded_loss.detach(),
            "decoded_soft_dice": decoded_dice.detach(),
        }
    )
    return total, diagnostics


@torch.no_grad()
def _validate(
    model: CanalVelocityTransformer,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    *,
    heun_steps: int,
    endpoint_weight: float,
    decoded_dice_weight: float,
    decoded_temperature_mm: float,
    amp: bool,
    compute_decoded_metrics: bool,
) -> dict[str, float]:
    model.eval()
    weighted = {
        "loss": 0.0,
        "rmse": 0.0,
        "final": 0.0,
        "dice": 0.0,
        "hd95": 0.0,
        "base_dice": 0.0,
        "base_hd95": 0.0,
    }
    total_samples = 0
    decoded_samples = 0
    validation_times = (0.125, 0.375, 0.625, 0.875)
    for raw in loader:
        batch = _to_device(raw, device)
        batch_size = len(batch["q0_local"])
        with _autocast(device, amp):
            if mode == "direct":
                validation_calls = [None]
            else:
                validation_calls = [
                    torch.full(
                        (batch_size,),
                        value,
                        device=device,
                        dtype=batch["q0_local"].dtype,
                    )
                    for value in validation_times
                ]
            batch_losses, batch_rmses = [], []
            for fixed_time in validation_calls:
                loss, diagnostics = _train_batch(
                    model,
                    batch,
                    mode,
                    endpoint_weight=endpoint_weight,
                    decoded_dice_weight=decoded_dice_weight,
                    decoded_temperature_mm=decoded_temperature_mm,
                    path_noise_fraction=0.0,
                    q0_jitter_fraction=0.0,
                    augment=False,
                    time_value=fixed_time,
                )
                batch_losses.append(loss)
                batch_rmses.append(diagnostics["surface_rmse_mm"])
            loss = torch.stack(batch_losses).mean()
            velocity_rmse = torch.stack(batch_rmses).mean()
            if mode == "direct":
                time_value = torch.zeros(len(batch["q0_local"]), device=device)
                delta_local, delta_global = model(
                    batch["q0_local"],
                    batch["q0_global"],
                    batch["shell"],
                    batch["side_id"],
                    batch["station_mask"],
                    time_value,
                )
                final_local = batch["q0_local"] + delta_local
                final_global = batch["q0_global"] + delta_global
                final_local, final_global = project_ell_gauge(
                    final_local, final_global, batch["station_mask"]
                )
            else:
                final_local, final_global = heun_rollout(
                    model,
                    batch["q0_local"],
                    batch["q0_global"],
                    batch["shell"],
                    batch["side_id"],
                    batch["station_mask"],
                    steps=heun_steps,
                    mode=mode,
                )
            final_rmse, _ = state_surface_error(
                final_local,
                final_global,
                batch["q1_local"],
                batch["q1_global"],
                batch["loss_mask"],
            )
        if compute_decoded_metrics:
            with torch.autocast(device_type=device.type, enabled=False):
                decoded_dice, decoded_hd = decoded_tube_metrics(
                    final_local.float(),
                    final_global.float(),
                    batch["q1_local"].float(),
                    batch["q1_global"].float(),
                    batch["arc_mm"].float(),
                    batch["metric_mask"],
                )
                decoded_base_dice, decoded_base_hd = decoded_tube_metrics(
                    batch["q0_local"].float(),
                    batch["q0_global"].float(),
                    batch["q1_local"].float(),
                    batch["q1_global"].float(),
                    batch["arc_mm"].float(),
                    batch["metric_mask"],
                )
        weighted["loss"] += float(loss) * batch_size
        weighted["rmse"] += float(velocity_rmse) * batch_size
        weighted["final"] += float(final_rmse) * batch_size
        total_samples += batch_size
        if compute_decoded_metrics:
            weighted["dice"] += float(decoded_dice) * batch_size
            weighted["hd95"] += float(decoded_hd) * batch_size
            weighted["base_dice"] += float(decoded_base_dice) * batch_size
            weighted["base_hd95"] += float(decoded_base_hd) * batch_size
            decoded_samples += batch_size
    total_samples = max(total_samples, 1)
    return {
        "val_loss": weighted["loss"] / total_samples,
        "val_velocity_surface_rmse_mm": weighted["rmse"] / total_samples,
        "val_final_surface_rmse_mm": weighted["final"] / total_samples,
        "val_tube_dice": weighted["dice"] / decoded_samples if decoded_samples else float("nan"),
        "val_tube_hd95_mm": weighted["hd95"] / decoded_samples if decoded_samples else float("nan"),
        "val_base_tube_dice": weighted["base_dice"] / decoded_samples if decoded_samples else float("nan"),
        "val_base_tube_hd95_mm": weighted["base_hd95"] / decoded_samples if decoded_samples else float("nan"),
    }


def _write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train(config: dict[str, Any], *, mode: str, refinement_fold: int, resume: bool = True) -> Path:
    if mode not in {"direct", "linear", "staged"}:
        raise ValueError("mode must be direct, linear or staged")
    training = config.get("training", {})
    seed = int(training.get("seed", 20260826)) + int(refinement_fold)
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not bool(training.get("allow_cpu_training", False)):
        raise RuntimeError("GPU not detected. Select a CUDA runtime or set allow_cpu_training=true for a smoke run.")

    train_dataset, val_dataset = split_datasets(config, refinement_fold)
    sample = train_dataset[0]
    shell_channels = int(sample["shell"].shape[0])
    run_root = Path(config["paths"]["run_dir"]).expanduser().resolve()
    run_dir = run_root / f"fold_{refinement_fold}" / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_file = run_dir / "state_stats.json"
    if stats_file.exists():
        stats = StateStats.from_dict(json.loads(stats_file.read_text(encoding="utf-8")))
    else:
        stats = compute_state_stats(train_dataset)
        save_stats(stats, stats_file)
    model = _build_model(config, stats, shell_channels).to(device)
    model.transport_mode = mode

    # Verify identity at initialization; training is still allowed to move it.
    model.eval()
    with torch.no_grad():
        probe = _to_device({key: value[None] if torch.is_tensor(value) else value for key, value in sample.items()}, device)
        zeros_local, zeros_global = model(
            probe["q0_local"],
            probe["q0_global"],
            probe["shell"],
            probe["side_id"],
            probe["station_mask"],
            torch.zeros(1, device=device),
        )
        if not (torch.equal(zeros_local, torch.zeros_like(zeros_local)) and torch.equal(zeros_global, torch.zeros_like(zeros_global))):
            raise AssertionError("Zero-initialized velocity head does not preserve the coarse state exactly")

    batch_size = int(training.get("batch_size", 8))
    workers = int(training.get("dataloader_workers", 2))
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_options)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs_direct" if mode == "direct" else "epochs_flow", 120))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except TypeError:  # PyTorch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    endpoint_weight = float(training.get("endpoint_weight", 1.0))
    decoded_dice_weight = float(training.get("decoded_dice_weight", 0.30))
    decoded_temperature_mm = float(training.get("decoded_temperature_mm", 0.10))
    path_noise_fraction = float(training.get("path_noise_fraction", 0.10))
    q0_jitter_fraction = float(training.get("q0_jitter_fraction", 0.20))
    if mode == "staged" and path_noise_fraction > 0.0:
        raise ValueError(
            "training.path_noise_fraction must be 0 for legacy staged mode. "
            "The v2 primary arm is linear flow."
        )
    heun_steps = int(training.get("heun_steps", 4))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    checkpoint_every = max(1, int(training.get("checkpoint_every_epochs", 1)))
    selection_every = max(1, int(training.get("selection_every_epochs", 1)))
    archive_every = max(
        1, int(training.get("archive_checkpoint_every_epochs", selection_every))
    )
    last_checkpoint, best_checkpoint = run_dir / "last.pt", run_dir / "best.pt"
    checkpoint_archive = run_dir / "checkpoints"
    checkpoint_archive.mkdir(exist_ok=True)
    training_contract = {
        "decoded_dice_weight": decoded_dice_weight,
        "decoded_temperature_mm": decoded_temperature_mm,
        "path_noise_fraction": path_noise_fraction,
        "q0_jitter_fraction": q0_jitter_fraction,
        "selection_fold_only": True,
    }
    start_epoch, best_metric, rows = 0, float("inf"), []
    best_rank = (2, float("inf"))

    if resume and last_checkpoint.exists():
        checkpoint = _torch_load(last_checkpoint, device)
        if checkpoint["mode"] != mode or int(checkpoint["refinement_fold"]) != int(refinement_fold):
            raise ValueError("Checkpoint mode/fold does not match this run")
        if checkpoint.get("training_contract") != training_contract:
            raise ValueError(
                "Checkpoint augmentation/objective contract differs from the current "
                "configuration; use a new run_dir or --no-resume."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        best_rank = tuple(checkpoint.get("best_rank", (0, best_metric)))
        log_path = run_dir / "training_log.csv"
        if log_path.exists():
            import pandas as pd

            rows = pd.read_csv(log_path).to_dict("records")

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"[CMF TRAIN] mode={mode} fold={refinement_fold} "
        f"train_shards={len(train_dataset)} val_shards={len(val_dataset)} "
        f"batches_per_epoch={len(train_loader)} params={trainable_parameters:,} "
        f"epochs={epochs} next_epoch={start_epoch + 1}",
        flush=True,
    )

    if start_epoch == 0:
        initial_validation = _validate(
            model,
            val_loader,
            device,
            mode,
            heun_steps=heun_steps,
            endpoint_weight=endpoint_weight,
            decoded_dice_weight=decoded_dice_weight,
            decoded_temperature_mm=decoded_temperature_mm,
            amp=amp,
            compute_decoded_metrics=True,
        )
        initial_row = {
            "epoch": -1,
            "mode": mode,
            "fold": refinement_fold,
            "train_loss": float("nan"),
            "train_physical_loss": float("nan"),
            "train_decoded_loss": float("nan"),
            "train_surface_rmse_mm": float("nan"),
            "train_endpoint_rmse_mm": float("nan"),
            **initial_validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": 0.0,
            "optimizer_steps": 0,
            "gradient_norm_mean": float("nan"),
        }
        rows.append(initial_row)
        _write_log(run_dir / "training_log.csv", rows)
        epoch0_state = {
            "schema_version": 3,
            "epoch": -1,
            "mode": mode,
            "refinement_fold": refinement_fold,
            "best_metric": float("inf"),
            "best_rank": (float("inf"), float("inf")),
            "model": model.state_dict(),
            "model_config": model.model_config,
            "state_stats": stats.to_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "seed": seed,
            "training_contract": training_contract,
            "velocity_parameterization": (
                "schedule_normalized_displacement" if mode == "staged" else "velocity"
            ),
        }
        torch.save(epoch0_state, run_dir / "epoch_0000_identity.pt")
        print(
            f"[CMF EPOCH 000/{epochs:04d}] "
            f"dice={initial_validation['val_tube_dice']:.6f} "
            f"hd95={initial_validation['val_tube_hd95_mm']:.6f}mm",
            flush=True,
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()
        losses, physical_losses, decoded_losses = [], [], []
        surface_rmses, endpoint_rmses, gradient_norms = [], [], []
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss, diagnostics = _train_batch(
                    model,
                    batch,
                    mode,
                    endpoint_weight=endpoint_weight,
                    decoded_dice_weight=decoded_dice_weight,
                    decoded_temperature_mm=decoded_temperature_mm,
                    path_noise_fraction=path_noise_fraction,
                    q0_jitter_fraction=q0_jitter_fraction,
                    augment=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            physical_losses.append(float(diagnostics["physical_loss"]))
            decoded_losses.append(float(diagnostics["decoded_loss"]))
            surface_rmses.append(float(diagnostics["surface_rmse_mm"]))
            endpoint_rmses.append(float(diagnostics["endpoint_rmse_mm"]))
            gradient_norms.append(float(gradient_norm.detach()))
        scheduler.step()
        validation = _validate(
            model,
            val_loader,
            device,
            mode,
            heun_steps=heun_steps,
            endpoint_weight=endpoint_weight,
            decoded_dice_weight=decoded_dice_weight,
            decoded_temperature_mm=decoded_temperature_mm,
            amp=amp,
            compute_decoded_metrics=((epoch + 1) % selection_every == 0 or epoch + 1 == epochs),
        )
        row = {
            "epoch": epoch,
            "mode": mode,
            "fold": refinement_fold,
            "train_loss": float(np.mean(losses)),
            "train_physical_loss": float(np.mean(physical_losses)),
            "train_decoded_loss": float(np.mean(decoded_losses)),
            "train_surface_rmse_mm": float(np.mean(surface_rmses)),
            "train_endpoint_rmse_mm": float(np.mean(endpoint_rmses)),
            **validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
            "optimizer_steps": (epoch + 1) * len(train_loader),
            "gradient_norm_mean": float(np.mean(gradient_norms)),
        }
        rows.append(row)
        _write_log(run_dir / "training_log.csv", rows)
        has_selection_metrics = np.isfinite(validation["val_tube_dice"])
        if has_selection_metrics:
            # Dice is primary and per-case HD95 is the deterministic tie-breaker.
            # The former non-inferiority gate could prefer a lower-Dice epoch
            # solely because it was closer to q0 under a pooled HD proxy.
            rank = (
                -validation["val_tube_dice"],
                validation["val_tube_hd95_mm"],
            )
            metric = -validation["val_tube_dice"]
        else:
            rank, metric = best_rank, best_metric
        is_new_best = has_selection_metrics and rank < best_rank
        state = {
            "schema_version": 3,
            "epoch": epoch,
            "mode": mode,
            "refinement_fold": refinement_fold,
            "best_metric": metric if is_new_best else best_metric,
            "best_rank": rank if is_new_best else best_rank,
            "model": model.state_dict(),
            "model_config": model.model_config,
            "state_stats": stats.to_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "seed": seed,
            "training_contract": training_contract,
            "velocity_parameterization": (
                "schedule_normalized_displacement" if mode == "staged" else "velocity"
            ),
        }
        if (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs:
            temporary = last_checkpoint.with_suffix(".tmp.pt")
            torch.save(state, temporary)
            temporary.replace(last_checkpoint)
        if (epoch + 1) % archive_every == 0 or epoch + 1 == epochs:
            archived = checkpoint_archive / f"epoch_{epoch + 1:04d}.pt"
            temporary = archived.with_suffix(".tmp.pt")
            torch.save(state, temporary)
            temporary.replace(archived)
        if is_new_best:
            best_rank = rank
            best_metric = metric
            state["best_metric"] = best_metric
            state["best_rank"] = best_rank
            temporary = best_checkpoint.with_suffix(".tmp.pt")
            torch.save(state, temporary)
            temporary.replace(best_checkpoint)
        print(
            f"[CMF EPOCH {epoch + 1:04d}/{epochs:04d}] "
            f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"dice={row['val_tube_dice']:.6f} "
            f"hd95={row['val_tube_hd95_mm']:.6f}mm "
            f"grad={row['gradient_norm_mean']:.4f} "
            f"time={row['seconds']:.1f}s "
            f"best={'yes' if is_new_best else 'no'}",
            flush=True,
        )
    return best_checkpoint


def load_trained_model(checkpoint_path: str | Path, device: torch.device) -> tuple[CanalVelocityTransformer, dict[str, Any]]:
    checkpoint = _torch_load(checkpoint_path, device)
    stats = StateStats.from_dict(checkpoint["state_stats"])
    model = CanalVelocityTransformer(stats=stats, **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.transport_mode = str(checkpoint.get("mode", "linear"))
    model.eval()
    return model, checkpoint
