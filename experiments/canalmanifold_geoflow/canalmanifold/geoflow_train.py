"""Training, rollout, and checkpoint I/O for Geometric Boundary Flow.

This module deliberately keeps two diagnostics separate:

* ``velocity_drift`` describes how much the predicted velocity changed during
  rollout; it is not a success threshold because the supervised linear path
  has a constant on-path target velocity.
* ``profile_remeasurement_rms`` verifies that the probability profile was
  actually read at a different moving-surface location.

Whether the profile feedback improves segmentation must ultimately be shown by
the exact paired evaluation (and an ablation), not by either diagnostic alone.
"""

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
from .geoflow_model import GeometricBoundaryFlow
from .surface_model import SurfaceStats
from .surface_train import compute_surface_stats, surface_shell_soft_dice_loss
from .train import _autocast, _to_device, _torch_load, seed_everything


def geoflow_physical_loss(
    predicted_h: torch.Tensor,
    predicted_endpoints: torch.Tensor,
    target_h: torch.Tensor,
    target_endpoints: torch.Tensor,
    current_radius_mm: torch.Tensor,
    ray_mask: torch.Tensor,
    *,
    reference_radius_mm: float,
    endpoint_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Area-weighted normal-velocity loss in physical millimetres.

    A ray is weighted by ``r_t / r_ref`` while the denominator remains the
    number of supervised rays.  Consequently, the same squared millimetre
    error has proportionally higher cost at a thicker surface element, which
    matches the first-order lateral-area change.
    """

    mask = ray_mask.to(predicted_h.dtype)
    area_weight = current_radius_mm.detach().clamp_min(0.12) / max(
        float(reference_radius_mm), 0.12
    )
    squared = (predicted_h - target_h).square()
    count = mask.sum().clamp_min(1.0)
    area = (squared * mask * area_weight).sum() / count
    unweighted = (squared * mask).sum() / count
    endpoint = (predicted_endpoints - target_endpoints).square().mean()
    loss = area + float(endpoint_weight) * endpoint
    return loss, {
        "area_velocity_rmse_mm": torch.sqrt(area.detach().clamp_min(0.0)),
        "surface_rmse_mm": torch.sqrt(unweighted.detach().clamp_min(0.0)),
        "endpoint_rmse_mm": torch.sqrt(endpoint.detach().clamp_min(0.0)),
    }


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, *, absolute: bool = False
) -> torch.Tensor:
    selected = values.abs() if absolute else values
    weights = mask.to(selected.dtype)
    while weights.ndim < selected.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(selected)
    return (selected * weights).sum() / weights.sum().clamp_min(1.0)


def _jitter_and_noise(
    model: GeometricBoundaryFlow,
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
            * model.h_output_scale.to(h0)[None, None]
            * float(q0_jitter_fraction)
        )
        h_origin = h_origin + jitter * station_mask[:, :, None].to(h0.dtype)
        endpoint_origin = endpoint_origin + (
            torch.randn_like(endpoints0)
            * model.endpoint_output_scale.to(endpoints0)[None]
            * float(q0_jitter_fraction)
        )

    noise_h = torch.zeros_like(h0)
    noise_endpoint = torch.zeros_like(endpoints0)
    if augment and path_noise_fraction > 0.0:
        noise_h = (
            torch.randn_like(h0)
            * model.h_output_scale.to(h0)[None, None]
            * float(path_noise_fraction)
        )
        noise_h = noise_h * station_mask[:, :, None].to(h0.dtype)
        noise_endpoint = (
            torch.randn_like(endpoints0)
            * model.endpoint_output_scale.to(endpoints0)[None]
            * float(path_noise_fraction)
        )
    return h_origin, endpoint_origin, noise_h, noise_endpoint


def _stratified_time(reference: torch.Tensor) -> torch.Tensor:
    batch = len(reference)
    time_value = (
        torch.arange(batch, device=reference.device, dtype=reference.dtype)
        + torch.rand(batch, device=reference.device, dtype=reference.dtype)
    ) / max(batch, 1)
    time_value = time_value[torch.randperm(batch, device=reference.device)]
    return time_value * 0.9998 + 0.0001


def _geoflow_batch(
    model: GeometricBoundaryFlow,
    batch: dict[str, Any],
    *,
    endpoint_weight: float,
    decoded_dice_weight: float,
    decoded_temperature_mm: float,
    path_noise_fraction: float,
    q0_jitter_fraction: float,
    dilation_penalty_weight: float,
    evidence_gain_penalty_weight: float,
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
        time_value = _stratified_time(h0)
    else:
        time_value = time_value.to(device=h0.device, dtype=h0.dtype)

    bridge = time_value * (1.0 - time_value)
    bridge_derivative = 1.0 - 2.0 * time_value
    delta_h = target_h - h0
    delta_endpoint = endpoint1 - endpoint0
    current_h = (
        h0
        + time_value[:, None, None] * delta_h
        + bridge[:, None, None] * noise_h
    )
    current_endpoint = (
        endpoint0
        + time_value[:, None] * delta_endpoint
        + bridge[:, None] * noise_endpoint
    )
    target_velocity_h = delta_h + bridge_derivative[:, None, None] * noise_h
    target_velocity_endpoint = (
        delta_endpoint + bridge_derivative[:, None] * noise_endpoint
    )

    predicted_h, predicted_endpoint, model_diagnostics = model(
        batch["q0_ray_radii_mm"],
        current_h,
        current_endpoint,
        batch["shell"],
        batch["side_id"],
        batch["station_mask"],
        time_value,
        batch.get("arc_mm"),
    )
    physical, diagnostics = geoflow_physical_loss(
        predicted_h,
        predicted_endpoint,
        target_velocity_h,
        target_velocity_endpoint,
        model_diagnostics["current_radius_mm"],
        batch["ray_loss_mask"],
        reference_radius_mm=model.radius_reference_mm,
        endpoint_weight=endpoint_weight,
    )

    # Analytic terminal extrapolation for the exact t(1-t) bridge.
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
    dilation_penalty = model_diagnostics["alpha_mm"].square().mean()
    gain_mask = model_diagnostics["profile_valid"] & batch["station_mask"][:, :, None]
    evidence_gain_penalty = _masked_mean(
        model_diagnostics["evidence_gain"].square(), gain_mask
    )
    total = (
        physical
        + float(decoded_dice_weight) * decoded
        + float(dilation_penalty_weight) * dilation_penalty
        + float(evidence_gain_penalty_weight) * evidence_gain_penalty
    )
    diagnostics.update(
        {
            "physical_loss": physical.detach(),
            "decoded_loss": decoded.detach(),
            "decoded_soft_dice": decoded_dice,
            "dilation_penalty": dilation_penalty.detach(),
            "evidence_gain_penalty": evidence_gain_penalty.detach(),
            "alpha_mm": model_diagnostics["alpha_mm"].detach().abs().mean(),
            "evidence_gain": _masked_mean(
                model_diagnostics["evidence_gain"].detach(), gain_mask
            ),
            "profile_valid_fraction": _masked_mean(
                model_diagnostics["profile_valid"].to(predicted_h.dtype),
                batch["station_mask"],
            ).detach(),
        }
    )
    return total, diagnostics


def _project_trust_region(
    model: GeometricBoundaryFlow,
    q0_radius: torch.Tensor,
    h: torch.Tensor,
    endpoints: torch.Tensor,
    origin_h: torch.Tensor,
    origin_endpoints: torch.Tensor,
    station_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = h - origin_h
    weights = station_mask.to(h.dtype)[:, :, None]
    rmse = torch.sqrt(
        (difference.square() * weights).sum(dim=(1, 2))
        / (weights.sum(dim=(1, 2)) * h.shape[2]).clamp_min(1.0)
    )
    limit = torch.as_tensor(
        model.trust_surface_rmse_mm, device=h.device, dtype=h.dtype
    )
    scale = torch.minimum(torch.ones_like(rmse), limit / rmse.clamp_min(1e-6))
    h = origin_h + scale[:, None, None] * difference
    h = torch.maximum(h, torch.as_tensor(0.12, device=h.device, dtype=h.dtype) - q0_radius)
    h = torch.minimum(
        h,
        torch.as_tensor(model.max_radius_mm, device=h.device, dtype=h.dtype)
        - q0_radius,
    )
    endpoint_delta = (endpoints - origin_endpoints).clamp(
        -model.trust_endpoint_mm, model.trust_endpoint_mm
    )
    return h, origin_endpoints + endpoint_delta


def _masked_rms(values: torch.Tensor, station_mask: torch.Tensor) -> torch.Tensor:
    weights = station_mask.to(values.dtype)[:, :, None].expand_as(values)
    return torch.sqrt((values.square() * weights).sum() / weights.sum().clamp_min(1.0))


@torch.no_grad()
def geoflow_heun_rollout(
    model: GeometricBoundaryFlow,
    q0_radius: torch.Tensor,
    endpoints0: torch.Tensor,
    shell: torch.Tensor,
    side_id: torch.Tensor,
    station_mask: torch.Tensor,
    arc_mm: torch.Tensor | None = None,
    *,
    steps: int,
    initial_h: torch.Tensor | None = None,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
]:
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    h = torch.zeros_like(q0_radius) if initial_h is None else initial_h.clone()
    endpoints = endpoints0.clone()
    origin_h = h.clone()
    origin_endpoints = endpoints.clone()
    batch = len(h)
    first_velocity = first_diagnostics = None
    for index in range(int(steps)):
        t0 = torch.full(
            (batch,), index / steps, device=h.device, dtype=h.dtype
        )
        t1 = torch.full(
            (batch,), (index + 1) / steps, device=h.device, dtype=h.dtype
        )
        velocity0_h, velocity0_endpoint, diagnostics0 = model(
            q0_radius,
            h,
            endpoints,
            shell,
            side_id,
            station_mask,
            t0,
            arc_mm,
        )
        if first_velocity is None:
            first_velocity = velocity0_h.clone()
            first_diagnostics = diagnostics0
        dt = 1.0 / steps
        euler_h = h + dt * velocity0_h
        euler_endpoint = endpoints + dt * velocity0_endpoint
        velocity1_h, velocity1_endpoint, _ = model(
            q0_radius,
            euler_h,
            euler_endpoint,
            shell,
            side_id,
            station_mask,
            t1,
            arc_mm,
        )
        h = h + 0.5 * dt * (velocity0_h + velocity1_h)
        endpoints = endpoints + 0.5 * dt * (
            velocity0_endpoint + velocity1_endpoint
        )
        h, endpoints = _project_trust_region(
            model,
            q0_radius,
            h,
            endpoints,
            origin_h,
            origin_endpoints,
            station_mask,
        )

    if not return_diagnostics:
        return h, endpoints
    final_time = torch.ones(batch, device=h.device, dtype=h.dtype)
    final_velocity, _, final_diagnostics = model(
        q0_radius,
        h,
        endpoints,
        shell,
        side_id,
        station_mask,
        final_time,
        arc_mm,
    )
    assert first_velocity is not None and first_diagnostics is not None
    profile_mask = final_diagnostics["profile_valid"] & station_mask[:, :, None]
    diagnostics = {
        "velocity_drift": _masked_rms(
            final_velocity - first_velocity, station_mask
        ),
        "profile_remeasurement_rms": _masked_rms(
            final_diagnostics["probability_at_surface"]
            - first_diagnostics["probability_at_surface"],
            station_mask,
        ),
        "newton_drift_mm": _masked_rms(
            final_diagnostics["newton_mm"] - first_diagnostics["newton_mm"],
            station_mask,
        ),
        "alpha_mm": 0.5
        * (
            first_diagnostics["alpha_mm"].abs().mean()
            + final_diagnostics["alpha_mm"].abs().mean()
        ),
        "evidence_gain": _masked_mean(
            final_diagnostics["evidence_gain"], profile_mask
        ),
        "evidence_gain_abs": _masked_mean(
            final_diagnostics["evidence_gain"], profile_mask, absolute=True
        ),
        "profile_valid_fraction": _masked_mean(
            final_diagnostics["profile_valid"].to(h.dtype), station_mask
        ),
        "newton_abs_mm": _masked_mean(
            final_diagnostics["newton_mm"], profile_mask, absolute=True
        ),
        "sharpness_per_mm": _masked_mean(
            final_diagnostics["sharpness_per_mm"], profile_mask
        ),
        "evidence_contribution_rms": _masked_rms(
            final_diagnostics["evidence_term_mm"], station_mask
        ),
        "learned_base_rms": _masked_rms(
            final_diagnostics["learned_base_mm"], station_mask
        ),
    }
    return h, endpoints, diagnostics


@torch.no_grad()
def _validate_geoflow(
    model: GeometricBoundaryFlow,
    loader: DataLoader,
    device: torch.device,
    *,
    endpoint_weight: float,
    decoded_dice_weight: float,
    decoded_temperature_mm: float,
    dilation_penalty_weight: float,
    evidence_gain_penalty_weight: float,
    heun_steps: int,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {
        "loss": 0.0,
        "dice": 0.0,
        "rmse": 0.0,
        "velocity_drift": 0.0,
        "profile_remeasurement_rms": 0.0,
        "newton_drift_mm": 0.0,
        "alpha_mm": 0.0,
        "evidence_gain": 0.0,
        "evidence_gain_abs": 0.0,
        "profile_valid_fraction": 0.0,
        "newton_abs_mm": 0.0,
        "sharpness_per_mm": 0.0,
    }
    count = 0
    validation_times = (0.125, 0.375, 0.625, 0.875)
    for raw in loader:
        batch = _to_device(raw, device)
        batch_size = len(batch["q0_local"])
        losses = []
        with _autocast(device, amp):
            for value in validation_times:
                fixed_time = torch.full(
                    (batch_size,),
                    value,
                    device=device,
                    dtype=batch["q0_local"].dtype,
                )
                loss, _ = _geoflow_batch(
                    model,
                    batch,
                    endpoint_weight=endpoint_weight,
                    decoded_dice_weight=decoded_dice_weight,
                    decoded_temperature_mm=decoded_temperature_mm,
                    path_noise_fraction=0.0,
                    q0_jitter_fraction=0.0,
                    dilation_penalty_weight=dilation_penalty_weight,
                    evidence_gain_penalty_weight=evidence_gain_penalty_weight,
                    augment=False,
                    time_value=fixed_time,
                )
                losses.append(loss)
            final_h, final_endpoint, rollout = geoflow_heun_rollout(
                model,
                batch["q0_ray_radii_mm"],
                batch["q0_global"][:, 1:3],
                batch["shell"],
                batch["side_id"],
                batch["station_mask"],
                batch.get("arc_mm"),
                steps=heun_steps,
                return_diagnostics=True,
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
        totals["loss"] += float(torch.stack(losses).mean()) * batch_size
        totals["dice"] += float(dice) * batch_size
        totals["rmse"] += float(rmse) * batch_size
        for key in totals:
            if key in {"loss", "dice", "rmse"}:
                continue
            totals[key] += float(rollout[key]) * batch_size
        count += batch_size
    denominator = max(count, 1)
    return {
        "val_loss": totals["loss"] / denominator,
        "val_soft_dice": totals["dice"] / denominator,
        "val_rmse_mm": totals["rmse"] / denominator,
        "val_velocity_drift": totals["velocity_drift"] / denominator,
        "val_profile_remeasurement_rms": totals["profile_remeasurement_rms"]
        / denominator,
        "val_newton_drift_mm": totals["newton_drift_mm"] / denominator,
        "val_alpha_mm": totals["alpha_mm"] / denominator,
        "val_evidence_gain": totals["evidence_gain"] / denominator,
        "val_evidence_gain_abs": totals["evidence_gain_abs"] / denominator,
        "val_profile_valid_fraction": totals["profile_valid_fraction"] / denominator,
        "val_newton_abs_mm": totals["newton_abs_mm"] / denominator,
        "val_sharpness_per_mm": totals["sharpness_per_mm"] / denominator,
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


def _build_geoflow(
    config: dict[str, Any], stats: SurfaceStats, shell_channels: int
) -> GeometricBoundaryFlow:
    geometry = config.get("geometry", {})
    # Surface options are fallback defaults only.  GeoFlow-specific values win.
    options = {**config.get("surface_model", {}), **config.get("geoflow_model", {})}
    return GeometricBoundaryFlow(
        shell_channels=shell_channels,
        stats=stats,
        stations=int(geometry.get("stations", 160)),
        n_angles=int(geometry.get("angles", 32)),
        shell_radii=int(geometry.get("shell_radii", 24)),
        max_radius_mm=float(geometry.get("max_radius_mm", 5.0)),
        profile_offsets_mm=list(
            options.get("profile_offsets_mm", [-0.6, -0.3, 0.0, 0.3, 0.6])
        ),
        d_model=int(options.get("d_model", 64)),
        n_heads=int(options.get("n_heads", 4)),
        n_layers=int(options.get("n_layers", 2)),
        angle_hidden=int(options.get("angle_hidden", 24)),
        angular_modes=int(options.get("angular_modes", 6)),
        dropout=float(options.get("dropout", 0.10)),
        probability_channel=int(options.get("probability_channel", 1)),
        evidence_threshold=float(options.get("evidence_threshold", 0.5)),
        minimum_slope_per_mm=float(options.get("minimum_slope_per_mm", 0.05)),
        maximum_newton_step_mm=float(
            options.get("maximum_newton_step_mm", 0.60)
        ),
        sharpness_cap_per_mm=float(options.get("sharpness_cap_per_mm", 4.0)),
        evidence_gain_limit=float(options.get("evidence_gain_limit", 2.0)),
        evidence_gain_init=float(options.get("evidence_gain_init", 0.0)),
        curvature_ratio_limit=float(options.get("curvature_ratio_limit", 4.0)),
        axial_curvature_clip=float(options.get("axial_curvature_clip", 4.0)),
        nominal_spacing_mm=float(options.get("nominal_spacing_mm", 0.30)),
        trust_surface_rmse_mm=float(options.get("trust_surface_rmse_mm", 1.0)),
        trust_endpoint_mm=float(options.get("trust_endpoint_mm", 2.0)),
    )


def train_geoflow(
    config: dict[str, Any], *, refinement_fold: int, resume: bool = True
) -> Path:
    training = {**config.get("training", {}), **config.get("geoflow_training", {})}
    seed = int(training.get("seed", 20260831)) + 200 + int(refinement_fold)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not bool(training.get("allow_cpu_training", False)):
        raise RuntimeError("GPU not detected; set allow_cpu_training=true only for a smoke run")

    train_dataset, val_dataset = split_datasets(config, refinement_fold)
    sample = train_dataset[0]
    run_dir = (
        Path(config["paths"]["run_dir"]).expanduser().resolve()
        / f"fold_{refinement_fold}"
        / "geoflow_newton"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = run_dir / "surface_stats.json"
    if stats_path.exists():
        stats = SurfaceStats.from_dict(json.loads(stats_path.read_text(encoding="utf-8")))
    else:
        stats = compute_surface_stats(train_dataset)
        stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    model = _build_geoflow(config, stats, int(sample["shell"].shape[0])).to(device)

    with torch.no_grad():
        probe = _to_device(
            {
                key: value[None] if torch.is_tensor(value) else value
                for key, value in sample.items()
            },
            device,
        )
        zero_h, zero_endpoint, _ = model(
            probe["q0_ray_radii_mm"],
            torch.zeros_like(probe["q0_ray_radii_mm"]),
            probe["q0_global"][:, 1:3],
            probe["shell"],
            probe["side_id"],
            probe["station_mask"],
            torch.zeros(1, device=device),
            probe.get("arc_mm"),
        )
        if float(model.model_config["evidence_gain_init"]) == 0.0 and not (
            torch.equal(zero_h, torch.zeros_like(zero_h))
            and torch.equal(zero_endpoint, torch.zeros_like(zero_endpoint))
        ):
            raise AssertionError("GeoFlow zero initialization is not exact identity")

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
    epochs = int(training.get("epochs_geoflow", 60))
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
    dilation_penalty = float(training.get("dilation_penalty_weight", 0.0))
    evidence_gain_penalty = float(
        training.get("evidence_gain_penalty_weight", 0.0)
    )
    heun_steps = int(training.get("heun_steps", 4))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    archive_every = max(1, int(training.get("archive_checkpoint_every_epochs", 1)))
    contract = {
        "evidence_version": model.evidence_version,
        "decoded_dice_weight": decoded_weight,
        "decoded_temperature_mm": decoded_temperature,
        "path_noise_fraction": path_noise,
        "q0_jitter_fraction": q0_jitter,
        "dilation_penalty_weight": dilation_penalty,
        "evidence_gain_penalty_weight": evidence_gain_penalty,
        "heun_steps": heun_steps,
    }

    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    archive_dir = run_dir / "checkpoints"
    archive_dir.mkdir(exist_ok=True)
    start_epoch, best_rank, rows = 0, (float("inf"), float("inf")), []
    if resume and last_path.exists():
        checkpoint = _torch_load(last_path, device)
        if checkpoint.get("training_contract") != contract:
            raise ValueError(
                "GeoFlow checkpoint contract differs from the current config; "
                "use --no-resume or a new run_dir"
            )
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
                loss, _ = _geoflow_batch(
                    model,
                    batch,
                    endpoint_weight=endpoint_weight,
                    decoded_dice_weight=decoded_weight,
                    decoded_temperature_mm=decoded_temperature,
                    path_noise_fraction=path_noise,
                    q0_jitter_fraction=q0_jitter,
                    dilation_penalty_weight=dilation_penalty,
                    evidence_gain_penalty_weight=evidence_gain_penalty,
                    augment=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation = _validate_geoflow(
            model,
            val_loader,
            device,
            endpoint_weight=endpoint_weight,
            decoded_dice_weight=decoded_weight,
            decoded_temperature_mm=decoded_temperature,
            dilation_penalty_weight=dilation_penalty,
            evidence_gain_penalty_weight=evidence_gain_penalty,
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
        rank = (-validation["val_soft_dice"], validation["val_rmse_mm"])
        is_best = rank < best_rank
        state = {
            "schema_version": 4,
            "representation": "geoflow_newton",
            "mode": "closed_loop_geometric_basis",
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
            f"[GEOFLOW {epoch + 1:03d}/{epochs:03d}] "
            f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"softDice={row['val_soft_dice']:.6f} "
            f"rmse={row['val_rmse_mm']:.4f}mm "
            f"drift={row['val_velocity_drift']:.4f} "
            f"profileMove={row['val_profile_remeasurement_rms']:.4f} "
            f"alpha={row['val_alpha_mm']:.4f}mm "
            f"best={'yes' if is_best else 'no'}",
            flush=True,
        )
    return best_path


def load_geoflow_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[GeometricBoundaryFlow, dict[str, Any]]:
    checkpoint = _torch_load(checkpoint_path, device)
    if checkpoint.get("representation") != "geoflow_newton":
        raise ValueError(
            "This evaluator only accepts corrected geoflow_newton checkpoints; "
            "legacy crossing-current checkpoints are intentionally incompatible"
        )
    stats = SurfaceStats.from_dict(checkpoint["surface_stats"])
    model = GeometricBoundaryFlow(
        stats=stats, **checkpoint["model_config"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
