"""Decoder-pullback physical velocity objective in millimetres."""

from __future__ import annotations

import math

import torch

from .constants import HARMONIC_START, HARMONICS, LOCAL_DIM


def physical_velocity_loss(
    predicted_local: torch.Tensor,
    predicted_global: torch.Tensor,
    target_local: torch.Tensor,
    target_global: torch.Tensor,
    reference_local: torch.Tensor,
    reference_global: torch.Tensor,
    station_mask: torch.Tensor,
    *,
    n_angles: int = 32,
    endpoint_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dtype, device = reference_local.dtype, reference_local.device
    angles = torch.linspace(0.0, 2.0 * math.pi, n_angles + 1, device=device, dtype=dtype)[:-1]
    basis = torch.stack(
        [
            function(harmonic * angles)
            for harmonic in HARMONICS
            for function in (torch.cos, torch.sin)
        ],
        dim=1,
    )
    log_radius = (
        reference_global[:, None, None, 0]
        + reference_local[:, :, None, 2]
        + torch.einsum(
            "bsk,ak->bsa",
            reference_local[:, :, HARMONIC_START:LOCAL_DIM],
            basis,
        )
    )
    radius = torch.exp(log_radius.clamp(math.log(0.12), math.log(8.0)))
    error_local = predicted_local - target_local
    error_global = predicted_global - target_global
    error_log_radius = (
        error_global[:, None, None, 0]
        + error_local[:, :, None, 2]
        + torch.einsum(
            "bsk,ak->bsa",
            error_local[:, :, HARMONIC_START:LOCAL_DIM],
            basis,
        )
    )
    radial_speed = radius * error_log_radius
    error_x = error_local[:, :, None, 0] + radial_speed * torch.cos(angles)[None, None]
    error_y = error_local[:, :, None, 1] + radial_speed * torch.sin(angles)[None, None]
    squared_mm = error_x.square() + error_y.square()
    weights = station_mask.to(dtype)[:, :, None]
    surface = (squared_mm * weights).sum() / (weights.sum() * n_angles).clamp_min(1.0)
    endpoint = error_global[:, 1:3].square().mean()
    loss = surface + float(endpoint_weight) * endpoint
    return loss, {
        "surface_rmse_mm": torch.sqrt(surface.detach().clamp_min(0.0)),
        "endpoint_rmse_mm": torch.sqrt(endpoint.detach().clamp_min(0.0)),
    }


@torch.no_grad()
def state_surface_error(
    predicted_local: torch.Tensor,
    predicted_global: torch.Tensor,
    target_local: torch.Tensor,
    target_global: torch.Tensor,
    station_mask: torch.Tensor,
    n_angles: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros_local = torch.zeros_like(predicted_local)
    zeros_global = torch.zeros_like(predicted_global)
    # q_pred - q_target is a displacement target evaluated at q_target.
    loss, _ = physical_velocity_loss(
        predicted_local - target_local,
        predicted_global - target_global,
        zeros_local,
        zeros_global,
        target_local,
        target_global,
        station_mask,
        n_angles=n_angles,
    )
    return torch.sqrt(loss.clamp_min(0.0)), loss


@torch.no_grad()
def decoded_tube_metrics(
    predicted_local: torch.Tensor,
    predicted_global: torch.Tensor,
    target_local: torch.Tensor,
    target_global: torch.Tensor,
    arc_mm: torch.Tensor,
    metric_mask: torch.Tensor,
    *,
    n_angles: int = 48,
    grid_size: int = 64,
    grid_extent_mm: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast decoded-tube Dice and a symmetric surface HD95 proxy.

    This is used only for checkpoint selection. Final reported metrics are
    evaluated on the original voxel grid against GT in evaluate.py.
    """
    dtype, device = predicted_local.dtype, predicted_local.device
    angles = torch.linspace(0.0, 2.0 * math.pi, n_angles + 1, device=device, dtype=dtype)[:-1]
    basis = torch.stack(
        [
            function(harmonic * angles)
            for harmonic in HARMONICS
            for function in (torch.cos, torch.sin)
        ],
        dim=1,
    )

    def radius(local, global_state, theta_basis):
        log_r = global_state[:, None, None, 0] + local[:, :, None, 2]
        log_r = log_r + torch.einsum(
            "bsk,ak->bsa",
            local[:, :, HARMONIC_START:LOCAL_DIM],
            theta_basis,
        )
        return torch.exp(log_r.clamp(math.log(0.12), math.log(8.0)))

    pred_radius = radius(predicted_local, predicted_global, basis)
    target_radius = radius(target_local, target_global, basis)
    pred_surface = torch.stack(
        (
            predicted_local[:, :, None, 0] + pred_radius * torch.cos(angles)[None, None],
            predicted_local[:, :, None, 1] + pred_radius * torch.sin(angles)[None, None],
        ),
        dim=-1,
    )
    target_surface = torch.stack(
        (
            target_local[:, :, None, 0] + target_radius * torch.cos(angles)[None, None],
            target_local[:, :, None, 1] + target_radius * torch.sin(angles)[None, None],
        ),
        dim=-1,
    )
    active_pred = (
        metric_mask
        & (arc_mm >= predicted_global[:, None, 1])
        & (arc_mm <= predicted_global[:, None, 2])
    )
    active_target = (
        metric_mask
        & (arc_mm >= target_global[:, None, 1])
        & (arc_mm <= target_global[:, None, 2])
    )
    both_active = active_pred & active_target
    pairwise = torch.cdist(
        pred_surface.reshape(-1, n_angles, 2),
        target_surface.reshape(-1, n_angles, 2),
    ).reshape(len(predicted_local), predicted_local.shape[1], n_angles, n_angles)
    nearest = torch.cat((pairwise.amin(dim=-1), pairwise.amin(dim=-2)), dim=-1)
    # Compute one HD95 per side, then average. Pooling all points across a batch
    # would let long/easy canals dominate short/hard cases during selection.
    hd95_values = []
    for sample_index in range(len(predicted_local)):
        selected = nearest[sample_index][both_active[sample_index]].reshape(-1)
        endpoints = torch.abs(
            predicted_global[sample_index, 1:3]
            - target_global[sample_index, 1:3]
        ).reshape(-1)
        distances = torch.cat((selected, endpoints)) if selected.numel() else endpoints
        hd95_values.append(torch.quantile(distances, 0.95))
    hd95 = torch.stack(hd95_values).mean()

    coordinates = torch.linspace(-grid_extent_mm, grid_extent_mm, grid_size, device=device, dtype=dtype)
    x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def occupancy(local, global_state, active):
        dx = x[None, None] - local[:, :, None, None, 0]
        dy = y[None, None] - local[:, :, None, None, 1]
        rho = torch.sqrt(dx.square() + dy.square() + 1e-10)
        theta = torch.remainder(torch.atan2(dy, dx), 2.0 * math.pi)
        log_r = global_state[:, None, None, None, 0] + local[:, :, None, None, 2]
        for offset, harmonic in enumerate(HARMONICS):
            a = local[:, :, None, None, HARMONIC_START + 2 * offset]
            b = local[:, :, None, None, HARMONIC_START + 2 * offset + 1]
            log_r = log_r + a * torch.cos(harmonic * theta) + b * torch.sin(harmonic * theta)
        radial = torch.exp(log_r.clamp(math.log(0.12), math.log(8.0)))
        return (rho <= radial) & active[:, :, None, None]

    pred_occ = occupancy(predicted_local, predicted_global, active_pred)
    target_occ = occupancy(target_local, target_global, active_target)
    intersection = (pred_occ & target_occ).sum(dim=(1, 2, 3)).to(dtype)
    denominator = pred_occ.sum(dim=(1, 2, 3)) + target_occ.sum(dim=(1, 2, 3))
    dice = torch.where(
        denominator > 0,
        2.0 * intersection / denominator.clamp_min(1),
        torch.ones_like(denominator, dtype=dtype),
    ).mean()
    return dice, hd95


def decoded_shell_soft_dice_loss(
    predicted_local: torch.Tensor,
    predicted_global: torch.Tensor,
    target_occupancy: torch.Tensor,
    radii_mm: torch.Tensor,
    arc_mm: torch.Tensor,
    station_mask: torch.Tensor,
    *,
    temperature_mm: float = 0.10,
    endpoint_temperature_mm: float = 0.15,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable voxel-Dice proxy on the cached polar shell.

    Unlike ``L_phys``, this objective weights a boundary error by the volume it
    changes.  GT occupancy is a supervision tensor only; it is never passed to
    the condition encoder.
    """
    dtype, device = predicted_local.dtype, predicted_local.device
    target = target_occupancy.to(device=device, dtype=dtype)
    if target.ndim != 4:
        raise ValueError(f"target_occupancy must be [B,S,A,R], got {target.shape}")
    batch, stations, n_angles, n_radii = target.shape
    if predicted_local.shape[:2] != (batch, stations):
        raise ValueError("State and target occupancy dimensions do not match")

    theta = torch.linspace(
        0.0, 2.0 * math.pi, n_angles + 1, device=device, dtype=dtype
    )[:-1]
    radial_grid = radii_mm.to(device=device, dtype=dtype)
    if radial_grid.ndim == 1:
        radial_grid = radial_grid[None].expand(batch, -1)
    if radial_grid.shape != (batch, n_radii):
        raise ValueError(
            f"radii_mm must be [R] or [B,R], got {tuple(radial_grid.shape)}"
        )

    x = radial_grid[:, None, None, :] * torch.cos(theta)[None, None, :, None]
    y = radial_grid[:, None, None, :] * torch.sin(theta)[None, None, :, None]
    dx = x - predicted_local[:, :, None, None, 0]
    dy = y - predicted_local[:, :, None, None, 1]
    rho = torch.sqrt(dx.square() + dy.square() + 1e-10)
    relative_theta = torch.remainder(torch.atan2(dy, dx), 2.0 * math.pi)
    log_radius = (
        predicted_global[:, None, None, None, 0]
        + predicted_local[:, :, None, None, 2]
    )
    for offset, harmonic in enumerate(HARMONICS):
        a = predicted_local[
            :, :, None, None, HARMONIC_START + 2 * offset
        ]
        b = predicted_local[
            :, :, None, None, HARMONIC_START + 2 * offset + 1
        ]
        log_radius = (
            log_radius
            + a * torch.cos(harmonic * relative_theta)
            + b * torch.sin(harmonic * relative_theta)
        )
    boundary_radius = torch.exp(
        log_radius.clamp(math.log(0.12), math.log(8.0))
    )
    soft_inside = torch.sigmoid(
        (boundary_radius - rho) / max(float(temperature_mm), 1e-4)
    )

    arc = arc_mm.to(device=device, dtype=dtype)
    axial_start = torch.sigmoid(
        (arc - predicted_global[:, None, 1])
        / max(float(endpoint_temperature_mm), 1e-4)
    )
    axial_end = torch.sigmoid(
        (predicted_global[:, None, 2] - arc)
        / max(float(endpoint_temperature_mm), 1e-4)
    )
    prediction = soft_inside * (axial_start * axial_end)[:, :, None, None]
    weights = station_mask.to(dtype)[:, :, None, None]
    prediction = prediction * weights
    target = target * weights
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean(), dice.mean().detach()
