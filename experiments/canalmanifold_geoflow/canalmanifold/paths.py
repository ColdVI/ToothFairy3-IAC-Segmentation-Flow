"""Canonical direct, linear-ablation and staged G/L/H transport paths."""

from __future__ import annotations

import math

import torch

from .constants import (
    HARMONIC_START,
    HARMONICS,
    HIGH_GROUP,
    LOCAL_DIM,
    LOW_GROUP,
)


def _smoothstep(z: torch.Tensor) -> torch.Tensor:
    z = z.clamp(0.0, 1.0)
    return 3.0 * z.square() - 2.0 * z.pow(3)


def _smoothstep_derivative(z: torch.Tensor) -> torch.Tensor:
    inside = ((z > 0.0) & (z < 1.0)).to(z.dtype)
    clipped = z.clamp(0.0, 1.0)
    return (6.0 * clipped - 6.0 * clipped.square()) * inside


def staged_coefficients(t: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return alpha_G/L/H and their exact time derivatives."""
    alpha_g = _smoothstep(2.0 * t)
    alpha_l = _smoothstep(t)
    alpha_h = _smoothstep(2.0 * t - 1.0)
    derivative_g = 2.0 * _smoothstep_derivative(2.0 * t)
    derivative_l = _smoothstep_derivative(t)
    derivative_h = 2.0 * _smoothstep_derivative(2.0 * t - 1.0)
    return alpha_g, alpha_l, alpha_h, derivative_g, derivative_l, derivative_h


def interpolate_path(
    q0_local: torch.Tensor,
    q0_global: torch.Tensor,
    q1_local: torch.Tensor,
    q1_global: torch.Tensor,
    t: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_local = q1_local - q0_local
    delta_global = q1_global - q0_global
    if mode == "linear":
        alpha = t[:, None, None]
        return (
            q0_local + alpha * delta_local,
            q0_global + t[:, None] * delta_global,
            delta_local,
            delta_global,
        )
    if mode != "staged":
        raise ValueError(f"Flow path must be 'linear' or 'staged', got {mode}")
    alpha_g, alpha_l, alpha_h, derivative_g, derivative_l, derivative_h = staged_coefficients(t)
    q_local = q0_local.clone()
    target_local = torch.zeros_like(delta_local)
    q_local[:, :, 0:2] += alpha_g[:, None, None] * delta_local[:, :, 0:2]
    target_local[:, :, 0:2] = derivative_g[:, None, None] * delta_local[:, :, 0:2]
    q_local[:, :, list(LOW_GROUP)] += alpha_l[:, None, None] * delta_local[:, :, list(LOW_GROUP)]
    target_local[:, :, list(LOW_GROUP)] = derivative_l[:, None, None] * delta_local[:, :, list(LOW_GROUP)]
    q_local[:, :, list(HIGH_GROUP)] += alpha_h[:, None, None] * delta_local[:, :, list(HIGH_GROUP)]
    target_local[:, :, list(HIGH_GROUP)] = derivative_h[:, None, None] * delta_local[:, :, list(HIGH_GROUP)]
    q_global = q0_global + alpha_g[:, None] * delta_global
    target_global = derivative_g[:, None] * delta_global
    return q_local, q_global, target_local, target_global


def interpolate_noisy_linear_path(
    q0_local: torch.Tensor,
    q0_global: torch.Tensor,
    q1_local: torch.Tensor,
    q1_global: torch.Tensor,
    t: torch.Tensor,
    noise_local: torch.Tensor | None = None,
    noise_global: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variance-shaped conditional flow path with an exact velocity target.

    q_t = (1-t)q0 + t q1 + t(1-t) eps
    u_t = q1-q0 + (1-2t) eps

    The noise vanishes at both endpoints, so training augmentation does not
    change the transport problem.  Passing zero/None noise recovers ordinary
    linear conditional flow matching exactly.
    """
    if noise_local is None:
        noise_local = torch.zeros_like(q0_local)
    if noise_global is None:
        noise_global = torch.zeros_like(q0_global)
    beta = t * (1.0 - t)
    derivative = 1.0 - 2.0 * t
    delta_local = q1_local - q0_local
    delta_global = q1_global - q0_global
    return (
        q0_local
        + t[:, None, None] * delta_local
        + beta[:, None, None] * noise_local,
        q0_global
        + t[:, None] * delta_global
        + beta[:, None] * noise_global,
        delta_local + derivative[:, None, None] * noise_local,
        delta_global + derivative[:, None] * noise_global,
    )


def staged_velocity_from_displacement(
    displacement_local: torch.Tensor,
    displacement_global: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a schedule-normalized displacement estimate to staged velocity.

    The network regresses q1-q0 at every time.  Known alpha derivatives are
    applied analytically here, avoiding the 9x target-variance inflation of
    directly regressing the narrow G/H velocity pulses.
    """
    _, _, _, derivative_g, derivative_l, derivative_h = staged_coefficients(t)
    velocity_local = torch.zeros_like(displacement_local)
    velocity_local[:, :, 0:2] = (
        derivative_g[:, None, None] * displacement_local[:, :, 0:2]
    )
    velocity_local[:, :, list(LOW_GROUP)] = (
        derivative_l[:, None, None]
        * displacement_local[:, :, list(LOW_GROUP)]
    )
    velocity_local[:, :, list(HIGH_GROUP)] = (
        derivative_h[:, None, None]
        * displacement_local[:, :, list(HIGH_GROUP)]
    )
    velocity_global = derivative_g[:, None] * displacement_global
    return velocity_local, velocity_global


def transport_increment_from_displacement(
    displacement_local: torch.Tensor,
    displacement_global: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a displacement estimate over one known path interval.

    For staged transport this uses exact alpha(t1)-alpha(t0) increments.  A
    four-step trapezoidal rollout therefore reaches q1 exactly when the model
    predicts a constant true displacement; applying ordinary Heun directly to
    the narrow quadratic alpha' pulses does not have this property.
    """
    if mode == "linear":
        interval = (t1 - t0)
        return (
            interval[:, None, None] * displacement_local,
            interval[:, None] * displacement_global,
        )
    if mode != "staged":
        raise ValueError(f"Unsupported flow mode: {mode}")
    alpha_g0, alpha_l0, alpha_h0, *_ = staged_coefficients(t0)
    alpha_g1, alpha_l1, alpha_h1, *_ = staged_coefficients(t1)
    delta_g = alpha_g1 - alpha_g0
    delta_l = alpha_l1 - alpha_l0
    delta_h = alpha_h1 - alpha_h0
    increment_local = torch.zeros_like(displacement_local)
    increment_local[:, :, 0:2] = (
        delta_g[:, None, None] * displacement_local[:, :, 0:2]
    )
    increment_local[:, :, list(LOW_GROUP)] = (
        delta_l[:, None, None] * displacement_local[:, :, list(LOW_GROUP)]
    )
    increment_local[:, :, list(HIGH_GROUP)] = (
        delta_h[:, None, None] * displacement_local[:, :, list(HIGH_GROUP)]
    )
    increment_global = delta_g[:, None] * displacement_global
    return increment_local, increment_global


def project_ell_gauge(
    local: torch.Tensor,
    global_state: torch.Tensor,
    station_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ell to zero mean while preserving g+ell exactly."""
    weights = station_mask.to(local.dtype)
    mean = (local[:, :, 2] * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    local = local.clone()
    global_state = global_state.clone()
    local[:, :, 2] -= mean[:, None]
    global_state[:, 0] += mean
    return local, global_state


def _surface_displacement_rmse(
    local: torch.Tensor,
    global_state: torch.Tensor,
    q0_local: torch.Tensor,
    q0_global: torch.Tensor,
    station_mask: torch.Tensor,
    n_angles: int = 32,
) -> torch.Tensor:
    """Per-sample decoded cross-sectional surface displacement in millimetres."""
    dtype, device = local.dtype, local.device
    angles = torch.linspace(
        0.0, 2.0 * math.pi, n_angles + 1, device=device, dtype=dtype
    )[:-1]
    basis = torch.stack(
        [
            function(harmonic * angles)
            for harmonic in HARMONICS
            for function in (torch.cos, torch.sin)
        ],
        dim=1,
    )

    def points(state_local, state_global):
        log_radius = (
            state_global[:, None, None, 0]
            + state_local[:, :, None, 2]
            + torch.einsum(
                "bsk,ak->bsa",
                state_local[:, :, HARMONIC_START:LOCAL_DIM],
                basis,
            )
        )
        radius = torch.exp(log_radius.clamp(math.log(0.12), math.log(8.0)))
        x = state_local[:, :, None, 0] + radius * torch.cos(angles)[None, None]
        y = state_local[:, :, None, 1] + radius * torch.sin(angles)[None, None]
        return x, y

    x, y = points(local, global_state)
    x0, y0 = points(q0_local, q0_global)
    squared = (x - x0).square() + (y - y0).square()
    weights = station_mask.to(dtype)[:, :, None]
    denominator = (weights.sum(dim=(1, 2)) * n_angles).clamp_min(1.0)
    return torch.sqrt((squared * weights).sum(dim=(1, 2)) / denominator)


def project_trust_region(
    local: torch.Tensor,
    global_state: torch.Tensor,
    q0_local: torch.Tensor,
    q0_global: torch.Tensor,
    station_mask: torch.Tensor,
    *,
    max_surface_rmse_mm: float,
    max_endpoint_mm: float,
    max_center_shift_mm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a rollout into a train-distribution geometric trust region."""
    rmse = _surface_displacement_rmse(
        local, global_state, q0_local, q0_global, station_mask
    )
    limit = torch.as_tensor(max_surface_rmse_mm, dtype=local.dtype, device=local.device)
    scale = torch.minimum(torch.ones_like(rmse), limit / rmse.clamp_min(1e-6))
    local = q0_local + scale[:, None, None] * (local - q0_local)
    global_state = q0_global + scale[:, None] * (global_state - q0_global)

    local = local.clone()
    centre = local[:, :, 0:2]
    centre_norm = torch.linalg.vector_norm(centre, dim=-1, keepdim=True)
    centre_scale = torch.minimum(
        torch.ones_like(centre_norm),
        torch.as_tensor(
            max_center_shift_mm, dtype=local.dtype, device=local.device
        )
        / centre_norm.clamp_min(1e-6),
    )
    local[:, :, 0:2] = centre * centre_scale

    global_state = global_state.clone()
    endpoint_delta = (global_state[:, 1:3] - q0_global[:, 1:3]).clamp(
        -float(max_endpoint_mm), float(max_endpoint_mm)
    )
    endpoints = q0_global[:, 1:3] + endpoint_delta
    midpoint = endpoints.mean(dim=1)
    half_length = ((endpoints[:, 1] - endpoints[:, 0]) * 0.5).clamp_min(0.125)
    global_state[:, 1] = midpoint - half_length
    global_state[:, 2] = midpoint + half_length
    return project_ell_gauge(local, global_state, station_mask)


@torch.no_grad()
def heun_rollout(
    model,
    q0_local: torch.Tensor,
    q0_global: torch.Tensor,
    shell: torch.Tensor,
    side_id: torch.Tensor,
    station_mask: torch.Tensor,
    *,
    steps: int = 4,
    mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    local, global_state = q0_local.clone(), q0_global.clone()
    origin_local, origin_global = q0_local.clone(), q0_global.clone()
    batch = len(local)
    for index in range(int(steps)):
        t0 = torch.full((batch,), index / steps, device=local.device, dtype=local.dtype)
        t1 = torch.full((batch,), (index + 1) / steps, device=local.device, dtype=local.dtype)
        displacement0_local, displacement0_global = model(
            local, global_state, shell, side_id, station_mask, t0
        )
        transport_mode = mode or getattr(model, "transport_mode", "linear")
        increment0_local, increment0_global = transport_increment_from_displacement(
            displacement0_local,
            displacement0_global,
            t0,
            t1,
            transport_mode,
        )
        euler_local = local + increment0_local
        euler_global = global_state + increment0_global
        euler_local, euler_global = project_ell_gauge(euler_local, euler_global, station_mask)
        euler_local, euler_global = project_trust_region(
            euler_local,
            euler_global,
            origin_local,
            origin_global,
            station_mask,
            max_surface_rmse_mm=float(getattr(model, "trust_surface_rmse_mm", 1.0e6)),
            max_endpoint_mm=float(getattr(model, "trust_endpoint_mm", 1.0e6)),
            max_center_shift_mm=float(getattr(model, "max_center_shift_mm", 3.0)),
        )
        displacement1_local, displacement1_global = model(
            euler_local, euler_global, shell, side_id, station_mask, t1
        )
        mean_displacement_local = 0.5 * (
            displacement0_local + displacement1_local
        )
        mean_displacement_global = 0.5 * (
            displacement0_global + displacement1_global
        )
        corrected_local, corrected_global = transport_increment_from_displacement(
            mean_displacement_local,
            mean_displacement_global,
            t0,
            t1,
            transport_mode,
        )
        local = local + corrected_local
        global_state = global_state + corrected_global
        local, global_state = project_ell_gauge(local, global_state, station_mask)
        local, global_state = project_trust_region(
            local,
            global_state,
            origin_local,
            origin_global,
            station_mask,
            max_surface_rmse_mm=float(getattr(model, "trust_surface_rmse_mm", 1.0e6)),
            max_endpoint_mm=float(getattr(model, "trust_endpoint_mm", 1.0e6)),
            max_center_shift_mm=float(getattr(model, "max_center_shift_mm", 3.0)),
        )
    return local, global_state
