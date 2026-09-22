#!/usr/bin/env python3
"""Data-free numerical contracts for corrected Geometric Boundary Flow."""

from __future__ import annotations

import torch
from torch import nn

from canalmanifold.geoflow_model import (
    GeometricBoundaryFlow,
    MovingSurfaceProfileSampler,
)
from canalmanifold.geoflow_train import (
    geoflow_heun_rollout,
    geoflow_physical_loss,
)
from canalmanifold.surface_model import SurfaceStats


def sigmoid_shell(
    *, stations: int, angles: int, radii: int, crossing_mm: float = 1.5
) -> torch.Tensor:
    radius = torch.linspace(0.0, 5.0, radii)
    probability = torch.sigmoid((crossing_mm - radius) / 0.12)
    probability = probability[None, None, None, :].expand(
        1, stations, angles, -1
    )
    image = torch.zeros_like(probability)
    return torch.stack((image, probability), dim=1)


def test_local_profile_is_not_crossing_minus_h() -> None:
    stations, angles, radii = 4, 8, 24
    sampler = MovingSurfaceProfileSampler(
        angles,
        5.0,
        [-0.6, -0.3, 0.0, 0.3, 0.6],
        shell_radii=radii,
        probability_channel=1,
        minimum_slope_per_mm=0.01,
    )
    shell = sigmoid_shell(stations=stations, angles=angles, radii=radii)
    q0 = torch.full((1, stations, angles), 1.5)
    h0 = torch.zeros_like(q0)
    h1 = torch.full_like(q0, 0.2)
    _, at_start = sampler(shell, q0, h0)
    _, after_move = sampler(shell, q0, h1)
    start_identity = at_start["newton_mm"] + h0
    moved_identity = after_move["newton_mm"] + h1
    difference = (moved_identity - start_identity).abs().mean()
    assert float(difference) > 0.01, (
        "Newton evidence collapsed to crossing-current algebra; "
        f"mean difference={float(difference):.6f}"
    )
    profile_change = (
        after_move["probability_at_surface"]
        - at_start["probability_at_surface"]
    ).abs().mean()
    assert float(profile_change) > 0.05
    assert not torch.allclose(
        at_start["sharpness_per_mm"], after_move["sharpness_per_mm"]
    )


def small_model(stations: int = 8, angles: int = 8) -> GeometricBoundaryFlow:
    stats = SurfaceStats(
        h_output_scale=[0.10] * angles,
        q0_radius_mean=1.5,
        q0_radius_std=0.3,
        endpoint_mean=[4.0, 54.0],
        endpoint_std=[1.0, 1.0],
        endpoint_output_scale=[0.2, 0.2],
    )
    return GeometricBoundaryFlow(
        shell_channels=2,
        stats=stats,
        stations=stations,
        n_angles=angles,
        shell_radii=24,
        d_model=32,
        n_heads=4,
        n_layers=1,
        angle_hidden=8,
        angular_modes=2,
        dropout=0.0,
    ).eval()


def test_zero_initialisation_and_basis_gauges() -> int:
    stations, angles = 8, 8
    model = small_model(stations, angles)
    shell = sigmoid_shell(stations=stations, angles=angles, radii=24)
    q0 = torch.full((1, stations, angles), 1.5)
    q0[:, :, 1::2] += 0.15
    h = torch.zeros_like(q0)
    endpoints = torch.tensor([[4.0, 54.0]])
    side = torch.zeros(1, dtype=torch.long)
    mask = torch.ones(1, stations, dtype=torch.bool)
    arc = torch.linspace(0.0, 50.0, stations)[None]
    with torch.no_grad():
        velocity, endpoint_velocity, _ = model(
            q0, h, endpoints, shell, side, mask, torch.zeros(1), arc
        )
    assert torch.equal(velocity, torch.zeros_like(velocity))
    assert torch.equal(endpoint_velocity, torch.zeros_like(endpoint_velocity))

    with torch.no_grad():
        model.alpha_head[-1].bias.fill_(1.0)
        model.axial_head.bias.fill_(1.0)
        model.ray_head.bias[0] = 1.0
        _, _, diagnostics = model(
            q0, h, endpoints, shell, side, mask, torch.zeros(1), arc
        )
    axial_mean = diagnostics["axial_term_mm"].mean(dim=1)
    angular_mean = diagnostics["curvature_term_mm"].mean(dim=2)
    assert torch.allclose(axial_mean, torch.zeros_like(axial_mean), atol=1e-6)
    assert torch.allclose(angular_mean, torch.zeros_like(angular_mean), atol=1e-6)
    return sum(parameter.numel() for parameter in model.parameters())


def test_area_weighting() -> None:
    predicted = torch.ones(1, 1, 1)
    target = torch.zeros_like(predicted)
    endpoint = torch.zeros(1, 2)
    mask = torch.ones_like(predicted, dtype=torch.bool)
    thin, _ = geoflow_physical_loss(
        predicted,
        endpoint,
        target,
        endpoint,
        torch.full_like(predicted, 0.5),
        mask,
        reference_radius_mm=1.0,
        endpoint_weight=0.0,
    )
    thick, _ = geoflow_physical_loss(
        predicted,
        endpoint,
        target,
        endpoint,
        torch.full_like(predicted, 2.0),
        mask,
        reference_radius_mm=1.0,
        endpoint_weight=0.0,
    )
    assert torch.allclose(thick / thin, torch.tensor(4.0))


class ConstantVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trust_surface_rmse_mm = 0.05
        self.trust_endpoint_mm = 2.0
        self.max_radius_mm = 5.0

    def forward(
        self,
        q0_radius,
        h,
        endpoints,
        shell,
        side_id,
        station_mask,
        time,
        arc_mm=None,
    ):
        del q0_radius, shell, side_id, station_mask, time, arc_mm
        return torch.ones_like(h), torch.zeros_like(endpoints), {}


def test_trust_region() -> None:
    stations, angles = 8, 8
    q0 = torch.full((1, stations, angles), 1.5)
    endpoints = torch.tensor([[4.0, 54.0]])
    shell = torch.zeros(1, 2, stations, angles, 24)
    side = torch.zeros(1, dtype=torch.long)
    mask = torch.ones(1, stations, dtype=torch.bool)
    h, _ = geoflow_heun_rollout(
        ConstantVelocity(),
        q0,
        endpoints,
        shell,
        side,
        mask,
        steps=4,
    )
    rms = torch.sqrt(h.square().mean())
    assert torch.allclose(rms, torch.tensor(0.05), atol=1e-6)


def test_terminal_extrapolation() -> None:
    torch.manual_seed(4)
    h0 = torch.randn(3, 5, 7) * 0.1
    target = torch.randn_like(h0)
    noise = torch.randn_like(h0) * 0.2
    time = torch.tensor([0.1, 0.5, 0.9])
    delta = target - h0
    current = (
        h0
        + time[:, None, None] * delta
        + (time * (1.0 - time))[:, None, None] * noise
    )
    velocity = delta + (1.0 - 2.0 * time)[:, None, None] * noise
    remaining = 1.0 - time
    terminal = (
        current
        + remaining[:, None, None] * velocity
        - remaining.square()[:, None, None] * noise
    )
    assert torch.allclose(terminal, target, atol=2e-6, rtol=2e-6)


def main() -> None:
    test_local_profile_is_not_crossing_minus_h()
    parameters = test_zero_initialisation_and_basis_gauges()
    test_area_weighting()
    test_trust_region()
    test_terminal_extrapolation()
    print(
        "GEOFLOW_CONTRACTS_OK: local Newton profile, exact zero-init, "
        f"gauge-fixed basis, area weighting, trust region, bridge terminal; "
        f"small-model parameters={parameters / 1e6:.3f}M"
    )


if __name__ == "__main__":
    main()
