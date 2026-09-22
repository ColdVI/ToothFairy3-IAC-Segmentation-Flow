#!/usr/bin/env python3
"""Data-free contract test; runs on CPU in under a minute."""

from __future__ import annotations

import torch

from canalmanifold.constants import LOCAL_DIM, STATE_DIM
from canalmanifold.data import StateStats
from canalmanifold.losses import physical_velocity_loss
from canalmanifold.model import CanalVelocityTransformer
from canalmanifold.paths import interpolate_path, staged_coefficients
from canalmanifold.surface_model import SurfaceDisplacementTransformer, SurfaceStats


def main() -> None:
    assert LOCAL_DIM == 17 and STATE_DIM == 2723
    batch, stations, angles, radii, channels = 2, 160, 32, 24, 5
    q0_local = torch.zeros(batch, stations, LOCAL_DIM)
    q1_local = torch.randn_like(q0_local) * 0.02
    q0_global = torch.tensor([[0.4, 4.0, 54.0], [0.5, 4.0, 54.0]])
    q1_global = q0_global + torch.tensor([[0.02, 0.2, -0.1], [-0.01, -0.1, 0.2]])
    for time, expected in ((torch.zeros(batch), (q0_local, q0_global)), (torch.ones(batch), (q1_local, q1_global))):
        local, global_state, _, _ = interpolate_path(
            q0_local, q0_global, q1_local, q1_global, time, "staged"
        )
        assert torch.allclose(local, expected[0], atol=1e-6)
        assert torch.allclose(global_state, expected[1], atol=1e-6)

    _, _, alpha_h_early, _, _, derivative_h_early = staged_coefficients(torch.tensor([0.25]))
    assert alpha_h_early.item() == 0.0 and derivative_h_early.item() == 0.0

    stats = StateStats(
        local_mean=[0.0] * LOCAL_DIM,
        local_std=[1.0] * LOCAL_DIM,
        global_mean=[0.0] * 3,
        global_std=[1.0] * 3,
        output_local_scale=[1.0] * LOCAL_DIM,
        output_global_scale=[1.0] * 3,
    )
    model = CanalVelocityTransformer(
        shell_channels=channels,
        stats=stats,
        d_model=64,
        n_heads=4,
        n_layers=1,
        angular_hidden=16,
        dropout=0.0,
    ).eval()
    shell = torch.randn(batch, channels, stations, angles, radii)
    mask = torch.ones(batch, stations, dtype=torch.bool)
    side = torch.tensor([0, 1])
    with torch.no_grad():
        velocity_local, velocity_global = model(
            q0_local, q0_global, shell, side, mask, torch.tensor([0.2, 0.8])
        )
    assert torch.equal(velocity_local, torch.zeros_like(velocity_local))
    assert torch.equal(velocity_global, torch.zeros_like(velocity_global))

    zero_loss, _ = physical_velocity_loss(
        velocity_local,
        velocity_global,
        torch.zeros_like(velocity_local),
        torch.zeros_like(velocity_global),
        q0_local,
        q0_global,
        mask,
    )
    assert zero_loss.item() == 0.0

    surface_stats = SurfaceStats(
        h_output_scale=[0.1] * angles,
        q0_radius_mean=1.5,
        q0_radius_std=0.3,
        endpoint_mean=[4.0, 54.0],
        endpoint_std=[1.0, 1.0],
        endpoint_output_scale=[0.2, 0.2],
    )
    surface_model = SurfaceDisplacementTransformer(
        shell_channels=channels,
        stats=surface_stats,
        d_model=64,
        n_heads=4,
        n_layers=1,
        angle_hidden=16,
        dropout=0.0,
    ).eval()
    q0_radius = torch.full((batch, stations, angles), 1.5)
    h = torch.zeros_like(q0_radius)
    endpoint = torch.tensor([[4.0, 54.0], [4.0, 54.0]])
    with torch.no_grad():
        velocity_h, velocity_endpoint = surface_model(
            q0_radius,
            h,
            endpoint,
            shell,
            side,
            mask,
            torch.tensor([0.2, 0.8]),
        )
    assert torch.equal(velocity_h, torch.zeros_like(velocity_h))
    assert torch.equal(velocity_endpoint, torch.zeros_like(velocity_endpoint))
    print(
        "SMOKE_OK: state=2723 (m<=8), staged contracts retained, "
        "R1/R2 identity-init exact, L_phys zero"
    )


if __name__ == "__main__":
    main()
