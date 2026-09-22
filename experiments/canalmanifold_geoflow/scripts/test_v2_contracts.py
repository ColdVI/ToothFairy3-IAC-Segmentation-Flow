#!/usr/bin/env python3
"""Data-free numerical contracts for CanalManifoldFlow v2."""

from __future__ import annotations

import math

import numpy as np
import torch

from canalmanifold.constants import LOCAL_DIM, N_LOCAL
from canalmanifold.dense_sdf import DenseSDFVelocityUNet
from canalmanifold.displacement import normal_displacement_decode
from canalmanifold.geometry import TubeFrame
from canalmanifold.losses import decoded_shell_soft_dice_loss
from canalmanifold.paths import interpolate_noisy_linear_path


def test_noisy_path_terminal() -> None:
    torch.manual_seed(1)
    batch = 3
    q0_local = torch.randn(batch, N_LOCAL, LOCAL_DIM)
    q1_local = torch.randn_like(q0_local)
    q0_global = torch.randn(batch, 3)
    q1_global = torch.randn_like(q0_global)
    noise_local = torch.randn_like(q0_local) * 0.1
    noise_global = torch.randn_like(q0_global) * 0.1
    time = torch.tensor([0.1, 0.45, 0.9])
    current_local, current_global, velocity_local, velocity_global = (
        interpolate_noisy_linear_path(
            q0_local,
            q0_global,
            q1_local,
            q1_global,
            time,
            noise_local,
            noise_global,
        )
    )
    remaining = 1.0 - time
    terminal_local = (
        current_local
        + remaining[:, None, None] * velocity_local
        - remaining.square()[:, None, None] * noise_local
    )
    terminal_global = (
        current_global
        + remaining[:, None] * velocity_global
        - remaining.square()[:, None] * noise_global
    )
    assert torch.allclose(terminal_local, q1_local, atol=2e-6, rtol=2e-6)
    assert torch.allclose(terminal_global, q1_global, atol=2e-6, rtol=2e-6)


def test_decoded_loss_gradient() -> None:
    batch, stations, angles, radii = 1, N_LOCAL, 32, 24
    local = torch.zeros(batch, stations, LOCAL_DIM, requires_grad=True)
    global_state = torch.tensor([[math.log(1.5), 5.0, 45.0]], requires_grad=True)
    radial_grid = torch.linspace(0.0, 5.0, radii)
    target = (radial_grid[None, None, None] <= 1.5).float().expand(
        batch, stations, angles, -1
    )
    arc = torch.linspace(0.0, 50.0, stations)[None]
    mask = torch.ones(batch, stations, dtype=torch.bool)
    loss, dice = decoded_shell_soft_dice_loss(
        local,
        global_state,
        target,
        radial_grid,
        arc,
        mask,
        temperature_mm=0.10,
    )
    loss.backward()
    assert torch.isfinite(loss) and 0.0 <= float(dice) <= 1.0
    assert local.grad is not None and torch.isfinite(local.grad).all()
    assert global_state.grad is not None and torch.isfinite(global_state.grad).all()


def straight_frame() -> TubeFrame:
    arc = np.linspace(10.0, 53.0, N_LOCAL, dtype=np.float32)
    center = np.column_stack(
        (
            np.full(N_LOCAL, 32.0),
            np.full(N_LOCAL, 32.0),
            arc,
        )
    ).astype(np.float32)
    tangent = np.tile(np.asarray([0.0, 0.0, 1.0], np.float32), (N_LOCAL, 1))
    normal1 = np.tile(np.asarray([1.0, 0.0, 0.0], np.float32), (N_LOCAL, 1))
    normal2 = np.tile(np.asarray([0.0, 1.0, 0.0], np.float32), (N_LOCAL, 1))
    frame = TubeFrame(center, tangent, normal1, normal2, arc, 10.0, 53.0)
    frame.validate()
    return frame


def test_normal_decode_identity() -> None:
    shape = (64, 64, 64)
    x, y, z = np.indices(shape)
    coarse = (((x - 32) ** 2 + (y - 32) ** 2 <= 9) & (z >= 10) & (z <= 53))
    frame = straight_frame()
    zero = np.zeros((N_LOCAL, 32), dtype=np.float32)
    identity = normal_displacement_decode(
        coarse,
        zero,
        frame,
        np.eye(4),
        (1.0, 1.0, 1.0),
        q0_endpoints_mm=(10.0, 53.0),
        predicted_endpoints_mm=(10.0, 53.0),
    )
    assert np.array_equal(identity, coarse)
    outward = normal_displacement_decode(
        coarse,
        np.full_like(zero, 0.4),
        frame,
        np.eye(4),
        (1.0, 1.0, 1.0),
        q0_endpoints_mm=(10.0, 53.0),
        predicted_endpoints_mm=(10.0, 53.0),
        max_abs_displacement_mm=0.5,
    )
    assert outward.sum() >= coarse.sum()


def test_dense_identity_initialisation() -> None:
    model = DenseSDFVelocityUNet(base_channels=8).eval()
    condition = torch.randn(1, 2, 16, 16, 16)
    coarse = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        velocity = model(condition, coarse, coarse, torch.tensor([0.5]))
    assert torch.equal(velocity, torch.zeros_like(velocity))


def main() -> None:
    test_noisy_path_terminal()
    test_decoded_loss_gradient()
    test_normal_decode_identity()
    test_dense_identity_initialisation()
    print("V2_CONTRACTS_OK: noisy endpoint, decoded gradient, phi0-H identity, dense identity")


if __name__ == "__main__":
    main()
