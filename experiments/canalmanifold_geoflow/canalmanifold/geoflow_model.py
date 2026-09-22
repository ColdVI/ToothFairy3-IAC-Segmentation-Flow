"""Closed-loop geometric boundary flow on a moving radial surface.

The transported state is the unrestricted normal displacement
``h(s, theta)`` from the frozen nnU-Net boundary.  Unlike the legacy
``crossing - current`` signal, the feedback features below are measured from
the *local profile at the current surface*: a safeguarded Newton correction
and the local radial probability slope.

The learned velocity is gauge-fixed as

    V = alpha + a_zero_mean(s) + b_zero_angle_mean(s, theta)
        + gain(s, theta) * newton(s, theta)

where ``b`` is parameterised through the dimensionless circumferential
curvature ratio ``r_ref / r_t``.  The centring constraints keep the global
learned dilation mode identifiable as ``alpha``; evidence-driven motion is
reported separately.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import (
    PhaseAwareFourierAngularEncoder,
    SinusoidalTimeEmbedding,
    fixed_position_encoding,
)
from .surface_model import SurfaceStats


class MovingSurfaceProfileSampler(nn.Module):
    """Sample a fixed polar shell at the moving boundary.

    ``newton`` is a local, safeguarded Newton step in millimetres.  It is zero
    when the local probability profile is flat, rising, clipped by the radial
    grid, or otherwise invalid.  ``sharpness`` is the non-negative local
    radial gradient magnitude in probability/mm.  It is a boundary-strength
    feature, not a calibrated uncertainty estimate.
    """

    def __init__(
        self,
        n_angles: int,
        max_radius_mm: float,
        profile_offsets_mm: list[float],
        *,
        shell_radii: int,
        probability_channel: int = 1,
        evidence_threshold: float = 0.5,
        minimum_slope_per_mm: float = 0.05,
        maximum_newton_step_mm: float = 0.60,
        sharpness_cap_per_mm: float = 4.0,
    ) -> None:
        super().__init__()
        del shell_radii  # The shell's actual last dimension is checked at runtime.
        offsets = list(map(float, profile_offsets_mm))
        if len(offsets) < 3:
            raise ValueError("profile_offsets_mm needs a negative, zero, and positive offset")
        zero_index = min(range(len(offsets)), key=lambda index: abs(offsets[index]))
        if abs(offsets[zero_index]) > 1e-6:
            raise ValueError("profile_offsets_mm must contain 0.0")
        lower = [index for index, value in enumerate(offsets) if value < 0.0]
        upper = [index for index, value in enumerate(offsets) if value > 0.0]
        if not lower or not upper:
            raise ValueError("profile_offsets_mm must bracket 0.0")
        minus_index = max(lower, key=lambda index: offsets[index])
        plus_index = min(upper, key=lambda index: offsets[index])
        derivative_span = offsets[plus_index] - offsets[minus_index]
        if derivative_span <= 0.0:
            raise ValueError("Invalid finite-difference offsets")

        angles = torch.linspace(0.0, 2.0 * math.pi, int(n_angles) + 1)[:-1]
        self.register_buffer("angles", angles, persistent=False)
        self.register_buffer(
            "profile_offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False
        )
        self.minus_index = int(minus_index)
        self.zero_index = int(zero_index)
        self.plus_index = int(plus_index)
        self.derivative_span_mm = float(derivative_span)
        self.max_radius_mm = float(max_radius_mm)
        self.probability_channel = int(probability_channel)
        self.evidence_threshold = float(evidence_threshold)
        self.minimum_slope_per_mm = float(minimum_slope_per_mm)
        self.maximum_newton_step_mm = float(maximum_newton_step_mm)
        self.sharpness_cap_per_mm = float(sharpness_cap_per_mm)

    def forward(
        self,
        shell: torch.Tensor,
        q0_radius: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if shell.ndim != 5:
            raise ValueError(f"shell must be [B,C,S,A,R], got {tuple(shell.shape)}")
        batch, channels, stations, n_angles, _ = shell.shape
        if q0_radius.shape != (batch, stations, n_angles) or h.shape != q0_radius.shape:
            raise ValueError("q0_radius/h and shell station-angle dimensions differ")
        if self.probability_channel >= channels:
            raise ValueError(
                f"Probability channel {self.probability_channel} is absent from "
                f"a {channels}-channel shell"
            )

        current_unclipped = q0_radius + h
        current = current_unclipped.clamp(0.12, self.max_radius_mm)
        offsets = self.profile_offsets.to(device=current.device, dtype=current.dtype)
        query_unclipped = current[..., None] + offsets[None, None, None]
        query = query_unclipped.clamp(0.0, self.max_radius_mm)

        grid_x = 2.0 * query / self.max_radius_mm - 1.0
        angle = self.angles.to(device=current.device, dtype=current.dtype)
        grid_y = (angle / math.pi - 1.0)[None, None, :, None].expand_as(grid_x)
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            batch * stations, 1, n_angles * len(offsets), 2
        )
        polar = shell.permute(0, 2, 1, 3, 4).reshape(
            batch * stations, channels, n_angles, -1
        )
        # Periodic angular interpolation at theta=0/2pi.
        polar = torch.cat((polar, polar[:, :, :1]), dim=2)
        sampled = F.grid_sample(
            polar,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled[:, :, 0].reshape(
            batch, stations, channels, n_angles, len(offsets)
        )
        sampled = sampled.permute(0, 1, 3, 2, 4)

        probability = sampled[..., self.probability_channel, :]
        p_minus = probability[..., self.minus_index]
        p_at = probability[..., self.zero_index]
        p_plus = probability[..., self.plus_index]
        slope = (p_plus - p_minus) / self.derivative_span_mm
        sharpness = (-slope).clamp(0.0, self.sharpness_cap_per_mm)

        interior = (
            (query_unclipped[..., self.minus_index] >= 0.0)
            & (query_unclipped[..., self.plus_index] <= self.max_radius_mm)
            & (current_unclipped >= 0.12)
            & (current_unclipped <= self.max_radius_mm)
        )
        finite = torch.isfinite(slope) & torch.isfinite(p_at)
        valid = interior & finite & (slope <= -self.minimum_slope_per_mm)
        safe_slope = torch.where(valid, slope, -torch.ones_like(slope))
        threshold = torch.as_tensor(
            self.evidence_threshold, device=p_at.device, dtype=p_at.dtype
        )
        newton = (threshold - p_at) / safe_slope
        newton = newton.clamp(
            -self.maximum_newton_step_mm, self.maximum_newton_step_mm
        )
        newton = torch.where(valid, newton, torch.zeros_like(newton))
        sharpness = torch.where(interior & finite, sharpness, torch.zeros_like(sharpness))

        flattened = sampled.reshape(batch, stations, n_angles, -1)
        return flattened, {
            "current_radius_mm": current,
            "newton_mm": newton,
            "sharpness_per_mm": sharpness,
            "profile_valid": valid,
            "probability_at_surface": p_at,
            "radial_slope_per_mm": slope,
        }


class GeometricBoundaryFlow(nn.Module):
    """Gauge-fixed geometric normal-velocity controller for ``h(s,theta)``."""

    evidence_version = "local_newton_sharpness_v1"

    def __init__(
        self,
        *,
        shell_channels: int,
        stats: SurfaceStats,
        stations: int = 160,
        n_angles: int = 32,
        shell_radii: int = 24,
        max_radius_mm: float = 5.0,
        profile_offsets_mm: list[float] | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        angle_hidden: int = 24,
        angular_modes: int = 6,
        dropout: float = 0.10,
        probability_channel: int = 1,
        evidence_threshold: float = 0.5,
        minimum_slope_per_mm: float = 0.05,
        maximum_newton_step_mm: float = 0.60,
        sharpness_cap_per_mm: float = 4.0,
        evidence_gain_limit: float = 2.0,
        evidence_gain_init: float = 0.0,
        curvature_ratio_limit: float = 4.0,
        axial_curvature_clip: float = 4.0,
        nominal_spacing_mm: float = 0.30,
        trust_surface_rmse_mm: float = 1.0,
        trust_endpoint_mm: float = 2.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        profile_offsets_mm = profile_offsets_mm or [-0.6, -0.3, 0.0, 0.3, 0.6]
        if evidence_gain_limit <= 0.0:
            raise ValueError("evidence_gain_limit must be positive")
        if abs(evidence_gain_init) >= evidence_gain_limit:
            raise ValueError("abs(evidence_gain_init) must be below evidence_gain_limit")
        self.model_config: dict[str, Any] = {
            "shell_channels": int(shell_channels),
            "stations": int(stations),
            "n_angles": int(n_angles),
            "shell_radii": int(shell_radii),
            "max_radius_mm": float(max_radius_mm),
            "profile_offsets_mm": list(map(float, profile_offsets_mm)),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "angle_hidden": int(angle_hidden),
            "angular_modes": int(angular_modes),
            "dropout": float(dropout),
            "probability_channel": int(probability_channel),
            "evidence_threshold": float(evidence_threshold),
            "minimum_slope_per_mm": float(minimum_slope_per_mm),
            "maximum_newton_step_mm": float(maximum_newton_step_mm),
            "sharpness_cap_per_mm": float(sharpness_cap_per_mm),
            "evidence_gain_limit": float(evidence_gain_limit),
            "evidence_gain_init": float(evidence_gain_init),
            "curvature_ratio_limit": float(curvature_ratio_limit),
            "axial_curvature_clip": float(axial_curvature_clip),
            "nominal_spacing_mm": float(nominal_spacing_mm),
            "trust_surface_rmse_mm": float(trust_surface_rmse_mm),
            "trust_endpoint_mm": float(trust_endpoint_mm),
        }
        self.max_radius_mm = float(max_radius_mm)
        self.evidence_gain_limit = float(evidence_gain_limit)
        self.curvature_ratio_limit = float(curvature_ratio_limit)
        self.axial_curvature_clip = float(axial_curvature_clip)
        self.nominal_spacing_mm = float(nominal_spacing_mm)
        self.trust_surface_rmse_mm = float(trust_surface_rmse_mm)
        self.trust_endpoint_mm = float(trust_endpoint_mm)
        self.profile_sampler = MovingSurfaceProfileSampler(
            n_angles,
            max_radius_mm,
            profile_offsets_mm,
            shell_radii=shell_radii,
            probability_channel=probability_channel,
            evidence_threshold=evidence_threshold,
            minimum_slope_per_mm=minimum_slope_per_mm,
            maximum_newton_step_mm=maximum_newton_step_mm,
            sharpness_cap_per_mm=sharpness_cap_per_mm,
        )

        # sampled + newton + sharpness + valid + h + q0 + current + axial kappa
        # + cos(theta) + sin(theta)
        angle_input = shell_channels * len(profile_offsets_mm) + 9
        self.angle_encoder = nn.Sequential(
            nn.LayerNorm(angle_input),
            nn.Linear(angle_input, 2 * angle_hidden),
            nn.GELU(),
            nn.Linear(2 * angle_hidden, angle_hidden),
        )
        self.phase_encoder = PhaseAwareFourierAngularEncoder(
            angle_hidden, angle_hidden, n_modes=angular_modes
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(32),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )
        self.endpoint_embedding = nn.Sequential(
            nn.Linear(2, 24), nn.GELU(), nn.Linear(24, 24)
        )
        self.side_embedding = nn.Embedding(2, 8)
        self.input_projection = nn.Linear(2 * angle_hidden + 32 + 24 + 8, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.axial = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.alpha_head = nn.Sequential(
            nn.Linear(d_model + 24, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.axial_head = nn.Linear(d_model, 1)
        self.ray_hidden = nn.Sequential(
            nn.Linear(d_model + angle_hidden + 2, d_model),
            nn.GELU(),
        )
        self.ray_head = nn.Linear(d_model, 2)  # beta coefficient, evidence gain
        self.endpoint_head = nn.Sequential(
            nn.Linear(d_model + 24, d_model), nn.GELU(), nn.Linear(d_model, 2)
        )

        self.register_buffer(
            "position", fixed_position_encoding(stations, d_model), persistent=False
        )
        self.register_buffer(
            "angles",
            torch.linspace(0.0, 2.0 * math.pi, n_angles + 1)[:-1],
            persistent=False,
        )
        self.register_buffer(
            "h_output_scale", torch.tensor(stats.h_output_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "endpoint_mean", torch.tensor(stats.endpoint_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "endpoint_std", torch.tensor(stats.endpoint_std, dtype=torch.float32)
        )
        self.register_buffer(
            "endpoint_output_scale",
            torch.tensor(stats.endpoint_output_scale, dtype=torch.float32),
        )
        self.q0_radius_mean = float(stats.q0_radius_mean)
        self.q0_radius_std = float(stats.q0_radius_std)
        self.radius_reference_mm = float(max(stats.q0_radius_mean, 0.12))
        self.velocity_scale_mm = float(
            max(sum(stats.h_output_scale) / max(len(stats.h_output_scale), 1), 0.02)
        )

        for final in (
            self.alpha_head[-1],
            self.axial_head,
            self.ray_head,
            self.endpoint_head[-1],
        ):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        gain_bias = math.atanh(float(evidence_gain_init) / self.evidence_gain_limit)
        with torch.no_grad():
            self.ray_head.bias[1] = gain_bias

    def _axial_curvature(
        self,
        radius: torch.Tensor,
        arc_mm: torch.Tensor | None,
        station_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dimensionless axial curvature proxy ``r_ref * d2r/ds2``."""
        batch, stations, _ = radius.shape
        if stations < 3:
            return torch.zeros_like(radius)
        if arc_mm is None:
            arc = torch.arange(
                stations, device=radius.device, dtype=radius.dtype
            )[None].expand(batch, -1) * self.nominal_spacing_mm
        else:
            arc = arc_mm.to(device=radius.device, dtype=radius.dtype)
            if arc.ndim == 1:
                arc = arc[None].expand(batch, -1)
            if arc.shape != (batch, stations):
                raise ValueError(f"arc_mm must be [B,S], got {tuple(arc.shape)}")
        ds_left = arc[:, 1:-1] - arc[:, :-2]
        ds_right = arc[:, 2:] - arc[:, 1:-1]
        minimum_ds = max(0.05 * self.nominal_spacing_mm, 1e-4)
        safe_left = ds_left.clamp_min(minimum_ds)
        safe_right = ds_right.clamp_min(minimum_ds)
        slope_left = (radius[:, 1:-1] - radius[:, :-2]) / safe_left[..., None]
        slope_right = (radius[:, 2:] - radius[:, 1:-1]) / safe_right[..., None]
        second = 2.0 * (slope_right - slope_left) / (
            safe_left + safe_right
        )[..., None]
        triplet = (
            station_mask[:, :-2].bool()
            & station_mask[:, 1:-1].bool()
            & station_mask[:, 2:].bool()
            & (ds_left > 0.0)
            & (ds_right > 0.0)
        )
        second = torch.where(triplet[..., None], second, torch.zeros_like(second))
        curvature = F.pad(second * self.radius_reference_mm, (0, 0, 1, 1))
        return curvature.clamp(-self.axial_curvature_clip, self.axial_curvature_clip)

    def forward(
        self,
        q0_radius: torch.Tensor,
        h: torch.Tensor,
        endpoints: torch.Tensor,
        shell: torch.Tensor,
        side_id: torch.Tensor,
        station_mask: torch.Tensor,
        time: torch.Tensor,
        arc_mm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        sampled, profile = self.profile_sampler(shell, q0_radius, h)
        current = profile["current_radius_mm"]
        axial_curvature = self._axial_curvature(current, arc_mm, station_mask)
        angle = self.angles.to(device=h.device, dtype=h.dtype)
        angle_features = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        angle_features = angle_features[None, None].expand(
            len(h), h.shape[1], -1, -1
        )
        h_scale = self.h_output_scale.to(h).clamp_min(0.02)[None, None]
        sharpness_scale = max(self.profile_sampler.sharpness_cap_per_mm, 1e-3)
        newton_scale = max(self.profile_sampler.maximum_newton_step_mm, 1e-3)
        features = torch.cat(
            (
                sampled,
                (profile["newton_mm"] / newton_scale)[..., None],
                (profile["sharpness_per_mm"] / sharpness_scale)[..., None],
                profile["profile_valid"].to(h.dtype)[..., None],
                (h / h_scale)[..., None],
                (
                    (q0_radius - self.q0_radius_mean)
                    / max(self.q0_radius_std, 0.05)
                )[..., None],
                (
                    (current - self.q0_radius_mean)
                    / max(self.q0_radius_std, 0.05)
                )[..., None],
                (axial_curvature / max(self.axial_curvature_clip, 1e-3))[..., None],
                angle_features,
            ),
            dim=-1,
        )
        angle_encoded = self.angle_encoder(features)
        phase = self.phase_encoder(angle_encoded)
        time_token = self.time_embedding(time)
        endpoint_normalized = (endpoints - self.endpoint_mean) / self.endpoint_std
        endpoint_token = self.endpoint_embedding(endpoint_normalized)
        side_token = self.side_embedding(side_id)
        station_token = torch.cat(
            (
                phase,
                time_token[:, None].expand(-1, h.shape[1], -1),
                endpoint_token[:, None].expand(-1, h.shape[1], -1),
                side_token[:, None].expand(-1, h.shape[1], -1),
            ),
            dim=-1,
        )
        station_token = (
            self.input_projection(station_token) + self.position[:, : h.shape[1]]
        )
        encoded = self.final_norm(
            self.axial(station_token, src_key_padding_mask=~station_mask.bool())
        )
        station_weights = station_mask.to(encoded.dtype)
        pooled = (encoded * station_weights[:, :, None]).sum(dim=1) / station_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

        alpha = self.alpha_head(torch.cat((pooled, endpoint_token), dim=-1))[:, 0]
        alpha = alpha * self.velocity_scale_mm
        axial_raw = self.axial_head(encoded)[..., 0] * self.velocity_scale_mm
        axial_mean = (axial_raw * station_weights).sum(dim=1, keepdim=True) / station_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        axial_term = (axial_raw - axial_mean) * station_weights

        context = encoded[:, :, None].expand(-1, -1, h.shape[2], -1)
        ray = self.ray_hidden(
            torch.cat((context, angle_encoded, angle_features), dim=-1)
        )
        ray_output = self.ray_head(ray)
        beta_coefficient = ray_output[..., 0] * h_scale
        curvature_ratio = (
            self.radius_reference_mm / current.clamp_min(0.12)
        ).clamp(1.0 / self.curvature_ratio_limit, self.curvature_ratio_limit)
        curvature_raw = beta_coefficient * curvature_ratio
        # Gauge: the angularly constant station mode belongs to a(s), not beta.
        curvature_term = curvature_raw - curvature_raw.mean(dim=2, keepdim=True)
        gain = self.evidence_gain_limit * torch.tanh(ray_output[..., 1])
        evidence_term = gain * profile["newton_mm"]

        velocity = (
            alpha[:, None, None]
            + axial_term[:, :, None]
            + curvature_term
            + evidence_term
        )
        velocity = velocity * station_weights[:, :, None]
        endpoint_velocity = self.endpoint_head(
            torch.cat((pooled, endpoint_token), dim=-1)
        ) * self.endpoint_output_scale
        diagnostics = {
            **profile,
            "alpha_mm": alpha,
            "axial_term_mm": axial_term,
            "beta_coefficient_mm": beta_coefficient,
            "curvature_ratio": curvature_ratio,
            "curvature_term_mm": curvature_term,
            "axial_curvature": axial_curvature,
            "evidence_gain": gain,
            "evidence_term_mm": evidence_term,
            "learned_base_mm": (
                alpha[:, None, None] + axial_term[:, :, None] + curvature_term
            ),
        }
        return velocity, endpoint_velocity, diagnostics
