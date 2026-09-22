"""R2: unrestricted normal-displacement flow in the Bishop surface chart."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import (
    PhaseAwareFourierAngularEncoder,
    SinusoidalTimeEmbedding,
    fixed_position_encoding,
)


@dataclass
class SurfaceStats:
    h_output_scale: list[float]
    q0_radius_mean: float
    q0_radius_std: float
    endpoint_mean: list[float]
    endpoint_std: list[float]
    endpoint_output_scale: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SurfaceStats:
        return cls(**raw)


class FixedRaySurfaceSampler(nn.Module):
    """Legacy R2 sampler with an absolute-crossing residual.

    The sampled profile channels genuinely move with ``h``.  The additional
    scalar ``crossing - (q0+h)`` does not; it is retained so old R2 checkpoints
    remain loadable.  New training should use the corrected GeoFlow sampler.
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
    ) -> None:
        super().__init__()
        angles = torch.linspace(0.0, 2.0 * math.pi, n_angles + 1)[:-1]
        self.register_buffer("angles", angles, persistent=False)
        self.register_buffer(
            "profile_offsets",
            torch.tensor(profile_offsets_mm, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "radius_grid",
            torch.linspace(0.0, float(max_radius_mm), int(shell_radii)),
            persistent=False,
        )
        self.max_radius_mm = float(max_radius_mm)
        self.probability_channel = int(probability_channel)
        self.evidence_threshold = float(evidence_threshold)

    def _crossing(self, probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        threshold = torch.as_tensor(
            self.evidence_threshold,
            dtype=probability.dtype,
            device=probability.device,
        )
        below = probability < threshold
        index = below.to(torch.int64).argmax(dim=-1)
        valid = below.any(dim=-1) & (index > 0) & (probability[..., 0] >= threshold)
        upper = index.clamp(1, probability.shape[-1] - 1)
        lower = upper - 1
        p0 = torch.gather(probability, -1, lower[..., None])[..., 0]
        p1 = torch.gather(probability, -1, upper[..., None])[..., 0]
        grid = self.radius_grid.to(probability)
        r0, r1 = grid[lower], grid[upper]
        fraction = ((p0 - threshold) / (p0 - p1).clamp_min(1e-6)).clamp(0.0, 1.0)
        crossing = r0 + fraction * (r1 - r0)
        return torch.where(valid, crossing, torch.zeros_like(crossing)), valid

    def forward(
        self,
        shell: torch.Tensor,
        q0_radius: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, stations, n_angles, _ = shell.shape
        current = (q0_radius + h).clamp(0.12, self.max_radius_mm)
        query = current[..., None] + self.profile_offsets.to(current)[None, None, None]
        query = query.clamp(0.0, self.max_radius_mm)
        grid_x = 2.0 * query / self.max_radius_mm - 1.0
        angle = self.angles.to(current)
        grid_y = (angle / math.pi - 1.0)[None, None, :, None].expand_as(grid_x)
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            batch * stations, 1, n_angles * len(self.profile_offsets), 2
        )
        polar = shell.permute(0, 2, 1, 3, 4).reshape(
            batch * stations, channels, n_angles, -1
        )
        polar = torch.cat((polar, polar[:, :, :1]), dim=2)
        sampled = F.grid_sample(
            polar,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled[:, :, 0].reshape(
            batch, stations, channels, n_angles, len(self.profile_offsets)
        )
        sampled = sampled.permute(0, 1, 3, 2, 4).reshape(
            batch, stations, n_angles, -1
        )

        if self.probability_channel >= channels:
            raise ValueError("Configured OOF probability channel is absent")
        probability = shell[:, self.probability_channel]
        crossing, valid = self._crossing(probability)
        evidence = torch.where(valid, crossing - current, torch.zeros_like(current))
        return sampled, evidence


class SurfaceDisplacementTransformer(nn.Module):
    """Linear flow field for the unrestricted h(s,theta) representation."""

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
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        angle_hidden: int = 48,
        angular_modes: int = 8,
        dropout: float = 0.10,
        probability_channel: int = 1,
        evidence_threshold: float = 0.5,
        trust_surface_rmse_mm: float = 1.0,
        trust_endpoint_mm: float = 2.0,
    ) -> None:
        super().__init__()
        profile_offsets_mm = profile_offsets_mm or [-0.6, -0.3, 0.0, 0.3, 0.6]
        self.model_config = {
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
            "trust_surface_rmse_mm": float(trust_surface_rmse_mm),
            "trust_endpoint_mm": float(trust_endpoint_mm),
        }
        self.trust_surface_rmse_mm = float(trust_surface_rmse_mm)
        self.trust_endpoint_mm = float(trust_endpoint_mm)
        self.sampler = FixedRaySurfaceSampler(
            n_angles,
            max_radius_mm,
            profile_offsets_mm,
            shell_radii=shell_radii,
            probability_channel=probability_channel,
            evidence_threshold=evidence_threshold,
        )
        angle_input = shell_channels * len(profile_offsets_mm) + 5
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
            SinusoidalTimeEmbedding(64),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.endpoint_embedding = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 32)
        )
        self.side_embedding = nn.Embedding(2, 16)
        self.input_projection = nn.Linear(2 * angle_hidden + 64 + 32 + 16, d_model)
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
        self.h_head = nn.Sequential(
            nn.Linear(d_model + angle_hidden + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.endpoint_head = nn.Sequential(
            nn.Linear(d_model + 32, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
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
            "h_output_scale",
            torch.tensor(stats.h_output_scale, dtype=torch.float32),
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
        nn.init.zeros_(self.h_head[-1].weight)
        nn.init.zeros_(self.h_head[-1].bias)
        nn.init.zeros_(self.endpoint_head[-1].weight)
        nn.init.zeros_(self.endpoint_head[-1].bias)

    def forward(
        self,
        q0_radius: torch.Tensor,
        h: torch.Tensor,
        endpoints: torch.Tensor,
        shell: torch.Tensor,
        side_id: torch.Tensor,
        station_mask: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampled, evidence = self.sampler(shell, q0_radius, h)
        angle = self.angles.to(h)
        angle_features = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        angle_features = angle_features[None, None].expand(
            len(h), h.shape[1], -1, -1
        )
        features = torch.cat(
            (
                sampled,
                evidence[..., None],
                (
                    h / self.h_output_scale.clamp_min(0.02)[None, None]
                )[..., None],
                ((q0_radius - self.q0_radius_mean) / max(self.q0_radius_std, 0.05))[
                    ..., None
                ],
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
        station_token = self.input_projection(station_token) + self.position[:, : h.shape[1]]
        encoded = self.final_norm(
            self.axial(station_token, src_key_padding_mask=~station_mask.bool())
        )
        context = encoded[:, :, None].expand(-1, -1, h.shape[2], -1)
        angle_out = angle_features
        h_velocity = self.h_head(
            torch.cat((context, angle_encoded, angle_out), dim=-1)
        )[..., 0]
        h_velocity = h_velocity * self.h_output_scale[None, None]
        h_velocity = h_velocity * station_mask[:, :, None].to(h.dtype)
        weights = station_mask.to(encoded.dtype)
        pooled = (encoded * weights[:, :, None]).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        endpoint_velocity = self.endpoint_head(
            torch.cat((pooled, endpoint_token), dim=-1)
        ) * self.endpoint_output_scale
        return h_velocity, endpoint_velocity
