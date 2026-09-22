"""Phase-aware polar condition encoder and axial Transformer velocity field."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .constants import HARMONIC_START, HARMONICS, LOCAL_DIM, MAX_HARMONIC
from .data import StateStats


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(half - 1, 1)
        )
        phase = time[:, None] * frequency[None]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
        return F.pad(embedding, (0, self.dimension - embedding.shape[1]))


def fixed_position_encoding(length: int, dimension: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32)[:, None]
    divisor = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
    return encoding[None]


class PolarSurfaceSampler(nn.Module):
    """Legacy R1 sampler retained for checkpoint compatibility.

    Its scalar ``crossing - current`` channel changes when the state moves, but
    the absolute crossing is computed from the same fixed radial profile.
    Therefore it is algebraically a state-centred restoring term, not new
    image evidence.  The corrected GeoFlow arm uses local Newton and slope
    features from :mod:`canalmanifold.geoflow_model` instead.
    """

    def __init__(
        self,
        n_angles: int,
        max_radius_mm: float,
        profile_offsets_mm: list[float],
        *,
        probability_channel: int = 1,
        evidence_threshold: float = 0.5,
        evidence_radii: int = 32,
    ) -> None:
        super().__init__()
        angles = torch.linspace(0.0, 2.0 * math.pi, n_angles + 1)[:-1]
        basis = []
        for harmonic in HARMONICS:
            basis.extend((torch.cos(harmonic * angles), torch.sin(harmonic * angles)))
        self.register_buffer("angles", angles, persistent=False)
        self.register_buffer("basis", torch.stack(basis, dim=1), persistent=False)
        self.register_buffer(
            "profile_offsets", torch.tensor(profile_offsets_mm, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "evidence_radius_grid",
            torch.linspace(0.0, float(max_radius_mm), int(evidence_radii)),
            persistent=False,
        )
        self.max_radius_mm = float(max_radius_mm)
        self.probability_channel = int(probability_channel)
        self.evidence_threshold = float(evidence_threshold)

    def current_radius(self, local: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        log_radius = (
            global_state[:, None, None, 0]
            + local[:, :, None, 2]
            + torch.einsum(
                "bsk,ak->bsa",
                local[:, :, HARMONIC_START:LOCAL_DIM],
                self.basis,
            )
        )
        return torch.exp(log_radius.clamp(math.log(0.12), math.log(8.0)))

    def _sample(
        self,
        shell: torch.Tensor,
        local: torch.Tensor,
        query_radius: torch.Tensor,
    ) -> torch.Tensor:
        """Sample queries expressed about the moving local centre.

        ``query_radius`` has shape [B,S,A,Q]; output is [B,S,A,C,Q].
        """
        batch, channels, stations, n_angles, _ = shell.shape
        cos = torch.cos(self.angles).to(query_radius.dtype)[None, None, :, None]
        sin = torch.sin(self.angles).to(query_radius.dtype)[None, None, :, None]
        x = local[:, :, None, None, 0] + query_radius * cos
        y = local[:, :, None, None, 1] + query_radius * sin
        rho = torch.sqrt(x.square() + y.square() + 1e-10).clamp(0.0, self.max_radius_mm)
        theta = torch.remainder(torch.atan2(y, x), 2.0 * math.pi)

        grid_x = 2.0 * rho / self.max_radius_mm - 1.0
        grid_y = theta / math.pi - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            batch * stations, 1, n_angles * query_radius.shape[-1], 2
        )
        polar = shell.permute(0, 2, 1, 3, 4).reshape(batch * stations, channels, n_angles, -1)
        polar = torch.cat((polar, polar[:, :, :1]), dim=2)
        sampled = F.grid_sample(
            polar,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        queries = query_radius.shape[-1]
        sampled = sampled[:, :, 0].reshape(batch, stations, channels, n_angles, queries)
        return sampled.permute(0, 1, 3, 2, 4)

    def signed_radial_evidence(
        self,
        shell: torch.Tensor,
        local: torch.Tensor,
        current_radius: torch.Tensor,
    ) -> torch.Tensor:
        if self.probability_channel >= shell.shape[1]:
            raise ValueError(
                f"Probability channel {self.probability_channel} is absent from shell "
                f"with {shell.shape[1]} channels"
            )
        grid = self.evidence_radius_grid.to(
            device=current_radius.device, dtype=current_radius.dtype
        )
        query = grid[None, None, None, :].expand(
            len(local), local.shape[1], len(self.angles), -1
        )
        probability = self._sample(shell, local, query)[
            :, :, :, self.probability_channel, :
        ]
        threshold = torch.as_tensor(
            self.evidence_threshold,
            device=probability.device,
            dtype=probability.dtype,
        )
        below = probability < threshold
        first_below = below.to(torch.int64).argmax(dim=-1)
        has_crossing = below.any(dim=-1) & (first_below > 0)
        starts_inside = probability[..., 0] >= threshold
        upper = first_below.clamp(1, len(grid) - 1)
        lower = upper - 1
        p0 = torch.gather(probability, -1, lower[..., None])[..., 0]
        p1 = torch.gather(probability, -1, upper[..., None])[..., 0]
        r0 = grid[lower]
        r1 = grid[upper]
        fraction = ((p0 - threshold) / (p0 - p1).clamp_min(1e-6)).clamp(0.0, 1.0)
        crossing = r0 + fraction * (r1 - r0)
        valid = has_crossing & starts_inside
        delta = (crossing - current_radius).clamp(
            -self.max_radius_mm, self.max_radius_mm
        )
        return torch.where(valid, delta, torch.zeros_like(delta))

    def forward(self, shell: torch.Tensor, local: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        # shell [B,C,S,A,R] -> surface features [B,S,A,C*offsets+1]
        radius = self.current_radius(local, global_state)
        query_radius = (
            radius[:, :, :, None]
            + self.profile_offsets.to(radius.dtype)[None, None, None]
        )
        sampled = self._sample(shell, local, query_radius)
        sampled = sampled.reshape(
            len(local), local.shape[1], len(self.angles), -1
        )
        evidence = self.signed_radial_evidence(shell, local, radius)[..., None]
        return torch.cat((sampled, evidence), dim=-1)


class PhaseAwareFourierAngularEncoder(nn.Module):
    """Retain absolute angular phase instead of mean/max pooling it away.

    For every current-surface condition channel we retain the real DC term and
    real/imaginary parts of the configured modes.  The deterministic anatomical gauge and
    the right-side chart reflection make those phases comparable across cases.
    """

    def __init__(self, input_channels: int, hidden: int, n_modes: int = 4) -> None:
        super().__init__()
        self.n_modes = int(n_modes)
        feature_dim = int(input_channels) * (1 + 2 * self.n_modes)
        output_dim = 2 * int(hidden)
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, stations, angles, channels = values.shape
        x = values.reshape(batch * stations, angles, channels).float()
        coefficients = torch.fft.rfft(x, dim=1, norm="ortho")
        if coefficients.shape[1] <= self.n_modes:
            raise ValueError(
                f"Need at least {2 * self.n_modes + 1} angles, got {angles}"
            )
        dc = coefficients[:, 0].real
        modes = coefficients[:, 1 : self.n_modes + 1]
        features = torch.cat(
            (dc, modes.real.reshape(len(x), -1), modes.imag.reshape(len(x), -1)),
            dim=1,
        )
        encoded = self.network(features)
        return encoded.reshape(batch, stations, -1)


class CanalVelocityTransformer(nn.Module):
    def __init__(
        self,
        *,
        shell_channels: int,
        stats: StateStats,
        stations: int = 160,
        n_angles: int = 32,
        max_radius_mm: float = 5.0,
        profile_offsets_mm: list[float] | None = None,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        angular_hidden: int = 64,
        angular_modes: int = MAX_HARMONIC,
        dropout: float = 0.10,
        trust_surface_rmse_mm: float = 1.0e6,
        trust_endpoint_mm: float = 1.0e6,
        max_center_shift_mm: float = 3.0,
        angular_readout: str = "fourier_phase_m0_8",
        right_chart: str = "theta_reflection_at_model_boundary",
        probability_channel: int = 1,
        evidence_threshold: float = 0.5,
        evidence_radii: int = 32,
        max_harmonic: int = MAX_HARMONIC,
        local_dim: int = LOCAL_DIM,
    ) -> None:
        super().__init__()
        if int(max_harmonic) != MAX_HARMONIC or int(local_dim) != LOCAL_DIM:
            raise ValueError(
                f"Checkpoint state layout m<={max_harmonic}, local_dim={local_dim} "
                f"does not match this build (m<={MAX_HARMONIC}, local_dim={LOCAL_DIM})"
            )
        profile_offsets_mm = profile_offsets_mm or [-0.6, -0.3, 0.0, 0.3, 0.6]
        self.model_config = {
            "shell_channels": int(shell_channels),
            "stations": int(stations),
            "n_angles": int(n_angles),
            "max_radius_mm": float(max_radius_mm),
            "profile_offsets_mm": list(map(float, profile_offsets_mm)),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "angular_hidden": int(angular_hidden),
            "angular_modes": int(angular_modes),
            "dropout": float(dropout),
            "trust_surface_rmse_mm": float(trust_surface_rmse_mm),
            "trust_endpoint_mm": float(trust_endpoint_mm),
            "max_center_shift_mm": float(max_center_shift_mm),
            "angular_readout": str(angular_readout),
            "right_chart": str(right_chart),
            "probability_channel": int(probability_channel),
            "evidence_threshold": float(evidence_threshold),
            "evidence_radii": int(evidence_radii),
            "max_harmonic": int(MAX_HARMONIC),
            "local_dim": int(LOCAL_DIM),
        }
        self.trust_surface_rmse_mm = float(trust_surface_rmse_mm)
        self.trust_endpoint_mm = float(trust_endpoint_mm)
        self.max_center_shift_mm = float(max_center_shift_mm)
        self.surface_sampler = PolarSurfaceSampler(
            n_angles,
            max_radius_mm,
            profile_offsets_mm,
            probability_channel=probability_channel,
            evidence_threshold=evidence_threshold,
            evidence_radii=evidence_radii,
        )
        self.angular_encoder = PhaseAwareFourierAngularEncoder(
            shell_channels * len(profile_offsets_mm) + 1,
            angular_hidden,
            n_modes=angular_modes,
        )
        self.local_embedding = nn.Sequential(
            nn.Linear(LOCAL_DIM, 64), nn.GELU(), nn.Linear(64, 64)
        )
        self.global_embedding = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64))
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(64), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64)
        )
        self.side_embedding = nn.Embedding(2, 32)
        self.input_projection = nn.Linear(2 * angular_hidden + 64 + 64 + 64 + 32, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.axial = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.local_head = nn.Linear(d_model, LOCAL_DIM)
        self.global_head = nn.Sequential(nn.Linear(d_model + 64, d_model), nn.GELU(), nn.Linear(d_model, 3))
        self.register_buffer("position", fixed_position_encoding(stations, d_model), persistent=False)

        self.register_buffer("local_mean", torch.tensor(stats.local_mean, dtype=torch.float32))
        self.register_buffer("local_std", torch.tensor(stats.local_std, dtype=torch.float32))
        self.register_buffer("global_mean", torch.tensor(stats.global_mean, dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(stats.global_std, dtype=torch.float32))
        self.register_buffer("output_local_scale", torch.tensor(stats.output_local_scale, dtype=torch.float32))
        self.register_buffer("output_global_scale", torch.tensor(stats.output_global_scale, dtype=torch.float32))
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        final_global = self.global_head[-1]
        nn.init.zeros_(final_global.weight)
        nn.init.zeros_(final_global.bias)

    def forward(
        self,
        local: torch.Tensor,
        global_state: torch.Tensor,
        shell: torch.Tensor,
        side_id: torch.Tensor,
        station_mask: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditions = self.surface_sampler(shell, local, global_state)
        angular = self.angular_encoder(conditions)
        local_normalized = (local - self.local_mean) / self.local_std
        global_normalized = (global_state - self.global_mean) / self.global_std
        local_token = self.local_embedding(local_normalized)
        global_token = self.global_embedding(global_normalized)
        time_token = self.time_embedding(time)
        side_token = self.side_embedding(side_id)
        repeated = [
            angular,
            local_token,
            global_token[:, None].expand(-1, local.shape[1], -1),
            time_token[:, None].expand(-1, local.shape[1], -1),
            side_token[:, None].expand(-1, local.shape[1], -1),
        ]
        token = self.input_projection(torch.cat(repeated, dim=-1)) + self.position[:, : local.shape[1]]
        encoded = self.final_norm(self.axial(token, src_key_padding_mask=~station_mask.bool()))
        local_output = self.local_head(encoded) * self.output_local_scale
        weights = station_mask.to(encoded.dtype)
        pooled = (encoded * weights[:, :, None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_output = self.global_head(torch.cat((pooled, global_token), dim=1)) * self.output_global_scale
        local_output = local_output * weights[:, :, None]
        return local_output, global_output
