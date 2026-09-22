"""Side-canonical tube chart transforms.

The physical Bishop frame stored in the cache remains right handed.  Only the
network chart is reflected for right canals so that both sides use the same
angular sense.  The transform is an involution and therefore also converts a
predicted canonical velocity/state back to the native cache chart.
"""

from __future__ import annotations

import numpy as np
import torch

from .constants import HARMONICS

# theta -> -theta leaves cosine coefficients unchanged and negates sine
# coefficients.  d2 follows the reflected angular axis as well.
RIGHT_REFLECTION_SIGNS = np.asarray(
    [1.0, -1.0, 1.0]
    + [sign for _ in HARMONICS for sign in (1.0, -1.0)],
    dtype=np.float32,
)


def reflected_angle_indices(n_angles: int) -> np.ndarray:
    """Indices implementing theta -> -theta with theta=0 kept fixed."""
    indices = (-np.arange(int(n_angles), dtype=np.int64)) % int(n_angles)
    return indices


def reflect_local_numpy(local: np.ndarray) -> np.ndarray:
    output = np.asarray(local, dtype=np.float32).copy()
    output *= RIGHT_REFLECTION_SIGNS
    return output


def reflect_shell_numpy(shell: np.ndarray) -> np.ndarray:
    """Reflect a cache shell [C,S,A,R] along its angular axis."""
    output = np.asarray(shell, dtype=np.float32)
    return output[:, :, reflected_angle_indices(output.shape[2]), :].copy()


def reflect_surface_numpy(surface: np.ndarray) -> np.ndarray:
    """Reflect a station/angle tensor ``[S,A,...]`` into the canonical chart."""
    output = np.asarray(surface)
    if output.ndim < 2:
        raise ValueError(f"surface must have at least [S,A], got {output.shape}")
    return np.take(output, reflected_angle_indices(output.shape[1]), axis=1).copy()


def reflect_local_torch(local: torch.Tensor) -> torch.Tensor:
    signs = torch.as_tensor(
        RIGHT_REFLECTION_SIGNS,
        device=local.device,
        dtype=local.dtype,
    )
    return local * signs


def native_local_numpy(local: np.ndarray, side: str) -> np.ndarray:
    """Convert a canonical model state to the cache's native physical chart."""
    return reflect_local_numpy(local) if str(side).upper() == "R" else np.asarray(local).copy()
