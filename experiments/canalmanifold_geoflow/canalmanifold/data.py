"""PyTorch datasets and train-fold statistics for tube cache shards."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .chart import reflect_local_numpy, reflect_shell_numpy, reflect_surface_numpy
from .constants import HARMONICS
from .manifest import load_splits


@dataclass
class StateStats:
    local_mean: list[float]
    local_std: list[float]
    global_mean: list[float]
    global_std: list[float]
    output_local_scale: list[float]
    output_global_scale: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StateStats:
        return cls(**raw)


class TubeCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        case_ids: list[str] | set[str],
        *,
        exclude_fallback: bool = True,
        cache_in_ram: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        wanted = set(map(str, case_ids))
        paths = sorted(self.cache_dir.glob("fold_*/*.npz"))
        selected = []
        for path in paths:
            case_id = path.stem.rsplit("_", 1)[0]
            if case_id not in wanted:
                continue
            if exclude_fallback:
                with np.load(path, allow_pickle=False) as archive:
                    key = "training_excluded" if "training_excluded" in archive else "fallback"
                    if bool(archive[key]):
                        continue
            selected.append(path)
        if not selected:
            raise FileNotFoundError(
                f"No usable tube shards in {self.cache_dir} for {len(wanted)} requested cases"
            )
        self.paths = selected
        self.cache_in_ram = bool(cache_in_ram)
        self._memory = [self._load(path) for path in self.paths] if self.cache_in_ram else None

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as archive:
            side = path.stem.rsplit("_", 1)[1]
            q0_local = archive["q0_local"].astype(np.float32)
            q1_local = archive["q1_local"].astype(np.float32)
            shell = archive["shell"].astype(np.float32)
            if "representation_version" not in archive or int(archive["representation_version"]) < 2:
                raise RuntimeError(
                    f"{path} is a legacy cache shard. CanalManifoldFlow v2 requires "
                    "m<=8 states and target_occupancy_shell."
                )
            target_occupancy = archive["target_occupancy_shell"].astype(np.float32)
            q0_ray_radii = archive["q0_ray_radii_mm"].astype(np.float32)
            q1_ray_radii = archive["q1_ray_radii_mm"].astype(np.float32)
            q0_ray_valid = archive["q0_ray_valid"].astype(bool)
            q1_ray_valid = archive["q1_ray_valid"].astype(bool)

            # The cache remains in its native right-handed Bishop frame.  The
            # right side is reflected only at the model boundary, and both the
            # state and shell receive the same exact chart transform.
            if side == "R":
                q0_local = reflect_local_numpy(q0_local)
                q1_local = reflect_local_numpy(q1_local)
                shell = reflect_shell_numpy(shell)
                target_occupancy = reflect_surface_numpy(target_occupancy)
                q0_ray_radii = reflect_surface_numpy(q0_ray_radii)
                q1_ray_radii = reflect_surface_numpy(q1_ray_radii)
                q0_ray_valid = reflect_surface_numpy(q0_ray_valid)
                q1_ray_valid = reflect_surface_numpy(q1_ray_valid)

            q0_valid = (
                archive["q0_station_valid"].astype(bool)
                if "q0_station_valid" in archive
                else archive["station_mask"].astype(bool)
            )
            q1_valid = (
                archive["q1_station_valid"].astype(bool)
                if "q1_station_valid" in archive
                else archive["station_mask"].astype(bool)
            )
            sample = {
                "q0_local": torch.from_numpy(q0_local),
                "q0_global": torch.from_numpy(archive["q0_global"].astype(np.float32)),
                "q1_local": torch.from_numpy(q1_local),
                "q1_global": torch.from_numpy(archive["q1_global"].astype(np.float32)),
                # Causal model input: q1/GT validity never enters the network.
                "station_mask": torch.from_numpy(q0_valid),
                # Supervision is restricted to stations valid at both endpoints.
                "loss_mask": torch.from_numpy(q0_valid & q1_valid),
                "metric_mask": torch.from_numpy(q0_valid | q1_valid),
                "shell": torch.from_numpy(shell),
                # Supervision-only tensor; this is never concatenated to shell.
                "target_occupancy_shell": torch.from_numpy(target_occupancy),
                "shell_radii_mm": torch.from_numpy(
                    archive["shell_radii_mm"].astype(np.float32)
                ),
                "q0_ray_radii_mm": torch.from_numpy(q0_ray_radii),
                "q1_ray_radii_mm": torch.from_numpy(q1_ray_radii),
                "free_h_target_mm": torch.from_numpy(q1_ray_radii - q0_ray_radii),
                "ray_loss_mask": torch.from_numpy(q0_ray_valid & q1_ray_valid),
                "side_id": torch.as_tensor(int(archive["side_id"]), dtype=torch.long),
                "arc_mm": torch.from_numpy(archive["arc_mm"].astype(np.float32)),
                "path": str(path),
                "case_id": path.stem.rsplit("_", 1)[0],
                "side": side,
                "right_chart_reflected": side == "R",
            }
        return sample

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._memory is not None:
            return self._memory[index]
        return self._load(self.paths[index])


def split_datasets(config: dict[str, Any], refinement_fold: int) -> tuple[TubeCacheDataset, TubeCacheDataset]:
    paths = config["paths"]
    splits = load_splits(paths["splits_file"])
    fold = splits[int(refinement_fold)]
    training = config.get("training", {})
    common = {
        "exclude_fallback": bool(training.get("exclude_fallback", True)),
        "cache_in_ram": bool(training.get("cache_in_ram", False)),
    }
    return (
        TubeCacheDataset(paths["cache_dir"], fold["train"], **common),
        TubeCacheDataset(paths["cache_dir"], fold["val"], **common),
    )


def compute_state_stats(dataset: TubeCacheDataset) -> StateStats:
    local_values: list[np.ndarray] = []
    local_delta: list[np.ndarray] = []
    global_values: list[np.ndarray] = []
    global_delta: list[np.ndarray] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        mask = sample["loss_mask"].numpy().astype(bool)
        q0_local = sample["q0_local"].numpy()
        q1_local = sample["q1_local"].numpy()
        local_values.extend((q0_local[mask], q1_local[mask]))
        local_delta.append((q1_local - q0_local)[mask])
        q0_global = sample["q0_global"].numpy()
        q1_global = sample["q1_global"].numpy()
        global_values.extend((q0_global[None], q1_global[None]))
        global_delta.append((q1_global - q0_global)[None])

    local = np.concatenate(local_values, axis=0)
    delta_local = np.concatenate(local_delta, axis=0)
    global_state = np.concatenate(global_values, axis=0)
    delta_global = np.concatenate(global_delta, axis=0)
    local_floor = np.asarray(
        [0.05, 0.05, 0.02] + [0.01] * (2 * len(HARMONICS)),
        dtype=np.float64,
    )
    global_floor = np.asarray([0.02, 0.10, 0.10])
    return StateStats(
        local_mean=local.mean(axis=0).astype(float).tolist(),
        local_std=np.maximum(local.std(axis=0), local_floor).astype(float).tolist(),
        global_mean=global_state.mean(axis=0).astype(float).tolist(),
        global_std=np.maximum(global_state.std(axis=0), global_floor).astype(float).tolist(),
        output_local_scale=np.maximum(delta_local.std(axis=0), local_floor).astype(float).tolist(),
        output_global_scale=np.maximum(delta_global.std(axis=0), global_floor).astype(float).tolist(),
    )


def save_stats(stats: StateStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    temporary.replace(path)
