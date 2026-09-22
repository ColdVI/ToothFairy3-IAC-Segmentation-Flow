"""NIfTI, probability and label I/O with explicit geometry checks."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


@dataclass(frozen=True)
class Volume:
    data: np.ndarray
    affine: np.ndarray
    path: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def spacing(self) -> tuple[float, float, float]:
        linear = np.asarray(self.affine[:3, :3], dtype=np.float64)
        return tuple(float(v) for v in np.linalg.norm(linear, axis=0))


def load_nifti(path: str | Path, dtype=np.float32) -> Volume:
    path = Path(path)
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3-D NIfTI, got {image.shape}: {path}")
    data = np.asanyarray(image.dataobj).astype(dtype, copy=False)
    return Volume(data=data, affine=np.asarray(image.affine), path=str(path))


def load_label_on_reference(path: str | Path, reference: Volume) -> np.ndarray:
    image = nib.load(str(path))
    same_shape = tuple(image.shape) == reference.shape
    same_affine = np.allclose(image.affine, reference.affine, atol=1e-4, rtol=1e-5)
    if not (same_shape and same_affine):
        image = resample_from_to(image, (reference.shape, reference.affine), order=0)
    return np.rint(np.asanyarray(image.dataobj)).astype(np.int32, copy=False)


def normalize_label_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def resolve_iac_label_ids(dataset_json: str | Path) -> tuple[int, int]:
    """Resolve L/R IAC IDs by name; never assume numeric IDs."""
    with Path(dataset_json).open("r", encoding="utf-8-sig") as stream:
        metadata = json.load(stream)
    labels = metadata.get("labels")
    if not isinstance(labels, dict):
        raise KeyError("dataset.json does not contain a 'labels' mapping")

    left_candidates: list[tuple[str, int]] = []
    right_candidates: list[tuple[str, int]] = []
    for raw_name, raw_value in labels.items():
        name = normalize_label_name(str(raw_name))
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        has_iac = "inferior alveolar canal" in name or name.endswith(" iac")
        if has_iac and "left" in name:
            left_candidates.append((str(raw_name), value))
        if has_iac and "right" in name:
            right_candidates.append((str(raw_name), value))

    if len(left_candidates) != 1 or len(right_candidates) != 1:
        raise ValueError(
            "Could not uniquely resolve Left/Right Inferior Alveolar Canal "
            f"from dataset.json. left={left_candidates}, right={right_candidates}"
        )
    return left_candidates[0][1], right_candidates[0][1]


def _extract_npz_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        for key in ("probabilities", "softmax", "prob", "data"):
            if key in archive:
                return np.asarray(archive[key])
        arrays = [np.asarray(archive[key]) for key in archive.files if archive[key].ndim >= 3]
        if len(arrays) != 1:
            raise KeyError(f"Cannot choose probability tensor in {path}; keys={archive.files}")
        return arrays[0]


def _align_spatial(
    array: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    prefer_nnunet_zyx: bool = False,
) -> np.ndarray:
    # nnU-Net probability NPZ uses Z,Y,X while nibabel arrays use X,Y,Z.
    # Test that convention before accepting an accidentally matching cubic
    # shape, where shape-only permutation inference is ambiguous.
    if prefer_nnunet_zyx and tuple(array.shape[::-1]) == target_shape:
        return np.transpose(array, (2, 1, 0))
    if tuple(array.shape) == target_shape:
        return array
    matches = []
    for perm in permutations(range(3)):
        if tuple(array.shape[i] for i in perm) == target_shape:
            matches.append(perm)
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely align spatial probability shape {array.shape} to image {target_shape}; "
            f"matching permutations={matches}"
        )
    return np.transpose(array, matches[0])


def load_probability_channels(
    path: str | Path,
    target_shape: tuple[int, int, int],
    left_channel: int,
    right_channel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load exported nnU-Net probabilities and align them to NIfTI XYZ order."""
    path = Path(path)
    if path.suffix == ".npz":
        array = _extract_npz_array(path)
    elif path.name.endswith(".nii.gz") or path.suffix == ".nii":
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
    else:
        raise ValueError(f"Unsupported probability file: {path}")

    if array.ndim != 4:
        raise ValueError(f"Probability tensor must be 4-D, got {array.shape}: {path}")

    max_channel = max(left_channel, right_channel)
    first_matches = array.shape[0] > max_channel and sorted(array.shape[1:]) == sorted(target_shape)
    last_matches = array.shape[-1] > max_channel and sorted(array.shape[:3]) == sorted(target_shape)
    if first_matches and not last_matches:
        channels_first = array
    elif last_matches and not first_matches:
        channels_first = np.moveaxis(array, -1, 0)
    elif first_matches and last_matches:
        # A genuine ambiguity is possible only for unusually tiny/cubic data.
        # Prefer the axis with fewer entries, as class count is normally small.
        channels_first = array if array.shape[0] <= array.shape[-1] else np.moveaxis(array, -1, 0)
    else:
        raise ValueError(
            "Cannot align probability channel/spatial axes to the image. "
            f"requested L={left_channel}, R={right_channel}, tensor={array.shape}, image={target_shape}"
        )

    aligned = np.stack(
        [
            _align_spatial(
                np.asarray(channels_first[c]),
                target_shape,
                prefer_nnunet_zyx=(path.suffix == ".npz"),
            )
            for c in range(channels_first.shape[0])
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    if not np.isfinite(aligned).all():
        raise ValueError(f"Probability tensor has NaN/Inf: {path}")
    if aligned.min() < -1e-4 or aligned.max() > 1.0001:
        shifted = aligned - aligned.max(axis=0, keepdims=True)
        aligned = np.exp(shifted)
        aligned /= aligned.sum(axis=0, keepdims=True).clip(1e-8)
    else:
        denom = aligned.sum(axis=0, keepdims=True)
        if float(np.nanmedian(denom)) > 0.5:
            aligned = aligned / denom.clip(1e-8)

    p_left = np.clip(aligned[left_channel], 0.0, 1.0)
    p_right = np.clip(aligned[right_channel], 0.0, 1.0)
    entropy = -(aligned.clip(1e-7, 1.0) * np.log(aligned.clip(1e-7, 1.0))).sum(axis=0)
    entropy /= max(math.log(max(aligned.shape[0], 2)), 1e-8)
    return p_left.astype(np.float32), p_right.astype(np.float32), entropy.astype(np.float32)


def load_feature_channels(path: str | Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Load optional aligned decoder feature volumes as [C,X,Y,Z]."""
    path = Path(path)
    if path.suffix == ".npz":
        array = _extract_npz_array(path)
    elif path.name.endswith(".nii.gz") or path.suffix == ".nii":
        array = np.asanyarray(nib.load(str(path)).dataobj)
    else:
        raise ValueError(f"Unsupported feature file: {path}")
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(f"Feature tensor must be 3-D or 4-D, got {array.shape}")
    first_matches = sorted(array.shape[1:]) == sorted(target_shape)
    last_matches = sorted(array.shape[:3]) == sorted(target_shape)
    if first_matches and not last_matches:
        channels_first = array
    elif last_matches and not first_matches:
        channels_first = np.moveaxis(array, -1, 0)
    elif first_matches and last_matches:
        channels_first = array if array.shape[0] <= array.shape[-1] else np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"Cannot align feature tensor {array.shape} to image {target_shape}")
    return np.stack(
        [_align_spatial(np.asarray(channel), target_shape) for channel in channels_first], axis=0
    ).astype(np.float32, copy=False)


def robust_image_normalize(image: np.ndarray, seed: int = 0) -> np.ndarray:
    """Robust per-volume normalization suitable for scanner-variable CBCT."""
    flat = image.reshape(-1)
    if flat.size > 500_000:
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(flat.size, 500_000, replace=False)]
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("CBCT volume has no finite voxels")
    lo, hi = np.percentile(finite, (0.5, 99.5))
    clipped = np.clip(image, lo, hi).astype(np.float32)
    mean = float(clipped.mean())
    std = float(clipped.std())
    return np.clip((clipped - mean) / max(std, 1e-6), -5.0, 5.0).astype(np.float32)


def format_case_pattern(pattern: str, *, root: str, fold: int, case: str, side: str = "") -> Path:
    return Path(pattern.format(root=root, fold=fold, case=case, side=side)).expanduser().resolve()


def first_existing(paths: Iterable[Path]) -> Path:
    candidates = list(paths)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate paths exists: " + ", ".join(map(str, candidates)))
