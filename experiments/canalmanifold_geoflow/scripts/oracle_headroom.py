#!/usr/bin/env python3
"""O0-O6 headroom decomposition before any further refinement training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize_scalar

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from canalmanifold.data import TubeCacheDataset
from canalmanifold.io import load_label_on_reference, load_probability_channels
from canalmanifold.manifest import load_splits
from canalmanifold.metrics import dice_score


class TemperatureHistogram:
    """Exact-count, bounded-logit sufficient statistics for scalar scaling."""

    def __init__(self, bins: int = 4096, lower: float = -16.0, upper: float = 16.0):
        self.edges = np.linspace(lower, upper, int(bins) + 1, dtype=np.float64)
        self.counts = np.zeros(int(bins), dtype=np.float64)
        self.positives = np.zeros(int(bins), dtype=np.float64)

    def add(self, probability: np.ndarray, target: np.ndarray) -> None:
        p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        logit = np.log(p) - np.log1p(-p)
        self.counts += np.histogram(logit, bins=self.edges)[0]
        self.positives += np.histogram(logit[np.asarray(target, dtype=bool)], bins=self.edges)[0]

    def fit(self) -> float:
        centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        negatives = self.counts - self.positives

        def nll(log_temperature: float) -> float:
            temperature = math.exp(float(log_temperature))
            scaled = centres / temperature
            # stable -log(sigmoid(z)) and -log(1-sigmoid(z))
            positive_loss = np.logaddexp(0.0, -scaled)
            negative_loss = np.logaddexp(0.0, scaled)
            return float(
                (self.positives * positive_loss + negatives * negative_loss).sum()
                / max(self.counts.sum(), 1.0)
            )

        result = minimize_scalar(
            nll,
            bounds=(math.log(0.05), math.log(20.0)),
            method="bounded",
            options={"xatol": 1e-5},
        )
        if not result.success:
            raise RuntimeError(f"Temperature scaling failed: {result.message}")
        return float(math.exp(result.x))


def calibrated_probability(probability: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logit = (np.log(p) - np.log1p(-p)) / float(temperature)
    return np.where(
        logit >= 0,
        1.0 / (1.0 + np.exp(-logit)),
        np.exp(logit) / (1.0 + np.exp(logit)),
    ).astype(np.float32)


def bayes_dice_mask(probability: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fixed-point expected-Dice decision using a calibrated probability map."""
    p = np.asarray(probability, dtype=np.float64)
    total_probability = float(p.sum())
    threshold = 0.5
    expected_dice = 0.0
    for _ in range(64):
        selected = p > threshold
        expected_dice = (
            2.0 * float(p[selected].sum())
            / max(float(selected.sum()) + total_probability, 1e-12)
        )
        updated = 0.5 * expected_dice
        if abs(updated - threshold) < 1e-7:
            threshold = updated
            break
        threshold = updated
    return (p > threshold), float(threshold), float(expected_dice)


def threshold_counts(
    probability: np.ndarray,
    target: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    edges = np.concatenate(([-np.inf], thresholds, [np.inf]))
    all_hist = np.histogram(probability, bins=edges)[0]
    positive_hist = np.histogram(probability[np.asarray(target, dtype=bool)], bins=edges)[0]
    predicted = all_hist[::-1].cumsum()[::-1][1:]
    true_positive = positive_hist[::-1].cumsum()[::-1][1:]
    return predicted, true_positive, int(np.asarray(target, dtype=bool).sum())


def crop_union(left: np.ndarray, right: np.ndarray, pad: int) -> tuple[slice, ...]:
    union = np.asarray(left, dtype=bool) | np.asarray(right, dtype=bool)
    coordinates = np.argwhere(union)
    if not len(coordinates):
        return tuple(slice(0, size) for size in union.shape)
    lower = np.maximum(coordinates.min(axis=0) - int(pad), 0)
    upper = np.minimum(coordinates.max(axis=0) + 1 + int(pad), union.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))


def morphology_candidates(
    base: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    offsets_mm: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    max_offset = float(np.max(np.abs(offsets_mm)))
    pad = math.ceil(max_offset / max(min(spacing), 1e-6)) + 3
    crop = crop_union(base, target, pad)
    base_crop = np.asarray(base[crop], dtype=bool)
    outside_distance = distance_transform_edt(~base_crop, sampling=spacing)
    inside_distance = distance_transform_edt(base_crop, sampling=spacing)
    candidates: list[np.ndarray] = []
    scores = []
    for offset in offsets_mm:
        if offset >= 0.0:
            candidate_crop = base_crop | (outside_distance <= float(offset))
        else:
            candidate_crop = base_crop & (inside_distance > abs(float(offset)))
        candidate = np.zeros_like(base, dtype=bool)
        candidate[crop] = candidate_crop
        candidates.append(candidate)
        scores.append(dice_score(candidate, target))
    return candidates, np.asarray(scores, dtype=np.float64)


def load_case(
    item: dict,
    manifest: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray]:
    import nibabel as nib

    image_proxy = nib.load(item["image"])
    shape = tuple(map(int, image_proxy.shape[:3]))
    affine = np.asarray(image_proxy.affine, dtype=np.float64)
    spacing = tuple(map(float, np.linalg.norm(affine[:3, :3], axis=0)))
    reference = SimpleNamespace(shape=shape, affine=affine)
    label = load_label_on_reference(item["label"], reference)
    left, right, _ = load_probability_channels(
        item["probability"],
        shape,
        int(manifest["left_probability_channel"]),
        int(manifest["right_probability_channel"]),
    )
    target_left = label == int(manifest["left_label_id"])
    target_right = label == int(manifest["right_label_id"])
    return left, right, target_left, target_right, spacing, affine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--offset-min-mm", type=float, default=-0.60)
    parser.add_argument("--offset-max-mm", type=float, default=0.60)
    parser.add_argument("--offset-step-mm", type=float, default=0.03)
    parser.add_argument("--limit-calibration-cases", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["paths"]["manifest_file"]).read_text(encoding="utf-8"))
    items = {item["case_id"]: item for item in manifest["cases"]}
    split = load_splits(config["paths"]["splits_file"])[args.fold]
    train_ids = list(split["train"])
    if args.limit_calibration_cases:
        train_ids = train_ids[: int(args.limit_calibration_cases)]
    val_ids = list(split["val"])
    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + 0.5 * args.threshold_step,
        args.threshold_step,
        dtype=np.float64,
    )
    offsets = np.arange(
        args.offset_min_mm,
        args.offset_max_mm + 0.5 * args.offset_step_mm,
        args.offset_step_mm,
        dtype=np.float64,
    )

    threshold_score_sum = np.zeros_like(thresholds)
    threshold_samples = 0
    temperature = {"L": TemperatureHistogram(), "R": TemperatureHistogram()}
    for case_index, case_id in enumerate(train_ids):
        left, right, target_left, target_right, _, _ = load_case(items[case_id], manifest)
        for side, probability, target in (
            ("L", left, target_left),
            ("R", right, target_right),
        ):
            predicted, true_positive, target_size = threshold_counts(
                probability, target, thresholds
            )
            threshold_score_sum += (
                2.0 * true_positive / np.maximum(predicted + target_size, 1)
            )
            threshold_samples += 1
            temperature[side].add(probability, target)
        print(
            f"[ORACLE CAL] {case_index + 1}/{len(train_ids)} {case_id}",
            flush=True,
        )
    locked_threshold = float(thresholds[int(np.argmax(threshold_score_sum))])
    temperatures = {side: accumulator.fit() for side, accumulator in temperature.items()}

    val_dataset = TubeCacheDataset(
        config["paths"]["cache_dir"],
        val_ids,
        exclude_fallback=False,
        cache_in_ram=False,
    )
    cache_by_key = {
        (str(val_dataset[index]["case_id"]), str(val_dataset[index]["side"])):
        val_dataset[index]["path"]
        for index in range(len(val_dataset))
    }
    rows: list[dict[str, object]] = []
    for case_index, case_id in enumerate(val_ids):
        left, right, target_left, target_right, spacing, _ = load_case(
            items[case_id], manifest
        )
        probabilities = {"L": left, "R": right}
        targets = {"L": target_left, "R": target_right}
        o0_masks = {
            side: probability >= locked_threshold
            for side, probability in probabilities.items()
        }

        # O1: one patient-specific threshold shared by L and R.
        patient_threshold_score = np.zeros_like(thresholds)
        side_threshold_scores: dict[str, np.ndarray] = {}
        for side in ("L", "R"):
            predicted, tp, target_size = threshold_counts(
                probabilities[side], targets[side], thresholds
            )
            scores = 2.0 * tp / np.maximum(predicted + target_size, 1)
            side_threshold_scores[side] = scores
            patient_threshold_score += 0.5 * scores
        o1_index = int(np.argmax(patient_threshold_score))
        o1_threshold = float(thresholds[o1_index])

        morphology: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
        for side in ("L", "R"):
            morphology[side] = morphology_candidates(
                o0_masks[side], targets[side], spacing, offsets
            )
        patient_offset_score = 0.5 * (
            morphology["L"][1] + morphology["R"][1]
        )
        o2_index = int(np.argmax(patient_offset_score))
        o2_offset = float(offsets[o2_index])

        for side in ("L", "R"):
            target = targets[side]
            probability = probabilities[side]
            calibrated = calibrated_probability(probability, temperatures[side])
            o4_mask, bayes_threshold, expected_dice = bayes_dice_mask(calibrated)
            o3_index = int(np.argmax(morphology[side][1]))
            with np.load(cache_by_key[(case_id, side)], allow_pickle=False) as archive:
                o5 = float(archive["q1_ceiling_dice"])
                o6 = float(archive["q1_free_ceiling_dice"])
            rows.append(
                {
                    "case_id": case_id,
                    "side": side,
                    "O0_locked_threshold": locked_threshold,
                    "O0_global_threshold_dice": dice_score(o0_masks[side], target),
                    "O1_patient_threshold": o1_threshold,
                    "O1_patient_threshold_dice": float(side_threshold_scores[side][o1_index]),
                    "O2_patient_offset_mm": o2_offset,
                    "O2_patient_offset_dice": float(morphology[side][1][o2_index]),
                    "O3_side_offset_mm": float(offsets[o3_index]),
                    "O3_side_offset_dice": float(morphology[side][1][o3_index]),
                    "O4_temperature": temperatures[side],
                    "O4_bayes_threshold": bayes_threshold,
                    "O4_expected_dice": expected_dice,
                    "O4_probability_bayes_actual_dice": dice_score(o4_mask, target),
                    "O5_m8_tube_ceiling_dice": o5,
                    "O6_free_surface_ceiling_dice": o6,
                }
            )
        print(f"[ORACLE VAL] {case_index + 1}/{len(val_ids)} {case_id}", flush=True)

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "oracle_metrics.csv", index=False)
    metric_columns = [
        "O0_global_threshold_dice",
        "O1_patient_threshold_dice",
        "O2_patient_offset_dice",
        "O3_side_offset_dice",
        "O4_probability_bayes_actual_dice",
        "O4_expected_dice",
        "O5_m8_tube_ceiling_dice",
        "O6_free_surface_ceiling_dice",
    ]
    means = {column: float(frame[column].mean()) for column in metric_columns}
    o4_headroom = (
        means["O4_probability_bayes_actual_dice"]
        - means["O0_global_threshold_dice"]
    )
    o2_headroom = (
        means["O2_patient_offset_dice"] - means["O0_global_threshold_dice"]
    )
    if o4_headroom <= 0.02:
        decision = "IN_DOMAIN_HEADROOM_LOW: do not spend Set C; prioritize calibrated safety-envelope/OOD study"
    elif o2_headroom >= 0.005:
        decision = "HEADROOM_AND_GLOBAL_OFFSET_SIGNAL: run R1 m8 and R2 surface_h with locked cross-fold epoch"
    else:
        decision = "HEADROOM_IS_LOCAL: prioritize R2 surface_h and dense-SDF comparator over global calibre g"
    payload = {
        "fold": args.fold,
        "calibration_cases": len(train_ids),
        "validation_cases": len(val_ids),
        "locked_global_threshold": locked_threshold,
        "temperature_L": temperatures["L"],
        "temperature_R": temperatures["R"],
        "means": means,
        "O4_minus_O0_actual_dice": float(o4_headroom),
        "O2_minus_O0_dice": float(o2_headroom),
        "decision": decision,
        "interpretation_guard": (
            "O4 is the Bayes decision supported by the calibrated frozen OOF "
            "probability map, not a universal upper bound on all information in CBCT. "
            "O1-O3 are optimistic GT oracles and are never reportable test-time methods."
        ),
    }
    (output / "oracle_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
