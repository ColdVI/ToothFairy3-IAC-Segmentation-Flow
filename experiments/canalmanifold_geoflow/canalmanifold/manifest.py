"""Build a leakage-audited OOF manifest from nnU-Net splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import section
from .io import first_existing, format_case_pattern, resolve_iac_label_ids


def load_splits(path: str | Path) -> list[dict[str, list[str]]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if isinstance(raw, dict) and "folds" in raw:
        raw = raw["folds"]
    if isinstance(raw, dict):
        ordered = [raw[key] for key in sorted(raw, key=lambda value: int(str(value).replace("fold_", "")))]
        raw = ordered
    if not isinstance(raw, list):
        raise TypeError("splits_final.json must contain a list of fold mappings")
    output: list[dict[str, list[str]]] = []
    for index, fold in enumerate(raw):
        if not isinstance(fold, dict) or "train" not in fold or "val" not in fold:
            raise ValueError(f"Invalid fold {index}: expected train/val lists")
        output.append({"train": list(map(str, fold["train"])), "val": list(map(str, fold["val"]))})
    return output


def _case_image(dataset_root: Path, case_id: str) -> Path:
    return first_existing(
        (
            dataset_root / "imagesTr" / f"{case_id}_0000.nii.gz",
            dataset_root / "imagesTr" / f"{case_id}.nii.gz",
        )
    )


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    paths = section(config, "paths")
    manifest_options = section(config, "manifest")
    dataset_root = Path(paths["dataset_root"]).expanduser().resolve()
    splits_file = Path(paths["splits_file"]).expanduser().resolve()
    probability_root = str(Path(paths["oof_probability_root"]).expanduser().resolve())
    probability_pattern = paths.get(
        "oof_probability_pattern", "{root}/fold_{fold}/{case}.npz"
    )
    feature_root_value = paths.get("oof_feature_root")
    feature_pattern = paths.get("oof_feature_pattern", "{root}/fold_{fold}/{case}.npz")
    require_features = bool(manifest_options.get("require_decoder_features", False))

    dataset_json = dataset_root / "dataset.json"
    left_label, right_label = resolve_iac_label_ids(dataset_json)
    left_channel = int(manifest_options.get("probability_left_channel", left_label))
    right_channel = int(manifest_options.get("probability_right_channel", right_label))
    splits = load_splits(splits_file)

    val_owner: dict[str, int] = {}
    for fold_index, fold in enumerate(splits):
        overlap = set(fold["train"]) & set(fold["val"])
        if overlap:
            raise ValueError(f"Fold {fold_index} has train/val overlap: {sorted(overlap)[:5]}")
        for case_id in fold["val"]:
            if case_id in val_owner:
                raise ValueError(
                    f"Case {case_id} is validation in both fold {val_owner[case_id]} and {fold_index}"
                )
            val_owner[case_id] = fold_index

    development_cases = sorted(set().union(*(set(f["train"]) | set(f["val"]) for f in splits)))
    missing_oof = sorted(set(development_cases) - set(val_owner))
    if missing_oof:
        raise ValueError(
            "Every development case needs exactly one OOF owner fold. Missing: "
            + ", ".join(missing_oof[:10])
        )

    cases = []
    for case_id in development_cases:
        fold = val_owner[case_id]
        probability = format_case_pattern(
            probability_pattern, root=probability_root, fold=fold, case=case_id
        )
        if not probability.exists():
            raise FileNotFoundError(f"Missing fold-{fold} OOF probability for {case_id}: {probability}")
        feature_path = None
        if feature_root_value:
            feature_path = format_case_pattern(
                feature_pattern,
                root=str(Path(feature_root_value).expanduser().resolve()),
                fold=fold,
                case=case_id,
            )
            if not feature_path.exists():
                if require_features:
                    raise FileNotFoundError(f"Missing OOF decoder features for {case_id}: {feature_path}")
                feature_path = None
        elif require_features:
            raise ValueError("require_decoder_features=true but paths.oof_feature_root is unset")
        label = dataset_root / "labelsTr" / f"{case_id}.nii.gz"
        if not label.exists():
            raise FileNotFoundError(f"Missing label: {label}")
        cases.append(
            {
                "case_id": case_id,
                "oof_fold": fold,
                "image": str(_case_image(dataset_root, case_id)),
                "label": str(label),
                "probability": str(probability),
                "decoder_features": str(feature_path) if feature_path else None,
            }
        )

    return {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "dataset_json": str(dataset_json),
        "splits_file": str(splits_file),
        "left_label_id": left_label,
        "right_label_id": right_label,
        "left_probability_channel": left_channel,
        "right_probability_channel": right_channel,
        "folds": splits,
        "cases": cases,
    }


def write_manifest(config: dict[str, Any]) -> Path:
    manifest = build_manifest(config)
    output = Path(section(config, "paths")["manifest_file"]).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output

