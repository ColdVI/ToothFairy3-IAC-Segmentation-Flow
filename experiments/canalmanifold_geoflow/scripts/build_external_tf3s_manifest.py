#!/usr/bin/env python3
"""Build a locked manifest for the 52 ToothFairy3S cases.

The S cohort never enters Flow fitting, checkpoint selection, threshold
calibration, or trust-region estimation.  S_0040 is retained in the data but
marked as historically inspected so reports can use the remaining 51 cases as
the locked primary subset and all 52 as a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from canalmanifold.io import resolve_iac_label_ids


def case_id_from_image(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(path)
    return path.name[: -len(suffix)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--probability-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--expected", type=int, default=52)
    parser.add_argument("--historically-inspected", default="ToothFairy3S_0040")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    probability_root = Path(args.probability_root).expanduser().resolve()
    images = sorted((raw_root / "imagesTr").glob("ToothFairy3S_*_0000.nii.gz"))
    case_ids = [case_id_from_image(path) for path in images]
    if len(case_ids) != args.expected or len(set(case_ids)) != args.expected:
        raise RuntimeError(
            f"Expected {args.expected} unique ToothFairy3S cases, found {len(case_ids)}"
        )

    left_label, right_label = resolve_iac_label_ids(raw_root / "dataset.json")
    cases = []
    for case_id, image in zip(case_ids, images):
        label = raw_root / "labelsTr" / f"{case_id}.nii.gz"
        probability = probability_root / f"{case_id}.npz"
        if not label.exists():
            raise FileNotFoundError(label)
        if not probability.exists():
            raise FileNotFoundError(probability)
        cases.append(
            {
                "case_id": case_id,
                "oof_fold": 0,
                "image": str(image),
                "label": str(label),
                "probability": str(probability),
                "decoder_features": None,
                "historically_inspected": case_id == args.historically_inspected,
            }
        )

    splits = [{"train": [], "val": case_ids}]
    split_path = Path(args.splits).expanduser().resolve()
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "cohort": "ToothFairy3S_unused",
        "locked_primary_exclusion": args.historically_inspected,
        "dataset_root": str(raw_root),
        "dataset_json": str(raw_root / "dataset.json"),
        "splits_file": str(split_path),
        "left_label_id": int(left_label),
        "right_label_id": int(right_label),
        # The frozen refinement backbone is the 3-class {bg,L,R} model.
        "left_probability_channel": 1,
        "right_probability_channel": 2,
        "folds": splits,
        "cases": cases,
    }
    output = Path(args.manifest).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(output),
                "cases": len(cases),
                "left_label_id": left_label,
                "right_label_id": right_label,
                "locked_primary_cases": len(cases)
                - int(args.historically_inspected in case_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
