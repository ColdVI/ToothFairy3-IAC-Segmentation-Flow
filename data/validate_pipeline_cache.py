#!/usr/bin/env python3
"""Fail-fast validation for the persistent TrackB inputs."""
import argparse
import hashlib
import json
import os

import numpy as np


def ids_in(directory, suffix):
    return {name[:-len(suffix)] for name in os.listdir(directory) if name.endswith(suffix)}


def require_ids(label, directory, suffix, expected):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"{label} directory missing: {directory}")
    actual = ids_in(directory, suffix)
    missing = expected - actual
    if missing:
        raise ValueError(f"{label}: {len(missing)} cases missing; first={sorted(missing)[:5]}")
    print(f"[ok] {label}: {len(expected)} development cases")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--oof")
    ap.add_argument("--gt-sdf", required=True)
    ap.add_argument("--coarse-sdf", required=True)
    a = ap.parse_args()

    raw = open(a.splits, "rb").read()
    split = json.loads(raw)
    folds = split.get("folds", [])
    dev = set(split.get("development", []))
    if len(folds) != 5 or not dev:
        raise ValueError("splits must contain five folds and a non-empty development list")
    seen = set()
    for index, fold in enumerate(folds):
        train, val = set(fold["train"]), set(fold["val"])
        if train & val or train | val != dev:
            raise ValueError(f"fold {index} is not a disjoint train/val partition of development")
        if seen & val:
            raise ValueError(f"fold {index} repeats validation cases")
        seen |= val
    if seen != dev or dev & set(split.get("external_test", [])):
        raise ValueError("validation coverage or external-test isolation is invalid")
    print(f"[ok] splits: sha256={hashlib.sha256(raw).hexdigest()} dev={len(dev)}")

    require_ids("images", a.images, "_0000.nii.gz", dev)
    require_ids("labels", a.labels, ".nii.gz", dev)
    if a.oof:
        require_ids("OOF", a.oof, ".npz", dev)
    require_ids("GT SDF", a.gt_sdf, ".npz", dev)
    require_ids("coarse SDF", a.coarse_sdf, ".npz", dev)

    for label, directory, keys in [
        ("OOF", a.oof, {"prob_left", "prob_right"}),
        ("GT SDF", a.gt_sdf, {"sdf"}),
        ("coarse SDF", a.coarse_sdf, {"sdf", "prob_left", "prob_right"}),
    ]:
        if not directory:
            continue
        for sid in sorted(dev):
            try:
                with np.load(os.path.join(directory, sid + ".npz")) as item:
                    if not keys <= set(item.files):
                        raise ValueError(f"missing keys {keys - set(item.files)}")
            except Exception as exc:
                raise ValueError(f"invalid {label} cache for {sid}: {exc}") from exc
        print(f"[ok] {label}: all archives readable")


if __name__ == "__main__":
    main()
