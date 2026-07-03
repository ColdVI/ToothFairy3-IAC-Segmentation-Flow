#!/usr/bin/env python3
"""
evaluate_external.py — final OOD evaluation on the held-out S set.

Thin wrapper over evaluate_cv that (a) asserts every scored case is in the
external_test list of configs/splits.json (so the S set is never mixed with dev
cases by accident) and (b) writes a separate report. Run this ONCE, at the end.

    python evaluation/evaluate_external.py --pred outputs/preds_S \
        --gt .../labelsTr --splits configs/splits.json --out outputs/external_metrics.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluation import evaluate_cv                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--out", default="outputs/external_metrics.csv")
    a = ap.parse_args()

    external = set(json.load(open(a.splits))["external_test"])
    preds = [f[:-len(".nii.gz")] for f in os.listdir(a.pred) if f.endswith(".nii.gz")]
    leaked = [p for p in preds if p not in external]
    if leaked:
        raise SystemExit(f"[external] REFUSING: {len(leaked)} predicted cases are not in the "
                         f"external S set (e.g. {leaked[:5]}). This is not an OOD-clean run.")
    print(f"[external] {len(preds)} S-set cases, all confirmed held-out. Scoring...")
    sys.argv = ["evaluate_cv", "--pred", a.pred, "--gt", a.gt, "--out", a.out]
    evaluate_cv.main()


if __name__ == "__main__":
    main()
