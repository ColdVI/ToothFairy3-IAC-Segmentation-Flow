#!/usr/bin/env python3
"""
create_folds.py — development-set 5-fold CV split + external OOD test.

Split policy (v1.0 spec):
* Development set = P (same scanner) + F (same scanner, wider FOV) = 480 cases.
  These go into 5-fold cross-validation, STRATIFIED so every fold preserves the
  P/F ratio (F is small; naive random splitting would starve some folds of it).
* S (52 cases, different scanner) is a fully held-out external OOD test set. It
  never appears in any training/validation fold and is not touched until the
  final evaluation.

Writes a single splits.json:
    {
      "external_test": [ ...S ids... ],
      "folds": [ {"train": [...], "val": [...]}, ... x5 ]
    }
This file is the single source of truth consumed by both the nnU-Net OOF
training and the flow training, so there is exactly one definition of the split.
"""
import argparse
import json
import os
import re

CASE_RE = re.compile(r"(ToothFairy3([FPS])_\d+)")


def scan_ids(src):
    """Group case ids present in the source labelsTr by subset letter."""
    lab_dir = os.path.join(src, "labelsTr")
    groups = {"P": [], "F": [], "S": []}
    for f in sorted(os.listdir(lab_dir)):
        m = CASE_RE.search(f)
        if m and f.endswith(".nii.gz"):
            groups[m.group(2)].append(m.group(1))
    for k in groups:
        groups[k] = sorted(set(groups[k]))
    return groups


def stratified_folds(groups, k, seed):
    """
    k stratified folds over P+F. Within each subset the (shuffled) ids are dealt
    round-robin into folds, so fold sizes and the P/F ratio stay balanced.
    """
    import random

    rng = random.Random(seed)
    assign = {i: [] for i in range(k)}
    for subset in ("P", "F"):
        ids = groups[subset][:]
        rng.shuffle(ids)
        for i, cid in enumerate(ids):
            assign[i % k].append(cid)

    folds = []
    all_dev = sorted(groups["P"] + groups["F"])
    for i in range(k):
        val = sorted(assign[i])
        train = sorted(set(all_dev) - set(val))
        folds.append({"train": train, "val": val})
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ToothFairy3 root (reads labelsTr for ids)")
    ap.add_argument("--out", default="configs/splits.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ids-file", default=None,
                     help="JSON list of case ids to restrict to (e.g. a FAST/smoke-test slice). "
                          "Pass the SAME file to prepare_iac_dataset.py --ids-file so the raw "
                          "nnU-Net dataset and this split describe exactly the same cases.")
    a = ap.parse_args()

    groups = scan_ids(a.src)
    if a.ids_file:
        wanted = set(json.load(open(a.ids_file)))
        for k in groups:
            groups[k] = [c for c in groups[k] if c in wanted]
    print(f"[ids] P={len(groups['P'])} F={len(groups['F'])} S={len(groups['S'])}")
    folds = stratified_folds(groups, a.folds, a.seed)

    for i, fo in enumerate(folds):
        n_f = sum(1 for c in fo["val"] if "3F_" in c)
        print(f"  fold {i}: train={len(fo['train'])} val={len(fo['val'])} (F in val={n_f})")

    out = {
        "seed": a.seed,
        "external_test": groups["S"],
        "development": sorted(groups["P"] + groups["F"]),
        "folds": folds,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[splits] written -> {a.out}  (external OOD test S={len(groups['S'])})")


if __name__ == "__main__":
    main()
