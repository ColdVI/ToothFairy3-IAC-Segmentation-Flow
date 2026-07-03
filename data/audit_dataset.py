#!/usr/bin/env python3
"""
audit_dataset.py — per-case geometry + label audit table for ToothFairy3.

Runs on the SOURCE ToothFairy3 tree (IAC ids resolved by name) and writes a CSV
with one row per case. This is the P0 "prove the data is what we think it is"
step: catches shape/affine/orientation surprises, one-sided cases, and canals
that are already fragmented (more than one connected component per side) before
any training decision depends on them.

    python data/audit_dataset.py --src /path/ToothFairy3 --out outputs/tf3_audit.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import orientation_code, voxel_spacing            # noqa: E402
from prepare_iac_dataset import resolve_iac_ids, pair_cases     # noqa: E402


def n_components(mask):
    _, n = cc_label(mask)
    return int(n)


def audit_case(sid, img_path, lab_path, left_id, right_id):
    img = nib.load(img_path)
    lab = nib.load(lab_path)
    ld = np.asanyarray(lab.dataobj)
    left = ld == left_id
    right = ld == right_id
    subset = {"F": "F", "P": "P", "S": "S"}.get(sid.replace("ToothFairy3", "")[0], "?")
    return {
        "case_id": sid,
        "subset": subset,
        "shape": "x".join(map(str, ld.shape)),
        "spacing": "x".join(f"{s:.3f}" for s in voxel_spacing(lab)),
        "orientation": orientation_code(lab.affine),
        "shape_match": int(ld.shape == img.shape),
        "left_voxel_count": int(left.sum()),
        "right_voxel_count": int(right.sum()),
        "left_components": n_components(left),
        "right_components": n_components(right),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="outputs/tf3_audit.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    left_id, right_id = resolve_iac_ids(a.src)
    print(f"[ids] Left={left_id} Right={right_id}")
    cases, imgs, labs = pair_cases(a.src)
    if a.limit:
        cases = cases[: a.limit]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    for i, sid in enumerate(cases):
        rows.append(audit_case(sid, imgs[sid], labs[sid], left_id, right_id))
        if (i + 1) % 50 == 0:
            print(f"  audited {i + 1}/{len(cases)}", flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # quick summary of anything worth a human's attention
    bad_shape = [r["case_id"] for r in rows if not r["shape_match"]]
    one_sided = [r["case_id"] for r in rows if r["left_voxel_count"] == 0 or r["right_voxel_count"] == 0]
    frag = [r["case_id"] for r in rows if r["left_components"] > 1 or r["right_components"] > 1]
    non_rpi = [r["case_id"] for r in rows if r["orientation"] != "RPI"]
    print(f"[audit] {len(rows)} cases -> {a.out}")
    print(f"  shape mismatch : {len(bad_shape)}")
    print(f"  one-sided      : {len(one_sided)}")
    print(f"  non-RPI        : {len(non_rpi)}")
    print(f"  fragmented (>1 component per side): {len(frag)}  (informative, not fatal)")


if __name__ == "__main__":
    main()
