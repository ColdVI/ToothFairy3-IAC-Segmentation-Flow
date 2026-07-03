#!/usr/bin/env python3
"""
prepare_iac_dataset.py — build Dataset801_IAC_LR from ToothFairy3 (77 labels).

Output label schema (project convention, canonical):
    0 = background
    1 = Left  Inferior Alveolar Canal
    2 = Right Inferior Alveolar Canal

P0 correctness rules (v1.0 spec):
* IAC ids are resolved BY NAME from the source dataset.json
  ("left/right inferior alveolar canal"). If the names are not found we
  RAISE — the old silent fallback to ids 3/4 is gone, because a schema change
  would then quietly grab the wrong (incisive/lingual) canals.
* Every written case is validated: image/label shape, affine, spacing and
  orientation must match; orientation must be RPI; output labels must be a
  subset of {0,1,2}; both sides must have >0 voxels (else the case is dropped
  and reported, never silently kept half-empty).

Usage
-----
    python data/prepare_iac_dataset.py --src /path/ToothFairy3 \
        --dst $nnUNet_raw/Dataset801_IAC_LR [--copy-images] [--limit N] [--workers 8]
"""
import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import orientation_code, voxel_spacing, affines_match  # noqa: E402

CASE_RE = re.compile(r"(ToothFairy3[FPS]_\d+)")
L_NAME = "left inferior alveolar canal"
R_NAME = "right inferior alveolar canal"
REQUIRED_ORIENTATION = "RPI"


class LabelResolutionError(ValueError):
    pass


def resolve_iac_ids(src):
    """(left_id, right_id) resolved strictly by name. Raises if not found."""
    dj = os.path.join(src, "dataset.json")
    if not os.path.isfile(dj):
        raise LabelResolutionError(f"dataset.json not found at {dj}; cannot resolve IAC ids by name.")
    labels = json.load(open(dj)).get("labels", {})
    name2id = {str(k).strip().lower(): int(v) for k, v in labels.items()}
    if L_NAME not in name2id or R_NAME not in name2id:
        raise LabelResolutionError(
            "Could not find 'left/right inferior alveolar canal' in dataset.json labels.\n"
            f"Available label names: {sorted(name2id)}\n"
            "Refusing to fall back to hard-coded ids 3/4 — fix the source or the names."
        )
    return name2id[L_NAME], name2id[R_NAME]


def case_stem(fname):
    m = CASE_RE.search(os.path.basename(fname))
    return m.group(1) if m else None


def pair_cases(src):
    img_dir = os.path.join(src, "imagesTr")
    lab_dir = os.path.join(src, "labelsTr")
    imgs, labs = {}, {}
    for f in os.listdir(img_dir):
        s = case_stem(f)
        if s and f.endswith("_0000.nii.gz"):
            imgs[s] = os.path.join(img_dir, f)
    for f in os.listdir(lab_dir):
        s = case_stem(f)
        if s and f.endswith(".nii.gz"):
            labs[s] = os.path.join(lab_dir, f)
    return sorted(set(imgs) & set(labs)), imgs, labs


def convert_one(args):
    sid, img_path, lab_path, dst, left_id, right_id, copy_images = args
    lab_img = nib.load(lab_path)
    img_img = nib.load(img_path)
    ld = np.asanyarray(lab_img.dataobj)

    problems = []
    if ld.shape != img_img.shape:
        problems.append(f"shape img{img_img.shape} != lab{ld.shape}")
    if not affines_match(img_img.affine, lab_img.affine):
        problems.append("affine mismatch")
    if not np.allclose(voxel_spacing(img_img), voxel_spacing(lab_img), atol=1e-3):
        problems.append("spacing mismatch")
    ori = orientation_code(lab_img.affine)
    if ori != REQUIRED_ORIENTATION:
        problems.append(f"orientation {ori} != {REQUIRED_ORIENTATION}")
    if problems:
        return sid, -1, -1, "; ".join(problems)

    out = np.zeros(ld.shape, dtype=np.uint8)
    out[ld == left_id] = 1
    out[ld == right_id] = 2
    if not set(np.unique(out).tolist()).issubset({0, 1, 2}):
        return sid, -1, -1, f"output labels not subset of 0/1/2: {np.unique(out)}"

    n_l, n_r = int((out == 1).sum()), int((out == 2).sum())
    if n_l == 0 or n_r == 0:
        return sid, n_l, n_r, "missing-side"

    new_lab = nib.Nifti1Image(out, lab_img.affine, lab_img.header)
    new_lab.set_data_dtype(np.uint8)
    nib.save(new_lab, os.path.join(dst, "labelsTr", f"{sid}.nii.gz"))

    dst_img = os.path.join(dst, "imagesTr", f"{sid}_0000.nii.gz")
    if os.path.lexists(dst_img):
        os.remove(dst_img)
    if copy_images:
        shutil.copy2(img_path, dst_img)
    else:
        os.symlink(os.path.realpath(img_path), dst_img)
    return sid, n_l, n_r, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ToothFairy3 root (imagesTr/labelsTr/dataset.json)")
    ap.add_argument("--dst", required=True, help="output, e.g. $nnUNet_raw/Dataset801_IAC_LR")
    ap.add_argument("--copy-images", action="store_true", help="hard-copy images instead of symlinking")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()

    left_id, right_id = resolve_iac_ids(a.src)
    print(f"[ids] resolved by name -> Left={left_id}  Right={right_id}", flush=True)

    cases, imgs, labs = pair_cases(a.src)
    if a.limit:
        cases = cases[: a.limit]
    print(f"[cases] {len(cases)} paired image/label cases", flush=True)

    os.makedirs(os.path.join(a.dst, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(a.dst, "labelsTr"), exist_ok=True)

    jobs = [(sid, imgs[sid], labs[sid], a.dst, left_id, right_id, a.copy_images) for sid in cases]
    kept, dropped = 0, []

    def record(sid, n_l, n_r, status):
        nonlocal kept
        if status == "ok":
            kept += 1
        else:
            dropped.append((sid, n_l, n_r, status))

    use_pool = a.workers > 1
    if use_pool:
        try:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(convert_one, j): j[0] for j in jobs}
                for fut in as_completed(futs):
                    record(*fut.result())
        except (PermissionError, OSError) as e:
            print(f"[warn] pool unavailable ({e}); serial fallback", flush=True)
            use_pool = False
            kept, dropped = 0, []
    if not use_pool:
        for j in jobs:
            record(*convert_one(j))

    print(f"[done] kept {kept}; dropped {len(dropped)}", flush=True)
    for d in dropped[:30]:
        print("   dropped:", d)

    dataset_json = {
        "channel_names": {"0": "CBCT"},
        "labels": {"background": 0, "Left_IAC": 1, "Right_IAC": 2},
        "numTraining": kept,
        "file_ending": ".nii.gz",
        "name": "IAC_LR",
        "description": "Left/Right inferior alveolar canal reduced from ToothFairy3 (77-label).",
        "reference": "ToothFairy3 challenge, University of Modena (Bolelli et al.)",
    }
    with open(os.path.join(a.dst, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
    print("[dataset.json] written:", os.path.join(a.dst, "dataset.json"), flush=True)


if __name__ == "__main__":
    main()
