#!/usr/bin/env python3
"""
write_nnunet_splits.py — install configs/splits.json as nnU-Net's own
splits_final.json, so nnU-Net's automatic random 5-fold CV is REPLACED by the
project's stratified P/F split.

Why this matters: nnUNetv2_train does not know about configs/splits.json. Left
alone, on first training run it auto-generates its OWN random splits_final.json
in nnUNet_preprocessed/<dataset>/. predict_oof.py and flow/train.py both read
configs/splits.json's fold val lists and assume "the model trained for fold f
never saw its val cases" — but that is only true if nnU-Net actually trained
fold f on configs/splits.json's fold f, not its own random split. Without this
step, some "held-out" OOF cases were in fact in that model's training set
(leakage), silently invalidating the flow prior.

Must run AFTER `nnUNetv2_plan_and_preprocess` (so the preprocessed dataset
folder exists) and BEFORE any `nnUNetv2_train` call for this dataset.

    python nnunet/write_nnunet_splits.py --dataset 801 --splits configs/splits.json
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="801", help="nnU-Net dataset id (e.g. 801) or full name")
    ap.add_argument("--dataset-name", default="Dataset801_IAC_LR")
    ap.add_argument("--splits", default="configs/splits.json")
    a = ap.parse_args()

    preprocessed = os.environ.get("nnUNet_preprocessed")
    if not preprocessed:
        raise SystemExit("nnUNet_preprocessed env var not set — source your nnU-Net env first.")
    out_dir = os.path.join(preprocessed, a.dataset_name)
    if not os.path.isdir(out_dir):
        raise SystemExit(f"{out_dir} does not exist — run nnUNetv2_plan_and_preprocess first.")

    splits = json.load(open(a.splits))
    nnunet_splits = [{"train": fold["train"], "val": fold["val"]} for fold in splits["folds"]]

    out_path = os.path.join(out_dir, "splits_final.json")
    if os.path.isfile(out_path):
        print(f"[splits] {out_path} exists — overwriting with {a.splits}'s P/F split "
              "(nnU-Net's own auto-generated split, if any, is discarded)")
    with open(out_path, "w") as f:
        json.dump(nnunet_splits, f, indent=2)

    print(f"[splits] wrote {len(nnunet_splits)} fold(s) -> {out_path}")
    for i, fo in enumerate(nnunet_splits):
        overlap = set(fo["train"]) & set(fo["val"])
        assert not overlap, f"fold {i}: train/val overlap {overlap}"
        print(f"  fold {i}: train={len(fo['train'])} val={len(fo['val'])}")


if __name__ == "__main__":
    main()
