#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_toothfairy3_iac.py
==========================
Eğitilmiş sol/sağ IAC modeliyle yeni CBCT hacimleri üzerinde inference + basit
metrik (DSC) hesabı. nnU-Net v2 CLI'sini sarmalar; mirroring TTA kapalı tutulur.

Kullanım:
    # Sadece tahmin:
    python predict_toothfairy3_iac.py --in_dir /path/imagesTs --out_dir /path/pred \
        --dataset 111 --config 3d_fullres --trainer nnUNetTrainerIAC_NoMirror --folds 0

    # Tahmin + GT ile DSC:
    python predict_toothfairy3_iac.py ... --gt_dir /path/labelsTs
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
import nibabel as nib


def run_predict(in_dir, out_dir, dataset, config, trainer, folds):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_dir), "-o", str(out_dir),
        "-d", str(dataset), "-c", config,
        "-tr", trainer,
        "-f", *[str(f) for f in folds],
        "--disable_tta",   # sol/sağ ayrımı için mirroring-TTA istemiyoruz
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def dice(pred, gt, label):
    p, g = (pred == label), (gt == label)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def eval_dsc(pred_dir, gt_dir):
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    rows = []
    for pf in sorted(pred_dir.glob("*.nii.gz")):
        gf = gt_dir / pf.name
        if not gf.exists():
            continue
        pred = np.asarray(nib.load(str(pf)).dataobj)
        gt = np.asarray(nib.load(str(gf)).dataobj)
        dl, dr = dice(pred, gt, 1), dice(pred, gt, 2)
        rows.append((pf.name, dl, dr))
        print(f"  {pf.name:35s}  L-IAC DSC={dl:.3f}  R-IAC DSC={dr:.3f}")
    if rows:
        L = np.mean([r[1] for r in rows]); R = np.mean([r[2] for r in rows])
        print(f"\n[mean] Left-IAC DSC={L:.4f}  Right-IAC DSC={R:.4f}  (n={len(rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", type=int, default=801)
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--trainer", default="nnUNetTrainerIAC_NoMirror")
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--gt_dir", default=None, help="verilirse DSC hesaplar")
    args = ap.parse_args()

    run_predict(args.in_dir, args.out_dir, args.dataset, args.config, args.trainer, args.folds)
    if args.gt_dir:
        print("\n[eval] DSC hesaplanıyor...")
        eval_dsc(args.out_dir, args.gt_dir)


if __name__ == "__main__":
    main()
