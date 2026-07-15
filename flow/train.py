#!/usr/bin/env python3
"""
train.py — residual flow training with leakage-free data and real validation.

    python flow/train.py --config configs/flow.yaml --fold 0

Reads a fold from configs/splits.json, trains the residual velocity field on
OOF-derived coarse priors, and every `val_every` epochs runs validate() to
select best.pt by S_val = 0.5*Dice + 0.5*clDice. Training loss is logged for
monitoring only; it never drives checkpoint selection.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import ResidualVelocityUNet3D, COND_CH        # noqa: E402
from losses import total_loss                            # noqa: E402
from datasets import IACFlowDataset                      # noqa: E402
from validate import validate                            # noqa: E402


def load_yaml(path):
    try:
        import yaml
        return yaml.safe_load(open(path))
    except Exception:
        return json.load(open(path))       # allow a JSON config as fallback


def save_progress(out_dir, history):
    """Write progress.csv + progress.png (val metrics vs epoch), like nnU-Net's plot.

    History is the list of per-validation rows. The plot mirrors what we log to
    stdout so a disconnected Colab run still leaves a visual training curve.
    """
    import csv
    keys = ["epoch", "trainloss", "dice", "cldice", "hd95", "score"]
    with open(os.path.join(out_dir, "progress.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(history)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return                              # plotting is optional; CSV always written
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, [h["dice"] for h in history], "-o", ms=3, color="#3f8cf2", label="val Dice")
    ax1.plot(ep, [h["cldice"] for h in history], "-o", ms=3, color="#189f6f", label="val clDice")
    ax1.plot(ep, [h["score"] for h in history], "-o", ms=3, color="#ed4c54",
             label="score (0.5·Dice+0.5·clDice)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("Dice / clDice / score")
    ax1.set_ylim(0, 1); ax1.grid(alpha=.2)
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["trainloss"] for h in history], "--", color="#9aa0a6", label="train loss")
    ax2.plot(ep, [h["hd95"] for h in history], ":", color="#cf8a25", label="val HD95 (mm)")
    ax2.set_ylabel("train loss / HD95 (mm)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=.9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "progress.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flow.yaml")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--images", required=True, help="Dataset801_IAC_LR/imagesTr")
    ap.add_argument("--labels", required=True, help="Dataset801_IAC_LR/labelsTr")
    ap.add_argument("--gt-sdf", required=True)
    ap.add_argument("--coarse-sdf", required=True)
    ap.add_argument("--out", default="outputs/flow_fold0")
    ap.add_argument("--resume", action="store_true",
                    help="continue from outputs/.../last.pt if present (survives Colab disconnects)")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    splits = json.load(open(a.splits))
    fold = splits["folds"][a.fold]
    train_ids, val_ids = fold["train"], fold["val"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    print(f"[train] fold {a.fold}: {len(train_ids)} train / {len(val_ids)} val  device={dev}")

    ds = IACFlowDataset(train_ids, a.images, a.gt_sdf, a.coarse_sdf,
                        patch=cfg.get("patch", 96), fg_prob=cfg.get("fg_prob", 0.8))
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 2), shuffle=True,
                    num_workers=cfg.get("num_workers", 4), drop_last=True)

    model = ResidualVelocityUNet3D(cond_ch=COND_CH, base=cfg.get("base", 32)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4),
                            weight_decay=cfg.get("weight_decay", 1e-5))
    epochs = cfg.get("epochs", 500)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best = {"score": -1.0, "hd95": 1e9}
    start_epoch = 0
    history = []
    last_path = os.path.join(a.out, "last.pt")
    prog_csv = os.path.join(a.out, "progress.csv")
    wall_start = time.monotonic()
    if a.resume and os.path.isfile(last_path):
        ck = torch.load(last_path, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best = ck["best"]
        if os.path.isfile(prog_csv):        # keep the curve continuous across disconnects
            import csv
            history = [{k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()}
                       for row in csv.DictReader(open(prog_csv))]
        print(f"[resume] continuing from epoch {start_epoch} (best score {best['score']:.3f})")

    for ep in range(start_epoch, epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for cond, x0, x1 in dl:
            cond, x0, x1 = cond.to(dev), x0.to(dev), x1.to(dev)
            # noise schedule: some batches deterministic (sigma=0) so inference matches
            sigma = cfg.get("train_sigma", 0.1) * (np.random.rand() < cfg.get("noise_frac", 0.5))
            x0n = x0 + sigma * torch.randn_like(x0)
            t = torch.rand(x1.shape[0], device=dev)
            tb = t.view(-1, 1, 1, 1, 1)
            xt = (1 - tb) * x0n + tb * x1
            pred_v = model(xt, t, cond)
            loss, comp = total_loss(pred_v, x0n, x1, t, cfg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); run += comp["total"]; nb += 1
        sched.step()

        # resume checkpoint every epoch (cheap; optimizer+scheduler+epoch+best)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best, "cfg": cfg},
                   last_path)

        if ep % cfg.get("val_every", 25) == 0 or ep == epochs - 1:
            m = validate(model, val_ids, a.images, a.coarse_sdf, a.labels,
                         patch=cfg.get("patch", 96), steps=cfg.get("ode_steps", 8),
                         device=dev, max_cases=cfg.get("val_max_cases", 20))
            better = (m["score"] > best["score"] + 1e-5) or (
                abs(m["score"] - best["score"]) <= 1e-5 and m["hd95"] < best["hd95"])
            if better:
                best = m
                torch.save({"model": model.state_dict(), "cfg": cfg, "val": m},
                           os.path.join(a.out, "best.pt"))
            wall_elapsed = time.monotonic() - wall_start
            eta = wall_elapsed / max(ep + 1, 1) * (epochs - ep - 1)
            print(f"[ep {ep:4d}] trainloss {run/max(1,nb):.4f} | val Dice {m['dice']:.3f} "
                  f"clDice {m['cldice']:.3f} HD95 {m['hd95']:.2f} score {m['score']:.3f} "
                  f"{'*BEST*' if better else ''} | {ep + 1}/{epochs} "
                  f"({100 * (ep + 1) / epochs:.1f}%) | epoch {time.time()-t0:.0f}s | "
                  f"wall {wall_elapsed / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)
            history.append({"epoch": ep, "trainloss": run / max(1, nb),
                            "dice": m["dice"], "cldice": m["cldice"],
                            "hd95": m["hd95"], "score": m["score"]})
            save_progress(a.out, history)   # progress.csv + progress.png every val step
    print("[train] done. best:", best, "->", os.path.join(a.out, "best.pt"))


if __name__ == "__main__":
    main()
