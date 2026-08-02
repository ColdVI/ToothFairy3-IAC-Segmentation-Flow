#!/usr/bin/env python3
"""
train.py — residual flow training with leakage-free data and real validation.

    python flow/train.py --config configs/flow.yaml --fold 0

Reads a fold from configs/splits.json, trains the residual velocity field on
OOF-derived coarse priors, and every `val_every` epochs runs validate() to
write three distinct checkpoints: last.pt for resume, best_any.pt for debugging,
and best_safe.pt only after a predeclared Dice non-inferiority gate. Training
loss is logged for monitoring only; it never drives checkpoint selection.
"""
import argparse
import atexit
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import ResidualVelocityUNet3D                 # noqa: E402
from losses import compute_prompt3r_training_loss, total_loss  # noqa: E402
from datasets import IACFlowDataset                      # noqa: E402
from validate import validate                            # noqa: E402
from scripts.run_manifest import (default_run_dir, finish_manifest,             # noqa: E402
                                  start_manifest)
from channel_contract import (resolve_conditioning_spec,                        # noqa: E402
                              validate_checkpoint_contract)


PRIOR_METRICS = ("dice", "cldice", "hd95", "score")


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
    preferred = ["epoch", "trainloss", "fm_random_t", "t0_sdf",
                 "t0_narrowband", "t0_softdice", "t0_cldice",
                 "fm", "narrowband", "cldice_loss",
                 "laterality", "tv", "total", "dice", "cldice", "hd95",
                 "gap_mm", "betti0", "safe_eligible", "score"]
    present = {key for row in history for key in row}
    keys = [key for key in preferred if key in present]
    keys.extend(sorted(present.difference(keys)))
    progress_path = os.path.join(out_dir, "progress.csv")
    partial_path = progress_path + ".partial"
    with open(partial_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(history)
    os.replace(partial_path, progress_path)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return                              # plotting is optional; CSV always written
    ep = [h["epoch"] for h in history]
    fig, (ax1, ax_loss) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(ep, [h["dice"] for h in history], "-o", ms=3, color="#3f8cf2", label="val Dice")
    ax1.plot(ep, [h["cldice"] for h in history], "-o", ms=3, color="#189f6f", label="val clDice")
    ax1.plot(ep, [h["score"] for h in history], "-o", ms=3, color="#ed4c54",
             label="score (0.5·Dice+0.5·clDice)")
    ax1.set_ylabel("Dice / clDice / score")
    ax1.set_ylim(0, 1); ax1.grid(alpha=.2)
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["hd95"] for h in history], ":", color="#cf8a25", label="val HD95 (mm)")
    ax2.set_ylabel("HD95 (mm)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=.9)
    loss_styles = {
        "fm": ("#0072B2", "FM"),
        "narrowband": ("#E69F00", "narrow-band"),
        "cldice_loss": ("#009E73", "soft-clDice"),
        "laterality": ("#CC79A7", "laterality"),
        "tv": ("#56B4E9", "TV"),
        "total": ("#000000", "total"),
        "fm_random_t": ("#0072B2", "FM random-t"),
        "t0_sdf": ("#D55E00", "t0 SDF"),
        "t0_narrowband": ("#E69F00", "t0 narrow-band"),
        "t0_softdice": ("#009E73", "t0 soft-Dice"),
        "t0_cldice": ("#CC79A7", "t0 soft-clDice"),
    }
    for key, (colour, label) in loss_styles.items():
        if any(key in h for h in history):
            ax_loss.plot(ep, [h.get(key, np.nan) for h in history], "-o", ms=2,
                         color=colour, label=label)
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("training loss")
    ax_loss.grid(alpha=.2); ax_loss.legend(loc="best", fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "progress.png"), dpi=120)
    plt.close(fig)


def require_prior_floor(cfg):
    """Return a validated identity-prior floor, refusing unsafe training configs."""
    floor = cfg.get("prior_floor")
    if not isinstance(floor, dict):
        raise ValueError("config prior_floor is missing; run identity_baseline.py --write-config")
    if floor.get("complete_cv") is not True:
        raise ValueError("config prior_floor is partial; complete_cv must be true")
    missing = [key for key in PRIOR_METRICS
               if not isinstance(floor.get(key), (int, float))]
    if missing:
        raise ValueError("config prior_floor is incomplete "
                         f"({', '.join(missing)}); run full-CV identity_baseline.py --write-config")
    if not (0 <= floor["dice"] <= 1 and 0 <= floor["cldice"] <= 1
            and 0 <= floor["score"] <= 1 and floor["hd95"] >= 0):
        raise ValueError(f"invalid prior_floor values: {floor}")
    return {key: float(floor[key]) for key in PRIOR_METRICS}


def require_noninferiority_margin(cfg):
    margin = cfg.get("noninferiority_margin")
    if not isinstance(margin, (int, float)):
        raise ValueError("config noninferiority_margin is null; it requires an explicit "
                         "user decision after full-CV identity and before any B-run")
    if not 0 <= margin <= 1:
        raise ValueError(f"invalid noninferiority_margin: {margin}")
    return float(margin)


def is_safe(metrics, prior_floor, margin, tolerance=1e-8):
    dice_value = float(metrics["dice"])
    return math.isfinite(dice_value) and dice_value + tolerance >= prior_floor["dice"] - margin


def _finite(value, fallback):
    value = float(value)
    return value if math.isfinite(value) else fallback


def _selection_key(metrics, epoch, prior_floor, margin):
    """Higher tuple is better; Dice is an eligibility gate, not a weighted score."""
    required = ("dice", "gap_mm", "betti0", "hd95", "cldice")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"validation metrics missing selection fields: {missing}")
    return (int(is_safe(metrics, prior_floor, margin)),
            -_finite(metrics["gap_mm"], math.inf),
            -_finite(metrics["betti0"], math.inf),
            -_finite(metrics["hd95"], math.inf),
            _finite(metrics["cldice"], -math.inf), -int(epoch))


def checkpoint_is_better(metrics, epoch, best, prior_floor, margin, require_safe=False):
    """Lexicographic selection shared by best-any and best-safe checkpoints."""
    if require_safe and not is_safe(metrics, prior_floor, margin):
        return False
    if best is None:
        return True
    return _selection_key(metrics, epoch, prior_floor, margin) > _selection_key(
        best, best["epoch"], prior_floor, margin)


def update_checkpoint_selection(metrics, epoch, best_any, best_safe, prior_floor, margin):
    record = {**metrics, "epoch": int(epoch),
              "safe_eligible": is_safe(metrics, prior_floor, margin)}
    write_any = checkpoint_is_better(metrics, epoch, best_any, prior_floor, margin)
    write_safe = checkpoint_is_better(metrics, epoch, best_safe, prior_floor, margin,
                                      require_safe=True)
    return (record if write_any else best_any,
            record if write_safe else best_safe,
            write_any, write_safe)


def atomic_torch_save(payload, path):
    """Keep the previous checkpoint valid if a session dies during serialization."""
    partial = path + ".partial"
    torch.save(payload, partial)
    os.replace(partial, path)


def migrate_legacy_best(out_dir, map_location="cpu"):
    """Copy legacy best.pt into the new names once; keep the legacy file untouched."""
    legacy_path = os.path.join(out_dir, "best.pt")
    any_path = os.path.join(out_dir, "best_any.pt")
    safe_path = os.path.join(out_dir, "best_safe.pt")
    if not os.path.isfile(legacy_path) or os.path.isfile(any_path):
        return None, None
    checkpoint = torch.load(legacy_path, map_location=map_location, weights_only=False)
    metrics = checkpoint.get("val")
    if not isinstance(metrics, dict):
        raise ValueError(f"legacy checkpoint has no validation metrics: {legacy_path}")
    missing = [key for key in ("dice", "cldice", "hd95") if key not in metrics]
    if missing:
        raise ValueError(f"legacy validation metrics incomplete ({missing}): {legacy_path}")
    # Legacy checkpoints predate topology-aware selection. Preserve them for
    # debugging, but do not infer missing gap/Betti values or call them safe.
    epoch = int(checkpoint.get("epoch", metrics.get("epoch", -1)))
    record = {**metrics, "epoch": epoch, "safe_eligible": False,
              "legacy_missing_topology": True}
    migrated = {**checkpoint, "val": record, "migration_source": "best.pt"}
    atomic_torch_save(migrated, any_path)
    # Even if Dice happens to pass, best_safe is intentionally not created:
    # its topology rank and predeclared policy were absent in the legacy run.
    return record, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flow.yaml")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--images", required=True, help="Dataset801_IAC_LR/imagesTr")
    ap.add_argument("--labels", required=True, help="Dataset801_IAC_LR/labelsTr")
    ap.add_argument("--gt-sdf", required=True)
    ap.add_argument("--coarse-sdf", required=True)
    ap.add_argument("--out", default=None,
                    help="explicit legacy output directory (default: runs/<config_hash>)")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from outputs/.../last.pt if present (survives Colab disconnects)")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    conditioning_spec = resolve_conditioning_spec(cfg)
    prior_floor = require_prior_floor(cfg)
    noninferiority_margin = require_noninferiority_margin(cfg)
    seed = int(cfg.get("seed", 0) if a.seed is None else a.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    splits = json.load(open(a.splits))
    fold = splits["folds"][a.fold]
    train_ids, val_ids = fold["train"], fold["val"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.out is None:
        a.out = default_run_dir(a.runs_root, cfg, a.fold, seed)
    os.makedirs(a.out, exist_ok=True)
    start_manifest(a.out, cfg, a.fold, seed,
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
                   resume=a.resume, channel_contract=conditioning_spec.to_dict())
    manifest_finished = False

    def mark_interrupted():
        if not manifest_finished:
            finish_manifest(a.out, "interrupted")

    atexit.register(mark_interrupted)
    print(f"[train] fold {a.fold}: {len(train_ids)} train / {len(val_ids)} val  device={dev}")

    ds = IACFlowDataset(train_ids, a.images, a.gt_sdf, a.coarse_sdf,
                        patch=cfg.get("patch", 96), fg_prob=cfg.get("fg_prob", 0.8),
                        conditioning_spec=conditioning_spec)
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 2), shuffle=True,
                    num_workers=cfg.get("num_workers", 4), drop_last=True)

    model = ResidualVelocityUNet3D(
        cond_ch=conditioning_spec.conditioning_channels,
        state_ch=conditioning_spec.state_channels,
        base=cfg.get("base", 32),
        zero_init_head=cfg.get("zero_init_head", False)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4),
                            weight_decay=cfg.get("weight_decay", 1e-5))
    epochs = cfg.get("epochs", 500)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_any = None
    best_safe = None
    start_epoch = 0
    history = []
    last_path = os.path.join(a.out, "last.pt")
    best_any_path = os.path.join(a.out, "best_any.pt")
    best_safe_path = os.path.join(a.out, "best_safe.pt")
    prog_csv = os.path.join(a.out, "progress.csv")
    wall_start = time.monotonic()
    print(f"[train] identity prior floor: score={prior_floor['score']:.4f} "
          f"Dice={prior_floor['dice']:.4f} clDice={prior_floor['cldice']:.4f} "
          f"HD95={prior_floor['hd95']:.3f} mm")
    print(f"[train] Dice non-inferiority margin: {noninferiority_margin:.6f}")
    if a.resume and cfg.get("legacy_compatibility", False):
        migrated_any, _ = migrate_legacy_best(a.out, map_location=dev)
        if migrated_any is not None:
            print("[resume] preserved legacy best.pt as best_any.pt; it is not safe-eligible "
                  "because legacy topology ranks are unavailable")
    if a.resume and os.path.isfile(last_path):
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        validate_checkpoint_contract(
            ck, conditioning_spec,
            legacy_compatibility=cfg.get("legacy_compatibility", False))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_any = ck.get("best_any")
        best_safe = ck.get("best_safe")
        if "best" in ck and best_any is None:
            print("[resume] legacy last.pt best metadata ignored for new topology-aware ranking")
        if os.path.isfile(prog_csv):        # keep the curve continuous across disconnects
            import csv
            history = [{k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()}
                       for row in csv.DictReader(open(prog_csv))]
        best_label = "none" if best_any is None else f"epoch {best_any['epoch']}"
        print(f"[resume] continuing from epoch {start_epoch} (best_any {best_label})")

    for ep in range(start_epoch, epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        prompt3r_objective = cfg.get("training_objective") == "prompt3r_t0"
        component_keys = (("fm_random_t", "t0_sdf", "t0_narrowband",
                           "t0_softdice", "t0_cldice", "total")
                          if prompt3r_objective else
                          ("fm", "narrowband", "cldice", "laterality", "tv", "total"))
        component_sums = {key: 0.0 for key in component_keys}
        for cond, x0, x1 in dl:
            cond, x0, x1 = cond.to(dev), x0.to(dev), x1.to(dev)
            if prompt3r_objective:
                loss, comp = compute_prompt3r_training_loss(
                    model, cond, x0, x1, cfg)
            else:
                # Legacy schedule remains available only outside the Prompt-3R objective.
                sigma = cfg.get("train_sigma", 0.1) * (
                    np.random.rand() < cfg.get("noise_frac", 0.5))
                x0n = x0 + sigma * torch.randn_like(x0)
                t = torch.rand(x1.shape[0], device=dev)
                tb = t.view(-1, 1, 1, 1, 1)
                xt = (1 - tb) * x0n + tb * x1
                pred_v = model(xt, t, cond)
                loss, comp = total_loss(pred_v, x0n, x1, t, cfg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); run += comp["total"]; nb += 1
            for key in component_sums:
                component_sums[key] += comp.get(key, 0.0)
        sched.step()

        checkpoint_due = (ep + 1) % cfg.get("checkpoint_every", 10) == 0 or ep == epochs - 1
        wall_elapsed = time.monotonic() - wall_start
        eta = wall_elapsed / max(ep + 1, 1) * (epochs - ep - 1)
        is_val = ep % cfg.get("val_every", 25) == 0 or ep == epochs - 1
        if not is_val:
            print(f"[ep {ep + 1:4d}/{epochs}] trainloss {run/max(1,nb):.4f} | "
                  f"epoch {time.time()-t0:.0f}s | wall {wall_elapsed / 60:.1f} min | "
                  f"ETA {eta / 60:.1f} min | {'checkpoint saved' if checkpoint_due else 'running'}",
                  flush=True)
        if is_val:
            m = validate(model, val_ids, a.images, a.coarse_sdf, a.labels,
                         patch=cfg.get("patch", 96), steps=cfg.get("ode_steps", 8),
                         device=dev, max_cases=cfg.get("val_max_cases", 20),
                         conditioning_spec=conditioning_spec)
            best_any, best_safe, write_any, write_safe = update_checkpoint_selection(
                m, ep, best_any, best_safe, prior_floor, noninferiority_margin)
            checkpoint = {"model": model.state_dict(), "cfg": cfg,
                          "val": {**m, "epoch": ep,
                                  "safe_eligible": is_safe(
                                      m, prior_floor, noninferiority_margin)},
                          "fold": a.fold, "seed": seed,
                          "channel_contract": conditioning_spec.to_dict()}
            if write_any:
                atomic_torch_save(checkpoint, best_any_path)
            if write_safe:
                atomic_torch_save(checkpoint, best_safe_path)
            flags = " ".join(flag for flag, enabled in
                             (("*BEST_ANY*", write_any), ("*BEST_SAFE*", write_safe))
                             if enabled)
            print(f"[ep {ep:4d}] trainloss {run/max(1,nb):.4f} | val Dice {m['dice']:.3f} "
                  f"clDice {m['cldice']:.3f} HD95 {m['hd95']:.2f} "
                  f"gap {m['gap_mm']:.2f} Betti0 {m['betti0']:.2f} "
                  f"safe={is_safe(m, prior_floor, noninferiority_margin)} {flags} | "
                  f"{ep + 1}/{epochs} "
                  f"({100 * (ep + 1) / epochs:.1f}%) | epoch {time.time()-t0:.0f}s | "
                  f"wall {wall_elapsed / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)
            component_means = {key: value / max(1, nb)
                               for key, value in component_sums.items()}
            logged_components = dict(component_means)
            if not prompt3r_objective:
                logged_components["cldice_loss"] = logged_components.pop("cldice")
            history.append({"epoch": ep, "trainloss": run / max(1, nb),
                            **logged_components,
                            "dice": m["dice"], "cldice": m["cldice"],
                            "hd95": m["hd95"], "gap_mm": m["gap_mm"],
                            "betti0": m["betti0"],
                            "safe_eligible": int(is_safe(
                                m, prior_floor, noninferiority_margin)),
                            "score": m["score"]})
            save_progress(a.out, history)   # progress.csv + progress.png every val step

        # Save after validation so the resume checkpoint carries the current
        # best-gate state rather than lagging one validation behind.
        if checkpoint_due:
            atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(),
                               "sched": sched.state_dict(), "epoch": ep,
                               "best_any": best_any, "best_safe": best_safe,
                               "cfg": cfg, "fold": a.fold, "seed": seed,
                               "channel_contract": conditioning_spec.to_dict()}, last_path)
    print("[train] done. best_any:", best_any, "->",
          best_any_path if os.path.isfile(best_any_path) else "none")
    print("[train] done. best_safe:", best_safe, "->",
          best_safe_path if os.path.isfile(best_safe_path) else "no checkpoint passed safety gate")
    finish_manifest(a.out, "completed", {
        "best_any": best_any, "best_safe": best_safe,
        "best_any_written": os.path.isfile(best_any_path),
        "best_safe_written": os.path.isfile(best_safe_path),
    })
    manifest_finished = True
    atexit.unregister(mark_interrupted)


if __name__ == "__main__":
    main()
