#!/usr/bin/env python3
"""Resumable Flow-v2 Prompt-3R smoke/pilot runner.

This is deliberately separate from the legacy 500-epoch entry point. It uses
the same dataset, model, loss, sliding-window inference and paired validator,
while enforcing the new epoch-0, panel, safety, and immutable-history contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow.channel_contract import (resolve_conditioning_spec,  # noqa: E402
                                   validate_checkpoint_contract)
from flow.checkpointing import (CheckpointConflictError, atomic_write_trajectory,
                                build_checkpoint_payload, cleanup_checkpoint_partials,
                                save_immutable_checkpoint, save_resume_checkpoint,
                                sha256_file, upsert_epoch_record)
from flow.datasets import IACFlowDataset  # noqa: E402
from flow.losses import compute_prompt3r_training_loss  # noqa: E402
from flow.model import ResidualVelocityUNet3D  # noqa: E402
from flow.prompt3r_config import resolve_prompt3r_config  # noqa: E402
from flow.validate import paired_identity_safety_gate, paired_validation_rows  # noqa: E402
from scripts.run_manifest import (atomic_write_json, config_hash, finish_manifest,
                                  start_manifest)  # noqa: E402


COMPONENTS = ("fm_random_t", "t0_sdf", "t0_narrowband", "t0_softdice",
              "t0_cldice", "total")


def _git_sha():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            capture_output=True, check=True)
    return result.stdout.strip()


def _atomic_yaml(path, payload):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(yaml.safe_dump(payload, sort_keys=True))
    os.replace(partial, path)


def _write_csv_atomic(path, rows):
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def _csv_value(value):
    return "" if value is None else str(value)


def append_paired_rows(path, new_rows):
    """Append with an exact (epoch,tier,case,side) uniqueness contract."""
    path = Path(path)
    old = list(csv.DictReader(path.open())) if path.exists() else []
    keys = ("epoch", "validation_tier", "case_id", "side")
    indexed = {tuple(row[key] for key in keys): row for row in old}
    if len(indexed) != len(old):
        raise CheckpointConflictError(f"duplicate paired validation key in {path}")
    combined = list(old)
    for row in new_rows:
        normalized = {key: _csv_value(value) for key, value in row.items()}
        key = tuple(normalized[name] for name in keys)
        if key in indexed:
            existing = indexed[key]
            union = set(existing) | set(normalized)
            if any(existing.get(name, "") != normalized.get(name, "") for name in union):
                raise CheckpointConflictError(f"conflicting paired validation row {key}")
            continue
        indexed[key] = normalized
        combined.append(normalized)
    _write_csv_atomic(path, combined)


def _gate_summary(gate):
    return {"safe": bool(gate["safe"]), "criteria": gate["criteria"],
            "aggregate": gate.get("aggregate"),
            "invalid_cases": gate.get("invalid_cases", []),
            "case_count": len(gate.get("case_rows", []))}


def _rank(gate, epoch):
    aggregate = gate.get("aggregate") or {}
    return (float(aggregate.get("mean_delta_dice", -np.inf)),
            float(aggregate.get("mean_delta_cldice", -np.inf)),
            -float(aggregate.get("mean_delta_hd95_mm", np.inf)),
            -float(aggregate.get("mean_delta_gap_mm", np.inf)), -int(epoch))


def _checkpoint_files(directory):
    files = []
    for path in Path(directory).glob("epoch_*.pt"):
        name = path.name
        if name == "epoch_000_untrained.pt":
            epoch = 0
        else:
            try:
                epoch = int(name.removeprefix("epoch_").removesuffix(".pt"))
            except ValueError:
                continue
        files.append((epoch, path))
    return sorted(files)


def _load_resume_state(out_dir, spec, run_id, resolved_hash, selection_policy):
    candidates = _checkpoint_files(out_dir)
    last = Path(out_dir) / "last.pt"
    if last.exists():
        candidates.append((int(torch.load(last, map_location="cpu", weights_only=False)["epoch"]), last))
    if not candidates:
        return None
    epoch, path = max(candidates, key=lambda item: item[0])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_contract(checkpoint, spec, legacy_compatibility=False)
    expected = {"run_id": run_id, "config_hash": resolved_hash,
                "selection_policy": selection_policy, "epoch": epoch}
    mismatch = {key: (checkpoint.get(key), value) for key, value in expected.items()
                if checkpoint.get(key) != value}
    if mismatch:
        raise CheckpointConflictError(f"resume state mismatch: {mismatch}")
    return checkpoint


def _plot_outputs(out_dir, trajectory, full_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    epochs = [row["epoch"] for row in trajectory]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    for field, label, colour in (
            ("quick_mean_delta_dice", "Δ Dice", "#0072B2"),
            ("quick_mean_delta_cldice", "Δ clDice", "#009E73")):
        axes[0].plot(epochs, [row.get(field) for row in trajectory], "-o", ms=3,
                     label=label, color=colour)
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set_ylabel("paired overlap delta")
    axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[1].plot(epochs, [row.get("quick_mean_delta_hd95_mm") for row in trajectory],
                 "-o", ms=3, color="#D55E00", label="Δ HD95")
    axes[1].plot(epochs, [row.get("quick_mean_delta_gap_mm") for row in trajectory],
                 "-o", ms=3, color="#CC79A7", label="Δ gap")
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set(xlabel="completed epoch", ylabel="paired physical delta (mm)")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "trajectory_metrics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    valid = [row for row in full_rows if str(row.get("metric_valid")) == "True"]
    for axis, field, label, colour in (
            (axes[0], "delta_volume_ratio", "Δ volume ratio", "#56B4E9"),
            (axes[1], "delta_radius_bias_mm", "Δ radius bias (mm)", "#E69F00")):
        groups = {}
        for row in valid:
            groups.setdefault(int(row["epoch"]), []).append(float(row[field]))
        axis.boxplot([groups[key] for key in sorted(groups)],
                     tick_labels=[str(key) for key in sorted(groups)], patch_artist=True,
                     boxprops={"facecolor": colour, "alpha": .7}) if groups else None
        axis.axhline(0, color="black", lw=.7)
        axis.set(xlabel="completed epoch", ylabel=label)
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "geometry_bias.pdf")
    plt.close(fig)


def run(args):
    resolved = resolve_prompt3r_config(args.config, args.identity_json)
    if args.smoke:
        resolved.update({"epochs": 2, "patch": 32, "batch_size": 1,
                         "num_workers": 0, "checkpoint_every": 1})
        resolved["pilot_validation"] = {**resolved["pilot_validation"],
                                         "full_epochs": [0, 2]}
    fold = int(args.fold)
    seed = int(resolved["seed"] if args.seed is None else args.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_checkpoint_partials(out_dir)
    spec = resolve_conditioning_spec(resolved)
    resolved_hash = config_hash(resolved, fold, seed)
    mode = "smoke" if args.smoke else "pilot"
    run_id = f"prompt3r-{mode}-fold{fold}-seed{seed}-{resolved_hash}"
    selection_policy = resolved["selection_policy"]
    if selection_policy != "paired_identity":
        raise ValueError("Prompt-3R runner requires selection_policy=paired_identity")

    resolved_path = out_dir / "resolved_config.yaml"
    if resolved_path.exists():
        if yaml.safe_load(resolved_path.read_text()) != resolved:
            raise CheckpointConflictError("resolved config changed in an existing run")
    else:
        _atomic_yaml(resolved_path, resolved)
    if (out_dir / "manifest.json").exists() and not args.resume:
        raise FileExistsError("existing run requires --resume")
    manifest = start_manifest(out_dir, resolved, fold, seed, ROOT, resume=args.resume,
                              channel_contract=spec.to_dict())
    manifest.update({"run_id": run_id, "selection_policy": selection_policy,
                     "identity_baseline": {"path": str(Path(args.identity_json).resolve()),
                                           "sha256": sha256_file(args.identity_json)},
                     "mode": mode})
    atomic_write_json(out_dir / "manifest.json", manifest)

    splits = json.loads(Path(args.splits).read_text())
    panel = json.loads(Path(args.panel_config).read_text())
    if panel["source_splits_sha256"] != sha256_file(args.splits):
        raise ValueError("pilot panel split hash mismatch")
    train_ids = list(splits["folds"][fold]["train"])
    quick_ids = list(panel["quick_case_ids"])
    full_ids = list(panel["full_case_ids"])
    if args.smoke:
        train_ids = train_ids[:2]
        quick_ids = [quick_ids[0], quick_ids[2]]
        full_ids = quick_ids
    atomic_write_json(out_dir / "pilot_case_ids.json",
                      {"fold": fold, "mode": mode, "train_case_count": len(train_ids),
                       "quick_case_ids": quick_ids, "full_case_ids": full_ids,
                       "panel_sha256": sha256_file(args.panel_config)})

    dataset = IACFlowDataset(
        train_ids, args.images, args.gt_sdf, args.coarse_sdf,
        patch=resolved["patch"], fg_prob=resolved.get("fg_prob", .8),
        conditioning_spec=spec, cfg=resolved)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=resolved["batch_size"], shuffle=True,
                        num_workers=resolved["num_workers"], drop_last=True,
                        generator=generator)
    model = ResidualVelocityUNet3D(
        cond_ch=spec.conditioning_channels, state_ch=spec.state_channels,
        base=resolved["base"], zero_init_head=resolved["zero_init_head"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=resolved["lr"],
                                  weight_decay=resolved["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, resolved["epochs"])

    trajectory_path = out_dir / "epoch_trajectory.json"
    trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else []
    resume = _load_resume_state(out_dir, spec, run_id, resolved_hash, selection_policy)
    completed_epoch = -1
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["opt"])
        scheduler.load_state_dict(resume["sched"])
        completed_epoch = int(resume["epoch"])
        if resume.get("trajectory_record"):
            reconstructed = dict(resume["trajectory_record"])
            immutable = out_dir / ("epoch_000_untrained.pt" if completed_epoch == 0
                                   else f"epoch_{completed_epoch:03d}.pt")
            reconstructed["checkpoint_sha256"] = sha256_file(immutable)
            trajectory, _ = upsert_epoch_record(trajectory, reconstructed)
            atomic_write_trajectory(out_dir, trajectory)

    quick_csv = out_dir / "paired_validation_quick_all_epochs.csv"
    full_csv = out_dir / "paired_validation_full.csv"
    full_epochs = set(int(value) for value in resolved["pilot_validation"]["full_epochs"])
    best_any = None
    best_safe = None
    if resume:
        best_any = resume.get("best_any")
        best_safe = resume.get("best_safe")

    def evaluate_epoch(epoch, training_summary):
        nonlocal trajectory, best_any, best_safe
        quick_rows = paired_validation_rows(
            model, quick_ids, args.images, args.coarse_sdf, args.labels,
            patch=resolved["patch"], steps=resolved["ode_steps"], device=device,
            progress=True, conditioning_spec=spec, epoch=epoch,
            validation_tier="quick")
        append_paired_rows(quick_csv, quick_rows)
        quick_gate = paired_identity_safety_gate(
            quick_rows, resolved["paired_safety_gate"])
        quick_summary = _gate_summary(quick_gate)
        run_full = epoch in full_epochs or quick_gate["safe"]
        full_summary = None
        if run_full:
            full_rows = paired_validation_rows(
                model, full_ids, args.images, args.coarse_sdf, args.labels,
                patch=resolved["patch"], steps=resolved["ode_steps"], device=device,
                progress=True, conditioning_spec=spec, epoch=epoch,
                validation_tier="full")
            append_paired_rows(full_csv, full_rows)
            full_gate = paired_identity_safety_gate(
                full_rows, resolved["paired_safety_gate"])
            full_summary = _gate_summary(full_gate)

        record = {"epoch": int(epoch), **training_summary,
                  "quick_safe": quick_summary["safe"], "full_evaluated": run_full,
                  "full_safe": bool(full_summary and full_summary["safe"])}
        for prefix, summary in (("quick", quick_summary), ("full", full_summary)):
            if summary and summary.get("aggregate"):
                record.update({f"{prefix}_{key}": value
                               for key, value in summary["aggregate"].items()})
        validation_summaries = {"quick": quick_summary, "full": full_summary}
        candidate = {"epoch": epoch, "rank": _rank(quick_gate, epoch),
                     "validation": quick_summary}
        if best_any is None or tuple(candidate["rank"]) > tuple(best_any["rank"]):
            best_any = candidate
        if full_summary and full_summary["safe"]:
            safe_candidate = {"epoch": epoch,
                              "rank": _rank(full_gate, epoch),
                              "validation": full_summary,
                              "selection_policy": selection_policy}
            if best_safe is None or tuple(safe_candidate["rank"]) > tuple(best_safe["rank"]):
                best_safe = safe_candidate
        checkpoint = build_checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            config=resolved, config_hash=resolved_hash,
            channel_contract=spec.to_dict(), selection_policy=selection_policy,
            validation_summaries=validation_summaries, training_summary=training_summary,
            trajectory_record=record, git_sha=_git_sha(), run_id=run_id,
            fold=fold, seed=seed)
        checkpoint.update({"best_any": best_any, "best_safe": best_safe})
        immutable = save_immutable_checkpoint(out_dir, checkpoint, spec)
        record["checkpoint_sha256"] = immutable["sha256"]
        trajectory, _ = upsert_epoch_record(trajectory, record)
        atomic_write_trajectory(out_dir, trajectory)
        shutil.copyfile(out_dir / "epoch_trajectory.csv", out_dir / "progress.csv.partial")
        os.replace(out_dir / "progress.csv.partial", out_dir / "progress.csv")
        save_resume_checkpoint(out_dir / "last.pt", checkpoint)
        if best_any and best_any["epoch"] == epoch:
            save_resume_checkpoint(out_dir / "best_any.pt", checkpoint)
        if best_safe and best_safe["epoch"] == epoch:
            save_resume_checkpoint(out_dir / "best_safe.pt", checkpoint)

    if completed_epoch < 0:
        evaluate_epoch(0, {"trainloss": None, **{key: None for key in COMPONENTS}})
        completed_epoch = 0
        if args.stop_after_epoch == 0:
            finish_manifest(out_dir, "controlled_interruption", {"completed_epoch": 0})
            return

    for epoch in range(completed_epoch + 1, int(resolved["epochs"]) + 1):
        model.train()
        sums = {key: 0.0 for key in COMPONENTS}
        batches = 0
        for cond, x0, x1 in loader:
            cond, x0, x1 = cond.to(device), x0.to(device), x1.to(device)
            loss, components = compute_prompt3r_training_loss(model, cond, x0, x1, resolved)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batches += 1
            for key in COMPONENTS:
                sums[key] += float(components[key])
        if batches == 0:
            raise RuntimeError("training produced zero batches")
        scheduler.step()
        means = {key: value / batches for key, value in sums.items()}
        if not all(np.isfinite(value) for value in means.values()):
            raise RuntimeError(f"non-finite training loss at epoch {epoch}: {means}")
        evaluate_epoch(epoch, {"trainloss": means["total"], **means})
        if args.stop_after_epoch == epoch:
            finish_manifest(out_dir, "controlled_interruption", {"completed_epoch": epoch})
            return

    quick_rows = list(csv.DictReader(quick_csv.open()))
    full_rows = list(csv.DictReader(full_csv.open())) if full_csv.exists() else []
    _plot_outputs(out_dir, trajectory, full_rows)
    safe_epochs = [row["epoch"] for row in trajectory if row.get("full_safe")]
    decision = ("PROMOTE_TO_REVISED_STAGE1_ABLATION" if safe_epochs
                else "STOP_FLOW_V2_AND_REOPEN_DIAGNOSIS")
    atomic_write_json(out_dir / "pilot_decision.json",
                      {"decision": decision, "selection_policy": selection_policy,
                       "full_safe_epochs": safe_epochs,
                       "best_safe": best_safe, "diagnostic_scope": "short Fold-0 pilot"})
    report = (
        "# Prompt 3R short Fold-0 trajectory pilot\n\n"
        f"Mode: `{mode}`  \nCompleted epochs: `{resolved['epochs']}`  \n"
        f"Selection policy: `{selection_policy}`  \nDecision: `{decision}`\n\n"
        "This short pilot is diagnostic and does not establish a final causal proof.\n"
    )
    (out_dir / "pilot_report.md.partial").write_text(report)
    os.replace(out_dir / "pilot_report.md.partial", out_dir / "pilot_report.md")
    finish_manifest(out_dir, "completed", {
        "decision": decision, "best_any": best_any, "best_safe": best_safe,
        "selection_policy": selection_policy, "full_safe_epochs": safe_epochs,
        "paired_quick_rows": len(quick_rows), "paired_full_rows": len(full_rows)})
    manifest_path = out_dir / "manifest.json"
    final_manifest = json.loads(manifest_path.read_text())
    artifact_names = (
        "pilot_case_ids.json", "resolved_config.yaml", "progress.csv",
        "epoch_trajectory.csv", "epoch_trajectory.json",
        "paired_validation_quick_all_epochs.csv", "paired_validation_full.csv",
        "pilot_report.md", "pilot_decision.json", "trajectory_metrics.pdf",
        "geometry_bias.pdf",
    )
    final_manifest["artifacts"] = {
        name: sha256_file(out_dir / name) for name in artifact_names
        if (out_dir / name).is_file()
    }
    final_manifest["immutable_checkpoints"] = {
        path.name: sha256_file(path) for _, path in _checkpoint_files(out_dir)
    }
    atomic_write_json(manifest_path, final_manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flow_prompt3r.yaml")
    parser.add_argument("--identity-json", required=True)
    parser.add_argument("--splits", default="configs/splits.json")
    parser.add_argument("--panel-config", default="configs/prompt3r_pilot_cases.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--gt-sdf", required=True)
    parser.add_argument("--coarse-sdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
