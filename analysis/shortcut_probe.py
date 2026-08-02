#!/usr/bin/env python3
"""Deterministic, fail-closed limited endpoint shortcut diagnostics.

Historical per-epoch checkpoints were not saved.  This diagnostic compares the
two immutable endpoints that actually exist: ``best.pt`` under the explicit
label ``best_legacy_unknown_epoch`` and ``last.pt`` after verifying its internal
epoch is 129.  It never trains, migrates, derives, or rewrites a checkpoint.
Outputs are staged locally, validated, and only then published with
no-overwrite atomic copies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "data", ROOT / "flow"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from data.io_utils import (normalize_coords, physical_coord_grid,  # noqa: E402
                           sdf_stack_to_mask, voxel_spacing)
from evaluation.geometry_metrics import (radius_profile_summary,  # noqa: E402
                                         signed_surface_distance_summary)
from evaluation.metrics import _surface_distances, dice, hd95  # noqa: E402
from flow.conditioning import build_conditioning  # noqa: E402
from flow.channel_contract import (resolve_conditioning_spec,                # noqa: E402
                                   validate_checkpoint_contract)
from flow.model import COND_CH, FLOW_STATE_CH, ResidualVelocityUNet3D  # noqa: E402
from flow.sliding_window import predict_volume  # noqa: E402


CHECKPOINT_LABELS = ("best_legacy_unknown_epoch", "epoch_129")
T_GRID = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
          0.60, 0.70, 0.80, 0.90, 0.95)
SEEDS = {"global": 20260802, "patch": 1701, "noise": 2903, "bootstrap": 4409}
OUTPUT_NAMES = (
    "shortcut_probe.csv",
    "shortcut_probe_summary.json",
    "limited_endpoint_diagnostic.pdf",
    "thickening_probe.csv",
    "shortcut_probe_manifest.json",
)
PROTOCOL_FLAGS = {
    "protocol_deviation": True,
    "exact_epoch_trajectory_available": False,
    "historical_per_epoch_checkpoints_were_not_saved": True,
}
CLAIM = ("Diagnostic-only endpoint observations; no training trajectory or paper-proof "
         "claim is supported by this limited protocol.")


class ReadinessError(RuntimeError):
    """A missing, ambiguous, or incompatible immutable input."""


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def git_state(repo=ROOT):
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    dirty_lines = run("status", "--porcelain").splitlines()
    return {"branch": run("branch", "--show-current"), "head": run("rev-parse", "HEAD"),
            "dirty": bool(dirty_lines), "dirty_files": dirty_lines}


def relative_delta(full, ablated, eps=1e-8):
    """Relative L2 output change with an explicit zero-reference flag."""
    full = torch.as_tensor(full).detach().float()
    ablated = torch.as_tensor(ablated).detach().float()
    if full.shape != ablated.shape:
        raise ValueError(f"shape mismatch: {tuple(full.shape)} != {tuple(ablated.shape)}")
    denominator = float(torch.linalg.vector_norm(full))
    numerator = float(torch.linalg.vector_norm(full - ablated))
    return {"value": numerator / max(denominator, eps),
            "valid": denominator > eps, "flag": "ok" if denominator > eps else "zero_full_norm",
            "numerator": numerator, "denominator": denominator}


def cosine_r2(prediction, target, eps=1e-8):
    """Cosine similarity and voxelwise R2 without silently dropping degeneracy."""
    pred = torch.as_tensor(prediction).detach().double().reshape(-1)
    true = torch.as_tensor(target).detach().double().reshape(-1)
    if pred.numel() != true.numel():
        raise ValueError("prediction and target lengths differ")
    pred_norm = float(torch.linalg.vector_norm(pred))
    true_norm = float(torch.linalg.vector_norm(true))
    cosine_valid = pred_norm > eps and true_norm > eps
    cosine = (float(torch.dot(pred, true) / (pred_norm * true_norm))
              if cosine_valid else None)
    residual = float(torch.sum((true - pred) ** 2))
    total = float(torch.sum((true - torch.mean(true)) ** 2))
    r2_valid = total > eps
    r2 = 1.0 - residual / total if r2_valid else None
    flags = []
    if pred_norm <= eps:
        flags.append("zero_prediction_norm")
    if true_norm <= eps:
        flags.append("zero_target_norm")
    if not r2_valid:
        flags.append("degenerate_r2_target")
    return {"cosine": cosine, "r2": r2, "cosine_valid": cosine_valid,
            "r2_valid": r2_valid, "flag": "ok" if not flags else ";".join(flags),
            "prediction_norm": pred_norm, "target_norm": true_norm}


def center_crop_pad(array, target_shape, fill=0):
    """Centre crop/pad spatial dimensions and preserve leading dimensions."""
    is_torch = torch.is_tensor(array)
    value = array.detach().cpu().numpy() if is_torch else np.asarray(array)
    target_shape = tuple(int(x) for x in target_shape)
    if value.ndim < len(target_shape):
        raise ValueError("target has more dimensions than input")
    lead = value.shape[:-len(target_shape)]
    output = np.full(lead + target_shape, fill, dtype=value.dtype)
    source_slices, target_slices = [], []
    for size, target in zip(value.shape[-len(target_shape):], target_shape):
        take = min(size, target)
        source_start = (size - take) // 2
        target_start = (target - take) // 2
        source_slices.append(slice(source_start, source_start + take))
        target_slices.append(slice(target_start, target_start + take))
    output[(..., *target_slices)] = value[(..., *source_slices)]
    return torch.as_tensor(output, device=array.device) if is_torch else output


def extract_patch(array, start, patch, fill=0):
    value = np.asarray(array)
    spatial = value.shape[-3:]
    out = np.full(value.shape[:-3] + (patch, patch, patch), fill, dtype=value.dtype)
    src, dst = [], []
    for origin, size in zip(start, spatial):
        lo, hi = max(0, origin), min(size, origin + patch)
        src.append(slice(lo, hi)); dst.append(slice(lo - origin, hi - origin))
    out[(..., *dst)] = value[(..., *src)]
    return out


def apply_cbct_intervention(cond, intervention, *, generator=None, donor_cond=None):
    """Change CBCT only; callers can assert the state/prior/t invariants."""
    result = cond.clone()
    if intervention == "zero":
        result[:, 0].zero_()
    elif intervention == "noise":
        source = cond[:, 0]
        noise = torch.randn(source.shape, generator=generator, device=source.device,
                            dtype=source.dtype)
        noise = (noise - noise.mean()) / (noise.std(unbiased=False) + 1e-8)
        result[:, 0] = noise * source.std(unbiased=False) + source.mean()
    elif intervention == "shuffle":
        if donor_cond is None:
            raise ValueError("shuffle requires donor_cond")
        result[:, 0] = center_crop_pad(donor_cond[:, 0], cond.shape[-3:]).to(cond.device)
    else:
        raise ValueError(f"unknown CBCT intervention: {intervention}")
    return result


def apply_prior_intervention(cond, intervention, *, donor_cond=None):
    """Change conditioning coarse-SDF channels 3:5 only."""
    result = cond.clone()
    if intervention == "zero":
        result[:, 3:5].zero_()
    elif intervention == "swap":
        if donor_cond is None:
            raise ValueError("swap requires donor_cond")
        result[:, 3:5] = center_crop_pad(donor_cond[:, 3:5], cond.shape[-3:]).to(cond.device)
    else:
        raise ValueError(f"unknown prior intervention: {intervention}")
    return result


def assert_intervention_invariants(original_cond, changed_cond, *, kind):
    changed = {index for index in range(original_cond.shape[1])
               if not torch.equal(original_cond[:, index], changed_cond[:, index])}
    allowed = {0} if kind == "cbct" else {3, 4}
    if not changed <= allowed:
        raise AssertionError(f"{kind} intervention changed forbidden channels: {changed - allowed}")


def case_level_bootstrap(rows, value_key="value", *, iterations=2000, seed=0):
    """Bootstrap case means; patches are never treated as independent samples."""
    grouped = defaultdict(list)
    excluded = defaultdict(int)
    for row in rows:
        value = row.get(value_key)
        if row.get("valid", True) and value is not None and np.isfinite(float(value)):
            grouped[str(row["case_id"])].append(float(value))
        else:
            excluded[row.get("flag", "invalid")] += 1
    case_means = {case: float(np.mean(values)) for case, values in grouped.items()}
    if not case_means:
        return {"mean": None, "ci95": [None, None], "n_cases": 0,
                "n_patches": 0, "excluded": dict(excluded), "bootstrap_unit": "case"}
    values = np.asarray(list(case_means.values()), np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            "n_cases": len(values), "n_patches": sum(len(v) for v in grouped.values()),
            "excluded": dict(excluded), "bootstrap_unit": "case"}


def _checkpoint_epoch(checkpoint):
    value = checkpoint.get("epoch")
    if value is None and isinstance(checkpoint.get("val"), dict):
        value = checkpoint["val"].get("epoch")
    return None if value is None else int(value)


def validate_checkpoint_epoch(path, expected_epoch):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actual = _checkpoint_epoch(checkpoint)
    if actual is None:
        raise ReadinessError(f"checkpoint has no internal epoch value: {path}")
    if actual != int(expected_epoch):
        raise ReadinessError(f"checkpoint epoch mismatch: expected {expected_epoch}, got {actual}: {path}")
    return checkpoint


def validate_checkpoint_label(path, checkpoint_label):
    """Validate the two historical endpoint identities without inventing epochs."""
    path = Path(path)
    expected_names = {
        "best_legacy_unknown_epoch": {"best.pt"},
        "epoch_129": {"last.pt", "latest.pt"},
    }
    if checkpoint_label not in expected_names:
        raise ReadinessError(f"unsupported limited-endpoint checkpoint label: {checkpoint_label}")
    if path.name not in expected_names[checkpoint_label]:
        raise ReadinessError(
            f"{checkpoint_label} must reference {sorted(expected_names[checkpoint_label])}, got {path.name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actual = _checkpoint_epoch(checkpoint)
    if checkpoint_label == "best_legacy_unknown_epoch":
        if actual is not None:
            raise ReadinessError(
                f"legacy best checkpoint unexpectedly has internal epoch {actual}: {path}")
    elif checkpoint_label == "epoch_129":
        if actual != 129:
            raise ReadinessError(
                f"checkpoint epoch mismatch: expected 129, got {actual}: {path}")
    return checkpoint


def _resolved_model_contract(checkpoint):
    state = checkpoint.get("model")
    if not isinstance(state, dict) or "stem.weight" not in state or "head.2.weight" not in state:
        raise ReadinessError("checkpoint model state is missing required tensors")
    cfg = checkpoint.get("cfg") or {}
    stem = state["stem.weight"]
    head = state["head.2.weight"]
    base, total_in = int(stem.shape[0]), int(stem.shape[1])
    state_ch = int(head.shape[0]); cond_ch = total_in - state_ch
    spec = resolve_conditioning_spec({"cond_include_coarse_sdf": cond_ch == COND_CH})
    if state_ch != spec.state_channels or cond_ch != spec.conditioning_channels:
        raise ReadinessError(f"channel contract mismatch: state={state_ch}, cond={cond_ch}")
    try:
        validate_checkpoint_contract(
            checkpoint, spec, legacy_compatibility="channel_contract" not in checkpoint)
    except ValueError as error:
        raise ReadinessError(str(error)) from error
    model = ResidualVelocityUNet3D(cond_ch=cond_ch, state_ch=state_ch, base=base)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ReadinessError(f"model state is incompatible: {error}") from error
    contract = {"base": base, "state_ch": state_ch, "cond_ch": cond_ch,
                "patch": int(cfg.get("patch", 96)), "ode_steps": int(cfg.get("ode_steps", 8)),
                "config": cfg, "config_hash": canonical_hash(cfg),
                "channel_contract": spec.to_dict()}
    return model, contract


def audit_checkpoint(path, checkpoint_label):
    path = Path(path).resolve()
    if not path.is_file():
        raise ReadinessError(f"checkpoint is missing: {path}")
    checkpoint = validate_checkpoint_label(path, checkpoint_label)
    model, contract = _resolved_model_contract(checkpoint)
    del model
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "checkpoint_label": checkpoint_label,
            "internal_epoch": _checkpoint_epoch(checkpoint),
            "contract": contract}


def audit_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ReadinessError(f"identity baseline is missing: {path}")
    payload = json.loads(path.read_text())
    required = {"evaluated_cases": 480, "complete_cv": True, "identity_complete": True,
                "fold_provenance_errors": 0, "direct_vs_sdf_voxel_difference": 0,
                "direct_vs_full_path_voxel_difference": 0}
    mismatches = {key: {"expected": expected, "actual": payload.get(key)}
                  for key, expected in required.items() if payload.get(key) != expected}
    if mismatches:
        raise ReadinessError(f"identity baseline is not complete/clean: {mismatches}")
    overall = payload.get("overall", {}).get("per_case", {}).get("direct", {})
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            "evaluated_cases": 480, "complete_cv": True,
            "dice": overall.get("dice"), "cldice": overall.get("cldice"),
            "hd95": overall.get("hd95")}


def audit_run_candidates(runs_root):
    root = Path(runs_root)
    candidates = []
    if not root.is_dir():
        return candidates
    for directory in sorted(p for p in root.rglob("*") if p.is_dir() and "flow_fold0" in p.name):
        files = {}
        for name in ("manifest.json", "config.yaml", "progress.csv",
                     "best.pt", "last.pt", "latest.pt"):
            path = directory / name
            if path.is_file():
                files[name] = {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
                if name in ("manifest.json", "config.yaml", "progress.csv"):
                    files[name]["sha256"] = sha256_file(path)
                if name == "progress.csv":
                    try:
                        progress = list(csv.DictReader(path.open()))
                        files[name]["epochs"] = [int(row["epoch"]) for row in progress]
                        files[name]["rows"] = len(progress)
                    except Exception as error:
                        files[name]["read_error"] = f"{type(error).__name__}: {error}"
                if name.endswith(".pt"):
                    try:
                        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                        _, contract = _resolved_model_contract(checkpoint)
                        files[name].update({"sha256": sha256_file(path),
                                            "internal_epoch": _checkpoint_epoch(checkpoint),
                                            "keys": sorted(checkpoint),
                                            "contract": contract})
                    except Exception as error:
                        files[name]["read_error"] = f"{type(error).__name__}: {error}"
        candidates.append({"path": str(directory.resolve()), "files": files})
    return candidates


def readiness_audit(checkpoints, identity_path, *, runs_root=None, selected_run=None):
    state = git_state()
    selected = Path(selected_run).resolve() if selected_run else None
    if selected and not selected.is_dir():
        raise ReadinessError(f"selected run directory is missing: {selected}")
    if selected:
        for checkpoint_label, checkpoint in checkpoints.items():
            try:
                Path(checkpoint).resolve().relative_to(selected)
            except ValueError as error:
                raise ReadinessError(
                    f"{checkpoint_label} is outside selected run {selected}: {checkpoint}") from error
    audited = {label: audit_checkpoint(checkpoints[label], label)
               for label in CHECKPOINT_LABELS}
    contracts = [item["contract"] for item in audited.values()]
    comparison = [{key: contract[key] for key in ("base", "state_ch", "cond_ch", "patch")}
                  for contract in contracts]
    if any(item != comparison[0] for item in comparison[1:]):
        raise ReadinessError(f"checkpoint model/config contracts differ: {comparison}")
    return {"status": "ready", **PROTOCOL_FLAGS, "diagnostic_only": True,
            "git": state, "checkpoints": audited,
            "selected_run": None if selected is None else str(selected),
            "resolved_config": contracts[0]["config"],
            "resolved_config_hash": contracts[0]["config_hash"],
            "identity_baseline": audit_identity(identity_path),
            "run_candidates": audit_run_candidates(runs_root) if runs_root else []}


def _atomic_json_new(path, payload):
    """Create a resumable work shard, refusing to replace a completed shard."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite work shard: {path}")
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(partial, path)


def _load_patch_shard(path, item, checkpoint_label, internal_epoch):
    payload = json.loads(Path(path).read_text())
    expected = len(T_GRID) * 6
    if (payload.get("checkpoint_label") != checkpoint_label
            or payload.get("internal_epoch") != internal_epoch
            or payload.get("case_id") != item["case_id"]
            or payload.get("patch_index") != item["patch_index"]
            or tuple(payload.get("start", ())) != tuple(item["start"])
            or len(payload.get("rows", ())) != expected):
        raise ReadinessError(f"invalid/incompatible resumable patch shard: {path}")
    return payload["rows"]


def _prepare_work_dir(path, signature):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    signature_path = path / "signature.json"
    if signature_path.is_file():
        existing = json.loads(signature_path.read_text())
        if existing != signature:
            raise ReadinessError(f"work directory belongs to a different probe: {path}")
    else:
        unexpected = [item for item in path.iterdir() if not item.name.endswith(".partial")]
        if unexpected:
            raise ReadinessError(f"unsigned non-empty work directory: {path}")
        _atomic_json_new(signature_path, signature)
    (path / "patches").mkdir(exist_ok=True)
    (path / "thickening").mkdir(exist_ok=True)
    return path


def _load_case(case_id, images_dir, gt_sdf_dir, coarse_sdf_dir, labels_dir=None):
    image = nib.load(str(Path(images_dir) / f"{case_id}_0000.nii.gz"))
    cbct = np.asanyarray(image.dataobj).astype(np.float32)
    coords = normalize_coords(physical_coord_grid(cbct.shape, image.affine))
    with np.load(Path(coarse_sdf_dir) / f"{case_id}.npz") as item:
        coarse = item["sdf"].astype(np.float32)
        prob_l = item["prob_left"].astype(np.float32)
        prob_r = item["prob_right"].astype(np.float32)
    with np.load(Path(gt_sdf_dir) / f"{case_id}.npz") as item:
        target = item["sdf"].astype(np.float32)
    cond = build_conditioning(cbct, prob_l, prob_r, coarse[0], coarse[1], coords)
    label = None
    if labels_dir:
        label = np.asanyarray(nib.load(str(Path(labels_dir) / f"{case_id}.nii.gz")).dataobj)
    return {"cond": cond, "x0": coarse, "x1": target, "label": label,
            "spacing": voxel_spacing(image), "shape": cbct.shape}


def _starts_for_case(case, patch, patches_per_stratum, seed):
    shape = np.asarray(case["shape"]); rng = np.random.default_rng(seed)
    union_gt = np.minimum(case["x1"][0], case["x1"][1]) < 0
    union_prior = np.minimum(case["x0"][0], case["x0"][1]) < 0
    foreground_voxels = np.argwhere(union_gt)
    if not len(foreground_voxels):
        raise ReadinessError("case has no GT foreground")
    fg_indices = np.linspace(0, len(foreground_voxels) - 1, patches_per_stratum, dtype=int)
    starts = []
    for index in fg_indices:
        centre = foreground_voxels[index]
        start = np.clip(centre - patch // 2, 0, np.maximum(shape - patch, 0)).astype(int)
        starts.append(("foreground", tuple(start.tolist())))
    candidates = []
    maximum = np.maximum(shape - patch, 0)
    for _ in range(4096):
        start = np.array([rng.integers(0, value + 1) if value else 0 for value in maximum])
        gt_patch = extract_patch(union_gt, start, patch, False)
        prior_patch = extract_patch(union_prior, start, patch, False)
        if not gt_patch.any() and not prior_patch.any():
            coordinate = tuple(start.tolist())
            if coordinate not in candidates:
                candidates.append(coordinate)
            if len(candidates) == patches_per_stratum:
                break
    if len(candidates) != patches_per_stratum:
        raise ReadinessError(f"could not find {patches_per_stratum} pure-background patches")
    starts.extend(("pure_background", coordinate) for coordinate in candidates)
    return starts


def build_patch_grid(case_ids, images_dir, gt_sdf_dir, coarse_sdf_dir,
                     *, patch, cases, patches_per_stratum, seed):
    grid, exclusions = [], []
    for case_id in case_ids:
        if len({item["case_id"] for item in grid}) >= cases:
            break
        try:
            case = _load_case(case_id, images_dir, gt_sdf_dir, coarse_sdf_dir)
            starts = _starts_for_case(case, patch, patches_per_stratum,
                                      seed + int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16))
        except Exception as error:
            exclusions.append({"case_id": case_id, "reason": f"{type(error).__name__}: {error}"})
            continue
        for patch_index, (stratum, start) in enumerate(starts):
            grid.append({"case_id": case_id, "patch_index": patch_index,
                         "stratum": stratum, "start": start})
    selected = {item["case_id"] for item in grid}
    if len(selected) != cases:
        raise ReadinessError(f"requested {cases} usable cases, found {len(selected)}; exclusions={exclusions}")
    return grid, exclusions


def _safe_forward(model, batches, batch_size):
    """Retry CUDA OOM with a smaller batch; never skip an input."""
    outputs, offset = [], 0
    while offset < len(batches):
        current = min(batch_size, len(batches) - offset)
        try:
            xt = torch.cat([item[0] for item in batches[offset:offset + current]])
            time = torch.cat([item[1] for item in batches[offset:offset + current]])
            cond = torch.cat([item[2] for item in batches[offset:offset + current]])
            with torch.no_grad():
                outputs.extend(model(xt, time, cond).detach().cpu().split(1))
            offset += current
        except torch.cuda.OutOfMemoryError:
            if current == 1:
                raise
            batch_size = max(1, current // 2)
            torch.cuda.empty_cache()
    return outputs, batch_size


def _seeded_generator(seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _row_base(checkpoint_label, internal_epoch, item, t, seed):
    z, y, x = item["start"]
    return {"aggregation_level": "patch", "checkpoint_label": checkpoint_label,
            "checkpoint_epoch": internal_epoch,
            "case_id": item["case_id"], "fold": 0, "stratum": item["stratum"],
            "patch_index": item["patch_index"], "patch_z": z, "patch_y": y,
            "patch_x": x, "t": t, "seed": seed}


def probe_patch(model, checkpoint_label, internal_epoch, item, case, donor, t, device, batch_size):
    size = int(item["patch_size"]); start = item["start"]
    cond_np = extract_patch(case["cond"], start, size, 0)
    x0_np = extract_patch(case["x0"], start, size, 1)
    x1_np = extract_patch(case["x1"], start, size, 1)
    donor_np = extract_patch(donor["cond"], start, size, 0)
    cond = torch.from_numpy(cond_np[None]).float().to(device)
    x0 = torch.from_numpy(x0_np[None]).float().to(device)
    x1 = torch.from_numpy(x1_np[None]).float().to(device)
    donor_cond = torch.from_numpy(donor_np[None]).float().to(device)
    time = torch.tensor([t], dtype=torch.float32, device=device)
    tb = time.reshape(1, 1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * x1
    target_v = x1 - x0
    shortcut = (xt - x0) / t
    interventions = [("full", cond, None)]
    for name in ("zero", "noise", "shuffle"):
        changed = apply_cbct_intervention(
            cond, name, donor_cond=donor_cond,
            generator=_seeded_generator(item["noise_seed"], device) if name == "noise" else None)
        assert_intervention_invariants(cond, changed, kind="cbct")
        interventions.append((f"cbct_{name}", changed,
                              donor["case_id"] if name == "shuffle" else None))
    for name in ("zero", "swap"):
        changed = apply_prior_intervention(cond, name, donor_cond=donor_cond)
        assert_intervention_invariants(cond, changed, kind="prior")
        interventions.append((f"prior_{name}", changed,
                              donor["case_id"] if name == "swap" else None))
    # xt, x0 and t are deliberately shared by reference across every intervention.
    batches = [(xt, time, changed) for _, changed, _ in interventions]
    outputs, batch_size = _safe_forward(model, batches, batch_size)
    full = outputs[0]
    similarity = cosine_r2(full, shortcut.cpu())
    base = _row_base(checkpoint_label, internal_epoch, item, t, item["seed"])
    rows = []
    for index, (name, _, donor_case) in enumerate(interventions):
        delta = relative_delta(full, outputs[index])
        rows.append({**base, "intervention": name, "donor_case": donor_case,
                     "fm_loss": float(torch.mean((outputs[index] - target_v.cpu()) ** 2)),
                     "rel_delta": delta["value"], "rel_delta_valid": delta["valid"],
                     "cosine": similarity["cosine"] if name == "full" else None,
                     "cosine_valid": similarity["cosine_valid"] if name == "full" else None,
                     "r2": similarity["r2"] if name == "full" else None,
                     "r2_valid": similarity["r2_valid"] if name == "full" else None,
                     "validity_flag": similarity["flag"] if name == "full" else delta["flag"]})
    return rows, batch_size


def add_case_rows(patch_rows):
    keys = ("checkpoint_label", "checkpoint_epoch", "case_id", "fold", "stratum", "t", "intervention")
    grouped = defaultdict(list)
    for row in patch_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = list(patch_rows)
    for group, rows in grouped.items():
        base = dict(zip(keys, group))
        donor_cases = sorted({row["donor_case"] for row in rows if row.get("donor_case")})
        aggregate = {**base, "aggregation_level": "case", "patch_index": None,
                     "patch_z": None, "patch_y": None, "patch_x": None,
                     "seed": rows[0]["seed"], "donor_case": ";".join(donor_cases) or None}
        for metric in ("fm_loss", "rel_delta", "cosine", "r2"):
            values = [float(row[metric]) for row in rows
                      if row.get(metric) is not None and np.isfinite(float(row[metric]))]
            aggregate[metric] = float(np.mean(values)) if values else None
        aggregate["rel_delta_valid"] = all(row["rel_delta_valid"] for row in rows)
        aggregate["cosine_valid"] = all(row.get("cosine_valid") is not False for row in rows)
        aggregate["r2_valid"] = all(row.get("r2_valid") is not False for row in rows)
        aggregate["validity_flag"] = ";".join(sorted({row["validity_flag"] for row in rows}))
        output.append(aggregate)
    return output


def summarize_probe(rows, audit, exclusions, bootstrap_iterations):
    patch_rows = [row for row in rows if row["aggregation_level"] == "patch"]
    aggregate = {}
    for metric in ("fm_loss", "rel_delta", "cosine", "r2"):
        groups = defaultdict(list)
        for row in patch_rows:
            if row.get(metric) is None:
                continue
            key = (row["checkpoint_label"], row["stratum"], row["t"], row["intervention"])
            valid_key = {"rel_delta": "rel_delta_valid", "cosine": "cosine_valid",
                         "r2": "r2_valid"}.get(metric)
            groups[key].append({"case_id": row["case_id"], "value": row[metric],
                                "valid": row.get(valid_key, True), "flag": row["validity_flag"]})
        aggregate[metric] = {
            f"checkpoint={key[0]}|stratum={key[1]}|t={key[2]:.2f}|intervention={key[3]}":
            case_level_bootstrap(group, iterations=bootstrap_iterations,
                                 seed=SEEDS["bootstrap"] + index)
            for index, (key, group) in enumerate(sorted(groups.items()))}
    return {
        **PROTOCOL_FLAGS,
        "diagnostic_only": True,
        "claim_limit": CLAIM,
        "interpretation": {
            "probe_a": "Supporting diagnostic only; low large-t loss is not proof because x_t carries x1 information.",
            "probe_b": "Low CBCT sensitivity is evidence consistent with shortcut use, not mathematical proof.",
            "probe_c": "Shortcut similarity and prior interventions are empirical diagnostics.",
            "probe_d": "Erosion recovery may be described only as compatible with uniform thickening, not fully explained.",
        },
        "strata_definitions": {
            "foreground": "The extracted patch contains at least one GT foreground voxel (min GT SDF < 0).",
            "pure_background": "The extracted patch contains no GT foreground and no coarse-prior foreground voxel.",
        },
        "bootstrap": {"unit": "case", "iterations": bootstrap_iterations,
                      "patches_are_independent_units": False},
        "counts": {"patch_rows": len(patch_rows),
                   "case_rows": sum(row["aggregation_level"] == "case" for row in rows),
                   "cases": len({row["case_id"] for row in patch_rows}),
                   "excluded_candidates": len(exclusions)},
        "exclusions": exclusions, "aggregates": aggregate,
        "checkpoints": audit["checkpoints"], "resolved_config": audit["resolved_config"],
        "resolved_config_hash": audit["resolved_config_hash"],
        "identity_baseline": audit["identity_baseline"],
    }


def _physical_ball(spacing, radius_mm):
    spacing = np.asarray(spacing, np.float64)
    radii = np.maximum(1, np.ceil(radius_mm / spacing).astype(int))
    grid = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    distance2 = sum((axis * step) ** 2 for axis, step in zip(grid, spacing))
    return distance2 <= radius_mm ** 2 + 1e-12


def thickening_metrics(prediction, ground_truth, spacing, *, case_id,
                       checkpoint_label, internal_epoch):
    spacing = np.asarray(spacing, np.float64)
    radius_mm = float(np.mean(spacing))
    structure = _physical_ball(spacing, radius_mm)
    rows = []
    for side in (1, 2):
        pred = prediction == side; gt = ground_truth == side
        eroded = binary_erosion(pred, structure=structure)
        for stage, mask in (("before", pred), ("after_erosion", eroded)):
            surface = signed_surface_distance_summary(mask, gt, spacing)
            radius = radius_profile_summary(mask, gt, spacing)
            rows.append({"checkpoint_label": checkpoint_label,
                         "checkpoint_epoch": internal_epoch,
                         "case_id": case_id, "side": side,
                         "stage": stage, "erosion_radius_mm": radius_mm,
                         "spacing_z": spacing[0], "spacing_y": spacing[1],
                         "spacing_x": spacing[2], "dice": dice(mask, gt),
                         "hd95_mm": hd95(mask, gt, spacing),
                         "prediction_gt_volume_ratio": float(mask.sum() / max(int(gt.sum()), 1)),
                         **{f"signed_surface_{key}": value for key, value in surface.items()},
                         "radius_profile_signed_mean_mm": radius["signed_mean_mm"],
                         "radius_profile_mae_mm": radius["mae_mm"],
                         "radius_profile_valid": radius["valid"]})
    return rows


def _csv_write(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _json_write(path, payload):
    with Path(path).open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def render_figure(path, rows, thickening_rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                                 "xtick.labelsize": 8, "ytick.labelsize": 8,
                                 "legend.fontsize": 7, "pdf.fonttype": 42})
    palette = {"best_legacy_unknown_epoch": "#0072B2", "epoch_129": "#E69F00"}
    patch_rows = [r for r in rows if r["aggregation_level"] == "patch"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax = axes[0, 0]
    for checkpoint_label in CHECKPOINT_LABELS:
        for stratum, line in (("foreground", "-"), ("pure_background", "--")):
            points = [(t, np.mean([r["fm_loss"] for r in patch_rows
                                   if r["checkpoint_label"] == checkpoint_label and r["stratum"] == stratum
                                   and r["intervention"] == "full" and r["t"] == t]))
                      for t in T_GRID]
            ax.plot(*zip(*points), line, color=palette[checkpoint_label], marker="o", ms=2,
                    label=f"{checkpoint_label}, {stratum.replace('_', ' ')}")
    ax.set(xscale="log", yscale="log", xlabel="t", ylabel="FM MSE",
           title="A  Checkpoint-based t profile")
    ax.grid(alpha=.2); ax.legend(ncol=2)
    ax = axes[0, 1]
    styles = {"cbct_zero": "-", "cbct_noise": "--", "cbct_shuffle": ":"}
    for checkpoint_label in CHECKPOINT_LABELS:
        for intervention, line in styles.items():
            values = [(t, np.mean([r["rel_delta"] for r in patch_rows
                                   if r["checkpoint_label"] == checkpoint_label and r["stratum"] == "foreground"
                                   and r["intervention"] == intervention and r["t"] == t]))
                      for t in T_GRID]
            ax.plot(*zip(*values), line, color=palette[checkpoint_label],
                    label=f"{checkpoint_label} {intervention[5:]}")
    ax.set(xlabel="t", ylabel="relative output change", title="B  CBCT causal ablation")
    ax.grid(alpha=.2); ax.legend(ncol=3)
    ax = axes[1, 0]
    for checkpoint_label in CHECKPOINT_LABELS:
        cosine = [(t, np.mean([r["cosine"] for r in patch_rows
                               if r["checkpoint_label"] == checkpoint_label and r["stratum"] == "foreground"
                               and r["intervention"] == "full" and r["t"] == t
                               and r["cosine"] is not None])) for t in T_GRID]
        prior = [(t, np.mean([r["rel_delta"] for r in patch_rows
                              if r["checkpoint_label"] == checkpoint_label and r["stratum"] == "foreground"
                              and r["intervention"] == "prior_swap" and r["t"] == t]))
                 for t in T_GRID]
        ax.plot(*zip(*cosine), "-", color=palette[checkpoint_label],
                label=f"cosine {checkpoint_label}")
        ax.plot(*zip(*prior), "--", color=palette[checkpoint_label],
                label=f"prior swap {checkpoint_label}")
    ax.set(xlabel="t", ylabel="similarity / sensitivity", title="C  Shortcut and prior probes")
    ax.grid(alpha=.2); ax.legend(ncol=2)
    ax = axes[1, 1]
    if thickening_rows:
        values, labels = [], []
        for checkpoint_label in CHECKPOINT_LABELS:
            for stage in ("before", "after_erosion"):
                values.append([r["dice"] for r in thickening_rows
                               if r["checkpoint_label"] == checkpoint_label and r["stage"] == stage])
                labels.append(f"{checkpoint_label}\n{stage.replace('_', ' ')}")
        try:
            ax.boxplot(values, tick_labels=labels, showfliers=True)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot(values, labels=labels, showfliers=True)
        ax.tick_params(axis="x", labelrotation=20)
        ax.set(ylabel="Dice", title="D  Physical erosion compatibility")
    else:
        ax.text(.5, .5, "No thickening rows", ha="center", va="center")
        ax.set(title="D  Physical erosion compatibility", xticks=[], yticks=[])
    ax.grid(alpha=.2)
    fig.text(.01, .005, summary.get("claim_limit", CLAIM), fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(path, format="pdf")
    plt.close(fig)


def validate_staged_artifacts(directory):
    directory = Path(directory)
    for name in OUTPUT_NAMES:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing/empty staged artifact: {path}")
    if not (directory / "limited_endpoint_diagnostic.pdf").read_bytes().startswith(b"%PDF"):
        raise ValueError("figure is not a PDF")
    summary = json.loads((directory / "shortcut_probe_summary.json").read_text())
    manifest = json.loads((directory / "shortcut_probe_manifest.json").read_text())
    for payload in (summary, manifest):
        if any(payload.get(key) != value for key, value in PROTOCOL_FLAGS.items()):
            raise ValueError("protocol-deviation metadata is missing or incorrect")


def publish_artifacts(staged_directory, output_directory):
    """No-overwrite atomic publication from a validated local staging directory."""
    staged, output = Path(staged_directory), Path(output_directory)
    validate_staged_artifacts(staged)
    existing = [str(output / name) for name in OUTPUT_NAMES if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite analysis artifacts: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    published = []
    try:
        for name in OUTPUT_NAMES:
            destination = output / name
            partial = output / f".{name}.partial"
            shutil.copy2(staged / name, partial)
            os.replace(partial, destination)
            published.append(destination)
    except Exception:
        for partial in output.glob(".*.partial"):
            partial.unlink(missing_ok=True)
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    return published


def generate_smoke_artifacts(directory):
    """Minimal known-data output path used by tests, never presented as a real result."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    rows = []
    internal_epochs = {"best_legacy_unknown_epoch": None, "epoch_129": 129}
    for checkpoint_label in CHECKPOINT_LABELS:
        for stratum in ("foreground", "pure_background"):
            for t in T_GRID:
                for intervention, delta in (("full", 0.0), ("cbct_zero", .01),
                                            ("cbct_noise", .02), ("cbct_shuffle", .015),
                                            ("prior_zero", .5), ("prior_swap", .6)):
                    rows.append({"aggregation_level": "patch", "checkpoint_label": checkpoint_label,
                                 "checkpoint_epoch": internal_epochs[checkpoint_label],
                                 "case_id": "synthetic", "fold": 0, "stratum": stratum,
                                 "patch_index": 0, "patch_z": 0, "patch_y": 0, "patch_x": 0,
                                 "t": t, "seed": 1, "intervention": intervention,
                                 "donor_case": "donor" if "shuffle" in intervention or "swap" in intervention else None,
                                 "fm_loss": 1 / (1 + 10 * t), "rel_delta": delta,
                                 "rel_delta_valid": True, "cosine": .99 if intervention == "full" else None,
                                 "cosine_valid": True, "r2": .98 if intervention == "full" else None,
                                 "r2_valid": True, "validity_flag": "ok"})
    thick = []
    for checkpoint_label in CHECKPOINT_LABELS:
        for stage, value in (("before", .8), ("after_erosion", .9)):
            thick.append({"checkpoint_label": checkpoint_label,
                          "checkpoint_epoch": internal_epochs[checkpoint_label],
                          "case_id": "synthetic", "side": 1,
                          "stage": stage, "dice": value})
    summary = {**PROTOCOL_FLAGS, "diagnostic_only": True,
               "claim_limit": CLAIM, "synthetic_smoke_test": True}
    _csv_write(directory / "shortcut_probe.csv", rows)
    _csv_write(directory / "thickening_probe.csv", thick)
    _json_write(directory / "shortcut_probe_summary.json", summary)
    render_figure(directory / "limited_endpoint_diagnostic.pdf", rows, thick, summary)
    manifest = {**PROTOCOL_FLAGS, "diagnostic_only": True,
                "synthetic_smoke_test": True, "created_at": utcnow(), "artifacts": {}}
    for name in OUTPUT_NAMES[:-1]:
        manifest["artifacts"][name] = sha256_file(directory / name)
    _json_write(directory / "shortcut_probe_manifest.json", manifest)
    validate_staged_artifacts(directory)


def _parse_checkpoints(values):
    parsed = {}
    for value in values:
        try:
            checkpoint_label, path = value.split("=", 1)
            if checkpoint_label in parsed:
                raise ValueError("duplicate checkpoint label")
            parsed[checkpoint_label] = Path(path)
        except Exception as error:
            raise argparse.ArgumentTypeError(f"invalid checkpoint mapping {value!r}") from error
    if set(parsed) != set(CHECKPOINT_LABELS):
        raise ReadinessError(
            f"checkpoint mappings must be exactly {list(CHECKPOINT_LABELS)}; got {sorted(parsed)}")
    return parsed


def run(args):
    started_at = utcnow()
    checkpoints = _parse_checkpoints(args.checkpoint)
    audit = readiness_audit(checkpoints, args.identity_baseline, runs_root=args.runs_root,
                            selected_run=args.run_dir)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.readiness_only:
        return audit
    if not all((args.splits, args.images, args.labels, args.gt_sdf, args.coarse_sdf)):
        raise ReadinessError("probe execution requires splits/images/labels/gt-sdf/coarse-sdf")
    random.seed(SEEDS["global"]); np.random.seed(SEEDS["global"]); torch.manual_seed(SEEDS["global"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEEDS["global"])
    device = torch.device(args.device)
    with open(args.splits) as handle:
        fold = json.load(handle)["folds"][0]
    patch = int(audit["checkpoints"][CHECKPOINT_LABELS[0]]["contract"]["patch"])
    grid, exclusions = build_patch_grid(fold["val"], args.images, args.gt_sdf,
                                        args.coarse_sdf, patch=patch, cases=args.cases,
                                        patches_per_stratum=args.patches_per_stratum,
                                        seed=SEEDS["patch"])
    for item in grid:
        item["patch_size"] = patch; item["seed"] = SEEDS["patch"]
        item["noise_seed"] = SEEDS["noise"] + item["patch_index"]
    selected_ids = list(dict.fromkeys(item["case_id"] for item in grid))
    donor_map = {case: selected_ids[(index + 1) % len(selected_ids)]
                 for index, case in enumerate(selected_ids)}
    if len(selected_ids) < 2:
        raise ReadinessError("shuffle/swap probes require at least two cases")
    cases = {case_id: _load_case(case_id, args.images, args.gt_sdf, args.coarse_sdf,
                                 args.labels) for case_id in selected_ids}
    signature = {"schema_version": 1,
                 "checkpoints": {label: audit["checkpoints"][label]["sha256"]
                                 for label in CHECKPOINT_LABELS},
                 "identity": audit["identity_baseline"]["sha256"],
                 "config_hash": audit["resolved_config_hash"],
                 "patch_grid_hash": canonical_hash(grid), "donor_map": donor_map,
                 "t_grid": T_GRID, "seeds": SEEDS}
    work_dir = _prepare_work_dir(
        args.work_dir or (Path(args.output_dir).parent / ".limited_endpoint_probe_work"), signature)
    patch_rows, batch_size = [], args.batch_size
    for checkpoint_label in CHECKPOINT_LABELS:
        internal_epoch = audit["checkpoints"][checkpoint_label]["internal_epoch"]
        checkpoint = torch.load(checkpoints[checkpoint_label], map_location="cpu", weights_only=False)
        model, _ = _resolved_model_contract(checkpoint); model.to(device).eval()
        for item in grid:
            shard = work_dir / "patches" / (
                f"{checkpoint_label}_patch_{item['patch_index']:03d}_{item['case_id']}.json")
            if shard.is_file():
                patch_rows.extend(_load_patch_shard(
                    shard, item, checkpoint_label, internal_epoch))
                continue
            case = cases[item["case_id"]]
            donor_id = donor_map[item["case_id"]]
            donor = {**cases[donor_id], "case_id": donor_id}
            shard_rows = []
            for t in T_GRID:
                produced, batch_size = probe_patch(
                    model, checkpoint_label, internal_epoch, item, case, donor,
                    t, device, batch_size)
                shard_rows.extend(produced)
            _atomic_json_new(shard, {"checkpoint_label": checkpoint_label,
                                     "internal_epoch": internal_epoch,
                                     "case_id": item["case_id"],
                                     "patch_index": item["patch_index"],
                                     "start": item["start"], "rows": shard_rows})
            patch_rows.extend(shard_rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    rows = add_case_rows(patch_rows)
    # Probe D evaluates both historical endpoints independently on the same cases.
    thickening_rows = []
    for checkpoint_label in CHECKPOINT_LABELS:
        internal_epoch = audit["checkpoints"][checkpoint_label]["internal_epoch"]
        checkpoint = torch.load(checkpoints[checkpoint_label], map_location="cpu", weights_only=False)
        model, contract = _resolved_model_contract(checkpoint); model.to(device).eval()
        for case_id in selected_ids[:args.thickening_cases]:
            shard = work_dir / "thickening" / f"{checkpoint_label}_{case_id}.json"
            if shard.is_file():
                payload = json.loads(shard.read_text())
                if (payload.get("checkpoint_label") != checkpoint_label
                        or payload.get("internal_epoch") != internal_epoch
                        or payload.get("case_id") != case_id
                        or len(payload.get("rows", ())) != 4):
                    raise ReadinessError(f"invalid/incompatible thickening shard: {shard}")
                thickening_rows.extend(payload["rows"])
                continue
            case = cases[case_id]
            endpoint = predict_volume(model, case["cond"], case["x0"], patch=patch,
                                      steps=contract["ode_steps"], device=str(device))
            prediction = sdf_stack_to_mask(endpoint)
            shard_rows = thickening_metrics(
                prediction, case["label"], case["spacing"], case_id=case_id,
                checkpoint_label=checkpoint_label, internal_epoch=internal_epoch)
            _atomic_json_new(shard, {"checkpoint_label": checkpoint_label,
                                     "internal_epoch": internal_epoch,
                                     "case_id": case_id, "rows": shard_rows})
            thickening_rows.extend(shard_rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = summarize_probe(rows, audit, exclusions, args.bootstrap_iterations)
    summary["thickening"] = {
        "cases": len({row["case_id"] for row in thickening_rows}),
        "interpretation_limit": "Observed recovery may be compatible with uniform thickening; it is not fully explained by this probe."}
    temp_parent = Path(args.local_temp_parent); temp_parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix="limited_endpoint_probe_", dir=temp_parent))
    _csv_write(staged / "shortcut_probe.csv", rows)
    _csv_write(staged / "thickening_probe.csv", thickening_rows)
    _json_write(staged / "shortcut_probe_summary.json", summary)
    render_figure(staged / "limited_endpoint_diagnostic.pdf", rows, thickening_rows, summary)
    artifact_hashes = {name: sha256_file(staged / name) for name in OUTPUT_NAMES[:-1]}
    manifest = {**PROTOCOL_FLAGS, "diagnostic_only": True,
                "git": audit["git"], "start_time": started_at, "end_time": utcnow(),
                "python": platform.python_version(), "torch": torch.__version__,
                "cuda_version": torch.version.cuda, "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None), "device": str(device),
                "checkpoints": audit["checkpoints"], "config_hash": audit["resolved_config_hash"],
                "seeds": SEEDS, "identity_baseline": audit["identity_baseline"],
                "patch_grid_hash": canonical_hash(grid), "patch_grid": grid,
                "donor_map": donor_map, "work_dir": str(work_dir.resolve()),
                "resumable": True, "artifacts": artifact_hashes,
                "claim_limit": CLAIM}
    _json_write(staged / "shortcut_probe_manifest.json", manifest)
    validate_staged_artifacts(staged)
    # Immutable inputs must still match the readiness snapshot immediately before publication.
    for checkpoint_label, path in checkpoints.items():
        if sha256_file(path) != audit["checkpoints"][checkpoint_label]["sha256"]:
            raise ReadinessError(f"checkpoint changed during probe: {path}")
    if sha256_file(args.identity_baseline) != audit["identity_baseline"]["sha256"]:
        raise ReadinessError("identity baseline changed during probe")
    published = publish_artifacts(staged, args.output_dir)
    print(CLAIM)
    print("Published:", *(str(path) for path in published), sep="\n- ")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--identity-baseline", required=True)
    parser.add_argument("--runs-root")
    parser.add_argument("--run-dir", help="explicitly proven legacy flow_fold0 run directory")
    parser.add_argument("--splits"); parser.add_argument("--images")
    parser.add_argument("--labels"); parser.add_argument("--gt-sdf")
    parser.add_argument("--coarse-sdf")
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--work-dir", help="persistent resumable work-shard directory")
    parser.add_argument("--local-temp-parent", default="/tmp")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--patches-per-stratum", type=int, default=2)
    parser.add_argument("--thickening-cases", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--synthetic-smoke-dir")
    return parser


def main():
    args = build_parser().parse_args()
    if args.synthetic_smoke_dir:
        generate_smoke_artifacts(args.synthetic_smoke_dir)
        print(f"synthetic smoke artifacts: {args.synthetic_smoke_dir}")
        return
    try:
        run(args)
    except ReadinessError as error:
        raise SystemExit(f"READINESS FAILED: {error}") from error


if __name__ == "__main__":
    main()
