#!/usr/bin/env python3
"""Drive-backed, stage- and case-resumable runner for the Prompt-1 barrier.

The acceptance path inventories legacy hard OOF artifacts, imports their fold
and checkpoint provenance without claiming exact cross-runtime reproduction,
completes only missing/invalid physical-SDF caches, audits all 480 development
cases, and measures the three-path identity baseline. It never trains Track A
or Track B, never modifies legacy hard artifacts, and does not require true
softmax for identity acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "data", ROOT / "nnunet"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analysis.oof_prior_audit import (atomic_json, expected_fold_map,  # noqa: E402
                                      find_case_artifact, load_oof)
from data.compute_coarse_sdf import (  # noqa: E402
    cache_is_valid as coarse_cache_is_valid, compute_one as compute_coarse_one)
from data.compute_gt_sdf import (cache_is_valid as gt_cache_is_valid,  # noqa: E402
                                 compute_one as compute_gt_one)
from data.io_utils import sdf_stack_to_mask, voxel_spacing  # noqa: E402
from nnunet.predict_oof import (cache_is_valid as hard_cache_is_valid,  # noqa: E402
                                predict_fold, resolve_checkpoint,
                                softmax_cache_is_valid)

EXPECTED_CASES = 480
MAX_AUTOMATIC_RETRIES = 2
HEARTBEAT_EVERY = 10
CROSS_RUNTIME_CHANGED_VOXELS = {0: 3, 1: 1, 2: 6, 3: 2, 4: 2}
INVENTORY_SCHEMA_VERSION = 1


class NonRetryableCaseError(RuntimeError):
    """A deterministic case failure for which another identical run is wasteful."""


def load_splits_config(config, repo_root=ROOT):
    """Load the configured fold split, retaining the repository fallback."""
    configured = config.get("SPLITS_PATH")
    path = Path(configured) if configured else Path(repo_root) / "configs" / "splits.json"
    if not path.is_file():
        source = "configured SPLITS_PATH" if configured else "repository splits fallback"
        raise FileNotFoundError(f"{source} does not exist: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid splits JSON at {path}: {error}") from error
    return path, payload


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_inventory_metadata(path, kind):
    path = Path(path)
    if kind in ("image", "label"):
        image = nib.load(str(path))
        shape = list(image.shape)
        if len(shape) != 3:
            raise ValueError(f"{kind} must be 3-D, got {shape}")
        return {"shape": shape, "spacing": voxel_spacing(image).tolist(),
                "affine": np.asarray(image.affine).tolist(), "artifact_type": "nifti"}
    if kind == "hard_oof":
        probs, hard, artifact_type = load_oof(path)
        if artifact_type not in ("hard_segmentation", "derived_one_hot"):
            raise ValueError(f"hard OOF has unsupported artifact type: {artifact_type}")
        if hard.ndim != 3 or not np.isfinite(probs).all():
            raise ValueError(f"invalid hard OOF array: {hard.shape}")
        return {"shape": list(hard.shape), "spacing": None, "affine": None,
                "artifact_type": artifact_type}
    if kind not in ("coarse_sdf", "gt_sdf"):
        raise ValueError(f"unknown inventory kind: {kind}")
    validator = coarse_cache_is_valid if kind == "coarse_sdf" else gt_cache_is_valid
    if not validator(path):
        raise ValueError(f"invalid {kind} cache structure")
    with np.load(path) as item:
        sdf = item["sdf"]
        if not np.isfinite(sdf).all():
            raise ValueError(f"{kind} contains non-finite values")
        spacing = item.get("spacing")
        return {"shape": list(sdf.shape),
                "spacing": None if spacing is None else np.asarray(spacing).tolist(),
                "affine": None, "artifact_type": "physical_mm_sdf"}


def inventory_file_record(path, kind, previous=None, metadata_reader=None):
    """Return cached file metadata, reopening content only when stat data changed."""
    if path is None:
        return {"status": "missing", "path": None, "kind": kind}
    path = Path(path)
    if not path.is_file():
        return {"status": "missing", "path": str(path), "kind": kind}
    stat = path.stat()
    if (previous and previous.get("path") == str(path)
            and previous.get("size_bytes") == stat.st_size
            and previous.get("mtime_ns") == stat.st_mtime_ns
            and previous.get("sha256")):
        return previous
    record = {
        "status": "valid", "path": str(path), "kind": kind,
        "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }
    try:
        reader = metadata_reader or _read_inventory_metadata
        record.update(reader(path, kind))
    except Exception as error:
        record.update({"status": "invalid", "shape": None, "spacing": None,
                       "affine": None, "artifact_type": None,
                       "error": f"{type(error).__name__}: {error}"})
    return record


def _compute_staged_sdf_job(job):
    """Compute the missing SDF artifacts for one already-staged case."""
    (case_id, labels_dir, hard_dir, output_dir, need_gt, need_coarse,
     prob_threshold, clip_mm) = job
    output_dir = Path(output_dir)
    result = {"case_id": case_id, "gt_sdf": None, "coarse_sdf": None}
    if need_gt:
        gt_out = output_dir / "gt"
        gt_out.mkdir(parents=True, exist_ok=True)
        _, produced, _ = compute_gt_one(
            (str(labels_dir), f"{case_id}.nii.gz", str(gt_out), clip_mm))
        if not gt_cache_is_valid(produced):
            raise ValueError(f"local GT SDF validation failed for {case_id}")
        result["gt_sdf"] = str(produced)
    if need_coarse:
        coarse_out = output_dir / "coarse"
        coarse_out.mkdir(parents=True, exist_ok=True)
        _, produced, _ = compute_coarse_one(
            (str(hard_dir), str(labels_dir), f"{case_id}.npz",
             str(coarse_out), prob_threshold, clip_mm))
        if not coarse_cache_is_valid(produced):
            raise ValueError(f"local coarse SDF validation failed for {case_id}")
        result["coarse_sdf"] = str(produced)
    return result


def atomic_copy(source, destination, replace_invalid=False):
    """Publish a local file to persistent storage without exposing partial bytes."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace_invalid:
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, partial)
    os.replace(partial, destination)
    return sha256_file(destination)


def assert_voxelwise_match(reference_mask, reproduced_mask, case_id):
    reference_mask = np.asarray(reference_mask)
    reproduced_mask = np.asarray(reproduced_mask)
    if reference_mask.shape != reproduced_mask.shape:
        raise NonRetryableCaseError(
            f"legacy OOF bootstrap shape mismatch for {case_id}: "
            f"{reference_mask.shape} != {reproduced_mask.shape}")
    changed = int(np.count_nonzero(reference_mask != reproduced_mask))
    if changed:
        raise NonRetryableCaseError(
            f"legacy OOF bootstrap mismatch for {case_id}: {changed} changed voxels")
    return changed


def _under(child, parent):
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


class CompletionState:
    """Atomic state file with the exact fields needed for overnight recovery."""

    def __init__(self, path, pinned_sha, device, session_id=None):
        self.path = Path(path)
        active_session = session_id or os.environ.get(
            "COLAB_RELEASE_TAG", f"session-{uuid.uuid4().hex[:12]}")
        if self.path.is_file():
            self.data = json.loads(self.path.read_text())
            previous = self.data.get("pinned_git_sha")
            if previous and previous != pinned_sha:
                raise ValueError(f"state belongs to git SHA {previous}, current is {pinned_sha}")
        else:
            self.data = {
                "pinned_git_sha": pinned_sha,
                "current_stage": "initialized",
                "completed_cases": {},
                "failed_cases": {},
                "retry_counts": {},
                "attempt_counts": {},
                "timestamps": {"created_at": utcnow()},
                "checksums": {},
                "last_heartbeat": None,
                "current_fold": None,
                "current_case": None,
                "cuda_device": device,
                "colab_session_identifier": active_session,
                "complete_cv": False,
            }
        for name, default in (("completed_cases", {}), ("failed_cases", {}),
                              ("retry_counts", {}), ("attempt_counts", {}),
                              ("checksums", {}), ("timestamps", {})):
            self.data.setdefault(name, default)
        previous_session = self.data.get("colab_session_identifier")
        if previous_session and previous_session != active_session:
            self.data.setdefault("session_history", []).append(previous_session)
        self.data["colab_session_identifier"] = active_session
        self.data["cuda_device"] = device
        self.data["timestamps"]["last_opened_at"] = utcnow()
        self.save()

    def save(self):
        self.data["timestamps"]["last_update_at"] = utcnow()
        atomic_json(self.path, self.data)

    def begin(self, stage, case_id=None, fold=None):
        self.data["current_stage"] = stage
        self.data["current_case"] = case_id
        self.data["current_fold"] = fold
        self.save()

    def complete(self, stage, case_id, result=None):
        self.data["completed_cases"].setdefault(stage, {})[case_id] = {
            "completed_at": utcnow(), **(result or {})}
        self.data["failed_cases"].setdefault(stage, {}).pop(case_id, None)
        checksums = (result or {}).get("checksums")
        if checksums:
            self.data["checksums"].setdefault(stage, {})[case_id] = checksums
        self.data["current_case"] = None
        self.data["current_fold"] = None
        self.save()

    def fail(self, stage, case_id, error):
        attempts = self.data.setdefault("attempt_counts", {}).setdefault(stage, {})
        attempts[case_id] = int(attempts.get(case_id, 0)) + 1
        retries = self.data["retry_counts"].setdefault(stage, {})
        retries[case_id] = max(0, attempts[case_id] - 1)
        self.data["failed_cases"].setdefault(stage, {})[case_id] = {
            "failed_at": utcnow(), "error": f"{type(error).__name__}: {error}"}
        self.save()
        return attempts[case_id]

    def heartbeat(self, done, total):
        self.data["last_heartbeat"] = {
            "at": utcnow(), "stage": self.data["current_stage"],
            "completed_in_queue": done, "queue_size": total}
        self.save()
        print(f"[heartbeat] {self.data['current_stage']} {done}/{total} | {utcnow()}",
              flush=True)


def run_case_queue(stage, cases, state, worker, validator, fold_map=None,
                   retry_failed=True, heartbeat_every=HEARTBEAT_EVERY,
                   skip_initially_valid=True, progress_prefix=None):
    """Run a case queue; a failed case never prevents later cases from running."""
    cases = list(cases)
    completed = 0
    for case_id in cases:
        if skip_initially_valid and validator(case_id):
            state.complete(stage, case_id, {"status": "already_valid"})
            completed += 1
            if completed % heartbeat_every == 0 or completed == len(cases):
                state.heartbeat(completed, len(cases))
            if progress_prefix:
                print(f"{progress_prefix} {completed}/{len(cases)}", flush=True)
            continue
        while True:
            state.begin(stage, case_id, (fold_map or {}).get(case_id))
            try:
                result = worker(case_id) or {}
                if not validator(case_id):
                    raise ValueError(f"worker returned but {stage} artifact is invalid")
                state.complete(stage, case_id, result)
                completed += 1
                if progress_prefix:
                    print(f"{progress_prefix} {completed}/{len(cases)}", flush=True)
                break
            except KeyboardInterrupt:
                state.save()
                raise
            except Exception as error:  # continue queue after bounded retries
                failures = state.fail(stage, case_id, error)
                retries_used = failures - 1
                if (isinstance(error, NonRetryableCaseError) or not retry_failed
                        or retries_used >= MAX_AUTOMATIC_RETRIES):
                    print(f"[{stage}] FAILED {case_id}: {error}", flush=True)
                    break
                print(f"[{stage}] retry {retries_used + 1}/{MAX_AUTOMATIC_RETRIES} "
                      f"for {case_id}: {error}", flush=True)
        if completed % heartbeat_every == 0 or completed == len(cases):
            state.heartbeat(completed, len(cases))
    return completed


class Prompt1Runner:
    def __init__(self, config):
        self.config = config
        required = ("DRIVE_ROOT", "DATASET_ROOT", "NNUNET_RESULTS",
                    "TRACKB_CACHE_ROOT", "OUTPUT_ROOT", "PINNED_COMMIT",
                    "NUM_WORKERS", "DEVICE", "MAX_CASES", "FORCE_REBUILD",
                    "EXPORT_TRUE_SOFTMAX", "RETRY_FAILED")
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(f"configuration is missing: {missing}")
        self.drive_root = Path(config["DRIVE_ROOT"])
        self.dataset = Path(config["DATASET_ROOT"])
        self.results = Path(config["NNUNET_RESULTS"])
        self.cache_root = Path(config["TRACKB_CACHE_ROOT"])
        self.output_root = Path(config["OUTPUT_ROOT"])
        if not self.drive_root.is_dir():
            raise FileNotFoundError(
                f"DRIVE_ROOT is not mounted or does not exist: {self.drive_root}")
        for name, path in (("DATASET_ROOT", self.dataset),
                           ("NNUNET_RESULTS", self.results),
                           ("TRACKB_CACHE_ROOT", self.cache_root),
                           ("OUTPUT_ROOT", self.output_root)):
            if not _under(path, self.drive_root):
                raise ValueError(f"{name} must be persistent under DRIVE_ROOT: {path}")
        self.images = self.dataset / "imagesTr"
        self.labels = self.dataset / "labelsTr"
        for name, path in (("imagesTr", self.images), ("labelsTr", self.labels),
                           ("nnU-Net results", self.results)):
            if not path.is_dir():
                raise FileNotFoundError(f"required Drive input {name} is missing: {path}")
        self.hard_dir = self.cache_root / "oof_hard"
        self.legacy_hard_dir = self.cache_root / "oof_probs"
        self.softmax_dir = self.cache_root / "oof_softmax"
        self.coarse_dir = self.cache_root / "coarse_sdf"
        self.gt_sdf_dir = self.cache_root / "gt_sdf"
        self.provenance_path = self.cache_root / "oof_manifest.json"
        self.prompt_dir = self.output_root / "prompt1"
        self.analysis_dir = self.output_root / "analysis"
        self.baseline_dir = self.output_root / "baselines"
        self.state_path = None
        self.inventory_path = self.prompt_dir / "cache_inventory_480.json"
        self.inventory_receipt_path = self.prompt_dir / "cache_inventory_480_receipt.json"
        self.legacy_import_path = self.prompt_dir / "legacy_provenance_import_manifest.json"
        self.legacy_import_receipt_path = self.prompt_dir / "legacy_provenance_import_receipt.json"
        self.audit_480_path = self.analysis_dir / "oof_prior_audit_480.json"
        self.audit_480_receipt_path = self.prompt_dir / "oof_prior_audit_480_receipt.json"
        self.sdf_receipt_path = self.prompt_dir / "complete_missing_sdf_receipt.json"
        self.identity_480_receipt_path = self.prompt_dir / "identity_480_receipt.json"
        self.final_receipt_path = self.prompt_dir / "finalize_prompt1_receipt.json"
        for directory in (self.hard_dir, self.softmax_dir, self.coarse_dir,
                          self.gt_sdf_dir, self.prompt_dir, self.analysis_dir,
                          self.baseline_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.splits_path, self.splits = load_splits_config(config)
        self.fold_map = expected_fold_map(self.splits)
        self.case_ids = list(self.splits["development"])
        external = set(self.splits.get("external_test", []))
        if len(self.case_ids) != EXPECTED_CASES:
            raise ValueError(f"expected {EXPECTED_CASES} development cases, got {len(self.case_ids)}")
        if external & set(self.case_ids):
            raise ValueError("external S cohort overlaps development")
        self.max_cases = config["MAX_CASES"]
        if self.max_cases in (0, "", False):
            self.max_cases = None
        if self.max_cases is not None:
            self.max_cases = int(self.max_cases)
            if self.max_cases < 1:
                raise ValueError("MAX_CASES must be null or positive")
        self.workers = max(1, min(int(config["NUM_WORKERS"]), 2, os.cpu_count() or 1))
        self.device = str(config["DEVICE"])
        self.force = bool(config["FORCE_REBUILD"])
        self.export_softmax = bool(config["EXPORT_TRUE_SOFTMAX"])
        self.retry_failed = bool(config["RETRY_FAILED"])
        self.quick_cases_per_fold = int(config.get("QUICK_PREFLIGHT_CASES_PER_FOLD", 1))
        self.full_preflight_cases = int(config.get("FULL_PREFLIGHT_CASES", 40))
        self.preflight_mode = str(config.get("PREFLIGHT_MODE", "quick")).lower()
        self.sdf_batch_size = int(config.get("SDF_BATCH_SIZE", 12))
        if self.quick_cases_per_fold < 1:
            raise ValueError("QUICK_PREFLIGHT_CASES_PER_FOLD must be positive")
        if self.full_preflight_cases < 1:
            raise ValueError("FULL_PREFLIGHT_CASES must be positive")
        if self.preflight_mode not in ("quick", "full"):
            raise ValueError("PREFLIGHT_MODE must be 'quick' or 'full'")
        if not 8 <= self.sdf_batch_size <= 16:
            raise ValueError("SDF_BATCH_SIZE must be between 8 and 16")
        self.dataset_id = int(config.get("NNUNET_DATASET_ID", 801))
        self.nn_config = config.get("NNUNET_CONFIG", "3d_fullres")
        self.trainer = config.get("NNUNET_TRAINER", "nnUNetTrainerIAC_NoMirror")
        current_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        expected_sha = subprocess.check_output(
            ["git", "rev-parse", str(config["PINNED_COMMIT"])], cwd=ROOT, text=True).strip()
        if current_sha != expected_sha:
            raise ValueError(f"repository SHA {current_sha} != pinned SHA {expected_sha}")
        self.git_sha = current_sha
        # Namespace orchestration state by code SHA. Persistent case artifacts
        # and signature-aware audit/identity state still resume across sessions,
        # while an older runner state cannot block a code migration.
        self.state_path = self.prompt_dir / f"prompt1_stage_state_{current_sha[:12]}.json"
        self.state = CompletionState(self.state_path, current_sha, self.device,
                                     config.get("COLAB_SESSION_IDENTIFIER"))
        self._stop_requested = False
        self._install_signal_handlers()

    def _install_signal_handlers(self):
        def stop(signum, _frame):
            self._stop_requested = True
            self.state.data["timestamps"]["signal_received_at"] = utcnow()
            self.state.data["timestamps"]["signal"] = signal.Signals(signum).name
            self.state.save()
            raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")
        for name in ("SIGTERM", "SIGINT"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), stop)

    def _provenance(self):
        if not self.provenance_path.is_file():
            return {"contract_version": 1, "cases": {}}
        raw = json.loads(self.provenance_path.read_text())
        raw.setdefault("cases", {})
        return raw

    def _write_provenance(self, payload):
        payload["updated_at"] = utcnow()
        atomic_json(self.provenance_path, payload)

    @staticmethod
    def _load_stage_json(path, name):
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f"{name} artifact is missing: {path}")
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{name} artifact is unreadable: {path}: {error}") from error

    def _write_stage_receipt(self, path, stage, inputs, artifacts, summary=None):
        records = {}
        for name, artifact in artifacts.items():
            artifact = Path(artifact)
            if not artifact.is_file():
                raise RuntimeError(f"{stage} output is missing: {artifact}")
            records[name] = {"path": str(artifact), "sha256": sha256_file(artifact)}
        payload = {"stage": stage, "pinned_git_sha": self.git_sha,
                   "completed_at": utcnow(), "inputs": inputs,
                   "artifacts": records, "summary": summary or {}}
        path = Path(path)
        if path.is_file():
            try:
                existing = json.loads(path.read_text())
                comparable = dict(existing)
                comparable.pop("completed_at", None)
                expected = dict(payload)
                expected.pop("completed_at", None)
                if comparable == expected and self._receipt_valid(path, inputs):
                    return existing
            except Exception:
                pass
        atomic_json(path, payload)
        return payload

    def _receipt_valid(self, path, inputs=None):
        path = Path(path)
        if not path.is_file():
            return False
        try:
            receipt = json.loads(path.read_text())
            if receipt.get("pinned_git_sha") != self.git_sha:
                return False
            if inputs is not None and receipt.get("inputs") != inputs:
                return False
            for record in receipt.get("artifacts", {}).values():
                artifact = Path(record["path"])
                if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
                    return False
            return bool(receipt.get("artifacts"))
        except Exception:
            return False

    @staticmethod
    def _inventory_geometry(files):
        image, label = files["image"], files["label"]
        hard, coarse, gt = files["hard_oof"], files["coarse_sdf"], files["gt_sdf"]
        def same(left, right, atol=1e-3):
            return (left is not None and right is not None
                    and bool(np.allclose(left, right, atol=atol)))
        label_shape = label.get("shape")
        checks = {
            "image_label_shape": image.get("shape") == label_shape and label_shape is not None,
            "image_label_spacing": same(image.get("spacing"), label.get("spacing")),
            "image_label_affine": same(image.get("affine"), label.get("affine")),
            "hard_shape": hard.get("shape") == label_shape and label_shape is not None,
            "coarse_shape": (coarse.get("shape", [None])[1:] == label_shape
                             if coarse.get("shape") else False),
            "gt_sdf_shape": (gt.get("shape", [None])[1:] == label_shape
                             if gt.get("shape") else False),
            "coarse_spacing": same(coarse.get("spacing"), label.get("spacing")),
            "gt_sdf_spacing": same(gt.get("spacing"), label.get("spacing")),
        }
        return {"checks": checks, "valid": all(checks.values())}

    def build_inventory(self):
        previous = (json.loads(self.inventory_path.read_text())
                    if self.inventory_path.is_file() else {})
        previous_cases = previous.get("cases", {})
        cases = {}
        status_counts = {"valid": 0, "missing": 0, "invalid": 0}
        for index, case_id in enumerate(self.case_ids, start=1):
            old_files = previous_cases.get(case_id, {}).get("files", {})
            paths = {
                "image": self.images / f"{case_id}_0000.nii.gz",
                "label": self.labels / f"{case_id}.nii.gz",
                "hard_oof": self.hard_path(case_id),
                "coarse_sdf": self.coarse_dir / f"{case_id}.npz",
                "gt_sdf": self.gt_sdf_dir / f"{case_id}.npz",
            }
            files = {name: inventory_file_record(path, name, old_files.get(name))
                     for name, path in paths.items()}
            geometry = self._inventory_geometry(files)
            file_statuses = {record["status"] for record in files.values()}
            if "missing" in file_statuses:
                status = "missing"
            elif "invalid" in file_statuses or not geometry["valid"]:
                status = "invalid"
            else:
                status = "valid"
            status_counts[status] += 1
            cases[case_id] = {"case_id": case_id,
                              "expected_fold": self.fold_map[case_id],
                              "status": status, "files": files,
                              "geometry": geometry}
            if index % HEARTBEAT_EVERY == 0 or index == len(self.case_ids):
                self.state.data["current_stage"] = "build_inventory"
                self.state.data["last_heartbeat"] = {
                    "at": utcnow(), "stage": "build_inventory",
                    "completed_in_queue": index, "queue_size": len(self.case_ids)}
                self.state.save()
                print(f"[inventory] {index}/{len(self.case_ids)}", flush=True)
        summary = {"total_cases": len(cases), "status_counts": status_counts,
                   "external_s_cases": len(set(cases) & set(self.splits.get("external_test", [])))}
        stable = (previous.get("schema_version") == INVENTORY_SCHEMA_VERSION
                  and previous.get("pinned_git_sha") == self.git_sha
                  and previous.get("splits_sha256") == sha256_file(self.splits_path)
                  and previous.get("cases") == cases and previous.get("summary") == summary)
        if stable:
            payload = previous
        else:
            payload = {"schema_version": INVENTORY_SCHEMA_VERSION,
                       "created_at": utcnow(), "pinned_git_sha": self.git_sha,
                       "splits_path": str(self.splits_path),
                       "splits_sha256": sha256_file(self.splits_path),
                       "summary": summary, "cases": cases}
            atomic_json(self.inventory_path, payload)
        self._write_stage_receipt(
            self.inventory_receipt_path, "build-inventory",
            {"splits_sha256": sha256_file(self.splits_path)},
            {"inventory": self.inventory_path}, summary)
        return payload

    @staticmethod
    def _legacy_import_entry(case_id, fold, hard_path, hard_sha,
                             checkpoint_path, checkpoint_sha):
        changed = CROSS_RUNTIME_CHANGED_VOXELS[fold]
        return {
            "case_id": case_id, "expected_fold": fold,
            "fold_semantics": "expected_validation_fold_from_splits",
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": checkpoint_sha,
            "artifact_type": "derived_one_hot", "hard_artifact": str(hard_path),
            "hard_sha256": hard_sha, "provenance_source": "legacy_import",
            "verification_method": "legacy_import_from_split_and_checkpoint_inventory",
            "exact_reproduction_claim": False,
            "cross_runtime_validation": {
                "reference_runtime": "A100_legacy", "diagnostic_runtime": "L4_smoke",
                "interpretation": "cross_runtime_near_exact_diagnostic",
                "changed_voxels": changed},
        }

    def import_legacy_provenance(self):
        inventory = self.build_inventory()
        checkpoints = {}
        for fold in range(len(self.splits["folds"])):
            checkpoint = resolve_checkpoint(self.results, self.dataset_id, self.trainer,
                                            self.nn_config, fold)
            checkpoints[fold] = (checkpoint, self._cached_sha256(checkpoint))
        cases, missing = {}, []
        for case_id in self.case_ids:
            fold = self.fold_map[case_id]
            legacy = find_case_artifact(self.legacy_hard_dir, case_id)
            if legacy is None:
                missing.append(case_id)
                continue
            hard_record = inventory["cases"][case_id]["files"]["hard_oof"]
            hard_sha = (hard_record.get("sha256") if hard_record.get("path") == str(legacy)
                        else sha256_file(legacy))
            cases[case_id] = self._legacy_import_entry(
                case_id, fold, legacy, hard_sha, *checkpoints[fold])
        cross_runtime = {
            "status": "near_exact", "exact_reproduction_claim": False,
            "interpretation": "cross_runtime_near_exact_diagnostic",
            "per_fold_changed_voxels": {str(k): v for k, v in CROSS_RUNTIME_CHANGED_VOXELS.items()},
            "total_changed_voxels": sum(CROSS_RUNTIME_CHANGED_VOXELS.values()),
        }
        core = {"contract_version": 2,
                "pinned_git_sha": self.git_sha, "provenance_source": "legacy_import",
                   "splits_path": str(self.splits_path),
                   "splits_sha256": sha256_file(self.splits_path),
                   "exact_reproduction_claim": False,
                   "cross_runtime_validation": cross_runtime,
                   "summary": {"development_cases": len(self.case_ids),
                               "imported_cases": len(cases),
                               "missing_legacy_cases": len(missing),
                               "missing_legacy_case_ids": missing},
                   "cases": cases}
        previous = (json.loads(self.legacy_import_path.read_text())
                    if self.legacy_import_path.is_file() else None)
        if previous and {k: v for k, v in previous.items() if k != "created_at"} == core:
            payload = previous
        else:
            payload = {"created_at": utcnow(), **core}
            atomic_json(self.legacy_import_path, payload)
        if (not self.provenance_path.is_file()
                or json.loads(self.provenance_path.read_text()) != payload):
            atomic_json(self.provenance_path, payload)
        legacy_input_sha = hashlib.sha256(json.dumps(
            {case_id: {"hard_sha256": entry["hard_sha256"],
                       "checkpoint_sha256": entry["source_checkpoint_sha256"]}
             for case_id, entry in cases.items()},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        inputs = {"legacy_hard_and_checkpoint_sha256": legacy_input_sha,
                  "splits_sha256": sha256_file(self.splits_path)}
        self._write_stage_receipt(
            self.legacy_import_receipt_path, "import-legacy-provenance", inputs,
            {"import_manifest": self.legacy_import_path,
             "active_provenance_manifest": self.provenance_path}, payload["summary"])
        print(f"[legacy-import] {len(cases)}/{len(self.case_ids)} cases; "
              f"cross-runtime changed voxels={sum(CROSS_RUNTIME_CHANGED_VOXELS.values())}",
              flush=True)
        return payload

    def hard_path(self, case_id):
        return (find_case_artifact(self.legacy_hard_dir, case_id)
                or find_case_artifact(self.hard_dir, case_id))

    def _label_geometry(self, case_id):
        image_path = self.images / f"{case_id}_0000.nii.gz"
        label_path = self.labels / f"{case_id}.nii.gz"
        if not image_path.is_file() or not label_path.is_file():
            return None
        image, label = nib.load(str(image_path)), nib.load(str(label_path))
        image_spacing, label_spacing = voxel_spacing(image), voxel_spacing(label)
        return {
            "shape": tuple(label.shape), "spacing": label_spacing,
            "image_label_shape": image.shape == label.shape,
            "image_label_affine": bool(np.allclose(image.affine, label.affine, atol=1e-3)),
            "image_label_spacing": bool(np.allclose(image_spacing, label_spacing, atol=1e-3)),
            "affine": np.asarray(label.affine),
        }

    def hard_valid(self, case_id):
        path, geometry = self.hard_path(case_id), self._label_geometry(case_id)
        if path is None or geometry is None:
            return False
        try:
            _, mask, artifact_type = load_oof(path)
            return (artifact_type in ("hard_segmentation", "derived_one_hot")
                    and tuple(mask.shape) == geometry["shape"])
        except Exception:
            return False

    def softmax_valid(self, case_id):
        path = self.softmax_dir / f"{case_id}.npz"
        geometry = self._label_geometry(case_id)
        if geometry is None or not softmax_cache_is_valid(path):
            return False
        try:
            probs, _, artifact_type = load_oof(path)
            return artifact_type == "true_softmax" and tuple(probs.shape[1:]) == geometry["shape"]
        except Exception:
            return False

    def _sdf_valid(self, case_id, kind):
        directory = self.coarse_dir if kind == "coarse" else self.gt_sdf_dir
        validator = coarse_cache_is_valid if kind == "coarse" else gt_cache_is_valid
        path, geometry = directory / f"{case_id}.npz", self._label_geometry(case_id)
        if geometry is None or not validator(path):
            return False
        try:
            with np.load(path) as item:
                sdf = item["sdf"]
                spacing = item.get("spacing")
                valid = (tuple(sdf.shape[1:]) == geometry["shape"]
                         and spacing is not None
                         and bool(np.allclose(spacing, geometry["spacing"], atol=1e-3)))
                if kind == "coarse" and valid and self.hard_valid(case_id):
                    _, hard, _ = load_oof(self.hard_path(case_id))
                    valid = np.array_equal(sdf_stack_to_mask(sdf.astype(np.float32)), hard)
                return bool(valid)
        except Exception:
            return False

    def coarse_valid(self, case_id):
        return self._sdf_valid(case_id, "coarse")

    def gt_sdf_valid(self, case_id):
        return self._sdf_valid(case_id, "gt")

    def provenance_valid(self, case_id):
        case = self._provenance().get("cases", {}).get(case_id, {})
        checkpoint = case.get("source_checkpoint")
        actual = self.hard_path(case_id)
        if actual is None:
            return False
        try:
            recorded = Path(case.get("hard_artifact", ""))
            recorded_matches = recorded.is_file() and recorded.resolve() == actual.resolve()
            checksum_matches = bool(case.get("hard_sha256")) and (
                case["hard_sha256"] == sha256_file(actual))
            checkpoint_checksum_matches = bool(case.get("source_checkpoint_sha256")) and (
                case["source_checkpoint_sha256"] == self._cached_sha256(checkpoint))
            _, _, actual_type = load_oof(actual)
        except Exception:
            return False
        fold_matches = (case.get("expected_fold", case.get("prediction_fold"))
                        == self.fold_map[case_id])
        source_valid = case.get("provenance_source") in (None, "legacy_import", "runtime_prediction")
        return (fold_matches and source_valid
                and bool(checkpoint) and Path(checkpoint).is_file()
                and case.get("artifact_type") == actual_type
                and actual_type in ("hard_segmentation", "derived_one_hot")
                and recorded_matches and checksum_matches and checkpoint_checksum_matches)

    def bootstrap_provenance_valid(self, case_id):
        if not self.provenance_valid(case_id) or not self.softmax_valid(case_id):
            return False
        case = self._provenance().get("cases", {}).get(case_id, {})
        legacy = find_case_artifact(self.legacy_hard_dir, case_id)
        softmax = self.softmax_dir / f"{case_id}.npz"
        if legacy is None:
            return False
        try:
            return (case.get("verification_method") == "legacy_reprediction_voxelwise"
                    and case.get("legacy_voxelwise_match") is True
                    and case.get("legacy_changed_voxels") == 0
                    and case.get("legacy_hard_artifact") == str(legacy)
                    and case.get("legacy_hard_sha256") == sha256_file(legacy)
                    and bool(case.get("reproduced_hard_sha256"))
                    and case.get("softmax_artifact") == str(softmax)
                    and case.get("softmax_sha256") == sha256_file(softmax))
        except Exception:
            return False

    def create_hard_view(self, case_ids=None, name="prompt1_hard_view"):
        view = Path("/content") / name
        if not Path("/content").is_dir():
            view = Path(tempfile.gettempdir()) / name
        view.mkdir(parents=True, exist_ok=True)
        for case_id in self.case_ids if case_ids is None else case_ids:
            source = self.hard_path(case_id)
            if source is None:
                continue
            destination = view / source.name
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(source.resolve())
        return view

    def complete_cache_ids(self):
        return [case_id for case_id in self.case_ids
                if self.hard_valid(case_id) and self.coarse_valid(case_id)
                and self.gt_sdf_valid(case_id)]

    def legacy_complete_cache_ids(self):
        complete = []
        for case_id in self.complete_cache_ids():
            legacy = find_case_artifact(self.legacy_hard_dir, case_id)
            active = self.hard_path(case_id)
            if (legacy is not None and active is not None
                    and legacy.resolve() == active.resolve()):
                complete.append(case_id)
        return complete

    def quick_candidate_valid(self, case_id):
        """Validate one quick candidate without inspecting any other case."""
        legacy = find_case_artifact(self.legacy_hard_dir, case_id)
        geometry = self._label_geometry(case_id)
        if legacy is None or geometry is None:
            return False
        try:
            _, legacy_mask, artifact_type = load_oof(legacy)
            active = self.hard_path(case_id)
            if (artifact_type not in ("hard_segmentation", "derived_one_hot")
                    or tuple(legacy_mask.shape) != geometry["shape"] or active is None):
                return False
            _, active_mask, _ = load_oof(active)
            return (np.array_equal(active_mask, legacy_mask)
                    and self.coarse_valid(case_id) and self.gt_sdf_valid(case_id))
        except Exception:
            return False

    def select_quick_preflight_cases(self):
        selected = []
        for fold_index, fold in enumerate(self.splits["folds"]):
            fold_selected = []
            for case_id in fold["val"]:
                if self.quick_candidate_valid(case_id):
                    fold_selected.append(case_id)
                    print(f"[quick-preflight] fold={fold_index} case={case_id}", flush=True)
                    if len(fold_selected) == self.quick_cases_per_fold:
                        break
            if len(fold_selected) != self.quick_cases_per_fold:
                raise RuntimeError(
                    f"quick preflight fold {fold_index} requires "
                    f"{self.quick_cases_per_fold} legacy-complete cases; "
                    f"found {len(fold_selected)}")
            selected.extend(fold_selected)
        return selected

    def select_full_preflight_cases(self):
        candidates = self.legacy_complete_cache_ids()
        if len(candidates) < self.full_preflight_cases:
            raise RuntimeError(
                f"full preflight requires at least {self.full_preflight_cases} complete "
                f"legacy oof_probs cases; found {len(candidates)}")
        return candidates[:self.full_preflight_cases]

    def quick_preflight_paths(self):
        count = len(self.splits["folds"]) * self.quick_cases_per_fold
        return SimpleNamespace(
            ids=self.prompt_dir / f"quick_preflight_{count}_case_ids.json",
            receipt=self.prompt_dir / f"quick_preflight_{count}_receipt.json",
            provenance_receipt=self.prompt_dir / "quick_provenance_bootstrap_receipt.json",
            audit_prefix=self.analysis_dir / f"oof_prior_audit_quick{count}",
            identity_prefix=self.baseline_dir / f"identity_quick{count}",
            state=self.prompt_dir / "quick_preflight_state.json")

    def full_preflight_paths(self):
        count = self.full_preflight_cases
        return SimpleNamespace(
            ids=self.prompt_dir / f"full_preflight_{count}_case_ids.json",
            receipt=self.prompt_dir / f"full_preflight_{count}_receipt.json",
            provenance_receipt=self.prompt_dir / "full_provenance_bootstrap_receipt.json",
            audit_prefix=self.analysis_dir / f"oof_prior_audit_full{count}",
            identity_prefix=self.baseline_dir / f"identity_full_preflight_{count}",
            state=self.prompt_dir / "full_preflight_state.json")

    def bootstrap_legacy_provenance(self, case_ids, mode="full", receipt_path=None):
        case_ids = list(case_ids)
        if receipt_path is None:
            receipt_path = (self.quick_preflight_paths().provenance_receipt
                            if mode == "quick" else self.full_preflight_paths().provenance_receipt)
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("pinned_git_sha") != self.git_sha:
                raise RuntimeError("legacy bootstrap receipt belongs to another git SHA")
            if receipt.get("case_ids") != case_ids:
                raise RuntimeError("legacy bootstrap cohort changed after it was sealed")
        stage = f"{mode}_provenance_bootstrap"
        completed = run_case_queue(
            stage, case_ids, self.state,
            lambda case_id: self._predict_case(
                case_id, compare_existing=True,
                verification="legacy_reprediction_voxelwise", force_softmax=True),
            self.bootstrap_provenance_valid, self.fold_map, self.retry_failed,
            heartbeat_every=1 if mode == "quick" else HEARTBEAT_EVERY,
            progress_prefix="[quick-preflight] provenance" if mode == "quick" else None)
        failures = self.state.data.get("failed_cases", {}).get(stage, {})
        if completed != len(case_ids) or failures:
            self.state.data["complete_cv"] = False
            self.state.save()
            raise RuntimeError(f"legacy OOF provenance bootstrap failed: {failures}")
        provenance = self._provenance()["cases"]
        records = {case_id: {
            "prediction_fold": provenance[case_id]["prediction_fold"],
            "source_checkpoint": provenance[case_id]["source_checkpoint"],
            "source_checkpoint_sha256": provenance[case_id]["source_checkpoint_sha256"],
            "artifact_type": provenance[case_id]["artifact_type"],
            "legacy_hard_sha256": provenance[case_id]["legacy_hard_sha256"],
            "reproduced_hard_sha256": provenance[case_id]["reproduced_hard_sha256"],
            "softmax_sha256": provenance[case_id]["softmax_sha256"],
            "legacy_changed_voxels": provenance[case_id]["legacy_changed_voxels"],
        } for case_id in case_ids}
        atomic_json(receipt_path, {
            "pinned_git_sha": self.git_sha, "completed_at": utcnow(),
            "case_ids": case_ids, "records": records})
        print(f"[{mode}-provenance-bootstrap] {len(case_ids)} legacy cases verified", flush=True)

    def _load_or_select_preflight_ids(self, paths, mode):
        receipt = json.loads(paths.receipt.read_text()) if paths.receipt.is_file() else None
        if receipt:
            if receipt.get("pinned_git_sha") != self.git_sha:
                raise RuntimeError("preflight receipt belongs to a different pinned git SHA")
            ids = receipt.get("case_ids", [])
        elif paths.ids.is_file():
            ids = json.loads(paths.ids.read_text())
        else:
            ids = (self.select_quick_preflight_cases() if mode == "quick"
                   else self.select_full_preflight_cases())
            atomic_json(paths.ids, ids)
        expected = (len(self.splits["folds"]) * self.quick_cases_per_fold
                    if mode == "quick" else self.full_preflight_cases)
        if len(ids) != expected or len(set(ids)) != expected:
            raise RuntimeError(f"{mode} preflight cohort is not {expected} unique cases")
        if not paths.ids.is_file():
            atomic_json(paths.ids, ids)
        return ids, receipt

    def _validate_preflight_cohort(self, ids):
        incomplete = [case_id for case_id in ids if not (
            self.hard_valid(case_id) and self.coarse_valid(case_id)
            and self.gt_sdf_valid(case_id))]
        if incomplete:
            raise RuntimeError(f"sealed preflight artifacts became invalid: {incomplete[:5]}")
        bad_provenance = [case_id for case_id in ids if not self.provenance_valid(case_id)]
        if bad_provenance:
            raise RuntimeError("legacy hard OOF provenance is absent or invalid; refusing to infer it "
                               f"from filenames (first: {bad_provenance[:5]})")

    def _run_preflight_audit(self, ids, paths, mode):
        audit_json = paths.audit_prefix.with_suffix(".json")
        view = self.create_hard_view(ids, f"prompt1_hard_view_{mode}")
        if not audit_json.is_file():
            cmd = [sys.executable, str(ROOT / "analysis/oof_prior_audit.py"),
                   "--splits", str(self.splits_path), "--images", str(self.images),
                   "--labels", str(self.labels), "--oof-hard", str(view),
                   "--oof-softmax", str(self.softmax_dir), "--coarse-sdf", str(self.coarse_dir),
                   "--gt-sdf", str(self.gt_sdf_dir), "--provenance-manifest", str(self.provenance_path),
                   "--case-ids", str(paths.ids), "--out-prefix", str(paths.audit_prefix),
                   "--state", str(self.prompt_dir / f"{mode}_oof_audit_state.json"), "--resume"]
            if mode == "quick":
                cmd += ["--heartbeat-every", "1", "--progress-prefix",
                        "[quick-preflight] audit"]
            subprocess.run(cmd, cwd=ROOT, check=True)
        return json.loads(audit_json.read_text())["summary"]

    def _run_preflight_identity(self, ids, paths, mode):
        identity_json = paths.identity_prefix.with_suffix(".json")
        view = self.create_hard_view(ids, f"prompt1_hard_view_{mode}")
        if not identity_json.is_file():
            cmd = [sys.executable, str(ROOT / "scripts/identity_preflight.py"),
                   "--splits", str(self.splits_path), "--images", str(self.images),
                   "--labels", str(self.labels), "--oof-hard", str(view),
                   "--coarse-sdf", str(self.coarse_dir), "--gt-sdf", str(self.gt_sdf_dir),
                   "--provenance-manifest", str(self.provenance_path), "--case-ids", str(paths.ids),
                   "--expected-cases", str(len(ids)), "--device", self.device,
                   "--out-prefix", str(paths.identity_prefix),
                   "--state", str(self.prompt_dir / f"{mode}_identity_state.json"), "--resume"]
            if mode == "quick":
                cmd += ["--progress-prefix", "[quick-preflight] identity"]
            subprocess.run(cmd, cwd=ROOT, check=True)
        return json.loads(identity_json.read_text())["summary"]["decision"]

    def _preflight_checksums(self, ids):
        return {case_id: {
            "hard": sha256_file(self.hard_path(case_id)),
            "coarse_sdf": sha256_file(self.coarse_dir / f"{case_id}.npz"),
            "gt_sdf": sha256_file(self.gt_sdf_dir / f"{case_id}.npz")}
            for case_id in ids}

    def _preflight(self, mode):
        paths = self.quick_preflight_paths() if mode == "quick" else self.full_preflight_paths()
        ids, receipt = self._load_or_select_preflight_ids(paths, mode)
        atomic_json(paths.state, {"mode": mode, "current_stage": "provenance",
                                  "case_ids": ids, "updated_at": utcnow()})
        self.bootstrap_legacy_provenance(ids, mode, paths.provenance_receipt)
        self._validate_preflight_cohort(ids)
        atomic_json(paths.state, {"mode": mode, "current_stage": "audit",
                                  "case_ids": ids, "updated_at": utcnow()})
        audit = self._run_preflight_audit(ids, paths, mode)
        if audit.get("status_counts") != {"valid": len(ids)}:
            raise RuntimeError(f"OOF audit failed: {audit.get('status_counts')}")
        atomic_json(paths.state, {"mode": mode, "current_stage": "identity",
                                  "case_ids": ids, "updated_at": utcnow()})
        decision = self._run_preflight_identity(ids, paths, mode)
        if decision != "PASS":
            raise RuntimeError(f"identity preflight failed: {decision}")
        checksums = self._preflight_checksums(ids)
        if receipt and receipt.get("checksums") != checksums:
            raise RuntimeError(f"a sealed {mode} preflight artifact changed after acceptance")
        passed_at = receipt.get("passed_at") if receipt else utcnow()
        atomic_json(paths.receipt, {"pinned_git_sha": self.git_sha, "mode": mode,
                                    "passed_at": passed_at, "case_ids": ids,
                                    "checksums": checksums, "audit_decision": "PASS",
                                    "identity_decision": "PASS"})
        atomic_json(paths.state, {"mode": mode, "current_stage": "complete",
                                  "case_ids": ids, "updated_at": utcnow()})
        self.state.data["timestamps"][f"{mode}_preflight_passed_at"] = utcnow()
        self.state.save()
        print(f"[{mode}-preflight] {len(ids)}-case OOF audit and three-path identity: PASS",
              flush=True)

    def preflight_quick(self):
        return self._preflight("quick")

    def preflight_full(self):
        return self._preflight("full")

    def preflight(self):
        return self.preflight_quick() if self.preflight_mode == "quick" else self.preflight_full()

    def audit_480(self):
        inventory = self._load_stage_json(self.inventory_path, "cache inventory")
        provenance = self._load_stage_json(self.provenance_path, "legacy provenance import")
        inputs = {"inventory_sha256": sha256_file(self.inventory_path),
                  "provenance_sha256": sha256_file(self.provenance_path),
                  "splits_sha256": sha256_file(self.splits_path)}
        if self._receipt_valid(self.audit_480_receipt_path, inputs):
            return self._load_stage_json(self.audit_480_path, "480-case OOF audit")
        prefix = self.audit_480_path.with_suffix("")
        cmd = [sys.executable, str(ROOT / "analysis/oof_prior_audit.py"),
               "--splits", str(self.splits_path), "--images", str(self.images),
               "--labels", str(self.labels), "--legacy-oof", str(self.legacy_hard_dir),
               "--oof-softmax", str(self.softmax_dir), "--coarse-sdf", str(self.coarse_dir),
               "--gt-sdf", str(self.gt_sdf_dir),
               "--provenance-manifest", str(self.provenance_path),
               "--inventory", str(self.inventory_path), "--out-prefix", str(prefix),
               "--state", str(self.prompt_dir / "oof_prior_audit_480_state.json"),
               "--heartbeat-every", "10", "--resume", "--replace-reports"]
        subprocess.run(cmd, cwd=ROOT, check=True)
        payload = self._load_stage_json(self.audit_480_path, "480-case OOF audit")
        summary = payload.get("summary", {})
        if summary.get("total_cases") != EXPECTED_CASES:
            raise RuntimeError(
                f"audit-480 evaluated {summary.get('total_cases')} cases, expected {EXPECTED_CASES}")
        if inventory.get("summary", {}).get("external_s_cases") != 0:
            raise RuntimeError("audit inventory contains external S-cohort cases")
        if provenance.get("summary", {}).get("development_cases") != EXPECTED_CASES:
            raise RuntimeError("legacy provenance import is not the 480-case development cohort")
        self._write_stage_receipt(
            self.audit_480_receipt_path, "audit-480", inputs,
            {"audit_json": self.audit_480_path,
             "audit_csv": prefix.parent / f"{prefix.name}_cases.csv",
             "audit_report": prefix.with_suffix(".md")}, summary)
        return payload

    def _cached_sha256(self, path):
        path = Path(path)
        stat = path.stat()
        key = str(path.resolve())
        cache = self.state.data.setdefault("file_checksum_cache", {})
        cached = cache.get(key, {})
        if (cached.get("size_bytes") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns
                and cached.get("sha256")):
            return cached["sha256"]
        checksum = sha256_file(path)
        cache[key] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "sha256": checksum}
        return checksum

    def _case_file_record(self, path, valid):
        path = Path(path) if path else None
        if path is None or not path.is_file():
            return {"status": "missing", "path": str(path) if path else None,
                    "checksum": None, "last_update": None}
        stat = path.stat()
        return {"status": "valid" if valid else "invalid", "path": str(path),
                "checksum": self._cached_sha256(path),
                "last_update": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size}

    def build_manifest(self):
        provenance = self._provenance().get("cases", {})
        cases = {}
        required_valid = 0
        missing, invalid, provenance_errors, softmax_valid = [], [], [], 0
        for index, case_id in enumerate(self.case_ids, start=1):
            geometry = self._label_geometry(case_id)
            hard_path = self.hard_path(case_id)
            hard_ok, coarse_ok, gt_ok = (self.hard_valid(case_id), self.coarse_valid(case_id),
                                          self.gt_sdf_valid(case_id))
            soft_ok, prov_ok = self.softmax_valid(case_id), self.provenance_valid(case_id)
            image_path = self.images / f"{case_id}_0000.nii.gz"
            label_path = self.labels / f"{case_id}.nii.gz"
            geometry_ok = bool(geometry and all(geometry[name] for name in
                               ("image_label_shape", "image_label_affine", "image_label_spacing")))
            identity_valid = hard_ok and coarse_ok and gt_ok and geometry_ok and prov_ok
            if identity_valid:
                required_valid += 1
            elif any(not path.is_file() for path in (image_path, label_path,
                     self.coarse_dir / f"{case_id}.npz", self.gt_sdf_dir / f"{case_id}.npz")) or hard_path is None:
                missing.append(case_id)
            else:
                invalid.append(case_id)
            if not prov_ok:
                provenance_errors.append(case_id)
            softmax_valid += int(soft_ok)
            cases[case_id] = {
                "case_id": case_id, "expected_fold": self.fold_map[case_id],
                "image": self._case_file_record(image_path, image_path.is_file()),
                "gt_label": self._case_file_record(label_path, label_path.is_file()),
                "hard_oof": self._case_file_record(hard_path, hard_ok),
                "true_softmax": self._case_file_record(self.softmax_dir / f"{case_id}.npz", soft_ok),
                "coarse_sdf": self._case_file_record(self.coarse_dir / f"{case_id}.npz", coarse_ok),
                "gt_sdf": self._case_file_record(self.gt_sdf_dir / f"{case_id}.npz", gt_ok),
                "geometry": None if geometry is None else {
                    "shape": list(geometry["shape"]), "spacing": geometry["spacing"].tolist(),
                    "affine": geometry["affine"].tolist(),
                    "image_label_shape": geometry["image_label_shape"],
                    "image_label_affine": geometry["image_label_affine"],
                    "image_label_spacing": geometry["image_label_spacing"]},
                "fold_provenance": provenance.get(case_id),
                "fold_provenance_valid": prov_ok,
                "identity_cache_status": "valid" if identity_valid else (
                    "missing" if case_id in missing else "invalid"),
            }
            if index % HEARTBEAT_EVERY == 0:
                self.state.data["current_stage"] = "manifest"
                self.state.data["last_heartbeat"] = {
                    "at": utcnow(), "stage": "manifest",
                    "completed_in_queue": index, "queue_size": EXPECTED_CASES}
                self.state.save()
                print(f"[manifest] {index}/{EXPECTED_CASES}", flush=True)
        summary = {
            "development_cases": len(cases), "valid_cases": required_valid,
            "missing_cases": len(missing), "invalid_cases": len(invalid),
            "fold_provenance_errors": len(provenance_errors),
            "true_softmax_valid_cases": softmax_valid,
            "external_s_cases_in_manifest": len(set(cases) & set(self.splits.get("external_test", []))),
            "missing_case_ids": missing, "invalid_case_ids": invalid,
            "provenance_error_case_ids": provenance_errors,
        }
        payload = {"created_at": utcnow(), "pinned_git_sha": self.git_sha,
                   "summary": summary, "cases": cases}
        atomic_json(self.prompt_dir / "cache_manifest_480.json", payload)
        return payload

    def _predict_case(self, case_id, compare_existing=False, verification=None,
                      force_softmax=False):
        fold = self.fold_map[case_id]
        checkpoint = resolve_checkpoint(self.results, self.dataset_id, self.trainer,
                                        self.nn_config, fold)
        checkpoint_sha = self._cached_sha256(checkpoint)
        local_root = Path("/content/prompt1_work")
        if not Path("/content").is_dir():
            local_root = Path(tempfile.gettempdir()) / "prompt1_work"
        local_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_root) as tmp:
            hard_local, soft_local = Path(tmp) / "hard", Path(tmp) / "softmax"
            hard_local.mkdir()
            soft_local.mkdir()
            cmd = [sys.executable, str(ROOT / "nnunet/predict_oof.py"),
                   "--dataset", str(self.dataset_id), "--config", self.nn_config,
                   "--trainer", self.trainer, "--splits", str(self.splits_path),
                   "--images", str(self.images), "--out", str(hard_local),
                   "--device", self.device, "--fold", str(fold), "--case-id", case_id,
                   "--npp", str(self.workers), "--nps", str(self.workers)]
            want_softmax = self.export_softmax or force_softmax
            if want_softmax:
                cmd += ["--save-probabilities", "--softmax-out", str(soft_local)]
            env = dict(os.environ)
            env["nnUNet_results"] = str(self.results)
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            produced_hard = hard_local / f"{case_id}.npz"
            if not hard_cache_is_valid(produced_hard):
                raise ValueError(f"invalid local hard prediction: {produced_hard}")
            _, local_mask, kind = load_oof(produced_hard)
            geometry = self._label_geometry(case_id)
            if kind != "derived_one_hot" or tuple(local_mask.shape) != geometry["shape"]:
                raise ValueError(f"local hard artifact has wrong type/shape for {case_id}")
            existing = self.hard_path(case_id)
            if existing is not None and self.hard_valid(case_id):
                _, old_mask, _ = load_oof(existing)
                if compare_existing:
                    reference_mask = old_mask
                    if verification == "legacy_reprediction_voxelwise":
                        legacy_reference = find_case_artifact(self.legacy_hard_dir, case_id)
                        if legacy_reference is None:
                            raise NonRetryableCaseError(
                                f"legacy artifact is missing for {case_id}")
                        _, reference_mask, _ = load_oof(legacy_reference)
                    assert_voxelwise_match(reference_mask, local_mask, case_id)
                hard_sha = sha256_file(existing)
            else:
                target = self.hard_dir / f"{case_id}.npz"
                hard_sha = atomic_copy(produced_hard, target,
                                       replace_invalid=self.force and target.exists())
                existing = target
            soft_sha, soft_target = None, None
            if want_softmax:
                produced_soft = soft_local / f"{case_id}.npz"
                if not softmax_cache_is_valid(produced_soft):
                    raise ValueError(f"invalid local official softmax: {produced_soft}")
                soft_target = self.softmax_dir / f"{case_id}.npz"
                if self.softmax_valid(case_id):
                    soft_sha = sha256_file(soft_target)
                else:
                    soft_sha = atomic_copy(produced_soft, soft_target,
                                           replace_invalid=self.force and soft_target.exists())
            provenance = self._provenance()
            previous = provenance["cases"].get(case_id, {})
            entry = {
                "prediction_fold": fold, "source_checkpoint": checkpoint,
                "source_checkpoint_sha256": checkpoint_sha,
                "artifact_type": kind, "hard_artifact": str(existing),
                "softmax_artifact": str(soft_target) if soft_target else None,
                "hard_sha256": hard_sha, "softmax_sha256": soft_sha,
                "predicted_at": utcnow(), "pinned_git_sha": self.git_sha,
            }
            verification_keys = (
                "verification_method", "legacy_voxelwise_match",
                "legacy_changed_voxels", "legacy_hard_artifact",
                "legacy_hard_sha256", "reproduced_hard_sha256", "verified_at")
            if verification == "legacy_reprediction_voxelwise":
                legacy = find_case_artifact(self.legacy_hard_dir, case_id)
                if legacy is None:
                    raise NonRetryableCaseError(
                        f"legacy artifact is missing for {case_id}")
                entry.update({
                    "verification_method": verification,
                    "legacy_voxelwise_match": True,
                    "legacy_changed_voxels": 0,
                    "legacy_hard_artifact": str(legacy),
                    "legacy_hard_sha256": sha256_file(legacy),
                    "reproduced_hard_sha256": sha256_file(produced_hard),
                    "verified_at": utcnow(),
                })
            else:
                entry.update({key: previous[key] for key in verification_keys if key in previous})
            provenance["cases"][case_id] = entry
            self._write_provenance(provenance)
            return {"checksums": {"hard": hard_sha, "softmax": soft_sha},
                    "source_checkpoint": checkpoint, "fold": fold}

    def _predict_softmax_fold(self, fold, case_ids):
        case_ids = list(case_ids)
        if not case_ids:
            return
        local_root = Path("/content/prompt1_softmax_work")
        if not Path("/content").is_dir():
            local_root = Path(tempfile.gettempdir()) / "prompt1_softmax_work"
        local_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_root) as hard_output:
            previous_results = os.environ.get("nnUNet_results")
            os.environ["nnUNet_results"] = str(self.results)
            try:
                predict_fold(
                    fold, case_ids, str(self.images), self.dataset_id, self.nn_config,
                    self.trainer, hard_output, self.device, npp=self.workers,
                    nps=self.workers, save_probabilities=True,
                    softmax_out=str(self.softmax_dir))
            finally:
                if previous_results is None:
                    os.environ.pop("nnUNet_results", None)
                else:
                    os.environ["nnUNet_results"] = previous_results
        invalid = [case_id for case_id in case_ids if not self.softmax_valid(case_id)]
        if invalid:
            raise RuntimeError(f"fold {fold} softmax batch left invalid cases: {invalid[:5]}")

    def complete_true_softmax_by_fold(self):
        """Optional future queue: load the predictor only once per missing fold."""
        for fold, split in enumerate(self.splits["folds"]):
            missing = [case_id for case_id in split["val"]
                       if not self.softmax_valid(case_id)]
            if missing:
                print(f"[softmax] fold={fold} missing={len(missing)}", flush=True)
                self._predict_softmax_fold(fold, missing)

    def run_smoke_oof(self, smoke_ids):
        smoke_ids = list(smoke_ids)
        completed = run_case_queue(
            "oof_smoke", smoke_ids, self.state,
            lambda case_id: self._predict_case(case_id, compare_existing=True),
            lambda case_id: self.hard_valid(case_id) and self.provenance_valid(case_id)
                            and (not self.export_softmax or self.softmax_valid(case_id)),
            self.fold_map, self.retry_failed, heartbeat_every=1)
        failed = self.state.data.get("failed_cases", {}).get("oof_smoke", {})
        if completed != len(smoke_ids) or failed:
            raise RuntimeError(f"2-case GPU smoke failed: {failed}")
        self.state.data["oof_smoke_passed"] = True
        self.state.data["timestamps"]["oof_smoke_passed_at"] = utcnow()
        self.state.save()

    def run_oof_full(self):
        queue = [case_id for case_id in self.case_ids
                 if not (self.hard_valid(case_id) and self.provenance_valid(case_id)
                         and (not self.export_softmax or self.softmax_valid(case_id)))]
        run_case_queue(
            "oof", queue, self.state, self._predict_case,
            lambda case_id: self.hard_valid(case_id) and self.provenance_valid(case_id)
                            and (not self.export_softmax or self.softmax_valid(case_id)),
            self.fold_map, self.retry_failed)
        return self.case_ids

    def run_oof(self):
        """Compatibility wrapper for callers that previously used the full queue."""
        return self.run_oof_full()

    def _compute_sdf_case(self, case_id):
        local_root = Path("/content/prompt1_work")
        if not Path("/content").is_dir():
            local_root = Path(tempfile.gettempdir()) / "prompt1_work"
        local_root.mkdir(parents=True, exist_ok=True)
        checksums = {}
        with tempfile.TemporaryDirectory(dir=local_root) as tmp:
            tmp = Path(tmp)
            if not self.gt_sdf_valid(case_id):
                gt_local = tmp / "gt"
                gt_local.mkdir()
                _, produced, _ = compute_gt_one(
                    (str(self.labels), f"{case_id}.nii.gz", str(gt_local), 10.0))
                target = self.gt_sdf_dir / f"{case_id}.npz"
                checksums["gt_sdf"] = atomic_copy(
                    produced, target, replace_invalid=self.force and target.exists())
            else:
                checksums["gt_sdf"] = sha256_file(self.gt_sdf_dir / f"{case_id}.npz")
            if not self.coarse_valid(case_id):
                hard_view = tmp / "hard"
                hard_view.mkdir()
                source = self.hard_path(case_id)
                if source is None:
                    raise FileNotFoundError(f"missing hard OOF for coarse SDF: {case_id}")
                local_hard = hard_view / f"{case_id}.npz"
                if source.name.endswith(".npz"):
                    local_hard.symlink_to(source.resolve())
                else:
                    _, hard_mask, _ = load_oof(source)
                    np.savez_compressed(
                        local_hard,
                        prob_left=(hard_mask == 1).astype(np.float16),
                        prob_right=(hard_mask == 2).astype(np.float16))
                coarse_local = tmp / "coarse"
                coarse_local.mkdir()
                _, produced, _ = compute_coarse_one(
                    (str(hard_view), str(self.labels), f"{case_id}.npz",
                     str(coarse_local), 0.5, 10.0))
                target = self.coarse_dir / f"{case_id}.npz"
                checksums["coarse_sdf"] = atomic_copy(
                    produced, target, replace_invalid=self.force and target.exists())
            else:
                checksums["coarse_sdf"] = sha256_file(self.coarse_dir / f"{case_id}.npz")
        return {"checksums": checksums}

    def run_sdf(self, case_ids):
        queue = [case_id for case_id in case_ids if not (
            self.coarse_valid(case_id) and self.gt_sdf_valid(case_id))]
        run_case_queue(
            "sdf", queue, self.state, self._compute_sdf_case,
            lambda case_id: self.coarse_valid(case_id) and self.gt_sdf_valid(case_id),
            self.fold_map, self.retry_failed)

    def _sdf_worker_count(self):
        configured = self.config.get("SDF_WORKERS")
        if configured is not None:
            return max(1, int(configured))
        cores = os.cpu_count() or 2
        return max(1, min(8, cores // 2))

    def complete_missing_sdf(self):
        inventory = self.build_inventory()
        audited_cases = {}
        if self.audit_480_path.is_file():
            audited_cases = self._load_stage_json(
                self.audit_480_path, "480-case OOF audit").get("cases", {})
            if isinstance(audited_cases, list):
                audited_cases = {row["case_id"]: row for row in audited_cases}
        selected = []
        for case_id in self.case_ids:
            inventory_case = inventory["cases"][case_id]
            files = inventory_case["files"]
            checks = inventory_case["geometry"]["checks"]
            audit_case = audited_cases.get(case_id, {})
            need_coarse = (files["coarse_sdf"]["status"] != "valid"
                           or not checks["coarse_shape"] or not checks["coarse_spacing"]
                           or (audit_case.get("hard_vs_sdf_sign_voxel_difference") or 0) != 0)
            need_gt = (files["gt_sdf"]["status"] != "valid"
                       or not checks["gt_sdf_shape"] or not checks["gt_sdf_spacing"])
            if need_coarse or need_gt:
                if files["label"]["status"] != "valid":
                    raise RuntimeError(f"cannot complete SDF without valid label: {case_id}")
                if need_coarse and files["hard_oof"]["status"] != "valid":
                    raise RuntimeError(f"cannot complete coarse SDF without valid hard OOF: {case_id}")
                selected.append((case_id, need_gt, need_coarse))
        current_inputs = {"final_inventory_sha256": sha256_file(self.inventory_path)}
        if not selected and self._receipt_valid(self.sdf_receipt_path, current_inputs):
            return json.loads(self.sdf_receipt_path.read_text()).get("summary", {})
        stage = "complete_missing_sdf"
        for case_id in list(self.state.data.get("failed_cases", {}).get(stage, {})):
            if self.coarse_valid(case_id) and self.gt_sdf_valid(case_id):
                self.state.complete(stage, case_id, {"status": "reconciled_valid"})
        workers = self._sdf_worker_count()
        local_root = Path("/content/prompt1_sdf_stage")
        if not Path("/content").is_dir():
            local_root = Path(tempfile.gettempdir()) / "prompt1_sdf_stage"
        local_root.mkdir(parents=True, exist_ok=True)
        completed = 0
        for batch_start in range(0, len(selected), self.sdf_batch_size):
            batch = selected[batch_start:batch_start + self.sdf_batch_size]
            with tempfile.TemporaryDirectory(dir=local_root) as temporary:
                temporary = Path(temporary)
                staged_labels = temporary / "labels"
                staged_hard = temporary / "hard"
                staged_output = temporary / "output"
                staged_labels.mkdir(); staged_hard.mkdir(); staged_output.mkdir()
                jobs = []
                for case_id, need_gt, need_coarse in batch:
                    shutil.copy2(self.labels / f"{case_id}.nii.gz",
                                 staged_labels / f"{case_id}.nii.gz")
                    if need_coarse:
                        source = self.hard_path(case_id)
                        local_hard = staged_hard / f"{case_id}.npz"
                        if source.name.endswith(".npz"):
                            shutil.copy2(source, local_hard)
                        else:
                            _, hard_mask, _ = load_oof(source)
                            np.savez_compressed(
                                local_hard,
                                prob_left=(hard_mask == 1).astype(np.float16),
                                prob_right=(hard_mask == 2).astype(np.float16))
                    jobs.append((case_id, staged_labels, staged_hard, staged_output,
                                 need_gt, need_coarse, 0.5, 10.0))
                print(f"[complete-missing-sdf] staged batch "
                      f"{batch_start // self.sdf_batch_size + 1}: {len(batch)} cases | "
                      f"workers={workers}", flush=True)
                with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
                    futures = {executor.submit(_compute_staged_sdf_job, job): job[0]
                               for job in jobs}
                    for future in as_completed(futures):
                        case_id = futures[future]
                        self.state.begin(stage, case_id, self.fold_map[case_id])
                        try:
                            result = future.result()
                            checksums = {}
                            if result["gt_sdf"]:
                                target = self.gt_sdf_dir / f"{case_id}.npz"
                                checksums["gt_sdf"] = atomic_copy(
                                    result["gt_sdf"], target, replace_invalid=target.exists())
                            if result["coarse_sdf"]:
                                target = self.coarse_dir / f"{case_id}.npz"
                                checksums["coarse_sdf"] = atomic_copy(
                                    result["coarse_sdf"], target, replace_invalid=target.exists())
                            if not (self.coarse_valid(case_id) and self.gt_sdf_valid(case_id)):
                                raise ValueError(f"published SDF validation failed for {case_id}")
                            self.state.complete(stage, case_id, {"checksums": checksums})
                            completed += 1
                            print(f"[complete-missing-sdf] {completed}/{len(selected)} "
                                  f"{case_id}", flush=True)
                        except Exception as error:
                            self.state.fail(stage, case_id, error)
                            print(f"[complete-missing-sdf] FAILED {case_id}: {error}", flush=True)
        unresolved = [case_id for case_id, _, _ in selected if not (
            self.coarse_valid(case_id) and self.gt_sdf_valid(case_id))]
        if unresolved:
            raise RuntimeError(f"SDF completion left invalid cases: {unresolved[:10]}")
        final_inventory = self.build_inventory()
        inputs = {"final_inventory_sha256": sha256_file(self.inventory_path)}
        summary = {"requested_cases": len(selected), "completed_cases": completed,
                   "batch_size": self.sdf_batch_size, "workers": workers,
                   "inventory_status_counts": final_inventory["summary"]["status_counts"]}
        self._write_stage_receipt(
            self.sdf_receipt_path, "complete-missing-sdf", inputs,
            {"inventory": self.inventory_path}, summary)
        return summary

    def _validate_smoke_cases(self, case_ids):
        invalid = [case_id for case_id in case_ids if not (
            self.hard_valid(case_id) and self.provenance_valid(case_id)
            and (not self.export_softmax or self.softmax_valid(case_id))
            and self.coarse_valid(case_id) and self.gt_sdf_valid(case_id))]
        if invalid:
            raise RuntimeError(f"smoke artifacts failed validation: {invalid}")

    def reconcile_state(self):
        validators = {
            "oof_smoke": lambda case_id: self.hard_valid(case_id)
            and self.provenance_valid(case_id)
            and (not self.export_softmax or self.softmax_valid(case_id)),
            "oof": lambda case_id: self.hard_valid(case_id)
            and self.provenance_valid(case_id)
            and (not self.export_softmax or self.softmax_valid(case_id)),
            "sdf": lambda case_id: self.coarse_valid(case_id) and self.gt_sdf_valid(case_id),
            "quick_provenance_bootstrap": self.bootstrap_provenance_valid,
            "full_provenance_bootstrap": self.bootstrap_provenance_valid,
        }
        for stage, failures in list(self.state.data.get("failed_cases", {}).items()):
            validator = validators.get(stage)
            if validator is None:
                continue
            for case_id in list(failures):
                if validator(case_id):
                    self.state.complete(stage, case_id, {"status": "reconciled_valid"})

    @staticmethod
    def _metric_summary(side_rows, path):
        names = ("dice", "cldice", "hd95", "volume_ratio", "components")
        result = {}
        for name in names:
            values = [row[path][name] for row in side_rows if row[path][name] is not None]
            result[name] = None if not values else float(np.mean(values))
        if result["dice"] is not None and result["cldice"] is not None:
            result["score"] = 0.5 * result["dice"] + 0.5 * result["cldice"]
        return result

    def _identity_payload(self, raw, manifest):
        cases = raw["cases"]
        side_rows = [{**side, "case_id": case["case_id"], "fold": case["expected_fold"]}
                     for case in cases for side in case["sides"]]
        paths = ("direct", "sdf_decode", "full_path")
        per_side = {path: self._metric_summary(side_rows, path) for path in paths}
        case_rows = []
        for case_id in self.case_ids:
            rows = [row for row in side_rows if row["case_id"] == case_id]
            combined = {"case_id": case_id, "fold": self.fold_map[case_id]}
            for path in paths:
                combined[path] = {}
                for name in ("dice", "cldice", "hd95", "volume_ratio", "components"):
                    values = [row[path][name] for row in rows if row[path][name] is not None]
                    combined[path][name] = None if not values else float(np.mean(values))
            case_rows.append(combined)
        per_case = {path: self._metric_summary(case_rows, path) for path in paths}
        folds = []
        for fold in range(len(self.splits["folds"])):
            rows = [row for row in side_rows if row["fold"] == fold]
            folds.append({"fold": fold, "cases": len({row["case_id"] for row in rows}),
                          **{path: self._metric_summary(rows, path) for path in paths}})
        first20 = {case_id for item in self.splits["folds"] for case_id in item["val"][:20]}
        first20_rows = [row for row in side_rows if row["case_id"] in first20]
        summary = raw["summary"]
        cache_summary = manifest["summary"]
        payload = {
            "created_at": utcnow(), "pinned_git_sha": self.git_sha,
            "complete_cv": (summary["decision"] == "PASS"
                            and len(cases) == EXPECTED_CASES
                            and cache_summary["valid_cases"] == EXPECTED_CASES
                            and cache_summary["missing_cases"] == 0
                            and cache_summary["invalid_cases"] == 0
                            and cache_summary["fold_provenance_errors"] == 0),
            "evaluated_cases": len(cases), "cache_valid_cases": cache_summary["valid_cases"],
            "missing_cases": cache_summary["missing_cases"],
            "invalid_cases": cache_summary["invalid_cases"],
            "fold_provenance_errors": cache_summary["fold_provenance_errors"],
            "direct_vs_sdf_voxel_difference": summary["direct_vs_sdf_voxel_difference"],
            "direct_vs_full_path_voxel_difference": summary["direct_vs_full_path_voxel_difference"],
            "overall": {"per_side": per_side, "per_case": per_case},
            "folds": folds,
            "hypotheses": {
                "H1_val_max_cases_20": {
                    "first_20_per_fold": {path: self._metric_summary(first20_rows, path)
                                           for path in paths},
                    "full_cv": per_side},
                "H2_metric_aggregation": {"per_side": per_side, "per_case": per_case},
                "H3_hard_to_sdf_sign_roundtrip": {
                    "changed_voxels": summary["direct_vs_sdf_voxel_difference"]},
                "H4_full_path_identity": {
                    "changed_voxels": summary["direct_vs_full_path_voxel_difference"],
                    "max_sdf_absolute_error": summary["max_full_path_sdf_abs_error"]},
            },
            "softmax_status": {
                "requested": self.export_softmax,
                "valid_cases": cache_summary["true_softmax_valid_cases"],
                "required_for_hard_identity_floor": False},
        }
        export_rows = [
            {"case_id": row["case_id"], "fold": row["fold"],
             "aggregation": "per_side", "side": row["side"],
             **{path: row[path] for path in paths}}
            for row in side_rows]
        export_rows.extend(
            {"case_id": row["case_id"], "fold": row["fold"],
             "aggregation": "per_case", "side": "bilateral_mean",
             **{path: row[path] for path in paths}}
            for row in case_rows)
        return payload, export_rows, folds

    def _validate_final_audit(self):
        audit = self._load_stage_json(self.audit_480_path, "480-case OOF audit")
        summary = audit.get("summary", {})
        if summary.get("total_cases") != EXPECTED_CASES:
            raise RuntimeError(
                f"final audit requires {EXPECTED_CASES} cases, got {summary.get('total_cases')}")
        if summary.get("status_counts") != {"valid": EXPECTED_CASES}:
            raise RuntimeError(
                f"final audit does not have {EXPECTED_CASES} valid cases: "
                f"{summary.get('status_counts')}")
        if summary.get("provenance_errors", 0) != 0:
            raise RuntimeError(f"final audit has provenance errors: {summary['provenance_errors']}")
        if summary.get("split_leakage_errors", 0) != 0:
            raise RuntimeError(
                f"final audit has split leakage errors: {summary['split_leakage_errors']}")
        if summary.get("hard_vs_sdf_sign_voxel_difference", 0) != 0:
            raise RuntimeError("final audit hard-to-SDF round-trip changed voxels")
        return audit

    def _write_identity_report(self, payload, finalized=False):
        report = ["# Full-CV identity prior\n\n",
                  f"- complete_cv: `{payload.get('complete_cv', False)}`\n",
                  f"- identity_complete: `{payload.get('identity_complete', False)}`\n",
                  f"- cases: {payload.get('evaluated_cases')}/{EXPECTED_CASES}\n",
                  f"- provenance: `legacy_import`\n",
                  f"- cross-runtime diagnostic: `near_exact`, exact reproduction claim: `false`\n",
                  f"- direct vs SDF changed voxels: "
                  f"{payload.get('direct_vs_sdf_voxel_difference')}\n",
                  f"- direct vs full-path changed voxels: "
                  f"{payload.get('direct_vs_full_path_voxel_difference')}\n",
                  f"- finalized: `{finalized}`\n\n",
                  "| path | Dice | clDice | HD95 (mm) | score |\n",
                  "|---|---:|---:|---:|---:|\n"]
        for path in ("direct", "sdf_decode", "full_path"):
            values = payload.get("overall", {}).get("per_side", {}).get(path, {})
            def fmt(name):
                value = values.get(name)
                return "NA" if value is None else f"{value:.6f}"
            report.append(f"| {path} | {fmt('dice')} | {fmt('cldice')} | "
                          f"{fmt('hd95')} | {fmt('score')} |\n")
        report.append("\nThe non-inferiority margin remains null pending an explicit user decision.\n")
        self._atomic_text(self.baseline_dir / "identity_prior_report.md", "".join(report))

    def identity_480(self):
        audit = self._validate_final_audit()
        inputs = {"inventory_sha256": sha256_file(self.inventory_path),
                  "provenance_sha256": sha256_file(self.provenance_path),
                  "audit_sha256": sha256_file(self.audit_480_path)}
        final_json = self.baseline_dir / "identity_prior.json"
        if self._receipt_valid(self.identity_480_receipt_path, inputs):
            return self._load_stage_json(final_json, "identity prior")
        view = self.create_hard_view(self.case_ids, "prompt1_hard_view_identity_480")
        raw_prefix = self.baseline_dir / "identity_prior_work"
        raw_json = raw_prefix.with_suffix(".json")
        cmd = [sys.executable, str(ROOT / "scripts/identity_preflight.py"),
               "--splits", str(self.splits_path), "--images", str(self.images),
               "--labels", str(self.labels), "--oof-hard", str(view),
               "--coarse-sdf", str(self.coarse_dir), "--gt-sdf", str(self.gt_sdf_dir),
               "--provenance-manifest", str(self.provenance_path),
               "--inventory", str(self.inventory_path),
               "--expected-cases", str(EXPECTED_CASES), "--device", self.device,
               "--out-prefix", str(raw_prefix),
               "--state", str(self.prompt_dir / "identity_480_state.json"),
               "--resume", "--replace-reports"]
        subprocess.run(cmd, cwd=ROOT, check=True)
        raw = self._load_stage_json(raw_json, "identity work report")
        if raw.get("summary", {}).get("decision") != "PASS":
            raise RuntimeError(f"identity-480 failed: {raw.get('summary', {}).get('decision')}")
        audit_summary = audit["summary"]
        status_counts = audit_summary["status_counts"]
        manifest = {"summary": {
            "valid_cases": status_counts.get("valid", 0),
            "missing_cases": status_counts.get("missing", 0),
            "invalid_cases": status_counts.get("invalid", 0),
            "fold_provenance_errors": audit_summary.get("provenance_errors", 0),
            "true_softmax_valid_cases": 0}}
        payload, case_rows, folds = self._identity_payload(raw, manifest)
        payload["identity_complete"] = bool(
            len(raw.get("cases", [])) == EXPECTED_CASES
            and raw["summary"].get("decision") == "PASS")
        payload["complete_cv"] = False
        payload["provenance_source"] = "legacy_import"
        payload["true_softmax_required"] = False
        payload["cross_runtime_validation"] = self._provenance().get(
            "cross_runtime_validation")
        atomic_json(final_json, payload)
        self._write_csv(self.baseline_dir / "identity_prior_cases.csv", case_rows)
        self._write_csv(self.baseline_dir / "identity_prior_folds.csv", folds)
        self._write_identity_report(payload, finalized=False)
        raw_csv = raw_prefix.parent / f"{raw_prefix.name}_cases.csv"
        self._write_stage_receipt(
            self.identity_480_receipt_path, "identity-480", inputs,
            {"raw_identity_json": raw_json, "raw_identity_csv": raw_csv,
             "raw_identity_report": raw_prefix.with_suffix(".md"),
             "identity_state": self.prompt_dir / "identity_480_state.json"},
            {"decision": raw["summary"]["decision"],
             "evaluated_cases": len(raw.get("cases", []))})
        self.state.data["current_stage"] = "identity_480_complete"
        self.state.data["complete_cv"] = False
        self.state.save()
        return payload

    def finalize_prompt1(self):
        current_inventory_sha = sha256_file(self.inventory_path)
        current_provenance_sha = sha256_file(self.provenance_path)
        current_audit_sha = sha256_file(self.audit_480_path)
        upstream = {
            "inventory_receipt": self.inventory_receipt_path,
            "legacy_import_receipt": self.legacy_import_receipt_path,
            "audit_receipt": self.audit_480_receipt_path,
            "sdf_receipt": self.sdf_receipt_path,
            "identity_receipt": self.identity_480_receipt_path,
        }
        expected_inputs = {
            "inventory_receipt": {"splits_sha256": sha256_file(self.splits_path)},
            "audit_receipt": {"inventory_sha256": current_inventory_sha,
                              "provenance_sha256": current_provenance_sha,
                              "splits_sha256": sha256_file(self.splits_path)},
            "sdf_receipt": {"final_inventory_sha256": current_inventory_sha},
            "identity_receipt": {"inventory_sha256": current_inventory_sha,
                                 "provenance_sha256": current_provenance_sha,
                                 "audit_sha256": current_audit_sha},
        }
        invalid_receipts = [
            name for name, path in upstream.items()
            if not self._receipt_valid(path, expected_inputs.get(name))]
        if invalid_receipts:
            raise RuntimeError(f"finalize-prompt1 invalid receipts: {invalid_receipts}")
        inputs = {name: sha256_file(path) for name, path in upstream.items()}
        if self._receipt_valid(self.final_receipt_path, inputs):
            return self._load_stage_json(
                self.baseline_dir / "identity_prior.json", "final identity prior")
        inventory = self._load_stage_json(self.inventory_path, "cache inventory")
        inventory_counts = inventory.get("summary", {}).get("status_counts", {})
        if (inventory.get("summary", {}).get("total_cases") != EXPECTED_CASES
                or inventory_counts.get("valid") != EXPECTED_CASES
                or inventory_counts.get("missing", 0) != 0
                or inventory_counts.get("invalid", 0) != 0):
            raise RuntimeError(f"final inventory is not {EXPECTED_CASES} valid cases")
        provenance = self._load_stage_json(self.provenance_path, "legacy provenance import")
        if provenance.get("summary", {}).get("imported_cases") != EXPECTED_CASES:
            raise RuntimeError(f"legacy provenance import is not {EXPECTED_CASES} cases")
        self._validate_final_audit()
        identity_path = self.baseline_dir / "identity_prior.json"
        payload = self._load_stage_json(identity_path, "identity prior")
        if (not payload.get("identity_complete")
                or payload.get("evaluated_cases") != EXPECTED_CASES
                or payload.get("direct_vs_sdf_voxel_difference") != 0
                or payload.get("direct_vs_full_path_voxel_difference") != 0):
            raise RuntimeError("identity prior is not a passing 480-case three-path result")
        payload["complete_cv"] = True
        payload["finalized_at"] = utcnow()
        payload["finalization_receipts"] = inputs
        atomic_json(identity_path, payload)
        self._write_identity_report(payload, finalized=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/identity_baseline.py"),
                        "--write-config", "--write-config-from-report", str(identity_path),
                        "--config", str(ROOT / "configs/flow.yaml")], cwd=ROOT, check=True)
        config_text = (ROOT / "configs/flow.yaml").read_text()
        if "noninferiority_margin: null" not in config_text:
            raise RuntimeError("noninferiority_margin must remain null during Prompt-1 finalization")
        frozen_config = self.prompt_dir / "flow_with_identity_prior.yaml"
        atomic_copy(ROOT / "configs/flow.yaml", frozen_config, replace_invalid=True)
        self.state.data["complete_cv"] = True
        self.state.data["current_stage"] = "complete"
        self.state.data.setdefault("timestamps", {})["completed_at"] = utcnow()
        self.state.save()
        self._write_stage_receipt(
            self.final_receipt_path, "finalize-prompt1", inputs,
            {"identity_prior": identity_path,
             "identity_report": self.baseline_dir / "identity_prior_report.md",
             "flow_config": frozen_config},
            {"complete_cv": True, "evaluated_cases": EXPECTED_CASES,
             "noninferiority_margin": None})
        print("[finalize-prompt1] complete_cv=true; noninferiority_margin=null", flush=True)
        return payload

    def run_identity(self, manifest):
        gate = manifest["summary"]
        required = {"valid_cases": EXPECTED_CASES, "missing_cases": 0,
                    "invalid_cases": 0, "fold_provenance_errors": 0,
                    "external_s_cases_in_manifest": 0}
        failed = {name: (gate.get(name), expected) for name, expected in required.items()
                  if gate.get(name) != expected}
        if failed:
            raise RuntimeError(f"full identity gate failed: {failed}")
        view = self.create_hard_view()
        raw_prefix = self.baseline_dir / "identity_prior_work"
        raw_json = raw_prefix.with_suffix(".json")
        if not raw_json.is_file():
            cmd = [sys.executable, str(ROOT / "scripts/identity_preflight.py"),
                   "--splits", str(self.splits_path), "--images", str(self.images),
                   "--labels", str(self.labels), "--oof-hard", str(view),
                   "--coarse-sdf", str(self.coarse_dir), "--gt-sdf", str(self.gt_sdf_dir),
                   "--provenance-manifest", str(self.provenance_path),
                   "--expected-cases", str(EXPECTED_CASES), "--device", self.device,
                   "--out-prefix", str(raw_prefix),
                   "--state", str(self.prompt_dir / "identity_prior_state.json"), "--resume"]
            subprocess.run(cmd, cwd=ROOT, check=True)
        raw = json.loads(raw_json.read_text())
        if raw.get("summary", {}).get("decision") != "PASS":
            raise RuntimeError(f"full identity failed: {raw.get('summary', {}).get('decision')}")
        payload, case_rows, folds = self._identity_payload(raw, manifest)
        if not payload["complete_cv"] or payload["evaluated_cases"] != EXPECTED_CASES:
            raise RuntimeError("identity payload is not complete CV")
        final_json = self.baseline_dir / "identity_prior.json"
        atomic_json(final_json, payload)
        self._write_csv(self.baseline_dir / "identity_prior_cases.csv", case_rows)
        self._write_csv(self.baseline_dir / "identity_prior_folds.csv", folds)
        report = ["# Full-CV identity prior\n\n",
                  f"- complete_cv: `{payload['complete_cv']}`\n",
                  f"- cases: {payload['evaluated_cases']}/{EXPECTED_CASES}\n",
                  f"- cache: valid={payload['cache_valid_cases']}, "
                  f"missing={payload['missing_cases']}, invalid={payload['invalid_cases']}\n",
                  f"- fold provenance errors: {payload['fold_provenance_errors']}\n",
                  f"- direct vs SDF changed voxels: {payload['direct_vs_sdf_voxel_difference']}\n",
                  f"- direct vs full-path changed voxels: "
                  f"{payload['direct_vs_full_path_voxel_difference']}\n\n",
                  "| path | aggregation | Dice | clDice | HD95 (mm) | score |\n",
                  "|---|---|---:|---:|---:|---:|\n"]
        for path in ("direct", "sdf_decode", "full_path"):
            for aggregation in ("per_side", "per_case"):
                values = payload["overall"][aggregation][path]
                def metric(name):
                    value = values.get(name)
                    return "NA" if value is None else f"{value:.6f}"
                report.append(
                    f"| {path} | {aggregation} | {metric('dice')} | "
                    f"{metric('cldice')} | {metric('hd95')} | {metric('score')} |\n")
        report.extend(["\n",
                  "The non-inferiority margin remains null pending an explicit user decision.\n"]
        )
        self._atomic_text(self.baseline_dir / "identity_prior_report.md", "".join(report))
        subprocess.run([sys.executable, str(ROOT / "scripts/identity_baseline.py"),
                        "--write-config", "--write-config-from-report", str(final_json),
                        "--config", str(ROOT / "configs/flow.yaml")], cwd=ROOT, check=True)
        config_text = (ROOT / "configs/flow.yaml").read_text()
        if "noninferiority_margin: null" not in config_text:
            raise RuntimeError("identity write unexpectedly changed the non-inferiority margin")
        atomic_copy(ROOT / "configs/flow.yaml", self.prompt_dir / "flow_with_identity_prior.yaml",
                    replace_invalid=True)
        self.state.data["complete_cv"] = True
        self.state.data["current_stage"] = "complete"
        self.state.data["timestamps"]["completed_at"] = utcnow()
        self.state.save()

    @staticmethod
    def _atomic_text(path, content):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        partial.write_text(content)
        os.replace(partial, path)

    @classmethod
    def _write_csv(cls, path, rows):
        flattened = []
        for row in rows:
            flattened.append({key: (json.dumps(value, sort_keys=True)
                                    if isinstance(value, (dict, list)) else value)
                              for key, value in row.items()})
        if not flattened:
            raise ValueError(f"cannot write empty CSV: {path}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        with partial.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
        os.replace(partial, path)

    def smoke(self):
        self.preflight_quick()
        selected = self.case_ids[:2]
        if len(selected) != 2:
            raise RuntimeError(f"smoke requires two development cases, found {len(selected)}")
        self.run_smoke_oof(selected)
        self.run_sdf(selected)
        self._validate_smoke_cases(selected)
        self.state.data["current_stage"] = "smoke_complete"
        self.state.data["complete_cv"] = False
        self.state.data.setdefault("timestamps", {})["smoke_completed_at"] = utcnow()
        self.state.save()
        print("[prompt1] two-case smoke/resume validation: PASS", flush=True)

    def run_full(self):
        self.build_inventory()
        self.import_legacy_provenance()
        self.audit_480()
        self.complete_missing_sdf()
        self.build_inventory()
        self.audit_480()
        self.identity_480()
        self.finalize_prompt1()
        print("[prompt1] resumable 480-case Prompt-1 workflow complete", flush=True)

    def run(self):
        """Backward-compatible programmatic alias for the complete workflow."""
        return self.run_full()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON written by the Colab config cell")
    parser.add_argument("action", choices=(
        "build-inventory", "import-legacy-provenance", "audit-480",
        "complete-missing-sdf", "identity-480", "finalize-prompt1",
        "complete-true-softmax",
        "preflight-quick", "smoke", "preflight-full", "run-full", "manifest",
        "preflight", "run"))
    args = parser.parse_args()
    runner = Prompt1Runner(json.loads(Path(args.config).read_text()))
    if args.action == "build-inventory":
        result = runner.build_inventory()
        print(json.dumps(result["summary"], indent=2))
    elif args.action == "import-legacy-provenance":
        result = runner.import_legacy_provenance()
        print(json.dumps(result["summary"], indent=2))
    elif args.action == "audit-480":
        result = runner.audit_480()
        print(json.dumps(result["summary"], indent=2))
    elif args.action == "complete-missing-sdf":
        print(json.dumps(runner.complete_missing_sdf(), indent=2))
    elif args.action == "identity-480":
        result = runner.identity_480()
        print(json.dumps({"identity_complete": result.get("identity_complete"),
                          "complete_cv": result.get("complete_cv")}, indent=2))
    elif args.action == "finalize-prompt1":
        result = runner.finalize_prompt1()
        print(json.dumps({"complete_cv": result.get("complete_cv")}, indent=2))
    elif args.action == "complete-true-softmax":
        runner.complete_true_softmax_by_fold()
    elif args.action == "preflight-quick":
        runner.preflight_quick()
    elif args.action == "smoke":
        runner.smoke()
    elif args.action == "preflight-full":
        runner.preflight_full()
    elif args.action == "run-full":
        runner.run_full()
    elif args.action == "manifest":
        result = runner.build_manifest()
        print(json.dumps(result["summary"], indent=2))
    elif args.action == "preflight":
        print("[warning] 'preflight' is deprecated; using 'preflight-full'", file=sys.stderr)
        runner.preflight_full()
    else:
        print("[warning] 'run' is deprecated; using 'run-full'", file=sys.stderr)
        runner.run_full()


if __name__ == "__main__":
    main()
