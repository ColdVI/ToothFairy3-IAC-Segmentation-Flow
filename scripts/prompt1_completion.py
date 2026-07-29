#!/usr/bin/env python3
"""Drive-backed, case-resumable completion runner for the Prompt-1 barrier.

This script never trains Track A or Track B. It reuses frozen fold checkpoints
for leakage-free OOF inference, completes physical-SDF caches, validates all 480
development cases, and measures the three-path identity baseline. Persistent
artifacts are published through ``.partial`` files and state is saved per case.
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
from datetime import datetime, timezone
from pathlib import Path

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
                                resolve_checkpoint, softmax_cache_is_valid)

EXPECTED_CASES = 480
MAX_AUTOMATIC_RETRIES = 2
HEARTBEAT_EVERY = 10


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
                   skip_initially_valid=True):
    """Run a case queue; a failed case never prevents later cases from running."""
    cases = list(cases)
    completed = 0
    for case_id in cases:
        if skip_initially_valid and validator(case_id):
            state.complete(stage, case_id, {"status": "already_valid"})
            completed += 1
            if completed % heartbeat_every == 0 or completed == len(cases):
                state.heartbeat(completed, len(cases))
            continue
        while True:
            state.begin(stage, case_id, (fold_map or {}).get(case_id))
            try:
                result = worker(case_id) or {}
                if not validator(case_id):
                    raise ValueError(f"worker returned but {stage} artifact is invalid")
                state.complete(stage, case_id, result)
                completed += 1
                break
            except KeyboardInterrupt:
                state.save()
                raise
            except Exception as error:  # continue queue after bounded retries
                failures = state.fail(stage, case_id, error)
                retries_used = failures - 1
                if not retry_failed or retries_used >= MAX_AUTOMATIC_RETRIES:
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
        self.state_path = self.output_root / "prompt1_completion_state.json"
        for directory in (self.hard_dir, self.softmax_dir, self.coarse_dir,
                          self.gt_sdf_dir, self.prompt_dir, self.analysis_dir,
                          self.baseline_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.splits_path = ROOT / "configs" / "splits.json"
        self.splits = json.loads(self.splits_path.read_text())
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

    def hard_path(self, case_id):
        return (find_case_artifact(self.hard_dir, case_id)
                or find_case_artifact(self.legacy_hard_dir, case_id))

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
            _, _, actual_type = load_oof(actual)
        except Exception:
            return False
        return (case.get("prediction_fold") == self.fold_map[case_id]
                and bool(checkpoint) and Path(checkpoint).is_file()
                and case.get("artifact_type") == actual_type
                and actual_type in ("hard_segmentation", "derived_one_hot")
                and recorded_matches and checksum_matches)

    def create_hard_view(self):
        view = Path("/content/prompt1_hard_view")
        if not Path("/content").is_dir():
            view = Path(tempfile.gettempdir()) / "prompt1_hard_view"
        view.mkdir(parents=True, exist_ok=True)
        for case_id in self.case_ids:
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

    def preflight(self):
        receipt_path = self.prompt_dir / "identity_preflight_40_receipt.json"
        ids_path = self.prompt_dir / "identity_preflight_40_case_ids.json"
        audit_prefix = self.analysis_dir / "oof_prior_audit"
        identity_prefix = self.baseline_dir / "identity_preflight_40"
        audit_json = audit_prefix.with_suffix(".json")
        identity_json = identity_prefix.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
        if receipt:
            if receipt.get("pinned_git_sha") != self.git_sha:
                raise RuntimeError("preflight receipt belongs to a different pinned git SHA")
            ids = receipt.get("case_ids", [])
        elif ids_path.is_file() and audit_json.is_file() and identity_json.is_file():
            ids = json.loads(ids_path.read_text())
        else:
            ids = self.complete_cache_ids()
            if len(ids) != 40:
                raise RuntimeError(
                    f"initial preflight requires exactly 40 complete cases; found {len(ids)}")
            atomic_json(ids_path, ids)
        if len(ids) != 40 or len(set(ids)) != 40:
            raise RuntimeError("preflight cohort receipt is not 40 unique cases")
        currently_complete = set(self.complete_cache_ids())
        incomplete = [case_id for case_id in ids if case_id not in currently_complete]
        if incomplete:
            raise RuntimeError(f"sealed preflight artifacts became invalid: {incomplete[:5]}")
        bad_provenance = [case_id for case_id in ids if not self.provenance_valid(case_id)]
        if bad_provenance:
            raise RuntimeError("legacy hard OOF provenance is absent or invalid; refusing to infer it "
                               f"from filenames (first: {bad_provenance[:5]})")
        view = self.create_hard_view()
        if not audit_json.is_file():
            cmd = [sys.executable, str(ROOT / "analysis/oof_prior_audit.py"),
                   "--splits", str(self.splits_path), "--images", str(self.images),
                   "--labels", str(self.labels), "--oof-hard", str(view),
                   "--oof-softmax", str(self.softmax_dir), "--coarse-sdf", str(self.coarse_dir),
                   "--gt-sdf", str(self.gt_sdf_dir), "--provenance-manifest", str(self.provenance_path),
                   "--case-ids", str(ids_path), "--out-prefix", str(audit_prefix),
                   "--state", str(self.prompt_dir / "oof_audit_40_state.json"), "--resume"]
            subprocess.run(cmd, cwd=ROOT, check=True)
        audit = json.loads(audit_json.read_text())["summary"]
        if audit.get("status_counts") != {"valid": 40}:
            raise RuntimeError(f"OOF audit failed: {audit.get('status_counts')}")
        if not identity_json.is_file():
            cmd = [sys.executable, str(ROOT / "scripts/identity_preflight.py"),
                   "--splits", str(self.splits_path), "--images", str(self.images),
                   "--labels", str(self.labels), "--oof-hard", str(view),
                   "--coarse-sdf", str(self.coarse_dir), "--gt-sdf", str(self.gt_sdf_dir),
                   "--provenance-manifest", str(self.provenance_path), "--case-ids", str(ids_path),
                   "--expected-cases", "40", "--device", self.device,
                   "--out-prefix", str(identity_prefix),
                   "--state", str(self.prompt_dir / "identity_preflight_40_state.json"), "--resume"]
            subprocess.run(cmd, cwd=ROOT, check=True)
        decision = json.loads(identity_json.read_text())["summary"]["decision"]
        if decision != "PASS":
            raise RuntimeError(f"identity preflight failed: {decision}")
        checksums = {case_id: {
            "hard": sha256_file(self.hard_path(case_id)),
            "coarse_sdf": sha256_file(self.coarse_dir / f"{case_id}.npz"),
            "gt_sdf": sha256_file(self.gt_sdf_dir / f"{case_id}.npz")}
            for case_id in ids}
        if receipt and receipt.get("checksums") != checksums:
            raise RuntimeError("a sealed 40-case preflight artifact changed after acceptance")
        atomic_json(receipt_path, {"pinned_git_sha": self.git_sha,
                                   "passed_at": receipt.get("passed_at") if receipt else utcnow(),
                                   "case_ids": ids, "checksums": checksums,
                                   "audit_decision": "PASS", "identity_decision": "PASS"})
        self.state.data["timestamps"]["preflight_passed_at"] = utcnow()
        self.state.save()
        print("[preflight] 40-case OOF audit and three-path identity: PASS", flush=True)

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

    def _predict_case(self, case_id, compare_existing=False):
        fold = self.fold_map[case_id]
        checkpoint = resolve_checkpoint(self.results, self.dataset_id, self.trainer,
                                        self.nn_config, fold)
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
            if self.export_softmax:
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
                if compare_existing and not np.array_equal(local_mask, old_mask):
                    raise ValueError(f"smoke prediction differs from existing hard OOF for {case_id}")
                hard_sha = sha256_file(existing)
            else:
                target = self.hard_dir / f"{case_id}.npz"
                hard_sha = atomic_copy(produced_hard, target,
                                       replace_invalid=self.force and target.exists())
                existing = target
            soft_sha, soft_target = None, None
            if self.export_softmax:
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
            provenance["cases"][case_id] = {
                "prediction_fold": fold, "source_checkpoint": checkpoint,
                "artifact_type": kind, "hard_artifact": str(existing),
                "softmax_artifact": str(soft_target) if soft_target else None,
                "hard_sha256": hard_sha, "softmax_sha256": soft_sha,
                "predicted_at": utcnow(), "pinned_git_sha": self.git_sha,
            }
            self._write_provenance(provenance)
            return {"checksums": {"hard": hard_sha, "softmax": soft_sha},
                    "source_checkpoint": checkpoint, "fold": fold}

    def run_oof(self):
        smoke_ids = self.case_ids[:2]
        if not self.state.data.get("oof_smoke_passed"):
            completed = run_case_queue(
                "oof_smoke", smoke_ids, self.state,
                lambda case_id: self._predict_case(case_id, compare_existing=True),
                lambda case_id: self.hard_valid(case_id) and self.provenance_valid(case_id)
                                and (not self.export_softmax or self.softmax_valid(case_id)),
                self.fold_map, self.retry_failed, heartbeat_every=2,
                skip_initially_valid=False)
            failed = self.state.data.get("failed_cases", {}).get("oof_smoke", {})
            if completed != 2 or failed:
                raise RuntimeError(f"2-case GPU smoke failed: {failed}")
            self.state.data["oof_smoke_passed"] = True
            self.state.data["timestamps"]["oof_smoke_passed_at"] = utcnow()
            self.state.save()
        if self.max_cases is not None:
            return smoke_ids[:self.max_cases]
        queue = [case_id for case_id in self.case_ids
                 if not (self.hard_valid(case_id) and self.provenance_valid(case_id)
                         and (not self.export_softmax or self.softmax_valid(case_id)))]
        run_case_queue(
            "oof", queue, self.state, self._predict_case,
            lambda case_id: self.hard_valid(case_id) and self.provenance_valid(case_id)
                            and (not self.export_softmax or self.softmax_valid(case_id)),
            self.fold_map, self.retry_failed)
        return self.case_ids

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

    def reconcile_state(self):
        validators = {
            "oof_smoke": lambda case_id: self.hard_valid(case_id)
            and self.provenance_valid(case_id)
            and (not self.export_softmax or self.softmax_valid(case_id)),
            "oof": lambda case_id: self.hard_valid(case_id)
            and self.provenance_valid(case_id)
            and (not self.export_softmax or self.softmax_valid(case_id)),
            "sdf": lambda case_id: self.coarse_valid(case_id) and self.gt_sdf_valid(case_id),
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

    def run(self):
        self.preflight()
        self.build_manifest()
        selected = self.run_oof()
        self.run_sdf(selected)
        manifest = self.build_manifest()
        if self.max_cases is not None:
            self.state.data["current_stage"] = "smoke_complete"
            self.state.data["complete_cv"] = False
            self.state.save()
            print(f"[prompt1] smoke mode complete for {len(selected)} cases; "
                  "set MAX_CASES=None and RUN/RESUME for full completion", flush=True)
            return
        self.reconcile_state()
        failed = self.state.data.get("failed_cases", {})
        unresolved = {stage: values for stage, values in failed.items() if values}
        if unresolved:
            self.state.data["complete_cv"] = False
            self.state.save()
            raise RuntimeError(f"unresolved failed cases remain: {unresolved}")
        self.run_identity(manifest)
        print("[prompt1] complete-CV identity outputs written; Prompt-1 evidence is ready", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON written by the Colab config cell")
    parser.add_argument("action", choices=("preflight", "manifest", "run"))
    args = parser.parse_args()
    runner = Prompt1Runner(json.loads(Path(args.config).read_text()))
    if args.action == "preflight":
        runner.preflight()
    elif args.action == "manifest":
        result = runner.build_manifest()
        print(json.dumps(result["summary"], indent=2))
    else:
        runner.run()


if __name__ == "__main__":
    main()
