#!/usr/bin/env python3
"""Create and atomically update reproducibility manifests for Track B runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(config, fold, seed):
    """Hash every input that changes a run's data split or stochastic trajectory."""
    resolved = {"config": config, "fold": int(fold), "seed": int(seed)}
    return hashlib.sha256(_canonical_json(resolved).encode()).hexdigest()[:16]


def _git_info(repo_root):
    def run(*args):
        result = subprocess.run(["git", *args], cwd=repo_root, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    return sha, None if status is None else bool(status)


def _environment():
    try:
        import torch
        torch_version = torch.__version__
        if torch.cuda.is_available():
            accelerator = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "Apple Metal Performance Shaders (MPS)"
        else:
            accelerator = None
    except ImportError:
        torch_version, accelerator = None, None
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "torch": torch_version, "accelerator": accelerator,
            "platform": platform.platform()}


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(partial, path)


def start_manifest(out_dir, config, fold, seed, repo_root, resume=False,
                   channel_contract=None):
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    sha, dirty = _git_info(repo_root)
    digest = config_hash(config, fold, seed)
    previous = None
    if resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
    payload = {
        "git_commit": sha,
        "git_dirty": dirty,
        "resolved_config": config,
        "channel_contract": channel_contract,
        "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
        "config_hash": digest,
        "fold": int(fold),
        "seed": int(seed),
        "environment": _environment(),
        "start_time": previous.get("start_time") if previous else _utc_now(),
        "last_resume_time": _utc_now() if previous else None,
        "resume_count": int(previous.get("resume_count", 0)) + 1 if previous else 0,
        "end_time": None,
        "status": "running",
        "metrics": previous.get("metrics") if previous else None,
    }
    atomic_write_json(manifest_path, payload)
    return payload


def finish_manifest(out_dir, status, metrics=None):
    path = Path(out_dir) / "manifest.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text())
    payload["status"] = status
    payload["end_time"] = _utc_now()
    if metrics is not None:
        payload["metrics"] = metrics
    atomic_write_json(path, payload)


def default_run_dir(runs_root, config, fold, seed):
    return str(Path(runs_root) / config_hash(config, fold, seed))
