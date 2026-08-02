#!/usr/bin/env python3
"""Fail-closed Drive/data/provenance audit before Prompt-3R GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from flow.prompt3r_config import resolve_prompt3r_config
from scripts.run_manifest import atomic_write_json, config_hash


PROMPT2_FILES = (
    "shortcut_probe_manifest.json", "shortcut_probe_summary.json",
    "shortcut_probe.csv", "thickening_probe.csv",
    "limited_endpoint_diagnostic.pdf",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_unique(root, name):
    matches = [path for path in Path(root).rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(matches)}: {matches}")
    return matches[0]


def resolve_identity(root, explicit=None):
    path = Path(explicit) if explicit else _find_unique(root, "identity_prior.json")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def resolve_prompt2(root, explicit=None):
    if explicit:
        directory = Path(explicit)
        candidates = [directory]
    else:
        candidates = [path.parent for path in Path(root).rglob("shortcut_probe_manifest.json")]
    valid = [directory.resolve() for directory in candidates
             if all((directory / name).is_file() for name in PROMPT2_FILES)]
    if len(valid) != 1:
        raise RuntimeError(f"expected one complete Prompt-2 directory, found {valid}")
    return valid[0]


def audit_prompt2(directory, identity_sha):
    manifest_path = directory / "shortcut_probe_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required_flags = {
        "protocol_deviation": True,
        "exact_epoch_trajectory_available": False,
        "historical_per_epoch_checkpoints_were_not_saved": True,
        "diagnostic_only": True,
    }
    mismatch = {key: (manifest.get(key), expected)
                for key, expected in required_flags.items()
                if manifest.get(key) is not expected}
    if mismatch:
        raise RuntimeError(f"Prompt-2 protocol flags mismatch: {mismatch}")
    hashes = {name: sha256_file(directory / name) for name in PROMPT2_FILES}
    declared = manifest.get("artifacts", {})
    for name in PROMPT2_FILES[1:]:
        if declared.get(name) != hashes[name]:
            raise RuntimeError(f"Prompt-2 artifact checksum mismatch: {name}")
    identity = manifest.get("identity_baseline", {})
    if identity.get("sha256") != identity_sha:
        raise RuntimeError("Prompt-2 and current Prompt-1 identity SHA differ")
    summary = json.loads((directory / "shortcut_probe_summary.json").read_text())
    patch_grid = manifest.get("patch_grid", [])
    return {"directory": str(directory), "manifest": manifest_path.as_posix(),
            "manifest_git": manifest.get("git"),
            "config_hash": manifest.get("config_hash"),
            "checkpoints": manifest.get("checkpoints"),
            "gpu": manifest.get("gpu"), "artifacts": hashes,
            "case_counts": summary.get("counts"),
            "patch_grid_case_count": len({item.get("case_id") for item in patch_grid}),
            "protocol_flags": required_flags}


def run(args):
    roots = [Path(args.images), Path(args.labels), Path(args.gt_sdf),
             Path(args.coarse_sdf)]
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing dataset/cache directories: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("Prompt-3R notebook requires a CUDA GPU runtime")
    gpu = torch.cuda.get_device_name(0)
    if args.require_gpu_name and args.require_gpu_name.lower() not in gpu.lower():
        raise RuntimeError(f"expected GPU containing {args.require_gpu_name!r}, got {gpu!r}")
    identity = resolve_identity(args.drive_root, args.identity_json)
    resolved = resolve_prompt3r_config(args.config, identity)
    identity_sha = sha256_file(identity)
    prompt2_dir = resolve_prompt2(args.drive_root, args.prompt2_dir)
    prompt2 = audit_prompt2(prompt2_dir, identity_sha)
    usage = shutil.disk_usage(args.output_root)
    free_gb = usage.free / 1024 ** 3
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"only {free_gb:.1f} GiB free; require {args.min_free_gb:.1f}")
    receipt = {
        "ready": True, "gpu": gpu, "torch": torch.__version__,
        "cuda": torch.version.cuda, "free_gb": free_gb,
        "identity_baseline": {"path": str(identity), "sha256": identity_sha,
                              "evaluated_cases": 480},
        "prompt2": prompt2,
        "paths": {"images": str(roots[0]), "labels": str(roots[1]),
                  "gt_sdf": str(roots[2]), "coarse_sdf": str(roots[3])},
        "resolved_config_hash": config_hash(resolved, 0, int(resolved.get("seed", 0))),
        "resolved_prior_floor": resolved["prior_floor"],
    }
    atomic_write_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--gt-sdf", required=True)
    parser.add_argument("--coarse-sdf", required=True)
    parser.add_argument("--config", default="configs/flow_prompt3r.yaml")
    parser.add_argument("--identity-json")
    parser.add_argument("--prompt2-dir")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-gpu-name", default="L4")
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
