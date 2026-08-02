"""Resolve the Prompt-3R template only from a complete Prompt-1 identity JSON."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_prompt3r_config(template_path, identity_json_path):
    config = yaml.safe_load(Path(template_path).read_text())
    identity = json.loads(Path(identity_json_path).read_text())
    checks = {
        "complete_cv": identity.get("complete_cv") is True,
        "evaluated_cases": identity.get("evaluated_cases") == 480,
        "cache_valid_cases": identity.get("cache_valid_cases") == 480,
        "missing_cases": identity.get("missing_cases") == 0,
        "invalid_cases": identity.get("invalid_cases") == 0,
        "direct_vs_sdf": identity.get("direct_vs_sdf_voxel_difference") == 0,
        "direct_vs_full": identity.get("direct_vs_full_path_voxel_difference") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Prompt-1 identity artefact is incomplete: {failed}")
    source = identity.get("overall", {}).get("per_side", {}).get("direct")
    names = ("dice", "cldice", "hd95", "score")
    if not isinstance(source, dict) or any(not isinstance(source.get(name), (int, float))
                                           for name in names):
        raise ValueError("identity JSON lacks direct per-side dice/cldice/hd95/score")
    resolved = copy.deepcopy(config)
    resolved["prior_floor"] = {
        "complete_cv": True,
        **{name: float(source[name]) for name in names},
        "source_json": str(Path(identity_json_path).resolve()),
        "source_sha256": sha256_file(identity_json_path),
    }
    return resolved
