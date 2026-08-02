"""Deterministic, scanner-stratified Prompt-3R pilot panel selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flow.validate import scanner_group


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(case_id, seed):
    return hashlib.sha256(f"prompt3r|seed={seed}|{case_id}".encode()).hexdigest()


def select_pilot_cases(val_ids, *, seed=0, full_cases_per_scanner=5,
                       quick_cases_per_scanner=2):
    """Hash-rank Fold-0 validation IDs independently for F and P scanners."""
    if quick_cases_per_scanner > full_cases_per_scanner:
        raise ValueError("quick panel cannot exceed full panel")
    groups = {"F": [], "P": []}
    for case_id in val_ids:
        groups[scanner_group(case_id)].append(case_id)
    full, quick = [], []
    for group in ("F", "P"):
        ranked = sorted(groups[group], key=lambda case_id: (_rank(case_id, seed), case_id))
        if len(ranked) < full_cases_per_scanner:
            raise ValueError(f"scanner {group} has only {len(ranked)} validation cases")
        selected = ranked[:full_cases_per_scanner]
        full.extend(selected)
        quick.extend(selected[:quick_cases_per_scanner])
    return {"full_case_ids": full, "quick_case_ids": quick}


def build_pilot_case_manifest(splits_path, *, fold=0, seed=0,
                              full_cases_per_scanner=5,
                              quick_cases_per_scanner=2):
    splits_path = Path(splits_path)
    splits = json.loads(splits_path.read_text())
    selected = select_pilot_cases(
        splits["folds"][fold]["val"], seed=seed,
        full_cases_per_scanner=full_cases_per_scanner,
        quick_cases_per_scanner=quick_cases_per_scanner)
    case_list_payload = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    return {"fold": int(fold), "seed": int(seed),
            "selection_algorithm": "sha256('prompt3r|seed=0|' + case_id), scanner-stratified",
            "source_splits_sha256": file_sha256(splits_path),
            "case_list_sha256": hashlib.sha256(case_list_payload.encode()).hexdigest(),
            **selected}


def write_manifest_no_overwrite(path, manifest):
    """Allow resume only when an existing immutable panel is byte-equivalent."""
    path = Path(path)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"refusing to replace a different pilot panel: {path}")
        return False
    path.write_text(encoded)
    return True
