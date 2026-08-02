import json
from argparse import Namespace

import pytest

from scripts.prompt3r_preflight import audit_prompt2, resolve_prompt2, sha256_file


def test_prompt2_audit_requires_flags_hashes_and_same_identity(tmp_path):
    identity = tmp_path / "identity_prior.json"
    identity.write_text("identity")
    identity_sha = sha256_file(identity)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    names = ("shortcut_probe_summary.json", "shortcut_probe.csv",
             "thickening_probe.csv", "limited_endpoint_diagnostic.pdf")
    for name in names:
        (analysis / name).write_text(json.dumps({"counts": {"cases": 12}})
                                     if name.endswith("summary.json") else name)
    manifest = {
        "protocol_deviation": True, "exact_epoch_trajectory_available": False,
        "historical_per_epoch_checkpoints_were_not_saved": True,
        "diagnostic_only": True, "identity_baseline": {"sha256": identity_sha},
        "artifacts": {name: sha256_file(analysis / name) for name in names},
    }
    (analysis / "shortcut_probe_manifest.json").write_text(json.dumps(manifest))
    assert resolve_prompt2(tmp_path) == analysis.resolve()
    assert audit_prompt2(analysis, identity_sha)["protocol_flags"]["diagnostic_only"]


def test_prompt2_audit_rejects_protocol_drift(tmp_path):
    analysis = tmp_path
    for name in ("shortcut_probe_summary.json", "shortcut_probe.csv",
                 "thickening_probe.csv", "limited_endpoint_diagnostic.pdf"):
        (analysis / name).write_text(name)
    (analysis / "shortcut_probe_manifest.json").write_text(json.dumps({}))
    with pytest.raises(RuntimeError, match="protocol flags"):
        audit_prompt2(analysis, "identity-sha")
