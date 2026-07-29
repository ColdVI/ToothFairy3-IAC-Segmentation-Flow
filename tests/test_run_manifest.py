import json

import _pathsetup  # noqa: F401
from scripts.run_manifest import (config_hash, finish_manifest, start_manifest)


def test_config_hash_includes_fold_and_seed():
    cfg = {"base": 8, "epochs": 3}
    assert config_hash(cfg, 0, 1) == config_hash(cfg, 0, 1)
    assert config_hash(cfg, 0, 1) != config_hash(cfg, 1, 1)
    assert config_hash(cfg, 0, 1) != config_hash(cfg, 0, 2)


def test_manifest_start_resume_and_finish(tmp_path):
    out = tmp_path / "run"
    cfg = {"base": 8, "epochs": 3}
    first = start_manifest(out, cfg, fold=2, seed=7, repo_root=".")
    assert first["status"] == "running"
    assert first["resume_count"] == 0

    resumed = start_manifest(out, cfg, fold=2, seed=7, repo_root=".", resume=True)
    assert resumed["start_time"] == first["start_time"]
    assert resumed["resume_count"] == 1

    finish_manifest(out, "completed", {"dice": 0.91})
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["status"] == "completed"
    assert saved["end_time"] is not None
    assert saved["metrics"] == {"dice": 0.91}
