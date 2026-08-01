import json
from types import SimpleNamespace

import numpy as np
import pytest

import _pathsetup  # noqa: F401
from scripts.prompt1_completion import (CompletionState, NonRetryableCaseError,
                                         assert_voxelwise_match, atomic_copy,
                                         load_splits_config, run_case_queue)


def test_atomic_publish_refuses_existing_valid_artifact(tmp_path):
    source = tmp_path / "local.bin"
    source.write_bytes(b"complete")
    destination = tmp_path / "drive" / "case.bin"
    checksum = atomic_copy(source, destination)
    assert destination.read_bytes() == b"complete"
    assert len(checksum) == 64
    assert not (destination.parent / "case.bin.partial").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_copy(source, destination)


def test_two_case_interruption_then_resume_skips_completed_case(tmp_path):
    state_path = tmp_path / "state.json"
    state = CompletionState(state_path, "abc123", "cuda", "test-session")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    first_calls = []

    def interrupted_worker(case_id):
        first_calls.append(case_id)
        if case_id == "case_b":
            raise KeyboardInterrupt("simulated Colab interruption")
        (artifacts / case_id).write_text("valid")
        return {"checksums": {"hard": case_id}}

    def valid(case_id):
        return (artifacts / case_id).is_file()
    with pytest.raises(KeyboardInterrupt, match="simulated"):
        run_case_queue("oof_smoke", ["case_a", "case_b"], state,
                       interrupted_worker, valid, heartbeat_every=2)
    saved = json.loads(state_path.read_text())
    assert "case_a" in saved["completed_cases"]["oof_smoke"]
    assert "case_b" not in saved["completed_cases"]["oof_smoke"]

    resumed = CompletionState(state_path, "abc123", "cuda", "new-session")
    second_calls = []

    def resume_worker(case_id):
        second_calls.append(case_id)
        (artifacts / case_id).write_text("valid")

    assert run_case_queue("oof_smoke", ["case_a", "case_b"], resumed,
                          resume_worker, valid, heartbeat_every=2) == 2
    assert second_calls == ["case_b"]
    assert not list(tmp_path.rglob("*.partial"))


def test_case_failure_gets_only_two_automatic_retries(tmp_path):
    state = CompletionState(tmp_path / "state.json", "abc123", "cuda")
    calls = []

    def fail(case_id):
        calls.append(case_id)
        raise RuntimeError("GPU failure")

    completed = run_case_queue("oof", ["case_a"], state, fail,
                               lambda _case_id: False, retry_failed=True,
                               heartbeat_every=1)
    assert completed == 0
    assert calls == ["case_a", "case_a", "case_a"]  # initial + two retries
    assert state.data["retry_counts"]["oof"]["case_a"] == 2
    assert "GPU failure" in state.data["failed_cases"]["oof"]["case_a"]["error"]


def test_legacy_voxel_mismatch_is_nonretryable(tmp_path):
    state = CompletionState(tmp_path / "state.json", "abc123", "cuda")
    calls = []

    def mismatch(case_id):
        calls.append(case_id)
        assert_voxelwise_match(np.zeros((2, 2, 2), np.uint8),
                               np.ones((2, 2, 2), np.uint8), case_id)

    completed = run_case_queue("provenance_bootstrap", ["case_a"], state,
                               mismatch, lambda _case_id: False,
                               retry_failed=True, heartbeat_every=1)
    assert completed == 0
    assert calls == ["case_a"]
    error = state.data["failed_cases"]["provenance_bootstrap"]["case_a"]["error"]
    assert "8 changed voxels" in error
    assert state.data["retry_counts"]["provenance_bootstrap"]["case_a"] == 0


def test_legacy_voxel_match_accepts_identical_masks():
    mask = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    assert assert_voxelwise_match(mask, mask.copy(), "case_a") == 0
    with pytest.raises(NonRetryableCaseError, match="shape mismatch"):
        assert_voxelwise_match(mask, np.zeros((2, 2, 3), np.uint8), "case_a")


def _selection_runner(valid_ids):
    runner = object.__new__(__import__(
        "scripts.prompt1_completion", fromlist=["Prompt1Runner"]).Prompt1Runner)
    runner.splits = {"folds": [
        {"val": ["f0_bad", "f0_good", "f0_later"]},
        {"val": ["f1_good", "f1_later"]},
        {"val": ["f2_good"]},
        {"val": ["f3_good"]},
        {"val": ["f4_good"]},
    ]}
    runner.quick_cases_per_fold = 1
    checked = []

    def candidate(case_id):
        checked.append(case_id)
        return case_id in valid_ids

    runner.quick_candidate_valid = candidate
    return runner, checked


def test_quick_selection_is_one_per_fold_deterministic_and_skips_invalid_first():
    valid = {"f0_good", "f0_later", "f1_good", "f1_later", "f2_good",
             "f3_good", "f4_good"}
    runner, checked = _selection_runner(valid)
    expected = ["f0_good", "f1_good", "f2_good", "f3_good", "f4_good"]
    assert runner.select_quick_preflight_cases() == expected
    assert runner.select_quick_preflight_cases() == expected
    assert checked[:2] == ["f0_bad", "f0_good"]
    assert "f0_later" not in checked and "f1_later" not in checked


def test_quick_preflight_never_uses_global_scan_or_manifest(tmp_path):
    runner = object.__new__(__import__(
        "scripts.prompt1_completion", fromlist=["Prompt1Runner"]).Prompt1Runner)
    runner.quick_preflight_paths = lambda: SimpleNamespace(
        ids=tmp_path / "quick_ids.json", receipt=tmp_path / "quick_receipt.json",
        provenance_receipt=tmp_path / "quick_provenance.json",
        audit_prefix=tmp_path / "audit_quick5", identity_prefix=tmp_path / "identity_quick5",
        state=tmp_path / "quick_state.json")
    ids = [f"case_{index}" for index in range(5)]
    runner.splits = {"folds": [{"val": [case_id]} for case_id in ids]}
    runner.quick_cases_per_fold = 1
    runner.select_quick_preflight_cases = lambda: ids
    runner.git_sha = "abc"
    runner.bootstrap_legacy_provenance = lambda selected, mode, receipt: None
    runner._validate_preflight_cohort = lambda selected: None
    runner._run_preflight_audit = lambda selected, paths, mode: {"status_counts": {"valid": 5}}
    runner._run_preflight_identity = lambda selected, paths, mode: "PASS"
    runner._preflight_checksums = lambda selected: {case_id: {} for case_id in selected}
    runner.state = SimpleNamespace(data={"timestamps": {}}, save=lambda: None)
    runner.complete_cache_ids = lambda: (_ for _ in ()).throw(AssertionError("global scan"))
    runner.build_manifest = lambda: (_ for _ in ()).throw(AssertionError("manifest"))
    runner.preflight_quick()
    assert json.loads((tmp_path / "quick_ids.json").read_text()) == ids


def test_smoke_never_calls_manifest_or_full_preflight():
    runner = object.__new__(__import__(
        "scripts.prompt1_completion", fromlist=["Prompt1Runner"]).Prompt1Runner)
    calls = []
    runner.case_ids = ["a", "b", "c"]
    runner.preflight_quick = lambda: calls.append("quick")
    runner.preflight_full = lambda: (_ for _ in ()).throw(AssertionError("full preflight"))
    runner.build_manifest = lambda: (_ for _ in ()).throw(AssertionError("manifest"))
    runner.run_smoke_oof = lambda ids: calls.append(("oof", ids))
    runner.run_sdf = lambda ids: calls.append(("sdf", ids))
    runner._validate_smoke_cases = lambda ids: calls.append(("validate", ids))
    runner.state = SimpleNamespace(data={}, save=lambda: calls.append("save"))
    runner.smoke()
    assert calls[:4] == ["quick", ("oof", ["a", "b"]),
                         ("sdf", ["a", "b"]), ("validate", ["a", "b"])]
    assert runner.state.data["current_stage"] == "smoke_complete"


def test_second_smoke_oof_and_sdf_run_skips_valid_artifacts(tmp_path):
    runner = object.__new__(__import__(
        "scripts.prompt1_completion", fromlist=["Prompt1Runner"]).Prompt1Runner)
    runner.state = CompletionState(tmp_path / "state.json", "abc", "cpu")
    runner.export_softmax = True
    runner.fold_map = {"a": 0, "b": 1}
    runner.retry_failed = True
    runner.hard_valid = lambda case_id: True
    runner.provenance_valid = lambda case_id: True
    runner.softmax_valid = lambda case_id: True
    runner.coarse_valid = lambda case_id: True
    runner.gt_sdf_valid = lambda case_id: True
    runner._predict_case = lambda case_id, compare_existing=False: (
        _ for _ in ()).throw(AssertionError(f"regenerated OOF {case_id}"))
    runner._compute_sdf_case = lambda case_id: (
        _ for _ in ()).throw(AssertionError(f"regenerated SDF {case_id}"))
    runner.run_smoke_oof(["a", "b"])
    runner.run_sdf(["a", "b"])
    assert set(runner.state.data["completed_cases"]["oof_smoke"]) == {"a", "b"}


def test_quick_and_full_receipts_are_isolated(tmp_path):
    runner = object.__new__(__import__(
        "scripts.prompt1_completion", fromlist=["Prompt1Runner"]).Prompt1Runner)
    runner.prompt_dir = tmp_path / "prompt1"
    runner.analysis_dir = tmp_path / "analysis"
    runner.baseline_dir = tmp_path / "baselines"
    runner.quick_cases_per_fold = 1
    runner.full_preflight_cases = 40
    runner.splits = {"folds": [{"val": [str(index)]} for index in range(5)]}
    quick = runner.quick_preflight_paths()
    full = runner.full_preflight_paths()
    assert quick.receipt.name == "quick_preflight_5_receipt.json"
    assert quick.provenance_receipt.name == "quick_provenance_bootstrap_receipt.json"
    assert full.receipt.name == "full_preflight_40_receipt.json"
    assert full.provenance_receipt.name == "full_provenance_bootstrap_receipt.json"
    assert quick.receipt != full.receipt
    assert quick.provenance_receipt != full.provenance_receipt
    quick.receipt.parent.mkdir(parents=True)
    quick.receipt.write_text(json.dumps({"case_ids": ["quick"]}))
    assert not full.receipt.exists()


def test_splits_path_override_and_repo_fallback(tmp_path):
    drive_path = tmp_path / "drive" / "configs_cache" / "splits.json"
    drive_path.parent.mkdir(parents=True)
    payload = {"development": ["drive"], "folds": [{"val": ["drive"]}]}
    drive_path.write_text(json.dumps(payload))
    path, loaded = load_splits_config({"SPLITS_PATH": str(drive_path)}, tmp_path)
    assert path == drive_path and loaded == payload

    fallback = tmp_path / "configs" / "splits.json"
    fallback.parent.mkdir()
    fallback_payload = {"development": ["repo"], "folds": [{"val": ["repo"]}]}
    fallback.write_text(json.dumps(fallback_payload))
    path, loaded = load_splits_config({}, tmp_path)
    assert path == fallback and loaded == fallback_payload
    with pytest.raises(FileNotFoundError, match="SPLITS_PATH"):
        load_splits_config({"SPLITS_PATH": str(tmp_path / "missing.json")}, tmp_path)


def test_full_preflight_keeps_configured_40_case_gate(tmp_path):
    runner = object.__new__(__import__(
        "scripts.prompt1_completion", fromlist=["Prompt1Runner"]).Prompt1Runner)
    runner.full_preflight_cases = 40
    runner.legacy_complete_cache_ids = lambda: [f"case_{index:02d}" for index in range(40)]
    assert len(runner.select_full_preflight_cases()) == 40
    runner.legacy_complete_cache_ids = lambda: [f"case_{index:02d}" for index in range(39)]
    with pytest.raises(RuntimeError, match="at least 40"):
        runner.select_full_preflight_cases()
