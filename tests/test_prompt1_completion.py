import json

import pytest

import _pathsetup  # noqa: F401
from scripts.prompt1_completion import (CompletionState, atomic_copy,
                                         run_case_queue)


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
