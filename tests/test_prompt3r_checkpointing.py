import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from flow.channel_contract import resolve_conditioning_spec
from flow.checkpointing import (
    CheckpointConflictError,
    atomic_write_trajectory,
    build_checkpoint_payload,
    cleanup_checkpoint_partials,
    immutable_checkpoint_name,
    save_immutable_checkpoint,
    save_resume_checkpoint,
    sha256_file,
    upsert_epoch_record,
)


def _payload(run_id="run-a", epoch=0):
    model = torch.nn.Conv3d(2, 2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
    spec = resolve_conditioning_spec({"cond_include_coarse_sdf": False})
    payload = build_checkpoint_payload(
        model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
        config={"cond_include_coarse_sdf": False}, config_hash="cfg-a",
        channel_contract=spec.to_dict(), selection_policy="paired_identity",
        validation_summaries={"quick": {"safe": False}}, git_sha="abc123",
        run_id=run_id, fold=0, seed=0)
    return payload, spec


def test_epoch_names_distinguish_untrained_zero():
    assert immutable_checkpoint_name(0) == "epoch_000_untrained.pt"
    assert immutable_checkpoint_name(1) == "epoch_001.pt"
    assert immutable_checkpoint_name(15) == "epoch_015.pt"


def test_immutable_save_verifies_and_skips_exact_resume(tmp_path):
    payload, spec = _payload()
    first = save_immutable_checkpoint(tmp_path, payload, spec)
    original = Path(first["path"]).read_bytes()
    second = save_immutable_checkpoint(tmp_path, payload, spec)
    assert first["created"] is True and second["created"] is False
    assert first["sha256"] == second["sha256"] == sha256_file(first["path"])
    assert Path(first["path"]).read_bytes() == original


def test_existing_epoch_with_other_run_id_is_rejected_without_overwrite(tmp_path):
    payload, spec = _payload()
    saved = save_immutable_checkpoint(tmp_path, payload, spec)
    before = Path(saved["path"]).read_bytes()
    conflict, _ = _payload(run_id="run-b")
    with pytest.raises(CheckpointConflictError, match="run_id"):
        save_immutable_checkpoint(tmp_path, conflict, spec)
    assert Path(saved["path"]).read_bytes() == before


def test_mutable_last_checkpoint_is_atomically_replaceable(tmp_path):
    path = tmp_path / "last.pt"
    save_resume_checkpoint(path, {"epoch": 0})
    save_resume_checkpoint(path, {"epoch": 1})
    assert torch.load(path, weights_only=False)["epoch"] == 1
    assert not (tmp_path / "last.pt.partial").exists()


def test_trajectory_upsert_has_no_duplicate_epochs(tmp_path):
    records, created = upsert_epoch_record([], {"epoch": 0, "loss": None})
    records, repeated = upsert_epoch_record(records, {"epoch": 0, "loss": None})
    assert created is True and repeated is False and len(records) == 1
    with pytest.raises(CheckpointConflictError):
        upsert_epoch_record(records, {"epoch": 0, "loss": 1.0})
    json_path, csv_path = atomic_write_trajectory(tmp_path, records)
    assert json.loads(json_path.read_text()) == records
    assert csv_path.read_text().count("\n") == 2


def test_killed_subprocess_partial_is_cleaned_without_harming_prior_epoch(tmp_path):
    payload, spec = _payload(epoch=0)
    saved = save_immutable_checkpoint(tmp_path, payload, spec)
    before_sha = saved["sha256"]
    partial = tmp_path / "epoch_001.pt.partial"
    marker = tmp_path / "writer_ready"
    code = (
        "import pathlib,time,torch; "
        f"torch.save({{'epoch': 1}}, {str(partial)!r}); "
        f"pathlib.Path({str(marker)!r}).write_text('ready'); time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-c", code])
    deadline = time.time() + 10
    while not marker.exists() and time.time() < deadline:
        time.sleep(.05)
    assert marker.exists()
    process.terminate()
    process.wait(timeout=10)
    assert partial.exists()
    assert cleanup_checkpoint_partials(tmp_path) == [str(partial)]
    assert sha256_file(saved["path"]) == before_sha
