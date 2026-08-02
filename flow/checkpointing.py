"""Fail-closed immutable trajectory checkpoints and duplicate-safe ledgers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import torch

from flow.channel_contract import validate_checkpoint_contract


class CheckpointConflictError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_checkpoint_name(epoch):
    epoch = int(epoch)
    if epoch < 0:
        raise ValueError("checkpoint epoch cannot be negative")
    return "epoch_000_untrained.pt" if epoch == 0 else f"epoch_{epoch:03d}.pt"


def cleanup_checkpoint_partials(directory):
    removed = []
    for partial in Path(directory).glob("*.pt.partial"):
        partial.unlink()
        removed.append(str(partial))
    return removed


def build_checkpoint_payload(*, model, optimizer, scheduler, epoch, config,
                             config_hash, channel_contract, selection_policy,
                             validation_summaries, git_sha, run_id, fold, seed,
                             training_summary=None, trajectory_record=None):
    payload = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "epoch": int(epoch),
        "cfg": config,
        "config_hash": str(config_hash),
        "channel_contract": channel_contract,
        "selection_policy": str(selection_policy),
        "validation_summaries": validation_summaries,
        "git_sha": str(git_sha),
        "run_id": str(run_id),
        "fold": int(fold),
        "seed": int(seed),
    }
    if training_summary is not None:
        payload["training_summary"] = training_summary
    if trajectory_record is not None:
        payload["trajectory_record"] = trajectory_record
    return payload


def _validate_identity(checkpoint, *, epoch, config_hash, channel_spec,
                       selection_policy, run_id):
    expected = {"epoch": int(epoch), "config_hash": str(config_hash),
                "selection_policy": str(selection_policy), "run_id": str(run_id)}
    mismatches = {key: (checkpoint.get(key), value) for key, value in expected.items()
                  if checkpoint.get(key) != value}
    if mismatches:
        raise CheckpointConflictError(f"immutable checkpoint identity mismatch: {mismatches}")
    validate_checkpoint_contract(checkpoint, channel_spec, legacy_compatibility=False)


def _verify_or_create_receipt(path):
    receipt = path.with_suffix(path.suffix + ".sha256")
    actual = sha256_file(path)
    if receipt.exists():
        expected = receipt.read_text().strip()
        if actual != expected:
            raise CheckpointConflictError(
                f"immutable checkpoint checksum mismatch: {path} {actual} != {expected}")
    else:
        partial = receipt.with_suffix(receipt.suffix + ".partial")
        partial.write_text(actual + "\n")
        try:
            os.link(partial, receipt)
        except FileExistsError:
            if receipt.read_text().strip() != actual:
                raise CheckpointConflictError(f"conflicting checksum receipt: {receipt}")
        finally:
            partial.unlink(missing_ok=True)
    return actual


def verify_immutable_checkpoint(path):
    """Public checksum verification used before an immutable resume load."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return _verify_or_create_receipt(path)


def save_immutable_checkpoint(directory, payload, channel_spec):
    """Publish once; an exact resume encounter verifies and skips the epoch."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    epoch = int(payload["epoch"])
    path = directory / immutable_checkpoint_name(epoch)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        _validate_identity(existing, epoch=epoch,
                           config_hash=payload["config_hash"], channel_spec=channel_spec,
                           selection_policy=payload["selection_policy"],
                           run_id=payload["run_id"])
        return {"path": str(path), "sha256": _verify_or_create_receipt(path),
                "created": False}

    torch.save(payload, partial)
    try:
        os.link(partial, path)  # atomic no-overwrite publication on the same filesystem
    except FileExistsError:
        existing = torch.load(path, map_location="cpu", weights_only=False)
        _validate_identity(existing, epoch=epoch,
                           config_hash=payload["config_hash"], channel_spec=channel_spec,
                           selection_policy=payload["selection_policy"],
                           run_id=payload["run_id"])
        created = False
    else:
        created = True
    finally:
        partial.unlink(missing_ok=True)
    return {"path": str(path), "sha256": _verify_or_create_receipt(path),
            "created": created}


def save_resume_checkpoint(path, payload):
    """Atomically replace the sole mutable checkpoint, ``last.pt``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)
    return sha256_file(path)


def upsert_epoch_record(records, record, *, epoch_key="epoch"):
    """Skip an identical epoch record and reject any conflicting duplicate."""
    epoch = int(record[epoch_key])
    matching = [row for row in records if int(row[epoch_key]) == epoch]
    if len(matching) > 1:
        raise CheckpointConflictError(f"ledger already has duplicate epoch {epoch}")
    normalized = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    if matching:
        old = json.loads(json.dumps(matching[0], sort_keys=True, allow_nan=False))
        if old != normalized:
            raise CheckpointConflictError(f"conflicting ledger record for epoch {epoch}")
        return list(records), False
    return sorted([*records, record], key=lambda row: int(row[epoch_key])), True


def atomic_write_trajectory(directory, records):
    """Write CSV and JSON from one duplicate-checked in-memory ledger."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in records]
    if len(epochs) != len(set(epochs)):
        raise CheckpointConflictError("trajectory contains duplicate epochs")
    json_path = directory / "epoch_trajectory.json"
    json_partial = json_path.with_suffix(".json.partial")
    json_partial.write_text(json.dumps(records, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n")
    os.replace(json_partial, json_path)
    csv_path = directory / "epoch_trajectory.csv"
    csv_partial = csv_path.with_suffix(".csv.partial")
    fields = list(dict.fromkeys(key for row in records for key in row))
    with csv_partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(csv_partial, csv_path)
    return json_path, csv_path
