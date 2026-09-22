#!/usr/bin/env python3
"""Cheap contracts for the corrected chart and causal masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from canalmanifold.chart import (
    reflect_local_numpy,
    reflect_shell_numpy,
)
from canalmanifold.constants import LOCAL_DIM
from canalmanifold.data import TubeCacheDataset
from canalmanifold.paths import transport_increment_from_displacement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.cache).glob("fold_*/*.npz"))
    if not paths:
        raise FileNotFoundError(args.cache)

    right = next(path for path in paths if path.stem.endswith("_R"))
    with np.load(right, allow_pickle=False) as archive:
        local = archive["q0_local"].astype(np.float32)
        shell = archive["shell"].astype(np.float32)
    assert np.array_equal(reflect_local_numpy(reflect_local_numpy(local)), local)
    assert np.array_equal(reflect_shell_numpy(reflect_shell_numpy(shell)), shell)

    case_id = right.stem.rsplit("_", 1)[0]
    sample = TubeCacheDataset(args.cache, [case_id], exclude_fallback=False)[0]
    with np.load(sample["path"], allow_pickle=False) as archive:
        q0_valid = archive["q0_station_valid"].astype(bool)
        q1_valid = archive["q1_station_valid"].astype(bool)
    assert np.array_equal(sample["station_mask"].numpy(), q0_valid)
    assert np.array_equal(sample["loss_mask"].numpy(), q0_valid & q1_valid)
    assert np.array_equal(sample["metric_mask"].numpy(), q0_valid | q1_valid)

    # Four schedule-aware steps must integrate a constant displacement exactly.
    local_delta = torch.randn(2, 160, LOCAL_DIM)
    global_delta = torch.randn(2, 3)
    local_sum = torch.zeros_like(local_delta)
    global_sum = torch.zeros_like(global_delta)
    for step in range(4):
        t0 = torch.full((2,), step / 4)
        t1 = torch.full((2,), (step + 1) / 4)
        inc_local, inc_global = transport_increment_from_displacement(
            local_delta, global_delta, t0, t1, "staged"
        )
        local_sum += inc_local
        global_sum += inc_global
    assert torch.allclose(local_sum, local_delta, atol=1e-6, rtol=1e-6)
    assert torch.allclose(global_sum, global_delta, atol=1e-6, rtol=1e-6)
    print("Corrected chart involution and causal-mask contracts: PASS")


if __name__ == "__main__":
    main()
