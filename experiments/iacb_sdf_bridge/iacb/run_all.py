"""One command: cache -> diagnostics -> train -> infer. Colab "Run all" friendly.

Design rules for this entry point:
  * NOTHING stops training for a scientific reason. p0_anatomy and compare_external run as
    REPORTS. Their verdicts land in run_all_report.json; they never abort the pipeline.
  * Only real data errors stop it (missing files, shape/axis mismatch, disk).
  * One fixed configuration. No sweeps, no hyperparameter search.
  * Resume-safe: every stage is skipped if its output exists, and training resumes from
    last.pt. Re-running the same cell after a Colab disconnect continues where it stopped.

  python -m iacb.run_all --raw $RAW --prob_dir $OOF --splits $PRE/splits_final.json --work /content/iac

Optional: --teacher_dir with the supervisor's fold-0 predictions adds the same-cases
comparison rows. Leave it out and the pipeline runs without it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("cache", "diagnose", "train", "infer", "compare")


def run(cmd, log_path: Path, required: bool):
    """Stream a subprocess to stdout and a log file. Returns (ok, returncode)."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n"); log.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); log.write(line)
        p.wait()
    dt = time.time() - t0
    ok = p.returncode == 0
    print(f"[{'ok' if ok else 'FAILED'}] {log_path.stem} in {dt/60:.1f} min", flush=True)
    if not ok and required:
        raise SystemExit(f"Required stage failed: {log_path.stem}. See {log_path}")
    return ok, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="nnUNet_raw/DatasetXXX (expects imagesTr, labelsTr)")
    ap.add_argument("--prob_dir", required=True, help="OOF npz folder (nnU-Net --save_probabilities)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--work", required=True, help="working dir (use local disk on Colab, not Drive)")
    ap.add_argument("--teacher_dir", default=None, help="optional: supervisor's fold-0 predictions")
    ap.add_argument("--eval_fold", type=int, default=0)
    ap.add_argument("--label_left_id", type=int, default=3,
                    help="Left-IAC label in the full TF3 labelsTr")
    ap.add_argument("--label_right_id", type=int, default=4,
                    help="Right-IAC label in the full TF3 labelsTr")
    ap.add_argument("--prob_left_id", type=int, default=1,
                    help="Left-IAC probability channel from Dataset801 nnU-Net")
    ap.add_argument("--prob_right_id", type=int, default=2,
                    help="Right-IAC probability channel from Dataset801 nnU-Net")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--holdout_n", type=int, default=0,
                    help="bridge-only holdout from non-eval cases; 0 uses all 435 non-eval cases in fold-0 mode")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N cases end to end")
    ap.add_argument("--skip", default="", help="comma-separated stages to skip: " + ",".join(STAGES))
    ap.add_argument("--only", default="", help="comma-separated stages to run (others skipped)")
    a = ap.parse_args()

    py = sys.executable
    raw, work = Path(a.raw), Path(a.work)
    cache, run_dir, logs = work / "cache", work / "run", work / "logs"
    for d in (work, logs):
        d.mkdir(parents=True, exist_ok=True)
    skip = {s for s in a.skip.split(",") if s}
    only = {s for s in a.only.split(",") if s}

    def want(stage):
        return stage not in skip and (not only or stage in only)

    images, labels = raw / "imagesTr", raw / "labelsTr"
    for p in (images, labels, Path(a.prob_dir), Path(a.splits)):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")
    common = ["--labels_dir", str(labels), "--prob_dir", str(a.prob_dir), "--splits", str(a.splits)]
    rep = dict(config=vars(a), stages={})

    # -- 1. cache (idempotent: existing cases are skipped by cache.py itself)
    if want("cache"):
        cmd = [py, "-m", "iacb.cache", "--images_dir", str(images), *common, "--out", str(cache),
               "--label_left_id", str(a.label_left_id), "--label_right_id", str(a.label_right_id),
               "--prob_left_id", str(a.prob_left_id), "--prob_right_id", str(a.prob_right_id),
               "--workers", str(a.workers)]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        ok, dt = run(cmd, logs / "cache.log", required=True)
        rep["stages"]["cache"] = dict(ok=ok, minutes=round(dt / 60, 1))
    n_cached = len(list(cache.glob("*_meta.json")))
    print(f"[cache] {n_cached} cases ready", flush=True)
    if n_cached == 0:
        raise SystemExit("Cache is empty. Check --raw / --prob_dir / --splits and cache.log.")

    # -- 2. diagnostics: REPORT ONLY. A NO-GO verdict here does not stop training.
    if want("diagnose"):
        cmd = [py, "-m", "iacb.p0_anatomy", *common, "--folds", str(a.eval_fold), "--out", str(work / "p0"),
               "--label_left_id", str(a.label_left_id), "--label_right_id", str(a.label_right_id),
               "--prob_left_id", str(a.prob_left_id), "--prob_right_id", str(a.prob_right_id),
               "--workers", str(a.workers)]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        ok, dt = run(cmd, logs / "p0.log", required=False)
        rep["stages"]["diagnose"] = dict(ok=ok, minutes=round(dt / 60, 1), note="report only, never blocks")
        p0j = work / "p0" / "p0_summary.json"
        if p0j.exists():
            s = json.loads(p0j.read_text())
            rep["diagnostics"] = {k: s.get(k) for k in
                                  ("verdict", "gates", "pooled_miss_fraction", "hd95_headroom",
                                   "mean_oracle_dice_component", "prior_sides_with_topology_error")}
            print("\n[diagnostics] " + json.dumps(rep["diagnostics"].get("verdict", ""), ensure_ascii=False))
            print("[diagnostics] training continues regardless of this verdict.\n", flush=True)

    # -- 3. train (resumes from run/last.pt if present)
    if want("train"):
        cmd = [py, "-m", "iacb.train", "--cache", str(cache), "--out", str(run_dir),
               "--eval_fold", str(a.eval_fold), "--iters", str(a.iters),
               "--holdout_n", str(a.holdout_n), "--workers", str(min(a.workers, 6))]
        if a.limit:
            cmd += ["--holdout_n", "1", "--patch", "32,48,48", "--widths", "8,16,32", "--batch", "2",
                    "--val_every", "4", "--save_every", "4"]
        ok, dt = run(cmd, logs / "train.log", required=True)
        rep["stages"]["train"] = dict(ok=ok, minutes=round(dt / 60, 1))
    ckpt = run_dir / "last.pt"
    if not ckpt.exists() and want("infer"):
        raise SystemExit(f"No checkpoint at {ckpt}; cannot run inference.")

    # -- 4. inference on the untouched eval fold, one configuration
    if want("infer"):
        cmd = [py, "-m", "iacb.infer", "--cache", str(cache), "--ckpt", str(ckpt),
               "--out", str(work / "eval"), "--eval_fold", str(a.eval_fold)]
        if a.limit:
            cmd += ["--limit", str(a.limit), "--K", "3", "--nfe", "2"]
        ok, dt = run(cmd, logs / "infer.log", required=True)
        rep["stages"]["infer"] = dict(ok=ok, minutes=round(dt / 60, 1))
        ej = work / "eval" / "report.json"
        if ej.exists():
            e = json.loads(ej.read_text())
            rep["results"] = dict(summary=e.get("summary"), paired=e.get("paired"),
                                  sample_diversity=e.get("sample_diversity_mean_pairwise_dice"))

    # -- 5. same-cases comparison against the supervisor (optional, report only)
    per_side = work / "eval" / "per_side.csv"
    if want("compare") and a.teacher_dir:
        cmd = [py, "-m", "iacb.compare_external", "--pred_dir", a.teacher_dir, "--labels_dir", str(labels),
               "--splits", str(a.splits), "--folds", str(a.eval_fold), "--name", "teacher",
               "--left_id", str(a.label_left_id), "--right_id", str(a.label_right_id),
               "--out", str(work / "cmp_teacher"), "--workers", str(a.workers)]
        if per_side.exists():
            cmd += ["--merge_per_side", str(per_side)]
        ok, dt = run(cmd, logs / "compare.log", required=False)
        rep["stages"]["compare"] = dict(ok=ok, minutes=round(dt / 60, 1), note="report only")
        cj = work / "cmp_teacher" / "report.json"
        if cj.exists():
            c = json.loads(cj.read_text())
            rep["teacher_comparison"] = {k: c.get(k) for k in
                                         ("overlap", "per_side", "whole_binary", "merged", "paired_vs_external")}

    (work / "run_all_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("\n=== done ===")
    print(f"report        {work / 'run_all_report.json'}")
    print(f"checkpoint    {ckpt}")
    print(f"per-side csv  {per_side}")
    if "results" in rep and rep["results"].get("summary"):
        s = rep["results"]["summary"]
        print(f"\n{'method':24s} {'dice':>8s} {'hd95':>8s} {'topo_ok':>8s}")
        for k, v in s.items():
            print(f"{k:24s} {v.get('dice', float('nan')):8.4f} {v.get('hd95', float('nan')):8.3f} "
                  f"{v.get('sides_topo_ok', float('nan')):8.3f}")


if __name__ == "__main__":
    main()
