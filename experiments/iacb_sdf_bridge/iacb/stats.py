"""Paired statistics shared by infer.py and compare_external.py (no torch import)."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def paired(df, a, b, metric, n_boot=5000, seed=0):
    """Case-level paired bootstrap of mean(metric_b - metric_a) over common (case, side) rows."""
    x = df[df.method == a].set_index(["case", "side"])[metric]
    y = df[df.method == b].set_index(["case", "side"])[metric]
    j = x.index.intersection(y.index)
    if len(j) == 0:
        return dict(mean_diff=float("nan"), ci95=[float("nan")] * 2, n_cases=0)
    d = (y[j] - x[j]).groupby(level=0).mean().values           # case-level differences
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return dict(mean_diff=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                n_cases=int(len(d)))


def mcnemar_topology(df, a, b):
    """Exact McNemar on per-side 'beta0 correct' (beta0_err == 0)."""
    x = (df[df.method == a].set_index(["case", "side"]).beta0_err == 0)
    y = (df[df.method == b].set_index(["case", "side"]).beta0_err == 0)
    j = x.index.intersection(y.index)
    fixed = int((~x[j] & y[j]).sum()); broken = int((x[j] & ~y[j]).sum())
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    return dict(fixed=fixed, broken=broken, p_exact=float(p), n_sides=int(len(j)))
