"""Decoders operating on one side (bool masks, (Z,Y,X)).

prior_lcc        : keep largest component.
mcp_connect      : non-learned baseline. Connect each extra component to the main one
                   through a minimum-cost path on (1 - p); keep (soft) or drop (hard)
                   components that cannot be connected under the preregistered limits.
tcmbr            : topology-aware minimum-Bayes-risk decoding of K bridge samples.
                   Start from the sample marginal M; for every extra component j choose,
                   by expected utility against the samples, between dropping j, keeping it
                   (soft only) and grafting it through the corridor of a sample that
                   connects it. 'hard' enforces beta0 = 1.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

from .common import CONN26, bbox, components, radius_estimate_vox


def _crop_for(masks, margin_vox, shape):
    boxes = [bbox(m, margin_vox, shape) for m in masks]
    boxes = [b for b in boxes if b is not None]
    return tuple(slice(min(b[i].start for b in boxes), max(b[i].stop for b in boxes)) for i in range(3))


def _min_path(cost, start_mask, end_mask, sampling):
    """Cheapest path (list of (z,y,x)) from any start voxel to any end voxel, or None."""
    starts = np.argwhere(start_mask & ~ndi.binary_erosion(start_mask, CONN26))
    ends = np.argwhere(end_mask & ~ndi.binary_erosion(end_mask, CONN26))
    if len(starts) == 0 or len(ends) == 0:
        return None
    mcp = MCP_Geometric(cost, fully_connected=True, sampling=sampling)
    cum, _ = mcp.find_costs([tuple(s) for s in starts])
    ce = cum[tuple(ends.T)]
    if not np.isfinite(ce).any():
        return None
    end = tuple(ends[int(np.argmin(ce))])
    return mcp.traceback(end)


def _tube_around(path_mask, radius_vox, restrict=None):
    if not path_mask.any():
        return path_mask
    d = ndi.distance_transform_edt(~path_mask)
    t = d <= max(radius_vox, 1.0)
    return t & restrict if restrict is not None else t


# --------------------------------------------------------------------- baselines
def mcp_connect(mask, prob, spacing, max_gap_mm=10.0, min_mean_p=0.15, search_mm=15.0, hard=False):
    lab, n, sizes = components(mask)
    if n <= 1:
        return mask.copy(), dict(n_comp=n, grafted=0, dropped=0)
    order = np.argsort(-sizes)
    main_id = order[0] + 1
    out = lab == main_id
    r = radius_estimate_vox(out)
    sv = float(np.mean(spacing))
    grafted = dropped = 0
    for j in order[1:]:
        comp = lab == (j + 1)
        box = _crop_for([comp], int(np.ceil(search_mm / sv)), mask.shape)
        main_c = out[box]
        if not main_c.any():
            ok = False
        else:
            cost = (1.0 - prob[box]).astype(np.float64) + 0.01
            path = _min_path(cost, main_c, comp[box], spacing)
            ok = False
            if path is not None:
                pm = np.zeros(main_c.shape, bool)
                pm[tuple(np.asarray(path).T)] = True
                gap = pm & ~main_c & ~comp[box]
                gap_len_mm = gap.sum() * sv
                mean_p = float(prob[box][gap].mean()) if gap.any() else 1.0
                ok = gap_len_mm <= max_gap_mm and mean_p >= min_mean_p
        if ok:
            corridor = np.zeros_like(mask)
            corridor[box] = _tube_around(gap, r)
            out |= comp | corridor
            grafted += 1
        elif not hard:
            out |= comp
        else:
            dropped += 1
    return out, dict(n_comp=n, grafted=grafted, dropped=dropped)


# ------------------------------------------------------------------------ TC-MBR
def _beta0(mask):
    b = bbox(mask, 1)
    if b is None:
        return 0
    return components(mask[b])[1]


def expected_utility(candidate, samples, sample_beta0, lam):
    """E_k[ Dice(candidate, S_k) - lam * |beta0(candidate) - beta0(S_k)| ].
    lam = 0 gives plain expected Dice (the voxel-overlap MBR)."""
    inter = (samples & candidate[None]).reshape(len(samples), -1).sum(1)
    denom = samples.reshape(len(samples), -1).sum(1) + candidate.sum()
    d = np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1), 1.0)
    if lam:
        d = d - lam * np.abs(_beta0(candidate) - sample_beta0)
    return float(d.mean())


def tcmbr(samples, spacing, hard=False, lam=0.05, search_mm=15.0):
    """samples: bool (K, Z, Y, X) for one side. Returns (mask, info).

    soft : options {drop j, keep j unconnected, graft j via sample k}; utility = Dice minus
           lam * beta0 mismatch against each sample, so the samples' own topology decides
           whether a connection is believed (a corridor seen in 1/K samples can still win if
           every sample is connected somewhere).
    hard : options {drop j, graft j via sample k}; beta0 = 1 is enforced.
    """
    K = len(samples)
    sb0 = np.array([_beta0(s) for s in samples])
    marg = samples.mean(0) >= 0.5
    lab, n, sizes = components(marg)
    if n <= 1:
        return marg, dict(n_comp=n, grafted=0, dropped=0, kept=0)
    order = np.argsort(-sizes)
    Y = lab == (order[0] + 1)
    sv = float(np.mean(spacing))
    grafted = dropped = kept = 0
    for j in order[1:]:
        comp = lab == (j + 1)
        box = _crop_for([comp], int(np.ceil(search_mm / sv)), marg.shape)
        options = [(expected_utility(Y, samples, sb0, lam), Y, "drop")]
        if not hard:
            keep = Y | comp
            options.append((expected_utility(keep, samples, sb0, lam), keep, "keep"))
        for k in range(K):
            sk = samples[k][box]
            slab, sn, _ = components(sk)
            if sn == 0:
                continue
            touch_main = set(np.unique(slab[Y[box] & sk])) - {0}
            touch_j = set(np.unique(slab[comp[box] & sk])) - {0}
            common = touch_main & touch_j
            if not common:
                continue
            sc = np.isin(slab, list(common))
            cost = np.where(sc, 1.0, -1.0)                       # walk only inside sample k
            path = _min_path(cost, Y[box] & sc, comp[box] & sc, spacing)
            if path is None:
                continue
            pm = np.zeros(sk.shape, bool)
            pm[tuple(np.asarray(path).T)] = True
            gap = pm & ~Y[box] & ~comp[box]
            r = radius_estimate_vox(sc)
            corr = np.zeros_like(marg)
            corr[box] = _tube_around(gap, r, restrict=sc) | gap
            cand = Y | comp | corr
            options.append((expected_utility(cand, samples, sb0, lam), cand, "graft"))
        # MBR choice: maximise expected utility among constraint-satisfying options.
        # Ties keep the earlier option (drop < keep < graft order).
        _, Y, kind = max(options, key=lambda o: o[0])
        grafted += kind == "graft"; dropped += kind == "drop"; kept += kind == "keep"
    return Y, dict(n_comp=n, grafted=grafted, dropped=dropped, kept=kept)


def marginal(samples):
    return samples.mean(0) >= 0.5
