"""GT-free inference on the untouched eval fold + evaluation of every decoder.

Per case the bridge runs inside the prior-only ROI:
  det1   : K=1, NFE=1 (one-shot refiner, equals the network at t=0)
  detN   : K=1, NFE=N, temp=0 (deterministic multi-step path)
  samples: K stochastic samples, NFE=N, temp=T
Decoders compared on identical crops:
  prior, prior_lcc, prior_mcp_soft, prior_mcp_hard       (no learning)
  bridge_det1, bridge_detN, bridge_detN_lcc
  bridge_sample0, bridge_marg, bridge_marg_lcc
  bridge_mbr_dice (lam=0 control), bridge_tcmbr_soft, bridge_tcmbr_hard (proposed)
Plus sample diversity (mean pairwise Dice between samples) per side.

  python -m iacb.infer --cache /content/cache --ckpt run1/last.pt --out run1/eval_T1 --K 8 --nfe 4 --temp 1.0
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bridge import CorrelatedNoise, StateUNet, decode_labels, sample_bridge
from .common import dice, largest_component, prior_side_masks, side_metrics
from .stats import mcnemar_topology, paired
from .decode import marginal, mcp_connect, tcmbr


def run_bridge(model, img, x0, roi, K, nfe, sigma, temp, noise, patch, dev):
    sl = tuple(slice(a, b) for a, b in roi)
    im = torch.from_numpy(np.asarray(img[sl], np.float32))[None, None].to(dev)
    xs = torch.from_numpy(np.asarray(x0[(slice(None),) + sl], np.float32))[None].to(dev)
    xs = xs.expand(K, -1, -1, -1, -1).contiguous()
    x = sample_bridge(model, im, xs, nfe, sigma, temp, noise, patch)
    lab = decode_labels(x).cpu().numpy()
    full = np.zeros((K,) + img.shape, np.uint8)
    full[(slice(None),) + sl] = lab
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_fold", type=int, default=0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.05, help="TC-MBR topology weight (preregistered)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_cldice", action="store_true")
    ap.add_argument("--save_masks", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    cache, out = Path(a.cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = torch.load(a.ckpt, map_location=dev)
    model = StateUNet(tuple(s["widths"])).to(dev).eval()
    model.load_state_dict(s["model"])
    clip, sigma = s["clip_mm"], s["sigma_mm"] / s["clip_mm"]
    noise = CorrelatedNoise(s["noise_corr_vox"], dev)
    patch = s["patch"]
    metas = [json.loads(p.read_text()) for p in sorted(cache.glob("*_meta.json"))]
    cases = sorted(m["case"] for m in metas if m["fold"] == a.eval_fold)
    if a.limit:
        cases = cases[: a.limit]
    rows = []
    for ci, case in enumerate(cases):
        t0 = time.time()
        meta = json.loads((cache / f"{case}_meta.json").read_text())
        sp = tuple(meta["spacing"])
        img = np.load(cache / f"{case}_img.npy", mmap_mode="r")
        x0 = (
            np.asarray(
                np.load(cache / f"{case}_x0.npy", mmap_mode="r"),
                dtype=np.float32,
            )
            / float(clip)
        )
        p = np.load(cache / f"{case}_p.npy").astype(np.float32) / 255.0
        gt = np.load(cache / f"{case}_gt.npy")
        gt2 = np.stack([gt == 1, gt == 2])
        pr2 = prior_side_masks(p)
        roi = meta["roi_inference"]
        if roi is None:
            continue
        torch.manual_seed(1000 + ci)
        det1 = run_bridge(model, img, x0, roi, 1, 1, sigma, 0.0, noise, patch, dev)[0]
        detN = run_bridge(model, img, x0, roi, 1, a.nfe, sigma, 0.0, noise, patch, dev)[0]
        smp = run_bridge(model, img, x0, roi, a.K, a.nfe, sigma, a.temp, noise, patch, dev)
        t_bridge = time.time() - t0
        saved = {}
        for si, side in enumerate(("L", "R")):
            g = gt2[si]
            S = smp == (si + 1)
            m = {"prior": pr2[si]}
            m["prior_lcc"] = largest_component(pr2[si])
            m["prior_mcp_soft"], _ = mcp_connect(pr2[si], p[si], sp, hard=False)
            m["prior_mcp_hard"], _ = mcp_connect(pr2[si], p[si], sp, hard=True)
            m["bridge_det1"] = det1 == si + 1
            m["bridge_detN"] = detN == si + 1
            m["bridge_detN_lcc"] = largest_component(m["bridge_detN"])
            m["bridge_sample0"] = S[0]
            m["bridge_marg"] = marginal(S)
            m["bridge_marg_lcc"] = largest_component(m["bridge_marg"])
            m["bridge_mbr_dice"], _ = tcmbr(S, sp, hard=False, lam=0.0)      # lam=0 control
            m["bridge_tcmbr_soft"], info_s = tcmbr(S, sp, hard=False, lam=a.lam)
            m["bridge_tcmbr_hard"], info_h = tcmbr(S, sp, hard=True, lam=a.lam)
            div = [dice(S[i], S[j]) for i, j in itertools.combinations(range(len(S)), 2)] if len(S) > 1 else [1.0]
            for name, mask in m.items():
                r = dict(case=case, side=side, method=name, **side_metrics(mask, g, sp, not a.no_cldice))
                if name == "bridge_tcmbr_soft":
                    r.update(sample_div_dice=float(np.mean(div)), **{f"tc_{k}": v for k, v in info_s.items()})
                if name == "bridge_tcmbr_hard":
                    r.update({f"tc_{k}": v for k, v in info_h.items()})
                rows.append(r)
            if a.save_masks:
                for name in ("bridge_detN", "bridge_tcmbr_soft", "bridge_tcmbr_hard"):
                    saved[f"{name}_{side}"] = np.packbits(m[name])
        if a.save_masks:
            np.savez_compressed(out / f"{case}_masks.npz", shape=np.array(gt.shape), **saved)
        print(f"[{ci+1}/{len(cases)}] {case} bridge {t_bridge:.1f}s total {time.time()-t0:.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "per_side.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.groupby("method").agg(dice=("dice", "mean"), hd95=("hd95", "mean"),
                                    hd95_p90=("hd95", lambda x: np.percentile(x, 90)),
                                    assd=("assd", "mean"), signed_offset=("signed_offset", "mean"),
                                    vol_ratio=("vol_ratio", "mean"), beta0_err=("beta0_err", "mean"),
                                    sides_topo_ok=("beta0_err", lambda x: float((x == 0).mean())),
                                    **({} if a.no_cldice else dict(cldice=("cldice", "mean"))))
    report = dict(config=vars(a), n_cases=len(cases), summary=summ.round(5).to_dict(orient="index"))
    comps = [("prior", "bridge_tcmbr_soft"), ("prior_mcp_soft", "bridge_tcmbr_soft"),
             ("prior_mcp_hard", "bridge_tcmbr_hard"), ("prior_lcc", "bridge_tcmbr_hard"),
             ("bridge_detN", "bridge_tcmbr_soft"), ("bridge_marg", "bridge_tcmbr_soft"),
             ("bridge_mbr_dice", "bridge_tcmbr_soft"),
             ("bridge_det1", "bridge_detN")]
    report["paired"] = {f"{x} -> {y}": dict(dice=paired(df, x, y, "dice"), hd95=paired(df, x, y, "hd95"),
                                            topology=mcnemar_topology(df, x, y)) for x, y in comps}
    tc = df[df.method == "bridge_tcmbr_soft"]
    report["sample_diversity_mean_pairwise_dice"] = float(tc.sample_div_dice.mean())
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
