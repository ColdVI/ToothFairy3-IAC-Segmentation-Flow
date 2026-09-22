"""Contracts + synthetic end-to-end run (CPU, ~minutes). Run before touching real data:
    python -m iacb.selftest --work /tmp/iacb_selftest
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import components, dice, largest_component, sdf_mm
from .decode import marginal, tcmbr

SP = (0.3, 0.3, 0.3)


def tube(shape, cy, cz, r, x0=6, x1=None, yoff=None):
    Z, Y, X = shape
    x1 = x1 or X - 6
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    off = np.zeros(X) if yoff is None else yoff
    cyx = cy + 3 * np.sin(np.arange(X) / 14.0) + off
    czx = cz + 2 * np.cos(np.arange(X) / 18.0)
    m = (yy - cyx[xx]) ** 2 + (zz - czx[xx]) ** 2 <= r ** 2
    return m & (xx >= x0) & (xx < x1)


def ramp(X, a, b, width, shift):
    x = np.arange(X, dtype=float)
    up = np.clip((x - (a - width)) / width, 0, 1)
    down = np.clip(((b + width) - x) / width, 0, 1)
    return shift * np.minimum(up, down)


def test_contracts():
    shape = (32, 64, 96)
    m = tube(shape, 24, 16, 3)
    s = sdf_mm(m, SP, 4.0)
    assert np.array_equal(s < 0, m), "SDF sign round-trip failed"
    # correlated noise variance + identity contract of the untrained network
    import torch
    from .bridge import CorrelatedNoise, StateUNet, decode_labels, sample_bridge
    n = CorrelatedNoise(3.0, torch.device("cpu"))(( 1, 1, 64, 64, 64), torch.device("cpu"))
    std = float(n[..., 16:-16, 16:-16, 16:-16].std())
    assert 0.8 < std < 1.2, f"noise std {std}"
    model = StateUNet((8, 16)).eval()
    x0 = torch.from_numpy(np.stack([s, np.full(shape, 4.0, np.float32)]) / 4.0)[None]
    img = torch.zeros((1, 1) + shape)
    out = sample_bridge(model, img, x0, nfe=4, sigma=0.3, temp=0.0,
                        noise=CorrelatedNoise(3.0, torch.device("cpu")), patch=(32, 64, 64), amp=False)
    lab = decode_labels(out)[0].numpy()
    assert np.array_equal(lab == 1, m) and not (lab == 2).any(), "identity contract (zero-init, temp 0) failed"
    print("[ok] SDF round trip, noise variance %.3f, identity contract" % std)


def test_tcmbr():
    shape = (32, 72, 120)
    X = shape[2]
    shifts = [-14, -7, 0, 7, 14]
    S = np.stack([tube(shape, 36, 16, 3, yoff=ramp(X, 48, 72, 8, s)) for s in shifts])
    gt = S[2]
    assert all(components(s)[1] == 1 for s in S)
    M = marginal(S)
    nM = components(M)[1]
    soft, info_s = tcmbr(S, SP, hard=False)
    hard, info_h = tcmbr(S, SP, hard=True)
    lcc = largest_component(M)
    print(f"[tcmbr] marginal comps={nM} dice={dice(M, gt):.3f} | lcc dice={dice(lcc, gt):.3f} "
          f"| soft comps={components(soft)[1]} dice={dice(soft, gt):.3f} {info_s} "
          f"| hard comps={components(hard)[1]} dice={dice(hard, gt):.3f} {info_h}")
    assert nM >= 2, "test construction should produce a marginal gap"
    assert components(soft)[1] == 1 and components(hard)[1] == 1
    assert dice(soft, gt) > dice(lcc, gt)
    # evidence-absent case: every sample itself has the gap -> soft must NOT invent a bridge
    S2 = np.stack([tube(shape, 36, 16, 3) & ~((np.arange(X) >= 55) & (np.arange(X) < 65))[None, None, :]
                   for _ in range(5)])
    soft2, info2 = tcmbr(S2, SP, hard=False)
    assert components(soft2)[1] == 2 and info2["kept"] == 1, info2
    print("[ok] TC-MBR grafts multimodal gaps, keeps evidence-absent gaps (soft)")


def test_gap_classes():
    from scipy.ndimage import gaussian_filter
    from .p0_anatomy import gap_anatomy
    shape = (40, 72, 104); X = shape[2]; a = 45
    L = tube(shape, 30, 20, 3)
    seg = np.zeros(X, bool); seg[a:a + 14] = True

    def prob(kind):
        pl = gaussian_filter(L.astype(np.float32), 1.0)
        if kind == "smeared":
            alt = [gaussian_filter(tube(shape, 30, 20, 3, yoff=ramp(X, a + 2, a + 12, 2, s)).astype(np.float32), 1.0)
                   for s in (-7, 7)]
            pl[..., seg] = 0.45 * (alt[0][..., seg] + alt[1][..., seg])
        elif kind == "absent":
            pl[..., seg] *= 0.05
        else:
            alt = gaussian_filter(tube(shape, 30, 20, 3, yoff=ramp(X, a + 2, a + 12, 2, 7)).astype(np.float32), 1.0)
            pl[..., seg] = alt[..., seg]
        return np.clip(pl, 0, 1)

    for kind in ("smeared", "absent", "displaced"):
        p = prob(kind)
        o = gap_anatomy(p >= 0.5, p, L, SP)
        got = max(("displaced", "smeared", "absent"), key=lambda c: o[f"miss_{c}_mm"])
        assert o["miss_len_mm"] > 0 and got == kind, (kind, o)
    print("[ok] P0 gap classifier recovers constructed smeared / absent / displaced gaps")


def test_oracle_hd95():
    from .p0_anatomy import gap_anatomy
    shape = (40, 72, 104)
    L = tube(shape, 30, 20, 3)
    prior = L.copy()
    prior[..., 50:62] = False                         # missed segment
    prior[5:9, 60:64, 10:14] = True                   # spurious blob far from the canal
    prob = prior.astype(np.float32)
    o = gap_anatomy(prior, prob, L, SP)
    assert o["oracle_hd95_drop_spurious"] < o["hd95_prior"], o
    assert o["oracle_hd95_component"] <= o["oracle_hd95_drop_spurious"] + 1e-9, o
    from .common import dice as _d
    assert o["oracle_dice_component"] > _d(prior, L), o   # boundary-band FN at gap ends stay (by design)
    print(f"[ok] oracle HD95 decomposition: prior {o['hd95_prior']:.2f} -> drop spurious "
          f"{o['oracle_hd95_drop_spurious']:.2f} -> component oracle {o['oracle_hd95_component']:.2f} mm")


def make_synthetic(root: Path, n=10, seed=0):
    import SimpleITK as sitk
    rng = np.random.default_rng(seed)
    shape = (40, 72, 104)
    X = shape[2]
    for d in ("imagesTr", "labelsTr", "oof"):
        (root / d).mkdir(parents=True, exist_ok=True)
    from scipy.ndimage import gaussian_filter
    names = []
    for i in range(n):
        name = f"SYN_{i:03d}"; names.append(name)
        L = tube(shape, 20, 20, 3); R = tube(shape, 52, 20, 3)
        lab = np.zeros(shape, np.uint8); lab[L] = 1; lab[R] = 2
        img = np.where(lab > 0, -1.0, 1.0) + rng.normal(0, 0.6, shape)
        a = int(rng.integers(35, 60))
        img[..., a:a + 10] = rng.normal(0, 0.6, img[..., a:a + 10].shape)  # faded segment
        pl = gaussian_filter(L.astype(np.float32), 1.0)
        pr = gaussian_filter(R.astype(np.float32), 1.0)
        if i % 2 == 0:                      # smeared gap on the left canal
            alt = [gaussian_filter(tube(shape, 20, 20, 3, yoff=ramp(X, a, a + 10, 6, s)).astype(np.float32), 1.0)
                   for s in (-7, 7)]
            seg = np.zeros(X, bool); seg[a - 6:a + 16] = True
            pl[..., seg] = 0.45 * (alt[0][..., seg] + alt[1][..., seg])
        pl = np.clip(pl, 0, 1); pr = np.clip(pr, 0, 1)
        s = np.clip(pl + pr, 0, 1)
        prob = np.stack([1 - s, pl, pr]).astype(np.float16)
        for arr, path in ((img.astype(np.float32), root / "imagesTr" / f"{name}_0000.nii.gz"),
                          (lab, root / "labelsTr" / f"{name}.nii.gz")):
            im = sitk.GetImageFromArray(arr); im.SetSpacing(SP[::-1]); sitk.WriteImage(im, str(path))
        np.savez_compressed(root / "oof" / f"{name}.npz", probabilities=prob)
    splits = [dict(train=[c for c in names if c not in names[k::5]], val=names[k::5]) for k in range(5)]
    (root / "splits_final.json").write_text(json.dumps(splits))


def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"failed: {cmd[2]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/iacb_selftest")
    ap.add_argument("--skip_e2e", action="store_true")
    a = ap.parse_args()
    test_contracts()
    test_tcmbr()
    test_gap_classes()
    test_oracle_hd95()
    if a.skip_e2e:
        return
    root = Path(a.work)
    make_synthetic(root)
    py = sys.executable
    common = ["--labels_dir", str(root / "labelsTr"), "--prob_dir", str(root / "oof"),
              "--splits", str(root / "splits_final.json")]
    o = sh([py, "-m", "iacb.p0_anatomy", *common, "--folds", "0", "--out", str(root / "p0"), "--workers", "1",
            "--min_topo_sides", "1", "--min_remaining", "0"])
    print(o[-1500:])
    sh([py, "-m", "iacb.cache", "--images_dir", str(root / "imagesTr"), *common, "--out", str(root / "cache"),
        "--workers", "1", "--margin_mm", "3"])
    o = sh([py, "-m", "iacb.train", "--cache", str(root / "cache"), "--out", str(root / "run"), "--iters", "12",
            "--patch", "32,48,48", "--batch", "2", "--widths", "8,16,32", "--workers", "0", "--holdout_n", "2",
            "--val_every", "6", "--save_every", "6", "--device", "cpu"])
    print(o[-800:])
    o = sh([py, "-m", "iacb.infer", "--cache", str(root / "cache"), "--ckpt", str(root / "run" / "last.pt"),
            "--out", str(root / "eval"), "--K", "3", "--nfe", "2", "--device", "cpu"])
    rep = json.loads((root / "eval" / "report.json").read_text())
    print(json.dumps(rep["summary"], indent=1)[:2500])
    # external (supervisor-style binary) predictions evaluated with the same metric code
    import SimpleITK as sitk
    ext = root / "ext_binary"; ext.mkdir(exist_ok=True)
    fold0 = json.loads((root / "splits_final.json").read_text())[0]["val"]
    for c in fold0:
        pr = np.load(root / "oof" / f"{c}.npz")["probabilities"].astype(np.float32)
        b = (pr[1:].sum(0) >= 0.5).astype(np.uint8)
        b[2:5, 2:5, 2:5] = 1                                  # a spurious component
        im = sitk.GetImageFromArray(b); im.SetSpacing(SP[::-1]); sitk.WriteImage(im, str(ext / f"{c}.nii.gz"))
    o = sh([py, "-m", "iacb.compare_external", "--pred_dir", str(ext), "--labels_dir", str(root / "labelsTr"),
            "--splits", str(root / "splits_final.json"), "--folds", "0", "--name", "teacher_like",
            "--pred_kind", "binary", "--out", str(root / "cmp"), "--workers", "1",
            "--merge_per_side", str(root / "eval" / "per_side.csv")])
    crep = json.loads((root / "cmp" / "report.json").read_text())
    assert crep["merged"]["n_common_sides"] > 0 and "prior" in crep["paired_vs_external"], crep.keys()
    assert crep["paired_vs_external"]["prior"]["topology"]["fixed"] > 0, crep["paired_vs_external"]["prior"]
    print("[ok] compare_external: binary preds split, paired against infer per_side.csv")
    o = sh([py, "-m", "iacb.ae_gate", "--cache", str(root / "cache"), "--out", str(root / "ae"), "--factor", "2",
            "--zc", "4", "--width", "8", "--iters", "6", "--patch", "16,24,24", "--batch", "2", "--workers", "0",
            "--limit_eval", "1", "--device", "cpu"])
    g = json.loads((root / "ae" / "ae_gate.json").read_text())
    assert g["n_sides"] > 0 and "verdict" in g
    print("[ok] ae_gate ran (verdict on 6 iterations is meaningless, only the plumbing is tested)")
    # run_all on a fresh work dir: one command, cache -> diagnose -> train -> infer
    o = sh([py, "-m", "iacb.run_all", "--raw", str(root), "--prob_dir", str(root / "oof"),
            "--splits", str(root / "splits_final.json"), "--work", str(root / "ra"),
            "--teacher_dir", str(ext), "--iters", "8", "--workers", "1", "--limit", "4"])
    ra = json.loads((root / "ra" / "run_all_report.json").read_text())
    assert ra["stages"]["train"]["ok"] and ra["stages"]["infer"]["ok"], ra["stages"]
    assert "diagnostics" in ra and "results" in ra and "teacher_comparison" in ra, sorted(ra)
    print("[ok] run_all: single command completed, diagnostics reported without blocking")
    o = sh([py, "-m", "iacb.run_all", "--raw", str(root), "--prob_dir", str(root / "oof"),
            "--splits", str(root / "splits_final.json"), "--work", str(root / "ra"),
            "--iters", "8", "--workers", "1", "--limit", "4"])
    print("[ok] run_all is re-runnable (resume after a Colab disconnect)")
    print("[ok] end-to-end synthetic run finished")


if __name__ == "__main__":
    main()
