"""Shared IO, geometry and metrics. Pure numpy/scipy/skimage (no torch).

Axis convention: every array is (Z, Y, X) as returned by SimpleITK.GetArrayFromImage.
nnU-Net v2 `--save_probabilities` npz files use the same order, so no permutation is
applied anywhere. A Dice sanity check (prior vs GT) catches silent axis swaps that
shape checks cannot (square in-plane volumes).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

CONN26 = np.ones((3, 3, 3), dtype=bool)


# ----------------------------------------------------------------------------- IO
def read_image(path: Path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)                      # (Z, Y, X)
    spacing_zyx = tuple(float(s) for s in img.GetSpacing()[::-1])
    return arr, spacing_zyx




def resolve_oof_npz(prob_dir: Path, case: str, fold: int | None = None) -> Path:
    """Resolve a frozen nnU-Net probability file without recursively scanning Drive.

    Development cases (fold >= 0) use their true out-of-fold probability:
      <prob_dir>/fold_<k>/<case>.npz

    The 52 ToothFairy3S cases were never seen by *any* of the five frozen nnU-Net
    folds.  They are assigned one source fold each and exported once to:
      <prob_dir>/external_singlefold/<case>.npz

    A flat <prob_dir>/<case>.npz layout is also accepted for compatibility.
    """
    prob_dir = Path(prob_dir)
    candidates = [prob_dir / f"{case}.npz"]
    if fold is not None and fold < 0:
        candidates += [
            prob_dir / "external_singlefold" / f"{case}.npz",
            prob_dir / "external_ensemble" / f"{case}.npz",
        ]
    elif fold is not None:
        candidates += [prob_dir / f"fold_{fold}" / f"{case}.npz",
                       prob_dir / str(fold) / f"{case}.npz"]
    else:
        for k in range(5):
            candidates.append(prob_dir / f"fold_{k}" / f"{case}.npz")
        candidates += [
            prob_dir / "external_singlefold" / f"{case}.npz",
            prob_dir / "external_ensemble" / f"{case}.npz",
        ]
    for p in candidates:
        if p.exists():
            return p
    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"nnU-Net probability npz not found for {case}. Tried:\n  - {tried}")

def load_oof_probs(npz_path: Path, left_id: int, right_id: int) -> np.ndarray:
    """Returns float32 (2, Z, Y, X): [P(left), P(right)]."""
    with np.load(npz_path) as z:
        key = "probabilities" if "probabilities" in z.files else z.files[0]
        p = z[key]
    if p.ndim != 4:
        raise ValueError(f"{npz_path}: expected (C,Z,Y,X), got {p.shape}")
    return np.stack([p[left_id], p[right_id]]).astype(np.float32)


def side_masks(labels: np.ndarray, left_id: int, right_id: int) -> np.ndarray:
    return np.stack([labels == left_id, labels == right_id])


def prior_side_masks(p2: np.ndarray) -> np.ndarray:
    """3-class argmax from [P(L), P(R)] with P(bg) = 1 - P(L) - P(R)."""
    bg = 1.0 - p2.sum(0)
    full = np.concatenate([bg[None], p2], 0)
    am = full.argmax(0)
    return np.stack([am == 1, am == 2])


def case_folds(splits_json: Path, include_external: bool = True) -> dict:
    """Return {case_id: source/eval fold} for the project split file.

    Development cases receive their validation fold 0..4, which identifies the
    nnU-Net model that did *not* train on that case.  When ``include_external`` is
    true, the 52 ``external_test`` cases are included with fold ``-1``.  Those cases
    were unseen by all five frozen nnU-Net folds and are therefore safe to use as
    extra refiner-training cases once a frozen prior has been exported for them.

    Legacy list-of-folds JSON is also supported; it has no external cases.
    """
    obj = json.loads(Path(splits_json).read_text())
    external = []
    if isinstance(obj, dict):
        splits = obj.get("folds")
        if not isinstance(splits, list):
            raise ValueError(f"{splits_json}: dict split file has no list-valued 'folds'")
        if include_external:
            external = list(obj.get("external_test", []))
    elif isinstance(obj, list):
        splits = obj
    else:
        raise ValueError(f"{splits_json}: unsupported split JSON type {type(obj).__name__}")

    out = {}
    for k, split in enumerate(splits):
        if not isinstance(split, dict) or "val" not in split:
            raise ValueError(f"{splits_json}: fold {k} has no 'val' list")
        for case in split["val"]:
            if case in out:
                raise ValueError(f"{splits_json}: case {case} appears in multiple validation folds")
            out[case] = k
    for case in external:
        if case in out:
            raise ValueError(f"{splits_json}: external case {case} also appears in a development fold")
        out[case] = -1
    return out


# ----------------------------------------------------------------------- geometry
def bbox(mask: np.ndarray, margin_vox=0, shape=None):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    shape = shape or mask.shape
    m = np.broadcast_to(np.asarray(margin_vox), (3,))
    lo = np.maximum(idx.min(0) - m, 0)
    hi = np.minimum(idx.max(0) + 1 + m, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def union_bbox(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return tuple(slice(min(b[i].start for b in boxes), max(b[i].stop for b in boxes)) for i in range(3))


def sdf_mm(mask: np.ndarray, spacing, clip_mm: float) -> np.ndarray:
    """Signed distance in mm, negative inside, clipped to +-clip_mm."""
    if not mask.any():
        return np.full(mask.shape, clip_mm, np.float32)
    if mask.all():
        return np.full(mask.shape, -clip_mm, np.float32)
    dout = ndi.distance_transform_edt(~mask, sampling=spacing)
    din = ndi.distance_transform_edt(mask, sampling=spacing)
    # voxel-centre convention: inside voxels get -(din - 0.5 vox), outside +(dout - 0.5 vox)
    half = 0.5 * float(np.mean(spacing))
    s = np.where(mask, -(din - half), dout - half)
    return np.clip(s, -clip_mm, clip_mm).astype(np.float32)


def components(mask: np.ndarray):
    lab, n = ndi.label(mask, structure=CONN26)
    if n == 0:
        return lab, 0, np.zeros(0, np.int64)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    return lab, n, sizes


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n, sizes = components(mask)
    if n <= 1:
        return mask.copy()
    return lab == (int(np.argmax(sizes)) + 1)


def radius_estimate_vox(mask: np.ndarray) -> float:
    """Robust tube radius in voxels: 90th pct of the interior EDT."""
    if not mask.any():
        return 0.0
    d = ndi.distance_transform_edt(mask)
    return float(np.percentile(d[mask], 90))


def skeleton(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    if not mask.any():
        return mask.copy()
    return skeletonize(mask).astype(bool)


# ------------------------------------------------------------------------ metrics
def dice(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = int(a.sum()), int(b.sum())
    if sa + sb == 0:
        return 1.0
    return 2.0 * int((a & b).sum()) / (sa + sb)


def _surface(m):
    return m & ~ndi.binary_erosion(m, structure=CONN26, border_value=0)


def surface_stats(pred: np.ndarray, gt: np.ndarray, spacing, empty_value_mm: float = 50.0):
    """HD95 (symmetric, pooled), ASSD and signed mean offset of the predicted surface
    w.r.t. GT (+ = prediction surface outside GT = over-segmentation / 'şişme')."""
    if not pred.any() or not gt.any():
        v = 0.0 if (not pred.any() and not gt.any()) else empty_value_mm
        return dict(hd95=v, assd=v, signed_offset=np.nan)
    box = union_bbox([bbox(pred, 3), bbox(gt, 3)])
    p, g = pred[box], gt[box]
    sp, sg = _surface(p), _surface(g)
    dg = ndi.distance_transform_edt(~sg, sampling=spacing)
    dp = ndi.distance_transform_edt(~sp, sampling=spacing)
    d_pg, d_gp = dg[sp], dp[sg]
    allv = np.concatenate([d_pg, d_gp])
    sign = np.where(g[sp], -1.0, 1.0)
    return dict(hd95=float(np.percentile(allv, 95)), assd=float(allv.mean()),
                signed_offset=float((sign * d_pg).mean()))


def cldice(pred: np.ndarray, gt: np.ndarray):
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    box = union_bbox([bbox(pred, 2), bbox(gt, 2)])
    p, g = pred[box], gt[box]
    skp, skg = skeleton(p), skeleton(g)
    tprec = (skp & g).sum() / max(skp.sum(), 1)
    tsens = (skg & p).sum() / max(skg.sum(), 1)
    return float(2 * tprec * tsens / max(tprec + tsens, 1e-8))


def side_metrics(pred: np.ndarray, gt: np.ndarray, spacing, with_cldice=True):
    _, npred, _ = components(pred)
    _, ngt, _ = components(gt)
    out = dict(dice=dice(pred, gt), beta0_pred=npred, beta0_gt=ngt,
               beta0_err=abs(npred - ngt),
               vol_ratio=float(pred.sum() / max(gt.sum(), 1)))
    out.update(surface_stats(pred, gt, spacing))
    if with_cldice:
        out["cldice"] = cldice(pred, gt)
    return out


def case_dice_3class(pred2: np.ndarray, gt2: np.ndarray) -> float:
    """Mean of left/right Dice (the per-case number used in earlier tables)."""
    return float(np.mean([dice(pred2[i], gt2[i]) for i in range(2)]))
