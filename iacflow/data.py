from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.data import Dataset

from .core import atomic_json, decode_sdf, fingerprint, read_json


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


def clipped_sdf(mask, spacing, delta_mm=3.0):
    """Exactly truncated full-grid EDT; skip the provably constant far field.

    This is GT target preprocessing, never an inference ROI. The ROI has at
    least delta+one voxel halo, and is only clipped at the original grid edge.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.ones(mask.shape, dtype=np.float32)
    if not mask.any():
        return out
    coords = np.where(mask)
    halo = np.ceil(delta_mm/np.asarray(spacing)).astype(int)+1
    lo = np.maximum([int(v.min()) for v in coords]-halo, 0)
    hi = np.minimum(np.asarray([int(v.max()) for v in coords])+halo+1, mask.shape)
    sl = tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
    roi = mask[sl]
    dist = distance_transform_edt(~roi, sampling=spacing)
    dist -= distance_transform_edt(roi, sampling=spacing)
    np.clip(dist, -delta_mm, delta_mm, out=dist)
    out[sl] = (dist/delta_mm).astype(np.float32)
    return out


def load_preprocessed(folder, case):
    root = Path(folder)
    if (root/f"{case}.npy").exists() and (root/f"{case}_seg.npy").exists():
        image = np.load(root/f"{case}.npy", mmap_mode="r", allow_pickle=False)
        seg = np.load(root/f"{case}_seg.npy", mmap_mode="r", allow_pickle=False)
    elif (root/f"{case}.npz").exists():
        with np.load(root/f"{case}.npz", allow_pickle=False) as f:
            image, seg = f["data"], f["seg"]
    elif (root/f"{case}.b2nd").exists():
        import blosc2
        image = blosc2.open(str(root/f"{case}.b2nd"), mode="r", dparams={"nthreads":1})[:]
        seg = blosc2.open(str(root/f"{case}_seg.b2nd"), mode="r", dparams={"nthreads":1})[:]
    else:
        raise FileNotFoundError(f"Missing preprocessed case {case} in {root}; raw NIfTI is not a substitute.")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image [1,D,H,W], got {image.shape}")
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.shape != image.shape[1:]:
        raise ValueError(f"Image/label shape mismatch for {case}")
    return image, seg


def load_splits(path, fold, patient_map=None):
    obj = read_json(path)
    split = obj[int(fold)] if isinstance(obj, list) else obj
    train, val = list(split["train"]), list(split["val"])
    if not train or not val or set(train)&set(val) or len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Split must contain nonempty, disjoint, unique train/val case lists")
    mapping = read_json(patient_map) if patient_map else {}
    patient = lambda c: mapping.get(c, re.sub(r"_(L|R)$", "", c))
    overlap = {patient(c) for c in train}&{patient(c) for c in val}
    if overlap:
        raise ValueError(f"Patient leakage across split: {sorted(overlap)[:8]}")
    for case in train+val:
        if Path(case).name != case or case in (".", ".."):
            raise ValueError("Case IDs must be filenames without directory components")
    return {"train": train, "val": val}


def source_signature(folder, case):
    root = Path(folder)
    out = []
    for suffix in (".npy", "_seg.npy", ".npz", ".b2nd", "_seg.b2nd"):
        p = root/f"{case}{suffix}"
        if p.exists():
            stat = p.stat()
            out.append([p.name, stat.st_size, stat.st_mtime_ns])
    if not out:
        raise FileNotFoundError(case)
    return out


def _prepare_case(job):
    source, dest, case, spacing, delta, ids = job
    case_dir = Path(dest)/case
    case_dir.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({"source":source_signature(source,case), "spacing":spacing,
                             "delta_mm":delta, "class_ids":ids, "format":1})
    done = case_dir/"case.json"
    if done.exists() and read_json(done).get("signature") == signature:
        needed = [case_dir/x for x in ("image.npy","label.npy","sdf.npy","locations.npz")]
        if all(p.exists() for p in needed):
            return {"case":case,"cached":True,"seconds":0}
    t0 = time.perf_counter()
    image, seg = load_preprocessed(source,case)
    if not np.isfinite(image).all() or not set(np.unique(seg).tolist()).issubset({-1,0,1,2}):
        raise ValueError(f"Invalid image/labels for {case}")
    seg = seg.astype(np.int8)
    sdf = np.stack([clipped_sdf(seg == c, spacing, delta) for c in ids]).astype(np.float16)
    recovered = decode_sdf(sdf, *ids)
    if not np.array_equal(recovered[seg>=0], seg[seg>=0]):
        raise RuntimeError(f"GT SDF round-trip failed for {case}; training must not start")
    atomic_npy(case_dir/"image.npy", image.astype(np.float32, copy=False))
    atomic_npy(case_dir/"label.npy", seg)
    atomic_npy(case_dir/"sdf.npy", sdf)
    rng = np.random.default_rng(int(hashlib.sha256(case.encode()).hexdigest()[:8],16))
    loc = {}
    for c in ids:
        coords = np.argwhere(seg == c)
        if len(coords)>20000:
            coords = coords[rng.choice(len(coords),20000,replace=False)]
        loc[str(c)] = coords.astype(np.int32)
    with open(case_dir/"locations.npz.partial","wb") as f:
        np.savez(f,**loc)
    os.replace(case_dir/"locations.npz.partial",case_dir/"locations.npz")
    # A source change invalidates any previous baseline weights as well.
    (case_dir/"reference.json").unlink(missing_ok=True)
    meta = {"case":case,"shape":list(seg.shape),"spacing":list(spacing),"delta_mm":delta,
            "signature":signature,"roundtrip_exact":True,"source":str(Path(source).resolve()),
            "seconds":time.perf_counter()-t0}
    atomic_json(done,meta)
    return {**meta,"cached":False}


def prepare_cpu_cache(config, info, splits):
    root = Path(config["cache_dir"])
    root.mkdir(parents=True,exist_ok=True)
    # No resampling/renormalization; source must match the checkpoint's preprocessed configuration.
    declared = config.get("preprocessed_data_identifier")
    if declared != info["data_identifier"]:
        raise ValueError(f"Set preprocessed_data_identifier to {info['data_identifier']!r} only after checking the source folder.")
    cases = splits["train"]+splits["val"]
    jobs = [(config["preprocessed_dir"],str(root),c,info["spacing"],config["delta_mm"],
             [config["left_id"],config["right_id"]]) for c in cases]
    manifest_path = root/"manifest.json"
    contract = {"plans_sha256":info["plans_sha256"],"dataset_sha256":info["dataset_sha256"],
                "spacing":info["spacing"],"delta_mm":config["delta_mm"],"ids":[config["left_id"],config["right_id"]],
                "split":splits,"configuration":info["configuration"]}
    if manifest_path.exists() and read_json(manifest_path)["contract"] != contract:
        raise ValueError("Cache contract differs. Use a new cache directory; do not overwrite another experiment.")
    t0 = time.perf_counter()
    workers = int(config.get("cache_workers",1))
    pool = ProcessPoolExecutor(workers) if workers>1 else None
    try:
        iterator = pool.map(_prepare_case,jobs) if pool else map(_prepare_case,jobs)
        for i,result in enumerate(iterator,1):
            elapsed = time.perf_counter()-t0
            print(f"[CPU CACHE {i}/{len(cases)}] {result['case']} {'reuse' if result['cached'] else 'written'} "
                  f"elapsed={elapsed/60:.1f}m ETA={(len(cases)-i)*elapsed/max(i,1)/60:.1f}m",flush=True)
    finally:
        if pool:
            pool.shutdown()
    manifest = {"contract":contract,"cases":cases,"complete":True}
    atomic_json(manifest_path,manifest)
    return manifest


def write_reference(cache_dir, case, mask, info, config):
    root = Path(cache_dir)/case
    meta = read_json(root/"case.json")
    mask = np.asarray(mask)
    if tuple(mask.shape) != tuple(meta["shape"]) or not set(np.unique(mask)).issubset({0,1,2}):
        raise ValueError(f"Reference grid/labels mismatch for {case}; no automatic resampling")
    fields = [clipped_sdf(mask == c,info["spacing"],config["delta_mm"])
              for c in (config["left_id"],config["right_id"])]
    weights = np.stack([config["weight_floor"]+(1-config["weight_floor"])*
                        np.exp(-0.5*(d*config["delta_mm"]/config["band_mm"])**2) for d in fields])
    atomic_npy(root/"weight.npy",weights.astype(np.float16))
    atomic_npy(root/"reference.npy",mask.astype(np.uint8))
    atomic_json(root/"reference.json",{"checkpoint_sha256":info["checkpoint_sha256"],
                "case_signature":meta["signature"],"weight_floor":config["weight_floor"],
                "band_mm":config["band_mm"],"overlap":config["overlap"],"precision":config["precision"],
                "patch_size":config["patch_size"],"tta":False,
                "provenance":"same-fold original checkpoint; image-only"})


def reference_complete(cache_dir,case,info,config):
    root=Path(cache_dir)/case
    p=root/"reference.json"
    if not p.exists() or not (root/"weight.npy").exists() or not (root/"reference.npy").exists():
        return False
    m=read_json(p)
    return (m["checkpoint_sha256"] == info["checkpoint_sha256"] and
            m["case_signature"] == read_json(root/"case.json")["signature"] and
            m["weight_floor"] == config["weight_floor"] and m["band_mm"] == config["band_mm"] and
            m.get("overlap") == config["overlap"] and m.get("patch_size") == config["patch_size"] and
            m.get("precision") == config["precision"])


def take_patch(array,start,shape,fill):
    spatial=np.asarray(array.shape[-3:]); start=np.asarray(start); shape=np.asarray(shape)
    lo=np.maximum(start,0); hi=np.minimum(start+shape,spatial)
    out=np.full((*array.shape[:-3],*shape),fill,dtype=array.dtype)
    if np.any(hi<=lo):
        return out
    src=(...,)+tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
    dst=(...,)+tuple(slice(int(a),int(b)) for a,b in zip(lo-start,hi-start))
    out[dst]=array[src]
    return out


class PatchDataset(Dataset):
    def __init__(self,cache_dir,cases,patch_size,samples,seed=123,foreground_probability=.5):
        self.root=Path(cache_dir); self.cases=list(cases); self.patch=np.asarray(patch_size)
        self.samples=int(samples); self.seed=int(seed); self.fg=float(foreground_probability)
        self.opened=OrderedDict()

    def __len__(self):
        return self.samples

    def _case(self,case):
        if case not in self.opened:
            root=self.root/case
            arrays={k:np.load(root/f"{k}.npy",mmap_mode="r",allow_pickle=False)
                    for k in ("image","label","sdf","weight")}
            with np.load(root/"locations.npz",allow_pickle=False) as f:
                arrays["locations"]=[f[k].copy() for k in f.files if len(f[k])]
            self.opened[case]=arrays
            while len(self.opened)>2:
                self.opened.popitem(last=False)
        self.opened.move_to_end(case)
        return self.opened[case]

    def __getitem__(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(index)]))
        case=self.cases[int(rng.integers(len(self.cases)))]; a=self._case(case)
        if rng.random()<self.fg and a["locations"]:
            points=a["locations"][int(rng.integers(len(a["locations"])))]
            center=points[int(rng.integers(len(points)))]
        else:
            center=rng.integers(np.asarray(a["label"].shape))
        start=center-rng.integers(self.patch)
        label=take_patch(a["label"],start,self.patch,-1)
        return {"image":torch.from_numpy(take_patch(a["image"],start,self.patch,0)),
                "target":torch.from_numpy(take_patch(a["sdf"],start,self.patch,1)),
                "weight":torch.from_numpy(take_patch(a["weight"],start,self.patch,0)),
                "valid":torch.from_numpy((label>=0)[None]),"sample_index":int(index)}


def worker_init(_):
    torch.set_num_threads(1)


def cache_ready(config,info,splits):
    root=Path(config["cache_dir"])
    manifest=read_json(root/"manifest.json")
    expected={"plans_sha256":info["plans_sha256"],"dataset_sha256":info["dataset_sha256"],
              "spacing":info["spacing"],"delta_mm":config["delta_mm"],"ids":[config["left_id"],config["right_id"]],
              "split":splits,"configuration":info["configuration"]}
    if not manifest.get("complete") or manifest["contract"] != expected:
        raise ValueError("Incomplete cache or wrong split")
    missing=[c for c in splits["train"]+splits["val"] if not reference_complete(root,c,info,config)]
    if missing:
        raise RuntimeError(f"Missing/stale image-only reference weights for {len(missing)} cases: {missing[:5]}. Run reference preparation first.")
    # Bind resume to individual source versions too, without rereading multi-GB arrays.
    versions={c:{"case":read_json(root/c/"case.json")["signature"],"reference":read_json(root/c/"reference.json")}
              for c in splits["train"]+splits["val"]}
    return fingerprint({"manifest":manifest,"versions":versions})
