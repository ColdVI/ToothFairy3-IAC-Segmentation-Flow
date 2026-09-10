from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path
import time

import numpy as np
import torch

from .core import amp_context, decode_sdf, read_json
from .data import reference_complete, write_reference


def tile_starts(shape, patch, overlap=.5):
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0,1)")
    axes = []
    for n, p in zip(shape, patch):
        if n < p:
            raise ValueError("Pad volume before tiling")
        count = int(math.ceil((n-p)/(p*(1-overlap))))+1
        axes.append(np.unique(np.round(np.linspace(0,n-p,count)).astype(int)).tolist())
    return list(itertools.product(*axes))


def gaussian_window(patch):
    grids = np.meshgrid(*[(np.arange(p)-(p-1)/2)/(p/8) for p in patch], indexing="ij")
    return np.maximum(np.exp(-.5*sum(g*g for g in grids)),1e-4).astype(np.float32)


def case_seed(seed, case):
    return (int(seed)+int(hashlib.sha256(case.encode()).hexdigest()[:8],16)) % (2**32)


class VolumeEngine:
    """Synchronous full-volume state, overlapping neural evaluations.

    CPU feature cache has fixed admission: once full, uncached tiles are recomputed.
    No eviction churn, no disk feature cache, and never reused after model updates.
    """
    def __init__(self, model, image, config):
        self.model = model
        self.device = next(model.parameters()).device
        self.precision = config["precision"]
        self.patch = tuple(config["patch_size"])
        self.original_shape = tuple(image.shape[1:])
        self.shape = tuple(max(a,b) for a,b in zip(self.original_shape,self.patch))
        estimate_gb = np.prod(self.shape)*40/2**30 + config["feature_cache_mb"]/1024
        if estimate_gb > config["max_inference_ram_gb"]:
            raise MemoryError(f"Estimated inference working RAM {estimate_gb:.1f} GiB exceeds configured limit. "
                              "Raise the RAM limit only if available; spatial resolution will not be changed.")
        padding = [(0,0)]+[(0,b-a) for a,b in zip(self.original_shape,self.shape)]
        self.image = np.pad(np.asarray(image,dtype=np.float32),padding)
        self.starts = tile_starts(self.shape,self.patch,config["overlap"])
        self.window = gaussian_window(self.patch)
        self.norm = np.zeros(self.shape,dtype=np.float32)
        for start in self.starts:
            self.norm[self._slice(start)] += self.window
        if np.min(self.norm) <= 0:
            raise RuntimeError("Tiling left uncovered voxels")
        self.cache = {}; self.cache_bytes = 0
        self.cache_limit = int(config["feature_cache_mb"]*2**20)
        self.encoder_calls = 0; self.decoder_calls = 0; self.cache_hits = 0
        self.last_sensitivity = None

    def _slice(self,start):
        return tuple(slice(s,s+p) for s,p in zip(start,self.patch))

    def _tensor(self,array):
        # Copy avoids negative-stride/read-only mmap issues, only one patch at a time.
        return torch.from_numpy(np.array(array,copy=True))[None].to(self.device)

    def _features(self,start):
        if start in self.cache:
            self.cache_hits += 1
            return [f.to(self.device) for f in self.cache[start]]
        image = self._tensor(self.image[(slice(None),)+self._slice(start)])
        with amp_context(self.device,self.precision):
            features = self.model.encode(image)
        self.encoder_calls += 1
        size = sum(f.numel()*f.element_size() for f in features)
        if self.cache_bytes+size <= self.cache_limit:
            self.cache[start] = [f.detach().cpu() for f in features]
            self.cache_bytes += size
        return features

    @torch.inference_mode()
    def original_mask(self):
        """Call only on a freshly loaded ORIGINAL checkpoint, before training."""
        logits = np.zeros((3,*self.shape),dtype=np.float32)
        for start in self.starts:
            sl = self._slice(start)
            inp = self._tensor(self.image[(slice(None),)+sl])
            with amp_context(self.device,self.precision):
                out = self.model.original_logits(inp)
            logits[(slice(None),)+sl] += out[0].float().cpu().numpy()*self.window
        logits /= self.norm[None]
        return logits.argmax(0)[tuple(slice(0,n) for n in self.original_shape)].astype(np.uint8)

    @torch.inference_mode()
    def clean_field(self,state,t,probe=False):
        clean = np.zeros_like(state,dtype=np.float32)
        tt = torch.tensor([t],device=self.device,dtype=torch.float32)
        sensitivity = []
        for j,start in enumerate(self.starts):
            sl = self._slice(start)
            features = self._features(start)
            inp = self._tensor(state[(slice(None),)+sl])
            with amp_context(self.device,self.precision):
                out = self.model.clean(features,inp,tt)
                self.decoder_calls += 1
                if probe and j == len(self.starts)//2:
                    # OOD sensitivity diagnostic, NOT evidence of better segmentation.
                    shuffled = torch.roll(inp,shifts=tuple(max(1,p//3) for p in self.patch),dims=(2,3,4))
                    other = self.model.clean(features,shuffled,tt)
                    delta = (out.float()-other.float()).square().mean().sqrt()
                    scale = out.float().square().mean().sqrt().clamp_min(1e-4)
                    sensitivity.append(float(delta/scale))
            clean[(slice(None),)+sl] += out[0].float().cpu().numpy()*self.window
        clean /= self.norm[None]
        if sensitivity:
            self.last_sensitivity = float(np.mean(sensitivity))
        return clean

    @torch.inference_mode()
    def flow(self,nfe,eta,seed,probe=False):
        if int(nfe) != nfe or nfe < 1 or not 0 < eta < 1:
            raise ValueError("Need integer NFE>=1 and 0<eta<1")
        self.model.eval()
        rng = np.random.default_rng(seed)
        state = rng.standard_normal((2,*self.shape),dtype=np.float32)
        grid = np.linspace(0,1,int(nfe)+1)
        start_time = time.perf_counter()
        for k,(t,tnext) in enumerate(zip(grid[:-1],grid[1:])):
            clean = self.clean_field(state,float(t),probe=probe and k == max(0,int(nfe)//2))
            s, snext = 1-(1-eta)*t, 1-(1-eta)*tnext
            # Algebraically identical to Euler x += dt*v, but avoids subtractive cancellation.
            # Every tile reads the SAME global state; update occurs only after blending.
            state *= np.float32(snext/s)
            state += np.float32((tnext-t)/s)*clean
        sl = (slice(None),)+tuple(slice(0,n) for n in self.original_shape)
        field = state[sl].copy()
        return decode_sdf(field,self.model.left_id,self.model.right_id), {
            "nfe":int(nfe),"seconds":time.perf_counter()-start_time,"tiles":len(self.starts),
            "state_sensitivity":self.last_sensitivity if probe else None,
            "encoder_calls_cumulative":self.encoder_calls,"decoder_calls_cumulative":self.decoder_calls,
            "cache_hits_cumulative":self.cache_hits,"cached_feature_mb":self.cache_bytes/2**20,
            "diagnostic_extra_decoder_calls":int(probe),
        }


def prepare_references(config,info,splits,model):
    if getattr(model,"training_updates",0):
        raise RuntimeError("Reference preparation requires a fresh original checkpoint, not a trained Flow model")
    missing = [c for c in splits["train"]+splits["val"]
               if not reference_complete(config["cache_dir"],c,info,config)]
    if not missing:
        print("Reference masks and weights: all reused; zero teacher forward passes.")
        return
    external = config.get("reference_dir")
    if external:
        provenance = read_json(Path(external)/"manifest.json")
        required = {"checkpoint_sha256":info["checkpoint_sha256"],"plans_sha256":info["plans_sha256"],
                    "grid":"nnunet_preprocessed", "configuration":info["configuration"],
                    "overlap":config["overlap"],"tta":False,"postprocessing":"none",
                    "precision":config["precision"],"patch_size":config["patch_size"]}
        if any(provenance.get(k) != v for k,v in required.items()):
            raise ValueError(f"External reference manifest must match {required}. Shape alone is insufficient.")
    print(f"{len(missing)} missing references. One-time original checkpoint inference + CPU weights; no retraining.")
    model.eval(); t0 = time.perf_counter()
    for i,case in enumerate(missing,1):
        root = Path(config["cache_dir"])/case
        if external:
            mask = np.load(Path(external)/f"{case}.npy",allow_pickle=False)
        else:
            image = np.load(root/"image.npy",mmap_mode="r",allow_pickle=False)
            engine = VolumeEngine(model,image,config)
            mask = engine.original_mask()
            del engine
        write_reference(config["cache_dir"],case,mask,info,config)
        elapsed = time.perf_counter()-t0
        print(f"[REFERENCE {i}/{len(missing)}] {case}: {elapsed/60:.1f}m, "
              f"ETA {(len(missing)-i)*elapsed/i/60:.1f}m",flush=True)
