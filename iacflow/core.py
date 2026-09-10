from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import pydoc
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_configuration(plans, name):
    seen = set()
    def resolve(key):
        if key in seen:
            raise ValueError("Circular plans inheritance")
        seen.add(key)
        child = copy.deepcopy(plans["configurations"][key])
        parent = child.pop("inherits_from", None)
        result = resolve(parent) if parent else {}
        result.update(child)
        return result
    return resolve(name)


def load_backbone(checkpoint_path, plans_path, dataset_json_path, configuration, fold):
    """Load a trusted nnU-Net v2 checkpoint. No non-strict/partial loading."""
    plans, dataset = read_json(plans_path), read_json(dataset_json_path)
    conf = resolve_configuration(plans, configuration)
    labels = dataset["labels"]
    if any(isinstance(v, (list, tuple)) for v in labels.values()):
        raise ValueError("Region-based checkpoints are not supported; need bg/L/R softmax labels.")
    if sorted(int(v) for v in labels.values()) != [0, 1, 2]:
        raise ValueError(f"Need exactly labels 0/1/2, found {labels}; binary/77-class checkpoints cannot be substituted.")
    if conf.get("previous_stage"):
        raise ValueError("Cascaded checkpoints require extra inputs and are not supported.")
    arch = conf.get("architecture")
    if arch is None:
        raise ValueError("This loader requires architecture metadata in plans (modern nnU-Net v2). Do not guess architecture.")
    cls = pydoc.locate(arch["network_class_name"])
    if cls is None or cls.__name__ not in ("PlainConvUNet", "ResidualEncoderUNet"):
        raise ValueError(f"Supported: PlainConvUNet / ResidualEncoderUNet, got {arch['network_class_name']}")
    kwargs = copy.deepcopy(arch["arch_kwargs"])
    for key in arch.get("_kw_requires_import", []):
        if kwargs.get(key) is not None:
            kwargs[key] = pydoc.locate(kwargs[key])
            if kwargs[key] is None:
                raise ImportError(f"Cannot resolve {key}")
    if kwargs.get("conv_op") is not nn.Conv3d:
        raise ValueError("Only 3D models are supported.")
    kwargs["deep_supervision"] = False
    channels = len(dataset.get("channel_names", dataset.get("modality", {})))
    if channels != 1:
        raise ValueError("This notebook expects a single CBCT input channel.")
    model = cls(input_channels=channels, num_classes=3, **kwargs)
    # PyTorch checkpoints can contain Python objects; only load your own trusted checkpoint.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("init_args", {})
    if "fold" in args and str(args["fold"]) != str(fold):
        raise ValueError(f"Checkpoint fold={args['fold']} but requested fold={fold}")
    if "configuration" in args and args["configuration"] != configuration:
        raise ValueError("Checkpoint configuration differs from requested configuration")
    state = ckpt.get("network_weights", ckpt.get("state_dict"))
    if state is None:
        raise ValueError("Missing network_weights/state_dict in checkpoint")
    for prefix in ("module.", "_orig_mod."):
        while state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    info = {
        "checkpoint_sha256": file_hash(checkpoint_path),
        "plans_sha256": file_hash(plans_path), "dataset_sha256": file_hash(dataset_json_path),
        "configuration": configuration, "fold": int(fold),
        "spacing": list(map(float, conf["spacing"])), "patch_size": list(conf["patch_size"]),
        "data_identifier": conf["data_identifier"], "labels": labels,
        "checkpoint_epoch": ckpt.get("current_epoch"),
        "fold_metadata_present": "fold" in args,
    }
    del ckpt, state
    return model, info


class StateAdapter(nn.Module):
    def __init__(self, output_channels, hidden=8):
        super().__init__()
        self.features = nn.Sequential(nn.Conv3d(8, hidden, 3, padding=1), nn.SiLU())
        self.out = nn.Conv3d(hidden, output_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, state, t, shape):
        x = F.interpolate(state, size=shape, mode="trilinear", align_corners=False)
        t = t.reshape(-1, 1)
        emb = torch.cat((t, 1-t, torch.sin(math.pi*t), torch.cos(math.pi*t),
                         torch.sin(2*math.pi*t), torch.cos(2*math.pi*t)), 1)
        emb = emb[:, :, None, None, None].expand(-1, -1, *shape)
        return self.out(self.features(torch.cat((x, emb), dim=1)))


class IACFlow(nn.Module):
    def __init__(self, backbone, left_id=1, right_id=2, margin_scale=0.1, adapter_hidden=8):
        super().__init__()
        self.backbone = backbone
        self.left_id, self.right_id = int(left_id), int(right_id)
        if {self.left_id, self.right_id} != {1, 2}:
            raise ValueError("left_id/right_id must be a permutation of 1,2")
        for p in backbone.parameters():
            p.requires_grad_(False)
        # decoder.encoder is a shared module reference in DNA. Never unfreeze decoder.parameters().
        for module in [backbone.encoder.stages[-1], *backbone.decoder.stages, *backbone.decoder.transpconvs]:
            for p in module.parameters():
                p.requires_grad_(True)
        channels = list(backbone.encoder.output_channels)
        self.adapters = nn.ModuleList([StateAdapter(c, adapter_hidden) for c in channels])
        old = backbone.decoder.seg_layers[-1]
        if not isinstance(old, nn.Conv3d) or old.out_channels != 3 or old.kernel_size != (1,1,1):
            raise ValueError("Expected a three-class 1x1x1 nnU-Net output head")
        self.sdf_head = nn.Conv3d(old.in_channels, 2, 1, bias=True)
        with torch.no_grad():
            for c, semantic in enumerate((self.left_id, self.right_id)):
                self.sdf_head.weight[c].copy_(margin_scale*(old.weight[0]-old.weight[semantic]))
                self.sdf_head.bias[c].copy_(margin_scale*(old.bias[0]-old.bias[semantic]))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.encoder.eval()
        self.backbone.encoder.stages[-1].train(mode)
        # Keep all unused original segmentation heads deterministic.
        self.backbone.decoder.seg_layers.eval()
        return self

    def encode(self, image):
        encoder = self.backbone.encoder
        skips = []
        with torch.no_grad():
            x = encoder.stem(image) if getattr(encoder, "stem", None) is not None else image
            for stage in list(encoder.stages)[:-1]:
                x = stage(x)
                skips.append(x)
        # Gradient begins at last stage parameters, not at frozen prefix outputs.
        skips.append(encoder.stages[-1](x))
        return skips

    def clean(self, skips, state, t, use_state=True):
        modified = []
        for skip, adapter in zip(skips, self.adapters):
            modified.append(skip + adapter(state, t, skip.shape[2:]) if use_state else skip)
        x = modified[-1]
        for i, (up, stage) in enumerate(zip(self.backbone.decoder.transpconvs, self.backbone.decoder.stages)):
            x = up(x)
            skip = modified[-i-2]
            if x.shape[2:] != skip.shape[2:]:
                raise ValueError("Patch shape must be divisible by all encoder strides; no silent crop/resample.")
            x = stage(torch.cat((x, skip), dim=1))
        return self.sdf_head(x)

    def forward(self, image, state, t):
        return self.clean(self.encode(image), state, t)

    def original_logits(self, image):
        return self.backbone(image)

    def parameter_groups(self, lr, encoder_lr_factor=0.1):
        deep = list(self.backbone.encoder.stages[-1].parameters())
        deep_ids = {id(p) for p in deep}
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in deep_ids]
        return [{"params": deep, "lr": lr*encoder_lr_factor, "base_lr": lr*encoder_lr_factor},
                {"params": other, "lr": lr, "base_lr": lr}]

    def trainable_report(self):
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "frozen": sum(p.numel() for p in self.parameters() if not p.requires_grad),
                "trainable_encoder_stages": [i for i,s in enumerate(self.backbone.encoder.stages)
                                             if any(p.requires_grad for p in s.parameters())]}


def sample_path(target, eta=0.01, generator=None):
    b = target.shape[0]
    u = torch.rand(b, device=target.device, generator=generator)
    choose = torch.rand(b, device=target.device, generator=generator) < 0.5
    # Equal mixture: Uniform(0,1), Beta(1,2). Full support, 62.5% below t=.5.
    t = torch.where(choose, u, 1-torch.sqrt(1-u))
    t = t.clamp(1e-6, 1-1e-6)
    noise = torch.randn(target.shape, device=target.device, dtype=torch.float32, generator=generator)
    s = (1-(1-eta)*t)[:,None,None,None,None]
    state = t[:,None,None,None,None]*target.float() + s*noise
    velocity = target.float()-(1-eta)*noise
    return state, t, velocity


def velocity_from_clean(clean, state, t, eta=0.01):
    s = (1-(1-eta)*t)[:,None,None,None,None]
    return (clean.float()-(1-eta)*state.float())/s


def fm_loss(clean, target, weight, valid):
    # Exactly s(t)^2 times velocity MSE, without numerically subtracting large velocities.
    w = weight.float()*valid.float()
    axes = tuple(range(1, clean.ndim))
    per_sample = (w*(clean.float()-target.float()).square()).sum(axes)/w.expand_as(clean).sum(axes).clamp_min(1)
    return per_sample.mean()


def decode_sdf(field, left_id=1, right_id=2):
    if torch.is_tensor(field):
        scores = torch.cat((torch.zeros_like(field[:, :1]), field), 1)
        out = scores.argmin(1)
        return torch.where(out == 1, left_id, torch.where(out == 2, right_id, 0))
    a, b = field
    out = np.zeros(a.shape, dtype=np.uint8)
    out[(a < 0) & (a <= b)] = left_id
    out[(b < 0) & (b < a)] = right_id
    return out


def amp_context(device, precision):
    if str(device).startswith("cuda") and precision != "fp32":
        return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
    return contextlib.nullcontext()


def validate_patch(model, patch):
    stride = np.prod(np.asarray(model.backbone.encoder.strides), axis=0)
    patch = np.asarray(patch)
    if len(patch) != 3 or np.any(patch % stride) or np.prod(patch//stride) <= 1:
        raise ValueError(f"Patch {patch.tolist()} incompatible with strides {stride.tolist()} / InstanceNorm bottleneck.")


@torch.no_grad()
def check_head_initialization(model, image):
    model.eval()
    logits = model.original_logits(image)
    pred = logits.argmax(1)
    zero = torch.zeros((image.shape[0],2,*image.shape[2:]), device=image.device)
    clean = model(image, zero, torch.zeros(image.shape[0], device=image.device))
    mapped = decode_sdf(clean, model.left_id, model.right_id)
    mismatch = mapped != pred
    # Floating point differences at near-tied logits are reported separately.
    top = logits.topk(2, dim=1).values
    confident_mismatch = mismatch & ((top[:,0]-top[:,1]) > 1e-4)
    if confident_mismatch.any():
        raise RuntimeError("Logit-margin initialization did not preserve confident baseline decisions")
    return {"clean_head_mismatch_voxels": int(mismatch.sum()),
            "confident_mismatch_voxels": int(confident_mismatch.sum()),
            "note": "Clean-head check only; eta-noisy ODE endpoint is not guaranteed identical."}
