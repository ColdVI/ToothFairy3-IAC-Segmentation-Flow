#!/usr/bin/env python3
"""
conditioning.py — assemble the 8-channel conditioning tensor for the flow.

Channel order (must match model.COND_CH and configs/flow.yaml):
    0 normalised CBCT ROI
    1 nnU-Net Left  probability
    2 nnU-Net Right probability
    3 coarse Left  SDF (normalised)
    4 coarse Right SDF (normalised)
    5 physical x (normalised)
    6 physical y (normalised)
    7 physical z (normalised)

The residual-flow starting state x0 = coarse L/R SDF (channels 3/4) plus optional
noise is built by the sampler/dataset, not here — this module only builds the
conditioning `c`, plus the lateral-coordinate channel used by the laterality loss.
"""
import numpy as np

try:
    from .channel_contract import (COND_CH, FLOW_STATE_CH, ConditioningSpec,
                                   resolve_conditioning_spec,
                                   validate_conditioning_tensor)
except ImportError:  # direct script imports from flow/ on sys.path
    from channel_contract import (COND_CH, FLOW_STATE_CH, ConditioningSpec,
                                  resolve_conditioning_spec,
                                  validate_conditioning_tensor)


CHANNELS = list(resolve_conditioning_spec().conditioning_channel_names)


def znorm(vol):
    v = vol.astype(np.float32)
    return (v - v.mean()) / (v.std() + 1e-6)


def build_conditioning(cbct, prob_l, prob_r, coarse_sdf_l, coarse_sdf_r, coords_norm,
                       *, spec: ConditioningSpec | None = None, cfg=None):
    """
    All inputs are numpy arrays of matching (D,H,W) except coords_norm (3,D,H,W).
    Returns (8, D, H, W) float32. CBCT is z-normalised here; the SDFs and probs
    are assumed already normalised by the caller.
    """
    spec = spec or resolve_conditioning_spec(cfg)
    channels = [znorm(cbct), prob_l.astype(np.float32), prob_r.astype(np.float32)]
    if spec.include_coarse_sdf:
        channels.extend((coarse_sdf_l.astype(np.float32),
                         coarse_sdf_r.astype(np.float32)))
    channels.extend(coords_norm[index].astype(np.float32) for index in range(3))
    stack = np.stack(channels, axis=0).astype(np.float32)
    validate_conditioning_tensor(stack, spec, channel_axis=0,
                                 context="built conditioning")
    return stack


def lateral_axis_index(affine):
    """
    Index (0/1/2) of the array axis most aligned with the physical Right<->Left
    direction, and its sign toward Left (+). Used to pick which coordinate
    channel drives the optional laterality side-consistency term.
    """
    import nibabel as nib
    codes = nib.aff2axcodes(affine)              # e.g. ('R','P','I')
    for i, c in enumerate(codes):
        if c in ("R", "L"):
            sign = +1.0 if c == "L" else -1.0    # coordinate increases toward L?
            return i, sign
    return 0, 1.0
