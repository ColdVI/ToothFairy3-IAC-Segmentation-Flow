#!/usr/bin/env python3
"""
losses.py — flow-matching + topology-aware losses for the residual IAC flow.

All topology/boundary terms are computed on the ENDPOINT estimate

    x1_hat = x_t + (1 - t) * v_theta(x_t, t, c)

not on the raw velocity, so they constrain the object the model will actually
produce at t=1. SDF tensors are (B, 2, D, H, W): channel 0 = Left, 1 = Right,
normalised to ~[-1, 1] (negative = inside).

Total (first stable version):
    L = L_fm + w_cldice*L_clDice + w_nb*L_narrowband + w_lat*L_laterality
The plain total-variation smoothness term is available but is NOT the topology
loss and is off by default (name it what it is: TV/smoothness).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# SDF <-> soft occupancy
# --------------------------------------------------------------------------- #
def sdf_to_occupancy(sdf: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """Soft inside-probability from a normalised SDF: sigmoid(-sdf/tau) in (0,1)."""
    return torch.sigmoid(-sdf / tau)


# --------------------------------------------------------------------------- #
# Flow matching
# --------------------------------------------------------------------------- #
def flow_matching_loss(pred_v: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """E || v_theta - (x1 - x0) ||^2 for the residual path x_t=(1-t)x0 + t x1."""
    return F.mse_loss(pred_v, x1 - x0)


# --------------------------------------------------------------------------- #
# Narrow-band boundary loss
# --------------------------------------------------------------------------- #
def narrow_band_loss(x1_hat: torch.Tensor, x1: torch.Tensor, band: float = 0.2) -> torch.Tensor:
    """
    Squared error between predicted and GT SDF, weighted to emphasise the region
    near the GT zero level-set (weight = exp(-(x1/band)^2)). Fixes the boundary
    the mask decoding actually reads, rather than spending capacity deep in
    background where the truncated SDF is flat.
    """
    w = torch.exp(-(x1 / band) ** 2)
    num = (w * (x1_hat - x1) ** 2).sum()
    den = w.sum().clamp_min(1.0)
    return num / den


# --------------------------------------------------------------------------- #
# Soft-clDice (3D) — centerline connectivity
# --------------------------------------------------------------------------- #
def _soft_erode3d(img):
    p1 = -F.max_pool3d(-img, (3, 1, 1), 1, (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), 1, (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), 1, (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate3d(img):
    return F.max_pool3d(img, (3, 3, 3), 1, (1, 1, 1))


def _soft_open3d(img):
    return _soft_dilate3d(_soft_erode3d(img))


def soft_skeleton3d(img, iters: int = 10):
    """Differentiable morphological skeleton (Shit et al., clDice), 3D variant."""
    img1 = _soft_open3d(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode3d(img)
        img1 = _soft_open3d(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(occ_pred, occ_true, iters: int = 10, eps: float = 1e-6):
    """
    1 - clDice over soft occupancy maps (B,1,D,H,W in [0,1]). Rewards topological
    (centerline) agreement, which is exactly what a broken thin canal violates.
    """
    sp = soft_skeleton3d(occ_pred, iters)
    st = soft_skeleton3d(occ_true, iters)
    tprec = (sp * occ_true).sum() / (sp.sum() + eps)
    tsens = (st * occ_pred).sum() / (st.sum() + eps)
    cldice = 2.0 * tprec * tsens / (tprec + tsens + eps)
    return 1.0 - cldice


def soft_dice_loss(occ_pred, occ_true, eps: float = 1e-6):
    """Differentiable overlap loss, averaged over batch and anatomical channel."""
    dims = tuple(range(2, occ_pred.ndim))
    intersection = (occ_pred * occ_true).sum(dim=dims)
    denominator = occ_pred.sum(dim=dims) + occ_true.sum(dim=dims)
    score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - score.mean()


def sample_mixed_low_t(batch_size, cfg, device, *, generator=None):
    """Draw disjoint low-t and ordinary-uniform groups deterministically by seed."""
    fraction = float(cfg.get("low_t_fraction", 0.0))
    low_max = float(cfg.get("low_t_max", 0.25))
    if not 0 <= fraction <= 1 or not 0 < low_max <= 1:
        raise ValueError("invalid mixed-low-t sampling configuration")
    count = int(round(int(batch_size) * fraction))
    low_group = torch.zeros(int(batch_size), dtype=torch.bool, device=device)
    if count:
        selected = torch.randperm(int(batch_size), device=device, generator=generator)[:count]
        low_group[selected] = True
    times = torch.rand(int(batch_size), device=device, generator=generator)
    if count:
        times[low_group] = torch.rand(count, device=device, generator=generator) * low_max
    return times, low_group


def predict_t0_endpoint(model, state0, conditioning):
    """Predict the actual inference endpoint without accepting GT as an input."""
    t0 = torch.zeros(state0.shape[0], dtype=state0.dtype, device=state0.device)
    return state0 + model(state0, t0, conditioning)


def _two_channel_endpoint_losses(endpoint, target, cfg):
    tau = float(cfg.get("occ_tau", 0.05))
    iters = int(cfg.get("cldice_iters", 10))
    sdf = F.mse_loss(endpoint, target)
    narrowband = narrow_band_loss(
        endpoint, target, float(cfg.get("narrowband_band", 0.2)))
    dice_terms, cldice_terms = [], []
    for channel in range(endpoint.shape[1]):
        predicted = sdf_to_occupancy(endpoint[:, channel:channel + 1], tau)
        truth = sdf_to_occupancy(target[:, channel:channel + 1], tau)
        dice_terms.append(soft_dice_loss(predicted, truth))
        cldice_terms.append(soft_cldice_loss(predicted, truth, iters))
    return {
        "t0_sdf": sdf,
        "t0_narrowband": narrowband,
        "t0_softdice": torch.stack(dice_terms).mean(),
        "t0_cldice": torch.stack(cldice_terms).mean(),
    }


def compute_prompt3r_training_loss(model, conditioning, state0, target, cfg,
                                   *, generator=None):
    """Mixed-t FM plus a separate, GT-free-input t=0 endpoint branch."""
    times, _ = sample_mixed_low_t(
        target.shape[0], cfg, target.device, generator=generator)
    tb = times.view(-1, 1, 1, 1, 1)
    state_t = (1 - tb) * state0 + tb * target
    predicted_velocity = model(state_t, times, conditioning)
    if cfg.get("random_t_aux_enabled", False):
        random_loss, _ = total_loss(
            predicted_velocity, state0, target, times, cfg)
    else:
        random_loss = flow_matching_loss(predicted_velocity, state0, target)
    components = {"fm_random_t": float(random_loss.detach())}
    endpoint_weight = float(cfg.get("w_t0_endpoint", 0.0))
    if endpoint_weight == 0:
        components.update({name: 0.0 for name in
                           ("t0_sdf", "t0_narrowband", "t0_softdice", "t0_cldice")})
        components["total"] = float(random_loss.detach())
        return random_loss, components

    endpoint = predict_t0_endpoint(model, state0, conditioning)
    losses = _two_channel_endpoint_losses(endpoint, target, cfg)
    t0_loss = (
        float(cfg.get("w_t0_sdf", 1.0)) * losses["t0_sdf"]
        + float(cfg.get("w_t0_narrowband", 1.0)) * losses["t0_narrowband"]
        + float(cfg.get("w_t0_softdice", 1.0)) * losses["t0_softdice"]
        + float(cfg.get("w_t0_cldice", 0.5)) * losses["t0_cldice"]
    )
    total = random_loss + endpoint_weight * t0_loss
    components.update({name: float(value.detach()) for name, value in losses.items()})
    components["total"] = float(total.detach())
    return total, components


# --------------------------------------------------------------------------- #
# Laterality loss — disjoint L/R, no side swap
# --------------------------------------------------------------------------- #
def laterality_loss(x1_hat, tau: float = 0.05, lateral_coord: torch.Tensor = None,
                    coord_weight: float = 0.0):
    """
    Two effects:
    * overlap: the Left and Right occupancy maps must be disjoint — penalise
      their product (a voxel claimed as both sides is anatomically impossible).
    * (optional) side consistency: if `lateral_coord` (B,1,D,H,W, the normalised
      physical Right->Left axis, +1 on the Left side) is given, encourage Left
      occupancy where coord>0 and Right where coord<0, discouraging L/R swaps.
    """
    occ_l = sdf_to_occupancy(x1_hat[:, 0:1], tau)
    occ_r = sdf_to_occupancy(x1_hat[:, 1:2], tau)
    overlap = (occ_l * occ_r).mean()
    if lateral_coord is not None and coord_weight > 0:
        # penalise Left mass on the right half and Right mass on the left half
        wrong = occ_l * F.relu(-lateral_coord) + occ_r * F.relu(lateral_coord)
        return overlap + coord_weight * wrong.mean()
    return overlap


# --------------------------------------------------------------------------- #
# Optional TV smoothness (NOT a topology loss)
# --------------------------------------------------------------------------- #
def tv_smoothness(field):
    s = torch.tanh(field)
    dz = (s[:, :, 1:] - s[:, :, :-1]).abs().mean()
    dy = (s[:, :, :, 1:] - s[:, :, :, :-1]).abs().mean()
    dx = (s[:, :, :, :, 1:] - s[:, :, :, :, :-1]).abs().mean()
    return dz + dy + dx


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def total_loss(pred_v, x0, x1, t, cfg, lateral_coord=None):
    """
    Weighted sum per the config. `t` is (B,) in [0,1]; the endpoint estimate uses
    it per-sample. Returns (loss, components-dict) for logging.
    """
    tb = t.view(-1, 1, 1, 1, 1)
    x_t = (1 - tb) * x0 + tb * x1
    x1_hat = x_t + (1 - tb) * pred_v            # endpoint estimate at t=1

    l_fm = flow_matching_loss(pred_v, x0, x1)
    comp = {"fm": float(l_fm.detach())}
    total = l_fm

    if cfg.get("w_narrowband", 0) > 0:
        l_nb = narrow_band_loss(x1_hat, x1, cfg.get("narrowband_band", 0.2))
        total = total + cfg["w_narrowband"] * l_nb
        comp["narrowband"] = float(l_nb.detach())

    if cfg.get("w_cldice", 0) > 0:
        tau = cfg.get("occ_tau", 0.05)
        iters = cfg.get("cldice_iters", 10)
        l_cl = 0.0
        for c in (0, 1):
            occ_p = sdf_to_occupancy(x1_hat[:, c:c + 1], tau)
            occ_t = sdf_to_occupancy(x1[:, c:c + 1], tau)
            l_cl = l_cl + soft_cldice_loss(occ_p, occ_t, iters)
        l_cl = l_cl / 2.0
        total = total + cfg["w_cldice"] * l_cl
        comp["cldice"] = float(l_cl.detach() if torch.is_tensor(l_cl) else l_cl)

    if cfg.get("w_laterality", 0) > 0:
        l_lat = laterality_loss(x1_hat, cfg.get("occ_tau", 0.05),
                                lateral_coord, cfg.get("laterality_coord_weight", 0.0))
        total = total + cfg["w_laterality"] * l_lat
        comp["laterality"] = float(l_lat.detach())

    if cfg.get("w_tv", 0) > 0:
        l_tv = tv_smoothness(x1_hat)
        total = total + cfg["w_tv"] * l_tv
        comp["tv"] = float(l_tv.detach())

    comp["total"] = float(total.detach())
    return total, comp
