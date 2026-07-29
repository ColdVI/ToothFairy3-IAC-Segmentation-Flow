# CLAUDE.md — ToothFairy3 IAC Left/Right Segmentation with Residual Flow

Persistent context for Claude Code sessions in this repository. Read this fully
before touching any file. Everything here is measured, not assumed.

---

## 1. What this project is

Two-track pipeline for **left/right Inferior Alveolar Canal (IAC)** segmentation
in dental CBCT, built on ToothFairy3 (`Dataset801_IAC_LR`, 480 development cases,
0.3 mm isotropic, LPS).

- **Track A — discriminative baseline.** nnU-Net v2, `3d_fullres`, trainer
  `nnUNetTrainerIAC_NoMirror`, 5-fold CV, 1000 epochs/fold. **DONE. Do not retrain.**
- **Track B — generative refinement.** A continuous-time flow that transports the
  coarse nnU-Net SDF (`x0`) toward the GT SDF (`x1`), conditioned on the CBCT.

The research claim is **not** "higher Dice than nnU-Net". It is:

> Naive residual flow matching on top of a strong discriminative prior *degrades*
> that prior; the cause is a degenerate probability path; a bridge formulation
> fixes it and yields fewer topological breaks plus calibrated uncertainty at
> non-inferior Dice.

Clinical motivation: the IAC is a thin tubular structure whose *continuity* and
*boundary upper bound* matter for nerve-injury risk in implant planning and molar
extraction. Voxel overlap alone is the wrong success criterion.

---

## 2. Measured state (do not re-derive from memory — these are the real numbers)

### Track A, 5-fold CV, `Mean Validation Dice`

| fold | Dice   | epochs | wall (A100) |
|------|--------|--------|-------------|
| 0    | 0.9050 | 1000   | ~11.0 h     |
| 1    | 0.9104 | 1000   | ~10.8 h     |
| 2    | 0.9094 | 1000   | ~11.0 h     |
| 3    | 0.9147 | 1000   | ~10.7 h     |
| 4    | 0.9111 | 1000   | ~10.7 h     |

**Mean 0.9101 ± 0.0032.** Total ≈ 54 A100-hours already spent. Literature anchor:
ToothFairy2 challenge report (MIA 2026) — the five best methods all land near
0.89 DSC on IACs. The baseline is healthy and consistent.

### Track B, fold 0, 125/500 epochs (`progress.csv`)

| epoch | trainloss | dice   | cldice | hd95 (mm) | score  |
|-------|-----------|--------|--------|-----------|--------|
| 0     | 0.2674    | 0.8867 | 0.9927 | 0.430     | 0.9397 |
| 25    | 0.1771    | 0.7401 | 0.9940 | 0.803     | 0.8670 |
| 50    | 0.1902    | 0.8050 | 0.9941 | 2.082     | 0.8995 |
| 75    | 0.1804    | 0.7940 | 0.9941 | 2.069     | 0.8941 |
| 100   | 0.1838    | 0.7982 | 0.9928 | 2.052     | 0.8955 |
| 125   | 0.1691    | 0.7969 | 0.9931 | 2.058     | 0.8950 |

Signature: **train loss falls, validation Dice converges 0.09 BELOW the untouched
prior, HD95 degrades 5×, clDice never moves.** This is not underfitting. It is a
structural misalignment between the trained objective and the evaluated one.

---

## 3. THE CORE DIAGNOSIS — read this before proposing any fix

### 3.1 The probability path is algebraically invertible

In `flow/datasets.py`, `x0 = coarse_sdf` and **the same array** is also passed to
`build_conditioning(...)` as channels 3 and 4. So the network sees both `x_t` and
`x0`. Then:

```
x_t = (1-t)*x0 + t*x1
x_t - x0 = t*(x1 - x0) = t*u
=>  u = (x_t - x0) / t
```

The regression target is a **closed-form algebraic function of the network's own
inputs**. The model can drive the flow-matching MSE toward zero by computing
`(state_channels - cond_channels[3:5]) / t` without ever looking at the CBCT.
It is learning division, not segmentation.

### 3.2 The inference rollout collapses to a single evaluation at t=0

Assume the shortcut `v = (x - x0)/t` is learned. Euler integration:

```
k=0 : t=0,   x=x0                -> v = 0/0, network emits some w
                                    x_1 = x0 + dt*w
k=1 : t=dt,  x=x0+dt*w           -> v = (dt*w)/dt = w
                                    x_2 = x0 + 2*dt*w
...
k=N : x_N = x0 + w
```

The entire 8-step Heun/Euler ODE reduces to `x0 + w`, where `w` is the network
output at `t=0`. **All training at t>0 is wasted.** And `t≈0` — the only regime
that determines the output — is exactly where `x_t ≈ x0` carries zero information
about `x1` and the objective is worst-conditioned. This explains falling train
loss with collapsing validation.

`train_sigma=0.1, noise_frac=0.5` partially breaks this: with `x_t` built from a
noised `x0`, the shortcut estimate is off by `sigma*eps/t`. But it is active in
only half the batches and the model cannot tell which batch it is in, so it
learns a blend of the shortcut and the conditional mean.

### 3.3 The loss has no overlap term, and its only shape term rewards thickening

`total_loss` = FM + 1.0·narrowband + 0.5·clDice + 0.5·laterality. There is **no
Dice/overlap term at all**. Meanwhile soft-clDice:

```
T_sens = sum(S_gt * V_pred) / sum(S_gt)
```

increases monotonically with predicted mass, while `T_prec` stays flat under
uniform thickening (the skeleton remains central). So soft-clDice **rewards
inflating the tube**. The original clDice paper (Shit et al., CVPR 2021) uses
`L = (1-a)*L_softDice + a*L_softclDice` for exactly this reason; the soft-Dice
leg was dropped here.

Quantitative check: IAC radius ≈ 1.58 mm ≈ 5.3 voxels. Under uniform dilation by
`d` voxels, `Dice = 2 / (1 + (1 + d/r)^2)`. Observed Dice 0.798 implies
`d ≈ 1.2` voxels ≈ 0.36 mm. **The Dice drop is fully explained by ~1 voxel of
uniform thickening**, and so is clDice staying pinned at 0.993.

### 3.4 HD95 = 2.05 mm is NOT explained by thickening — it needs localized outliers

0.36 mm of dilation cannot produce 2.05 mm HD95. Two candidates, both testable:

1. **Train/inference patch distribution mismatch.** `fg_prob: 0.8` means 80% of
   training patches are canal-centred. `flow/sliding_window.py` tiles the *whole*
   volume, including pure-background patches where the coarse SDF is saturated at
   +1 everywhere — a regime the model never trained on. It must emit `v≈0` there
   and was never taught to.
2. **Gaussian blend numerics.** `acc /= np.maximum(wsum, 1e-6)` with
   `sigma_scale=0.125` gives corner weights of `exp(-32) ≈ 1e-14`, so the `1e-6`
   floor crushes those voxels toward 0. `sdf_stack_to_mask` decodes `min(L,R) < 0`,
   so a marginally negative value there becomes foreground.

Diagnostic: histogram of the distance from each false-positive connected component
to the nearest GT voxel. Bimodal (near + far) confirms this.

### 3.5 The auxiliary losses are mostly scoring the ground truth

```python
x_t    = (1-t)*x0 + t*x1
x1_hat = x_t + (1-t)*pred_v
```

At `t=0.9`, `x1_hat = 0.1*x0 + 0.9*x1 + 0.1*v` — **90% ground truth.** The
clDice / narrowband / laterality terms see a near-perfect shape regardless of `v`,
and their gradient w.r.t. `v` is damped by `(1-t)`. They only bite at small `t`.

Also: `laterality_loss` returns `mean(occ_l * occ_r)`. A 96³ patch spans 28.8 mm;
the two canals are ~30-40 mm apart and effectively never co-occur in one patch.
That term is **identically zero** — dead weight, `w_laterality=0.5`
notwithstanding. `laterality_coord_weight` is 0 *and* `lateral_coord` is never
passed to `total_loss` from the training loop, so side-consistency is entirely
unenforced; swap prevention currently relies on conditioning alone.

### 3.6 The checkpoint gate has no "do no harm" floor

`best.pt` can be written with a score below the untouched prior. The prior is a
fixed, known floor and must be encoded as a hard threshold.

### 3.7 The Track B baseline number does not match Track A

`validate()` reports 0.8867 at epoch 0; Track A CV is 0.9101. The SDF round-trip
is sign-preserving and therefore lossless, so the gap is either `val_max_cases=20`
sampling noise or a metric-definition mismatch (per-side vs per-case averaging).
**This number is the baseline row of the paper.** It must be reproduced on the
full CV with `v ≡ 0` before any comparison is trustworthy.

---

## 4. What is already correct — do not "improve" these

- **Leakage-free OOF prior** (`nnunet/predict_oof.py`). Fold `f` is predicted by
  the model trained on the other four. Most refinement papers get this wrong.
- **Physical millimetre SDF** — anisotropic `distance_transform_edt(sampling=spacing)`,
  single source of truth in `data/io_utils.py`.
- **Checkpoint selection on a segmentation metric**, not on training loss.
- **`nnUNetTrainerIAC_NoMirror`** — default L/R mirroring semantically swaps the
  two classes. ToothFairy2's top method (Isensee & Kirchhoff) disabled it too.
- **Endpoint-based auxiliary losses** — constraining the object actually produced
  at `t=1` is the right instinct, even though the current form is diluted by GT.
- **`evaluation/`** — Betti-0, centerline gap, false-branch length, NSD, L/R swap
  rate, `compare_bootstrap`. This is ahead of the field for this task and is
  effectively the paper's evaluation section already written.
- **Colab resume/persist discipline** — `checkpoint_every`, `persist_stage`,
  atomic `.partial` writes.

---

## 5. Non-negotiable invariants

1. **Never train the flow on in-sample nnU-Net predictions.** OOF only.
2. **Ground truth never enters any inference path.** Not `x0`, not the
   conditioning, not the sliding window. If a function needs `sdf_gt` to run at
   inference, that function is wrong.
3. **SDF distances are always in physical millimetres**, never voxel units.
4. **Never re-enable L/R mirror augmentation** anywhere in the pipeline.
5. **`best.pt` may only be written if the score beats the measured prior floor.**
6. **Every loss component is logged separately, every validation step.**
7. **Every run writes a manifest**: git commit SHA, full resolved config, config
   hash, fold, seed, start/end time, and the resulting metrics.
8. **Every training entry point must support `--resume` from the last checkpoint**
   and must be safe to kill at any moment (atomic writes, no partial state).
9. **Do not modify anything under `nnunet/` that would require retraining Track A.**
10. **No new feature, loss term, or architecture change may be added until the
    current stage's acceptance criteria pass.** Ablate before you tune.

---

## 6. Repo layout

```
configs/          flow.yaml (Track B hyperparameters), splits.json (fold definitions)
data/             io_utils.py (SDF, spacing, coords), compute_gt_sdf.py, compute_coarse_sdf.py,
                  validate_pipeline_cache.py
nnunet/           predict_oof.py — leakage-free OOF probability maps
flow/             model.py (ResidualVelocityUNet3D), losses.py, datasets.py, sampler.py,
                  sliding_window.py, train.py, validate.py, conditioning.py, selftest.py
evaluation/       metrics.py (Dice/HD95/clDice/NSD), topology_metrics.py, evaluate_cv.py
notebooks/        IAC_Colab_runner.ipynb, colab_trackB_only.ipynb, mac_local_oof_sdf.ipynb
archive/          v0_flat_flow/ — the abandoned noise-to-SDF formulation. Reference only.
docs/             project_notes.md
```

Channel contract (must stay in sync across `model.py`, `conditioning.py`, `flow.yaml`):

```
FLOW_STATE_CH = 2   # [Left SDF, Right SDF]
COND_CH       = 8   # [CBCT, prob_L, prob_R, coarse_SDF_L, coarse_SDF_R, x, y, z]
```

---

## 7. Compute and workflow constraints

- GPU is **rented Colab** (A100/T4, session-limited, disconnects without warning).
  Persistent storage is Google Drive. Local machine is an M4 MacBook (MPS for
  inference, CPU for SDF).
- Only three stages genuinely need a GPU: Track A training (done), OOF prediction
  (done), Track B flow training.
- **Never launch a long run without a smoke test first**: 3-5 epochs, tiny patch,
  2 cases, assert the loop completes, checkpoints write, and `--resume` works.
- Prefer many short cheap experiments over one long expensive one. A 50-epoch
  fold-0 ablation that answers a yes/no question beats a 500-epoch run that
  answers nothing.
- Assume the session dies mid-run. Design every script accordingly.

---

## 8. Working style expected of you (Claude Code)

- **Read the source before changing it.** This file describes the code; it is not
  a substitute for the code.
- **One concern per commit.** Message format: `stage/topic: what changed and why`.
- **Write the test before the feature** when the feature is a metric or a loss.
  A metric you have not validated against a known-answer input is not a metric.
- **State assumptions explicitly** and flag anything in this file that the code
  contradicts — this document can go stale.
- **Do not silently widen scope.** If a task implies a change to a file outside
  the stated scope, stop and say so.
- Explanations should be mechanistic: name the variable, trace the tensor shape,
  derive the formula. High-level summaries are not useful here.
