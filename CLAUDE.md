# CLAUDE.md — ToothFairy3 IAC Left/Right Segmentation with Residual Flow

Persistent context for coding agents working in this repository. Read the source
before changing it: this document separates observations from code facts,
conditional mathematics, untested explanations, and research decisions.

---

## 1. Project and research question

This is a two-track pipeline for left/right Inferior Alveolar Canal (IAC)
segmentation in dental CBCT using ToothFairy3.

- **Track A — discriminative prior.** nnU-Net v2, `3d_fullres`, trainer
  `nnUNetTrainerIAC_NoMirror`, five folds, 1000 epochs per fold. Track A training
  is complete and is not repeated for Track B debugging.
- **Track B — refinement under study.** A model transports or directly refines
  the out-of-fold nnU-Net SDF `x0` toward the GT SDF `x1`, conditioned on CBCT
  and prior-derived channels.

The intended claim is not assumed true. The research question is:

> We test whether a correctly derived stochastic bridge formulation can avoid
> prior degradation and improve topology while maintaining non-inferior overlap.

Any stochastic method must also be compared with a same-conditioning,
approximately same-capacity deterministic direct refiner. Until such controls
and the full-CV baseline exist, terms such as “fixes”, “calibrated uncertainty”,
and “clinical safety envelope” are conclusions, not premises.

Clinical motivation is continuity and boundary localisation of a thin tubular
structure. Outputs are candidate segmentations for research; uncertainty maps
may be called an **uncertainty envelope** or **candidate safety margin**, not a
clinically validated safety envelope.

---

## 2. VERIFIED MEASUREMENTS

These are observations from existing artifacts. They do not by themselves prove
a causal mechanism.

### 2.1 Track A, five-fold CV, `Mean Validation Dice`

| fold | Dice   | epochs | wall (A100) |
|------|--------|--------|-------------|
| 0    | 0.9050 | 1000   | ~11.0 h     |
| 1    | 0.9104 | 1000   | ~10.8 h     |
| 2    | 0.9094 | 1000   | ~11.0 h     |
| 3    | 0.9147 | 1000   | ~10.7 h     |
| 4    | 0.9111 | 1000   | ~10.7 h     |

Mean: **0.9101 ± 0.0032**. Approximately 54 A100-hours were spent. These
reported validation numbers are the Track A reference, but the paper baseline
still requires evaluation through the exact full-CV comparison pipeline.

### 2.2 Existing Track B progress artifact

The existing fold-0 progress artifact reports:

| epoch | trainloss | dice   | cldice | hd95 (mm) | score  |
|-------|-----------|--------|--------|-----------|--------|
| 0     | 0.2674    | 0.8867 | 0.9927 | 0.430     | 0.9397 |
| 25    | 0.1771    | 0.7401 | 0.9940 | 0.803     | 0.8670 |
| 50    | 0.1902    | 0.8050 | 0.9941 | 2.082     | 0.8995 |
| 75    | 0.1804    | 0.7940 | 0.9941 | 2.069     | 0.8941 |
| 100   | 0.1838    | 0.7982 | 0.9928 | 2.052     | 0.8955 |
| 125   | 0.1691    | 0.7969 | 0.9931 | 2.058     | 0.8950 |

The artifact shows falling training loss together with lower validation Dice and
higher HD95 than its epoch-0 row. It does **not** establish why. Its precise case
set and OOF provenance must be audited before using it as paper evidence.
Historical immutable per-epoch checkpoints were not saved: the progress rows do
not make an exact checkpoint trajectory recoverable, and no epoch-0, epoch-25,
or epoch-125 checkpoint may be inferred or reconstructed from them.

### 2.3 Prompt-1 complete identity baseline

- The configured development cohort contains 480 P/F cases; the held-out S
  scanner cohort contains 52 separate cases.
- All 480 development cases passed the OOF provenance and geometry audit.
- Direct OOF hard mask, direct coarse-SDF sign decode, and zero-velocity
  sliding-window full-path outputs differed by zero voxels.
- The complete per-case identity baseline measured Dice 0.910125, clDice
  0.991210, and HD95 0.829462 mm. Its immutable report SHA256 is
  `356f7dae6904e2dbe2d3f81ff21e2151d57623dc9f2f7928caeffcc6e1ad83e5`.

This supersedes the earlier partial 40-case preflight as the measured identity
reference. It does not set the Dice non-inferiority margin; that remains an
explicit pre-run user decision.

### 2.4 Prompt-2 limited endpoint diagnostic

The historical training did not retain immutable per-epoch checkpoints, so an
exact training trajectory is unavailable. The only audited endpoints are:

- `best.pt`, labelled only `best_legacy_unknown_epoch`; its internal epoch is
  absent. Calling it epoch 0 or epoch 1 is invalid. Whether it is early or
  prior-like is a hypothesis.
- `last.pt`, labelled `epoch_129` after its internal epoch value was verified as
  129.

The diagnostic was produced on an NVIDIA L4 from git
`7b6a32686fab381a0840d6f512a6ca9647f9593a`, using 12 cases, the same 48-patch
grid (24 foreground and 24 pure-background patches) for both endpoints, and a
case-level bootstrap. Its manifest records `protocol_deviation=true`,
`exact_epoch_trajectory_available=false`, and
`historical_per_epoch_checkpoints_were_not_saved=true`.

Measured endpoint observations, limited to the sampled cases and strata:

- The legacy best endpoint produced better full-volume segmentation than epoch
  129 on the five thickening-probe cases (10 sides): mean Dice 0.849671 versus
  0.732484 and mean HD95 0.581389 versus 6.335279 mm.
- Epoch 129 nevertheless had lower checkpoint-based FM loss averaged across the
  diagnostic t-grid and strata (0.000491 versus 0.002787). Low FM loss therefore
  did not imply better full-volume segmentation in this endpoint comparison.
- Epoch-129 predictions showed a geometry signal compatible with thickening:
  mean prediction/GT volume ratio was 1.5926 before erosion, and physical
  erosion improved Dice and HD95 on 9/10 sides. Mean Dice rose to 0.809608.
- One epoch-129 outlier, `ToothFairy3F_011` side 1, retained approximately
  55.3 mm HD95 after erosion. Uniform thickening does not explain that failure.
- At epoch 129, coarse-prior zero/swap interventions changed the output more
  than CBCT zero/shuffle interventions. This ordering was not uniform for every
  intervention at the unknown-epoch legacy best endpoint; CBCT Gaussian-noise
  sensitivity was also substantial at epoch 129.
- Direct analytic-shortcut similarity was not close to an exact implementation
  (mean cosine was 0.139 at epoch 129 and -0.068 at the legacy best endpoint;
  mean R² was negative). Prior dependence and endpoint behaviour are diagnostic
  evidence, not mathematical proof that the algebraic shortcut is implemented.

These observations motivate a new, prospectively checkpointed early-epoch
Fold-0 trajectory pilot. They are diagnostic-only: they are not a paper proof,
do not identify the legacy best epoch or the epoch where degradation began, and
do not establish monotonic thickening throughout historical training.

---

## 3. CODE-VERIFIED FACTS

These statements follow from the current source. They are not claims about what
the trained model internally learned.

### 3.1 Current state and conditioning path

`flow/datasets.py` loads `coarse_sdf` as `x0` and also supplies the same clean
coarse SDF as conditioning channels 3 and 4. `flow/train.py` may add noise to
the state start, constructs `x_t` from that noised start, and leaves the
conditioning coarse-SDF channels clean.

For a deterministic batch (`sigma=0`):

```text
x_t = (1-t)x0 + t x1
x_t - x0 = t(x1-x0)
u = x1-x0 = (x_t-x0)/t,  t>0
```

The FM target is therefore an algebraic function of inputs available to the
network. This establishes that a shortcut **exists**. It does not establish
that a trained network uses it.

### 3.2 Current loss and endpoint estimate

The current loss combines FM, optional narrow-band, soft-clDice, laterality,
and TV terms. It has no soft-Dice overlap term. For velocity parameterisation:

```text
x1_hat = x_t + (1-t) pred_v
```

At large `t`, this endpoint estimate includes a large fraction of GT through
`x_t`; gradients from endpoint auxiliary terms to `pred_v` are multiplied by
`1-t`. The coordinate-aware laterality option exists in `flow/losses.py`, but
the current training loop does not pass `lateral_coord`. The overlap-only
laterality term is not mathematically identical to zero, although it can be
numerically negligible in patches containing only one canal.

### 3.3 Sliding-window numerics

`flow/sliding_window.py` uses a separable Gaussian with `sigma_scale=0.125` and
divides accumulated output by `max(wsum, 1e-6)`. The one-axis endpoint weight is
approximately `exp(-32)`; a three-axis patch corner is approximately
`exp(-96)`. Whether this creates observed false positives is an empirical
hypothesis.

### 3.4 Physical SDF and evaluation

- `data/io_utils.py` computes SDFs using
  `distance_transform_edt(sampling=spacing)`, in physical millimetres.
- `sdf_stack_to_mask` decodes negative SDF values as foreground and selects the
  more-negative L/R channel.
- Evaluation implements Dice, HD95, NSD, clDice, Betti-0 error, centerline gap,
  false-branch length, empty prediction, and L/R swap rate.

### 3.5 OOF prior artifact semantics

`nnunet/predict_oof.py` currently runs each validation case with its expected
fold model, but converts the resulting hard segmentation into:

```python
prob_left  = (seg == 1).astype(np.float16)
prob_right = (seg == 2).astype(np.float16)
```

Therefore the existing `prob_left`/`prob_right` arrays are **derived one-hot hard
masks**, not calibrated softmax probabilities. Consequences:

- Values are expected to be binary, not continuous confidences.
- They carry no nnU-Net confidence or entropy information.
- Conditioning channels 1–2 are largely redundant with coarse-SDF channels
  3–4, because both are derived from the same hard segmentation.
- The filenames alone do not encode artifact type, source checkpoint, or fold;
  provenance must be supplied by a manifest and audited.
- A fold-aware script is leakage-free only if the actual split, checkpoint, and
  per-case provenance match. Existing artifacts are not trusted by filename.

Hard and soft artifacts must remain distinct:

```text
oof_hard/       hard segmentation or explicitly derived one-hot arrays
oof_softmax/    true nnU-Net softmax export with class probabilities
```

A backward-compatible adapter may read legacy `oof_probs/`, but must label its
contents from evidence. It must never relabel a hard mask as true softmax.

---

## 4. CONDITIONAL THEOREMS / DERIVATIONS

These results are mathematically valid under their stated assumptions. They are
not measurements of a trained checkpoint.

### 4.1 Noised-state shortcut error

Let `x0n = x0 + sigma*eps`, let training use
`x_t=(1-t)x0n+t*x1`, and let conditioning expose clean `x0`. The analytic
clean-prior shortcut differs from the FM target by:

```text
v_shortcut = (x_t-x0)/t
u          = x1-x0n
v_shortcut-u = sigma*eps/t
```

Thus squared shortcut error from this noise term scales as
`sigma^2/t^2` in expectation. This does not show that the network implements the
shortcut.

### 4.2 Rollout collapse if the shortcut is learned exactly

Assume inference starts at `x=x0`, the network emits a finite `w` at `t=0`, and
for every later step exactly implements `v=(x-x0)/t`. Euler gives:

```text
x(dt)   = x0 + dt*w
v(dt)   = w
x(2dt)  = x0 + 2dt*w
...
x(1)    = x0 + w
```

The same cancellation applies to the corresponding Heun evaluations under the
exact assumption. Therefore the multi-step rollout collapses to the first
finite endpoint direction **if the shortcut is learned exactly**. Whether the
checkpoint satisfies this assumption is tested with causal ablations and
similarity measures; it is not a measured fact.

### 4.3 Auxiliary-loss GT mixture

For the current rectified path and velocity endpoint estimate:

```text
x1_hat = (1-t)x0 + t*x1 + (1-t)pred_v
```

At `t=0.9`, 90% of `x1_hat` is explicitly `x1`, and the derivative with respect
to `pred_v` is `1-t=0.1`. This proves the mixture and gradient scaling. Its
practical effect on optimisation remains an empirical question.

### 4.4 Geometric dilation compatibility calculation

For an ideal cylindrical cross-section of radius `r`, uniform dilation by `d`
has the approximate overlap:

```text
Dice = 2 / (1 + (1+d/r)^2)
```

This can show that an observed Dice loss is **compatible with** a dilation. It
cannot show that thickening fully explains an actual prediction without volume,
surface-distance, erosion, and radius-profile measurements.

---

## 5. HYPOTHESES TO TEST

1. **Learned shortcut use.** The trained velocity model relies on `x_t` and clean
   `x0` while being insensitive to CBCT. Test with CBCT zero/noise/shuffle,
   cross-case coarse-SDF swaps, analytic-shortcut cosine similarity and R²,
   prospectively saved early-epoch checkpoints, foreground/background strata,
   and case bootstrap CIs. The historical run cannot supply checkpoint
   progression; its limited endpoint evidence is mixed and diagnostic-only.
2. **Small-t profile.** Deterministic small-t loss may plateau because `x_t`
   contains little target information; noised shortcut error may grow like
   `sigma²/t²`. A low large-t loss alone is not evidence of shortcut use because
   `x_t` already contains `x1` information.
3. **Uniform thickening.** The Dice/clDice pattern may be compatible with tube
   inflation. Test physical erosion, volume ratios, signed surface distances,
   and radius profiles.
4. **Patch-distribution mismatch.** Foreground-biased training may cause errors
   on pure-background patches encountered by whole-volume inference.
5. **Gaussian blending failure.** Very small boundary weights plus the `1e-6`
   denominator floor may change signs or shrink SDF magnitude near volume edges.
6. **Localised HD95 outliers.** Far false-positive components rather than uniform
   thickening may drive HD95. Test component-to-GT distance distributions.
7. **Auxiliary-loss dilution.** GT mixture and `(1-t)` gradients may make current
   auxiliary losses ineffective at large `t`.
8. **Stochastic bridge utility.** A correctly derived bridge may avoid prior
   degradation and improve topology at non-inferior overlap; it must first pass
   a known-distribution 1-D toy test and deterministic-refiner controls.
9. **Curve-space feasibility.** A longest-path spline may compactly represent
   ordinary canals, but can erase bifid/accessory anatomy. Fit failures,
   rasterised connectivity, self-intersections, endpoint errors, and lost branch
   length must be measured before making a contribution claim.

---

## 6. RESEARCH DECISIONS AND CONSTRAINTS

### 6.1 Non-negotiable invariants

1. **Never train Track B on in-sample nnU-Net predictions.** OOF only.
2. **Ground truth never enters inference.** If inference needs `sdf_gt`, it is
   invalid.
3. **SDF distances remain physical millimetres**, never voxel units.
4. **Do not use naive L/R mirroring.** Label-aware class/channel swapping may be
   a separately predeclared ablation. Existing Track A models are not retrained
   for it because Track A is complete.
5. **Safe checkpoints require a measured complete-CV prior and a predeclared
   non-inferiority margin.** Partial baselines never set the floor.
6. **Log every loss component separately** at every validation step.
7. **Every run writes a manifest:** git SHA, dirty flag, resolved config and hash,
   fold, seed, environment, times, provenance, and resulting metrics.
8. **Every long entry point is resumable, idempotent, and atomic.** A session may
   die after any case or epoch.
9. **Do not modify Track A in a way that requires retraining.**
10. **Do not add a new feature, loss, or architecture before the current stage's
    acceptance criteria pass.** Ablate before tuning.
11. **The 52-case S scanner cohort is never used for debugging, tuning, model
    selection, qualitative case selection, or threshold selection.** It is
    opened once after the final development configuration is locked; failure is
    reported without returning to development for reselection.
12. **Hard one-hot OOF masks are never described as calibrated probabilities.**
    Every OOF artifact records whether it is hard segmentation, derived one-hot,
    or true softmax.
13. **Every final flow comparison includes a same-conditioning, approximately
    same-capacity deterministic direct-refinement control.**
14. **Verified measurements, code-verified facts, conditional derivations,
    hypotheses, and intended research claims remain explicitly separated.**
15. **No stochastic bridge sampler is used in 3-D before its derivation and a
    known-distribution 1-D toy test pass.**

### 6.2 Model-selection contract

- `last.pt`: always updated for resume.
- `best_any.pt`: best validation checkpoint within a run, even below the prior;
  retained for debugging.
- `best_safe.pt`: written only after the predeclared non-inferiority condition.
- Selection is not reduced to one weighted score. Eligibility checks Dice
  non-inferiority first, then ranks topology, HD95, clDice, and earlier epoch.
- The non-inferiority margin is `null` until the full 480-case identity baseline
  is complete and the user sets the margin before seeing any B-run result.

### 6.3 OOF and held-out-data contract

- Development is the 480-case P/F cohort used for five-fold OOF construction.
- S is the 52-case scanner-shift cohort and is excluded from all Prompt-1
  manifests and all model-development decisions.
- Every OOF case records expected fold, actual prediction fold, source
  checkpoint, artifact type, shape, affine, spacing, and checksum.
- `oof_hard/` and `oof_softmax/` are separate queues and artifacts. Missing true
  softmax is reported, never fabricated from one-hot masks.

### 6.4 CurveFlow and uncertainty language

A single spline does not guarantee a clinically or raster-topologically valid
canal. Connectivity is tested after rasterisation. The longest-path step can
delete bifid or accessory branches and is a primary feasibility risk. Report
bifid and non-bifid cases separately, including lost branch length and
failed-fit rate.

Stochastic sample variance is only one uncertainty method. It is compared with
true-softmax entropy, TTA variance, feasible MC-dropout/ensemble controls, and a
naive morphological envelope. Monte Carlo stability is measured; `K=16` is not
treated as clinical sufficiency.

---

## 7. Channel and artifact contracts

Current channel order:

```text
FLOW_STATE_CH = 2  [Left SDF, Right SDF]
COND_CH       = 8  [CBCT, hard_L, hard_R, coarse_SDF_L, coarse_SDF_R, x, y, z]
```

Legacy code may call channels 1–2 `prob_left` and `prob_right`; for current
derived-one-hot artifacts, documentation and manifests use `hard_L`/`hard_R`.
Changing conditioning channels requires a single-source channel contract and
tests across model, dataset, validation, and sliding-window inference.

---

## 8. Compute and workflow

- All dataset-scale work runs in Google Colab. Persistent inputs and outputs are
  on Google Drive; the repository may be cloned under `/content`.
- Local `/Users/...` paths are never embedded in runtime code or notebooks.
- The local Mac is for editing only. Do not assume local data, CUDA, MPS, or long
  CPU runs.
- Track A training is complete. Missing fold-aware OOF inference may reuse the
  existing Track A checkpoints but must never retrain them.
- Before a long queue, run a two-case smoke test covering prediction, validation,
  atomic Drive write, interruption, and resume.
- Every case is independently validated and persisted. Completed valid artifacts
  are skipped on restart; failures and retries remain visible.
- Prompt 1 is complete, but the non-inferiority margin remains an explicit user
  decision. The legacy Prompt-3 B0–B3 50-epoch grid is temporarily superseded:
  do not launch it, Prompt 4–6, or any 3-D bridge training until Prompt 3R code
  and the prospectively checkpointed short Fold-0 pilot pass their declared
  acceptance criteria.

---

## 9. Repository map and working style

```text
configs/          Track B configuration and fold definitions
data/             geometry, SDF, cache preparation and validation
nnunet/           completed Track A definitions and fold-aware OOF inference
flow/             current residual-flow model, losses, training and validation
evaluation/       overlap, boundary and topology metrics
analysis/         non-mutating audits and diagnostic probes
scripts/          manifests and orchestration helpers
notebooks/        Colab runners; persistent artifacts remain on Drive
archive/          abandoned/reference implementations only
```

- Read source before editing; this file may be stale.
- One concern per commit: `stage/topic: what changed and why`.
- Write known-answer tests before changing metrics or losses.
- State assumptions and provenance. Never turn a hypothesis into a fact through
  wording.
- Do not silently widen scope. Prompt 1 hardening must not implement Prompt 2–6.
