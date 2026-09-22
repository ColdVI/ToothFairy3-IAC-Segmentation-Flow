# CanalManifoldFlow v2 — corrected Geometric Boundary Flow

This bundle preserves the R1/R2/R3 comparators written after the
`CanalManifoldFlow_Corrected` runs and adds the corrected primary arm:
`geoflow_newton`. The transported state is the unrestricted boundary
displacement `h(s,theta)` in millimetres. The old absolute-crossing residual is
kept only for R1/R2 checkpoint compatibility; it is not described as
closed-loop image evidence.

The frozen five-fold nnU-Net OOF predictions remain the prior. No nnU-Net is
retrained.

## Corrected GeoFlow in one equation

```text
V = alpha + a_zero_mean(s)
    + [beta(s,theta) * r_ref/r_t]_zero_angle_mean
    + gain(s,theta) * newton_local(s,theta)
```

`newton_local=(0.5-p_at_surface)/(dp/dr)` and the local slope magnitude are
recomputed from offsets around the moving surface. Flat/rising/clipped profiles
are invalidated and the Newton step is capped. `alpha` is identifiable because
the station and angular branches are gauge-centred. `sharpness` is reported as
local boundary strength, not calibrated uncertainty.

The supervised linear bridge has constant on-path target velocity. Therefore
`velocity_drift` is a descriptive rollout statistic, not a GO/NO-GO rule.
`profile_remeasurement_rms` separately checks whether the moving query read a
different probability value; only exact paired Dice/HD95/clDice and ablations
establish benefit.

## What changed

1. **O0–O6 headroom is a parallel diagnostic.** `scripts/oracle_headroom.py` locks a
   global threshold on refinement-train cases, fits train-only temperature
   scaling, and measures patient/side threshold and morphology oracles,
   calibrated probability-map Bayes-Dice, the m<=8 tube ceiling, and the free
   radial-surface ceiling. O1–O3 are explicitly labelled GT oracles.
2. **R1 analytic tube now uses m=2..8.** The state is `160x17+3 = 2723`, which
   raises the representation ceiling while retaining positive radii.
3. **R2 removes the rank ceiling.** It transports an unrestricted
   `h(s,theta)` table (`160x32`) plus two endpoints. The Bishop frame is kept as
   an anatomical coordinate system; the harmonic tube is a comparator/prior,
   not the only output family.
4. **Identity-preserving decode is `phi0-H`.** `H` is a normal extension of the
   predicted surface displacement. A zero correction returns the raw nnU-Net
   mask bit-for-bit. The old `phi0 + psi(qhat) - psi(q0)` SDF-field anchor is
   no longer primary. The continuous reach argument motivates the trust limit;
   voxel topology is still audited rather than claimed by construction.
5. **Training closes the objective/path mismatch.** R1/R2 retain their legacy
   scalar channel for checkpoint compatibility. The new GeoFlow arm replaces
   it with safeguarded local Newton + slope features, uses endpoint-vanishing
   path noise, q0 jitter, radius-area-weighted physical-mm velocity loss, and
   terminal decoded soft-Dice. GT occupancy is supervision only and never
   enters the model.
6. **R3 is a real dense comparator.** The prior-coupled dense SDF arm uses the
   same linear flow, path augmentation, band-weighted physical objective,
   terminal soft-Dice, and an identity-preserving original-SDF delta decode.
7. **Topology reporting is fair.** Components below `0.27 mm^3` are removed
   from both raw and refined outputs. Raw, post-processed raw, tube, normal
   displacement, Dice, HD95, clDice, component error, bootstrap intervals,
   and paired sign tests are retained separately.
8. **Epoch selection is separated from evidence.** One fold locks the epoch;
   the other four folds are the primary report. Archived checkpoints make this
   protocol reproducible.

No clDice, persistent-homology, TV, or extra topology loss is added: R1/R2
already constrain the surface family, while the observed fragmentation came
from the former anchor decoder rather than the tube rollout.

## Cache migration (no CBCT read)

The old `cache_q0centered_v2` shards are not silently accepted because their
states stop at m=4 and they lack GT occupancy supervision. Upgrade them using
the existing OOF probability files and labels; the stored CBCT/image shell is
reused:

```bash
python scripts/upgrade_cache_v2.py \
  --config /path/to/canalmanifold_v2.yaml \
  --source-cache /path/to/cache_q0centered_v2 \
  --output-cache /path/to/cache_m8_v2
```

For a clean rebuild, `python -m canalmanifold.cli --config ... precompute` also
produces the v2 cache, but it rereads CBCT volumes.

## Locked execution order

The ready-to-run Colab entry point is
`notebooks/CanalManifoldFlow_GeoFlow_Newton_Colab.ipynb`. Its path resolver
prefers the Drive layout verified on 2026-09-01
(`iac_runs/dataset_cache_colab_v2`, `iac_runs/configs_cache/splits.json`, and
`iac_runs/canalmanifold_oof_softmax`). The project itself is resolved below
`ToothFairy/ToothFairy3`, not directly below `ToothFairy`.

```bash
# 0. Data-free shape/identity contracts
python scripts/smoke_test.py
python scripts/test_v2_contracts.py
python scripts/test_geoflow_contracts.py

# 1. Quantify in-domain headroom for the discussion (does not stop training)
python scripts/oracle_headroom.py \
  --config /path/to/canalmanifold_v2.yaml --fold 0 \
  --output /path/to/runs_v2/oracle_fold0

# 2. Corrected GeoFlow (oracle is diagnostic, not a training gate)
python -m canalmanifold.cli --config /path/to/canalmanifold_geoflow_newton.yaml \
  train-geoflow --fold 0

# Optional historical comparators: R1 and R2
python -m canalmanifold.cli --config /path/to/canalmanifold_v2.yaml \
  train --mode linear --fold 0
python -m canalmanifold.cli --config /path/to/canalmanifold_v2.yaml \
  train-surface --fold 0

# 3. Exact raw-vs-corrected-GeoFlow evaluation
python scripts/evaluate_v2_exact.py \
  --config /path/to/canalmanifold_geoflow_newton.yaml --fold 0 \
  --geoflow-checkpoint /path/to/runs_v2/fold_0/geoflow_newton/best.pt \
  --output /path/to/runs_v2/exact_geoflow_newton_fold0

# R1/R2 comparators can be included in the same exact call:
python scripts/evaluate_v2_exact.py \
  --config /path/to/canalmanifold_v2.yaml --fold 0 \
  --tube-checkpoint /path/to/runs_v2/fold_0/linear/best.pt \
  --surface-checkpoint /path/to/runs_v2/fold_0/surface_h/best.pt \
  --output /path/to/runs_v2/exact_fold0

# 4. Optional dense prior-coupled comparator
python -m canalmanifold.cli --config /path/to/canalmanifold_v2.yaml dense-precompute
python -m canalmanifold.cli --config /path/to/canalmanifold_v2.yaml train-dense --fold 0
python -m canalmanifold.cli --config /path/to/canalmanifold_v2.yaml evaluate-dense \
  --fold 0 --checkpoint /path/to/runs_v2/fold_0/dense_sdf/best.pt \
  --output /path/to/runs_v2/dense_exact_fold0

# Optional uncertainty deliverable (calibrate jitter on selection fold only)
python scripts/safety_envelope.py \
  --config /path/to/canalmanifold_v2.yaml \
  --checkpoint /path/to/runs_v2/fold_0/surface_h/best.pt \
  --fold 0 --samples 64 --coverage 0.95 \
  --output /path/to/runs_v2/safety_envelope_fold0
```

Do not spend the locked 52-case ToothFairy3S cohort after a fold-0-selected
checkpoint. First lock one epoch and report it on the other folds:

```bash
python scripts/lock_crossfold_epoch.py \
  --run-root /path/to/runs_v2 --selection-fold 0 \
  --representation geoflow_newton \
  --output /path/to/runs_v2/geoflow_newton_epoch_lock.json

python scripts/report_crossfold.py \
  --lock /path/to/runs_v2/geoflow_newton_epoch_lock.json \
  --metrics 1=/path/to/fold1/metrics.csv \
  --metrics 2=/path/to/fold2/metrics.csv \
  --metrics 3=/path/to/fold3/metrics.csv \
  --metrics 4=/path/to/fold4/metrics.csv \
  --method gbf_newton_normal_pp \
  --output /path/to/runs_v2/geoflow_newton_crossfold.json
```

Lock and report folds 1-4 before touching Set C, so no fold-0-selected
checkpoint leaks into the external cohort. Positive, null, and negative
cross-fold outcomes are all reported; the protocol is an evidence boundary,
not a training gate.
