# IAC-Flow — Left/Right Inferior Alveolar Canal segmentation (ToothFairy3)

**Goal (v1.0):** a strong nnU-Net baseline that segments the **left** and **right**
inferior alveolar canal (IAC) as separate classes, then a **topology-aware,
SDF-based Conditional Residual Flow Matching** model that *refines* the nnU-Net
prediction — improving canal continuity, boundary accuracy and cross-scanner
generalisation.

Output schema everywhere: `{0: background, 1: Left IAC, 2: Right IAC}`.

## Research question

> Can topology-aware **residual** flow matching, starting from nnU-Net's coarse
> Left/Right IAC prediction, improve canal connectivity (clDice), boundary
> accuracy (HD95) and generalisation to a different scanner, without hurting Dice?

The key design difference from prior flow-segmentation work (and from SEAL-Flow):
the flow does not start from noise, it starts from `x0 = SDF(nnU-Net prediction)`
and learns the residual transport to `x1 = SDF(ground truth)`. This is trained
**leakage-free** on out-of-fold nnU-Net predictions.

## Dataset — ToothFairy3, not ToothFairy4

| | ToothFairy3 | ToothFairy4 |
|---|---|---|
| Task | 77-class **segmentation** | **report generation** (CBCT→text) |
| Voxel IAC labels | **Yes** (L=id 3, R=id 4, resolved by name) | **No** |
| Size | 532 CBCT (P 417 / F 63 / S 52) | 625 CBCT + reports |
| Venue | MICCAI 2025 ODIN | MICCAI 2026 ODIN |

Development set = **P + F** (same scanner) → 5-fold CV. **S** (different scanner)
is a held-out **external OOD test set**, opened only for the final evaluation.

## Layout

```
configs/     dataset.yaml, nnunet.yaml, flow.yaml, (splits.json generated)
data/        io_utils.py            NIfTI/orientation/physical-SDF helpers (single source of truth)
             prepare_iac_dataset.py TF3 77-label → Dataset801_IAC_LR, ids resolved BY NAME, hard-fail
             audit_dataset.py       per-case geometry/label/component audit CSV
             create_folds.py        P/F stratified 5-fold + S external test → splits.json
             compute_gt_sdf.py      cache GT per-side physical (mm) SDF targets
             compute_coarse_sdf.py  cache coarse SDF from OOF nnU-Net probs (the flow's x0)
nnunet/      run_nnunet_iac.sh      Track A end-to-end
             nnUNetTrainerIAC.py    A1: all mirroring OFF (+ short-epoch variants)
             trainer_no_lr_mirror.py A2: only the physical L/R axis OFF (axis resolved + logged)
             predict_oof.py         leakage-free out-of-fold predictions (flow prior)
             predict_iac.py         generic inference + DSC wrapper
             postprocess.py         tunable connected-component cleanup
flow/        model.py               residual velocity U-Net (2 state + 8 cond → 2 velocity ch)
             conditioning.py        8-channel conditioning assembly (+ lateral axis)
             losses.py              FM + narrow-band + soft-clDice(3D) + laterality (endpoint-based)
             sampler.py             Heun ODE integration from coarse SDF
             sliding_window.py      coherent whole-volume inference (Gaussian blend, global noise)
             datasets.py            leakage-free foreground-centred patch dataset
             train.py               training + real validation + best.pt by 0.5·Dice+0.5·clDice
             validate.py            sliding-window validation metrics
             selftest.py            CPU proof the residual refinement works (seconds)
evaluation/  metrics.py             Dice, HD95, clDice, NSD (physical)
             topology_metrics.py    components, Betti-0, gap/false-branch length, swap/empty rate
             evaluate_cv.py         aggregate CV metrics + paired bootstrap CI
             evaluate_external.py   final S-set evaluation (asserts held-out)
tests/       5 unit suites (orientation, sdf, laterality, flow shapes, sliding window)
archive/     v0 flat flow, old ClaudeResponse duplicates, original browser MEMORY.md
```

## Pipeline order

```
1. audit + safe label extraction   data/audit_dataset.py ; data/prepare_iac_dataset.py
2. splits                          data/create_folds.py            → configs/splits.json
3. nnU-Net baselines A0/A1/A2      nnunet/run_nnunet_iac.sh
4. out-of-fold predictions         nnunet/predict_oof.py           → outputs/oof_probs
5. SDF caches                      data/compute_gt_sdf.py ; data/compute_coarse_sdf.py
6. residual flow (leakage-free)    flow/train.py --fold f
7. CV evaluation                   evaluation/evaluate_cv.py
8. external OOD test (S), once     evaluation/evaluate_external.py
```

## Quick start / verify (CPU, no data or GPU)

```bash
python flow/selftest.py            # residual refinement machinery (coarse Dice -> higher)
python tests/test_flow_shapes.py   # (and the other 4 test files)
```

## Domain rule

**Sagittal left/right mirror augmentation must be OFF** (it swaps class 1↔2).
A1 (`nnUNetTrainerIAC_NoMirror`) turns all mirroring off; A2
(`nnUNetTrainerIAC_NoLRMirror`) turns off only the physical L/R axis — the A1-vs-A2
comparison is an explicit experiment. TTA mirroring is disabled at inference.

## What is verified vs GPU/data-dependent

- **Verified on CPU now:** residual selftest (coarse→refined Dice up), all 5 unit
  suites, evaluation metrics/topology, fold stratification, postprocessing, every
  module imports and the losses backprop.
- **Needs the dataset + GPU:** nnU-Net training/OOF, SDF caching over 480 volumes,
  full flow training and the CV/external reports. Those scripts are complete but
  are exercised by import + unit tests, not end-to-end here.

## References
ToothFairy3/2 (Bolelli et al., MICCAI/CVPR/MIA); SEAL-Flow (2D, architectural
reference); FlowSDF (arXiv 2405.18087); clDice (arXiv 2003.07311). See
`docs/project_notes.md`.
