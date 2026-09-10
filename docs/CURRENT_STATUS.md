# Current research status — September 2026

This file separates measured results from active experiments. Scores are only
comparable when cohort, grid, post-processing and full-volume evaluator match.

## What is already measured

The original teacher implementation completed 1000 epochs. In the shared code,
the bridge used `T_CONST=1.0`; the bridge learning-rate assignment could also be
overwritten by the scheduler. Its best checkpoint was evaluated GT-free on the
107 Fold-0 validation cases:

| Run | Dice mean | Dice median | HD95 mean | HD95 median | clDice | Components |
|---|---:|---:|---:|---:|---:|---:|
| Original teacher, best | 0.901905 | 0.915134 | 1.249 mm | 0.300 mm | 0.989248 | 2.271 |

Most cases were well behaved: 95/107 predictions had exactly two connected
components. A few severe failures inflate mean HD95; for example `F_011` had
Dice 0.8206, HD95 43.27 mm and 9 components. The 0.300 mm median HD95 equals one
voxel on the evaluated grid, so sub-voxel ranking should not be the sole claim.

Historical CanalManifold/GeoFlow experiments used a different common exact
Fold-0 cohort (97 patients, 194 sides) and must not be subtracted directly from
the 107-case teacher result:

| Method | Dice | HD95 | clDice | Components | Interpretation |
|---|---:|---:|---:|---:|---|
| Raw nnU-Net | 0.905050 | 0.450542 mm | 0.990846 | 1.057 | Cohort baseline |
| CMF corrected linear | 0.905313 | 0.444735 mm | 0.982827 | 1.959 | Accuracy nearly neutral; topology worse |
| CMF corrected staged | 0.905744 | 0.443752 mm | 0.982682 | 2.119 | Small selected-checkpoint gain; topology worse |
| GeoFlow-Newton | 0.903412 | 0.454341 mm | 0.987628 | — | Worse than frozen-prior baseline |

The first GeoFlow formulation had a mathematical degeneracy and is not treated
as valid evidence. These experiments show why this repository no longer treats
post-hoc refinement of a frozen segmentation as the main path.

## Active runs

Two active experiments must not be conflated:

1. **Fixed teacher branch:** the historical code with the identified time,
   bridge/fusion and learning-rate-stage issues corrected. This is a bug-fix
   comparison, not a new architecture.
2. **`iacflow/` conditional FM:** a separate model initialized from the trained
   three-class nnU-Net. It transports Gaussian noise to a two-channel clipped,
   normalized physical SDF. The final L/R mask comes from the integrated SDF
   state, not from frozen nnU-Net logits.

The conditional FM run uses random time sampling and the closed-form target
velocity. Early encoder stages are frozen for memory; the deepest encoder stage,
decoder, zero-effect state/time adapters and SDF head are trainable. SDF/cache
work is performed once, and training reads memory-mapped arrays rather than
recomputing EDT per patch.

The current 5-case sentinel is an engineering/budget diagnostic, not a result.
At step 3000 its NFE=4 Dice was 0.853351 versus the sentinel baseline 0.885791;
at step 4000 it was 0.840998. State sensitivity was non-zero, which shows the
network uses the evolving state, but does not show that the dynamics are useful.
No final 107-case claim is recorded until the run finishes and the fixed
full-volume evaluator is used.

## Success criterion for the current flow

Training loss decreasing only proves SDF regression. A useful multi-step flow
requires the same checkpoint at NFE=4 to improve filtered Betti-0 error over
both the frozen baseline and NFE=1, while the paired Dice difference remains
within -0.001. clDice and missing-centerline fraction are supporting metrics.
If NFE=1 and NFE=4 are effectively identical, the model has collapsed to direct
image-to-SDF prediction even if its Dice is good.

The full decision is made on the 107-case, full-volume, GT-free protocol. The
small sentinel panel may stop an obviously unproductive GPU run, but cannot
establish superiority.
