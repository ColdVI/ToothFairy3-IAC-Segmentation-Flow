# Drive flow source snapshot — 2026-09-22

Selected source files preserved verbatim from Drive before any cleanup. This is an archival snapshot, not a complete standalone training package or a newly validated model. No data or weights are included. Existing repository training entry points are unchanged.

## Files and provenance

| Snapshot | Drive source | Purpose |
|---|---|---|
| rectified_flow/iac_rectified_flow_v1.py | iac_flow_v1/iac_rectified_flow_v1.py; file 1Tb4cOJZzBeHvpwqbdAVM-IPQdaylSbW8 | Two-channel SDF velocity U-Net, prior-to-GT rectified path, image/prior conditioning, Heun inference |
| iacb/bridge.py | iacb/iacb/bridge.py; file 1igWATjTu0WiV6vGpQBPASsMKh1jPjiAM | Stochastic SDF bridge; endpoint prediction from image, current SDF and time |
| iacb/train_epoch_casebalanced_v3.py | iacb_runs_532_rpi_v2/train_epoch_casebalanced_v3.py; file 1F9_jA8aHWe3HspgN0eg0bYPD-3pIP9jR | Case-balanced IAC-B training runner |

Validation: Python syntax compilation passed for all three files. No training, GPU test, metric validation, or dataset-scale evaluation was run. The files retain their original imports and Drive paths. The complete Drive iacb package (including common, cache and inference helpers) remains required; do not delete it on the assumption that this selected snapshot replaces it.

## Architecture distinctions

- Historical teacher version (per the 2026-09-11 project report): binary nnU-Net logits plus auxiliary SDF head, SDF feature encoder, bridge and SDF decoder. Normal forward emits base logits; auxiliary decoded SDF is not the normal inference mask. t=1 training and no-op common-logit fusion were reported for that version.
- Original dense SDF refinement: frozen left/right nnU-Net prediction -> two SDFs -> refinement/flow -> threshold SDF into labels.
- CanalManifoldFlow: prior -> centerline stations and shape parameters -> parameter corrections -> rasterized tube or anchored SDF correction.
- GeoFlow-Newton: moving surface -> resample frozen prior probabilities near current surface -> geometric/Newton update -> mask.
- Gaussian-to-SDF (existing repository iacflow): Gaussian state -> image-conditioned SDF prediction and integration; pretrained image backbone partly trainable.
- IAC-B snapshot: prior SDF -> noisy prior/GT intermediate state during training -> image-conditioned clean SDF endpoint predictor -> bridge sampling from prior at inference. No learned latent AE in StateUNet's path.
- Rectified snapshot: prior SDF -> current SDF state; velocity U-Net sees image, current state, clean prior SDF, two prior probabilities and uncertainty. Target velocity is GT SDF minus prior SDF. Heun integration produces final SDF. No noise or latent AE.
- The earlier segmentation/graph joint-flow proposal is a separate research design; do not describe these SDF files as its implementation.

GT participates in training targets/intermediates and evaluation. It must not be supplied as inference conditioning.

## Protocol difference

The old repository protocol reserves the 52 S cases for external testing. Drive IAC-B integration instead explicitly includes the 52 S cases in refiner training. The rectified snapshot partitions 403 training / 32 validation / 97 evaluation cases (default settings). This archived code does not silently supersede the repository protocol. Those trained-on S cases cannot be reported as an untouched external test of that refiner.

## Drive cleanup inventory

Paths below are relative to ToothFairy/ToothFairy3. This is a family-level inventory, not a recursive deletion audit. No files were deleted or moved.

| Path | Family / disposition |
|---|---|
| ToothFairy3/ | Original dataset; retain |
| iac_runs/nnUNet_results/ | Frozen baseline weights and metadata; retain |
| iac_runs/configs_cache/ | Split/config provenance; retain |
| iac_runs/canalmanifold_oof_softmax/ | Reused by newer IAC-B; retain |
| iac_runs/dataset_cache_colab_v1, dataset_cache_colab_v2, sdf_cache_backup, caches | Derived legacy caches; deletion candidates only after dependency and regeneration checks |
| iac_runs/prompt3r_stage1a_v2, prompt3r_stage1a_v3_overnight | Dense SDF trials; preserve metrics/configs/checkpoint selection before removing bulky outputs |
| iac_runs/code, outputs, old | Mixed content; not yet recursively audited |
| CanalManifoldFlow, CanalManifoldFlow_corrected_v1 | Original/corrected parameter-space flow source packages; archive whole code first |
| CanalManifoldFlow_v2_GeoFlow_Newton | GeoFlow-Newton source package; archive whole code first |
| canalmanifold/ | CMF caches, runs, audits and v2 outputs; preserve reports/configs/best checkpoints; regenerate-only caches are candidates |
| teacher_baseline_cache, teacher_baseline_results | Teacher artifacts; preserve measured baseline evidence and original checkpoint |
| IAC_Flow_Training_Package | Gaussian-to-SDF package and runs; repository already contains an iacflow track, byte parity not established |
| iacb/ | Newer SDF bridge package; retain (also supplies metrics to rectified flow) |
| iacb_runs_532, iacb_runs_532_aligned_v1 | Earlier 532-case run families; exact contents/config differences require further audit |
| iacb_runs_532_rpi_v2 | Recent case-balanced runner, cache, evaluation and epoch runs; retain for current workflow |
| iac_flow_v1, iac_flow_v1_runs | Latest rectified flow code and run family; retain |

## Future naming convention

Use explicit families rather than generic v1/v2 alone:
- code: Git repository, with a commit pinned per run
- priors/nnunet_lr_oof/<source-model-and-split-id>/
- cache/<representation>_<orientation>_<spacing>_<schema>/
- runs/YYYY-MM-DD_<method>_<cohort>_fold0_seed42/
- each run: manifest.json, config.json, split.json, checkpoints/, metrics/, predictions/, logs/

Record source commit, input/cache manifests, label mapping, orientation, physical spacing, source nnU-Net fold/checkpoint, training/validation/evaluation case lists, and selected checkpoint. Keep current Drive paths until consuming configurations are migrated together.
