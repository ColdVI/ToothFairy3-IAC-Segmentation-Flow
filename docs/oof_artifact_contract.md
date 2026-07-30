# OOF artifact contract

Track B keeps hard segmentations and true softmax exports as separate artifact
types and queues:

```text
oof_hard/<case_id>.npz       # derived one-hot, or .nii.gz hard segmentation
oof_softmax/<case_id>.npz    # true class probabilities from nnU-Net export
oof_manifest.json            # per-case fold/checkpoint/type/checksum provenance
```

Legacy `oof_probs/<case_id>.npz` files containing binary `prob_left` and
`prob_right` arrays are read through `--legacy-oof` and labelled
`derived_one_hot`. They are never renamed or treated as calibrated probability.

The manifest has a top-level `cases` mapping. Every case entry records at least:

```json
{
  "prediction_fold": 0,
  "source_checkpoint": ".../fold_0/checkpoint_final.pth",
  "source_checkpoint_sha256": "...",
  "artifact_type": "derived_one_hot",
  "hard_artifact": ".../oof_hard/CASE.npz",
  "softmax_artifact": null,
  "hard_sha256": "...",
  "softmax_sha256": null
}
```

Legacy provenance bootstrap does not infer a fold from an existing filename.
It re-runs the case with the independently expected validation-fold checkpoint,
requires voxel-wise equality with the untouched `oof_probs` artifact, and saves
the official softmax separately. Verified legacy entries additionally record:

```json
{
  "verification_method": "legacy_reprediction_voxelwise",
  "legacy_voxelwise_match": true,
  "legacy_changed_voxels": 0,
  "legacy_hard_artifact": ".../oof_probs/CASE.npz",
  "legacy_hard_sha256": "...",
  "reproduced_hard_sha256": "...",
  "verified_at": "..."
}
```

A mismatch is a non-retryable preflight failure. The legacy artifact is never
overwritten or moved into `oof_hard`.

`artifact_type` is one of `hard_segmentation`, `derived_one_hot`, or
`true_softmax`. The expected validation fold is independently reconstructed
from `configs/splits.json`; a manifest cannot redefine it.
