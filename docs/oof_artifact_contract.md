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
  "artifact_type": "derived_one_hot",
  "hard_artifact": ".../oof_hard/CASE.npz",
  "softmax_artifact": null,
  "hard_sha256": "...",
  "softmax_sha256": null
}
```

`artifact_type` is one of `hard_segmentation`, `derived_one_hot`, or
`true_softmax`. The expected validation fold is independently reconstructed
from `configs/splits.json`; a manifest cannot redefine it.
