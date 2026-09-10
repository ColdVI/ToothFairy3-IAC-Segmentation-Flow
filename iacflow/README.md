# `iacflow`: checkpoint-initialized conditional SDF Flow Matching

This package is the current experimental track. It is separate from the
historical prior-residual implementation in `flow/`.

Training samples a two-channel Gaussian source and a random time, constructs an
analytic Gaussian-to-SDF conditional path, and predicts the clean left/right
SDF using CBCT features plus the current state and time. Inference integrates a
shared full-volume state and decodes the final mask from the two SDF channels.

Key implementation contracts:

- clipped and normalized physical SDF targets are precomputed once;
- training time is random, and the target velocity is closed form;
- the final mask comes from the flow state, not a post-hoc logit fusion;
- state/time adapters have an algebraic zero-effect initialization;
- early nnU-Net encoder stages are frozen, while the deepest stage and decoder
  remain trainable;
- NFE=1 and NFE=4 use the same checkpoint, so this diagnostic needs no retrain;
- datasets, checkpoint files, reference masks and caches are never committed.

Run the notebook at `notebooks/IAC_Flow_Training.ipynb`. See
`docs/CURRENT_STATUS.md` for the measured baselines, non-comparable cohorts and
the predeclared success criterion.
