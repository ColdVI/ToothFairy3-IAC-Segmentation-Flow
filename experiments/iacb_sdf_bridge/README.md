# IAC-B stochastic SDF bridge

This is the reviewed Drive source snapshot, including the later case-balanced trainer. See ../../docs/FLOW_ARCHITECTURES_TR.md and ../../docs/FLOW_RUN_LAYOUT.md.

Run from this directory:

    python -m iacb.selftest --skip_e2e
    python train_epoch_casebalanced_v3.py --help

The default bridge predicts a clean SDF endpoint from CBCT, current SDF and time. It starts from the frozen nnU-Net prior and has no trained SDF autoencoder. ae_gate.py is a separate experiment.

The recorded case-balanced run has 403/32/97 train/validation/eval cases, 45 epochs and best patch-val loss at epoch 27. Its full-volume result was not found; older zero-Dice output belongs to a different run. S cases are used for training in this protocol.

Keep checkpoints, splits, OOF provenance and RPI cache preparation outside GitHub. Old raw-data notebooks do not reproduce the intervening alignment steps automatically.
