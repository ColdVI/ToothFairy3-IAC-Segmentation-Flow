#!/usr/bin/env bash
# =============================================================================
# run_nnunet_iac.sh  —  Left/Right IAC segmentation with nnU-Net v2
# =============================================================================
# This is the GPU stage. It CANNOT run on the macOS/16GB/no-GPU laptop that
# built the dataset; run it on a CUDA box (>=1 GPU, >=12 GB VRAM ideally 24 GB).
#
# Pipeline:
#   0. install nnU-Net v2 + torch (CUDA)
#   1. set nnU-Net env vars (raw / preprocessed / results roots)
#   2. build Dataset801_IAC_LR from ToothFairy3 (runs the python converter)
#   3. plan & preprocess
#   4. train 3D full-res (5 folds, or a single fold to start)
#   5. (optional) find best config + predict
# =============================================================================
set -euo pipefail

# Resolve repo root from this script's location (script lives in nnunet/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- EDIT THESE PATHS -------------------------------------------------------
export TF3_SRC="${TF3_SRC:-$HOME/Desktop/ToothFairy3}"        # source dataset
export nnUNet_raw="${nnUNet_raw:-$HOME/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$HOME/nnUNet_results}"
DATASET_ID=801
DATASET_NAME="Dataset${DATASET_ID}_IAC_LR"
# Domain rule: L/R IAC separation is destroyed by sagittal mirror augmentation,
# so we ALWAYS train with the no-mirroring trainer. See nnunet/nnUNetTrainerIAC.py.
TRAINER="nnUNetTrainerIAC_NoMirror"
# -----------------------------------------------------------------------------

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "== [0] install nnU-Net v2 (skip if present) =="
python -c "import nnunetv2" 2>/dev/null || pip install nnunetv2

echo "== [1] env =="
echo "  nnUNet_raw          = $nnUNet_raw"
echo "  nnUNet_preprocessed = $nnUNet_preprocessed"
echo "  nnUNet_results      = $nnUNet_results"

echo "== [2] build $DATASET_NAME from ToothFairy3 =="
# --copy-images is safer on clusters that don't follow symlinks into $HOME/Desktop.
python "$REPO_ROOT/data/prepare_iac_dataset.py" \
    --src "$TF3_SRC" \
    --dst "$nnUNet_raw/$DATASET_NAME" \
    --workers 8

echo "== [2b] install the custom trainers into the nnU-Net package =="
# nnU-Net discovers trainers by class name inside its installed package tree.
NNUNET_DA_DIR="$(python -c 'import os,nnunetv2; print(os.path.join(os.path.dirname(nnunetv2.__file__),"training","nnUNetTrainer","variants","data_augmentation"))')"
cp "$REPO_ROOT/nnunet/nnUNetTrainerIAC.py" "$NNUNET_DA_DIR/"      # A1: all mirroring off
cp "$REPO_ROOT/nnunet/trainer_no_lr_mirror.py" "$NNUNET_DA_DIR/"  # A2: only L/R axis off
echo "  installed trainers -> $NNUNET_DA_DIR/"

echo "== [3] plan & preprocess =="
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

echo "== [4] train 3d_fullres (mirroring OFF via $TRAINER) =="
# Train a single fold first to validate the setup end-to-end:
nnUNetv2_train $DATASET_ID 3d_fullres 0 -tr $TRAINER
# Then the remaining folds (can be parallelised across GPUs):
# for f in 1 2 3 4; do nnUNetv2_train $DATASET_ID 3d_fullres $f -tr $TRAINER; done

echo "== [5] (optional) pick best config =="
# nnUNetv2_find_best_configuration $DATASET_ID -c 3d_fullres -tr $TRAINER

echo "== [6] (optional) predict on a test folder (TTA mirroring disabled) =="
# nnUNetv2_predict -i /path/to/imagesTs -o /path/to/preds \
#     -d $DATASET_ID -c 3d_fullres -f 0 -tr $TRAINER --disable_tta

echo "DONE."
