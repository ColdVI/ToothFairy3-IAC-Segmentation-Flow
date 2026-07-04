# -*- coding: utf-8 -*-
"""
trainer_no_lr_mirror.py — nnU-Net trainer that disables ONLY the physical
left/right (sagittal) mirror axis, keeping the other mirror augmentations.

This is experiment A2: the hypothesis that the *anatomically meaningful* axis is
the only one that must not be mirrored (mirroring it swaps Left<->Right IAC),
while superior/inferior and anterior/posterior mirroring remain harmless and
useful augmentation.

Resolving WHICH tensor axis is left/right is the subtle part. The training array
is not in world orientation: nnU-Net applies `configuration_manager.transpose_forward`
to the (already reoriented) data. We therefore:
  1. take the source orientation (ToothFairy3 is RPI -> array axis 0 is R<->L),
  2. apply transpose_forward to find where that axis lands in the network tensor,
  3. remove only that index from the mirror list.

Because getting this wrong silently corrupts the experiment, the resolved axis is
LOGGED, and an env override `IAC_LR_AXIS` (0/1/2) forces a specific axis if you
have verified it against your plans.json. VERIFY the logged axis on first run.
"""
import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# ToothFairy3 volumes are RPI: the first spatial axis (index 0) is the
# Right<->Left (sagittal) axis in the reoriented source array.
SOURCE_LR_AXIS = 0


class nnUNetTrainerIAC_NoLRMirror(nnUNetTrainer):
    def _resolve_lr_axis(self):
        override = os.environ.get("IAC_LR_AXIS")
        if override is not None:
            ax = int(override)
            self.print_to_log_file(f"[IAC] L/R mirror axis forced by IAC_LR_AXIS={ax}")
            return ax
        try:
            transpose = list(self.configuration_manager.transpose_forward)
            ax = transpose.index(SOURCE_LR_AXIS)
        except Exception as e:
            ax = SOURCE_LR_AXIS
            self.print_to_log_file(f"[IAC] WARNING could not read transpose_forward ({e}); "
                                   f"defaulting L/R axis to {ax}. VERIFY against plans.json.")
        self.print_to_log_file(f"[IAC] resolved physical L/R mirror axis -> tensor index {ax}. "
                               f"VERIFY this is correct for your plans.json.")
        return ax

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy, initial_patch, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        if mirror_axes is not None:
            lr = self._resolve_lr_axis()
            kept = tuple(ax for ax in mirror_axes if ax != lr)
            mirror_axes = kept if kept else None
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation, dummy, initial_patch, mirror_axes


class nnUNetTrainerIAC_NoLRMirror_50ep(nnUNetTrainerIAC_NoLRMirror):
    # Base ile birebir aynı imza (bkz. nnUNetTrainerIAC.py notu: *args/**kwargs KeyError verir).
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
