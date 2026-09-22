#!/usr/bin/env python3
"""Five-fold nnU-Net ensemble probabilities for the unused TF3-S cohort."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch


def case_id_from_image(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(path)
    return path.name[: -len(suffix)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--expected", type=int, default=52)
    parser.add_argument("--tile-step", type=float, default=0.5)
    parser.add_argument("--preprocess-workers", type=int, default=2)
    parser.add_argument("--export-workers", type=int, default=2)
    args = parser.parse_args()

    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as error:
        raise RuntimeError("Install nnunetv2 before external inference") from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The five-fold 3-D ensemble requires a CUDA runtime")

    raw_root = Path(args.raw_root).expanduser().resolve()
    model_folder = Path(args.model_folder).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = sorted((raw_root / "imagesTr").glob("ToothFairy3S_*_0000.nii.gz"))
    if len(images) != args.expected:
        raise RuntimeError(f"Expected {args.expected} TF3-S images, found {len(images)}")
    for fold in range(5):
        checkpoint = model_folder / f"fold_{fold}" / args.checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    inputs, outputs = [], []
    for image in images:
        case_id = case_id_from_image(image)
        prefix = output / case_id
        # Resume safely. nnU-Net's probability output is the required artifact.
        if prefix.with_suffix(".npz").exists():
            continue
        inputs.append([str(image)])
        outputs.append(str(prefix))

    if inputs:
        constructor = inspect.signature(nnUNetPredictor).parameters
        options = {
            "tile_step_size": float(args.tile_step),
            "use_gaussian": True,
            # The frozen 3-class L/R model was trained without sagittal mirror
            # label swapping. Mirrored TTA would invalidate laterality.
            "use_mirroring": False,
            "perform_everything_on_device": True,
            "device": device,
            "verbose": False,
            "verbose_preprocessing": False,
            "allow_tqdm": True,
        }
        predictor = nnUNetPredictor(
            **{key: value for key, value in options.items() if key in constructor}
        )
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            use_folds=(0, 1, 2, 3, 4),
            checkpoint_name=args.checkpoint,
        )
        signature = inspect.signature(predictor.predict_from_files)
        kwargs = {"save_probabilities": True, "overwrite": False}
        if "num_processes_preprocessing" in signature.parameters:
            kwargs["num_processes_preprocessing"] = args.preprocess_workers
        if "num_processes_segmentation_export" in signature.parameters:
            kwargs["num_processes_segmentation_export"] = args.export_workers
        predictor.predict_from_files(inputs, outputs, **kwargs)

    probabilities = sorted(output.glob("ToothFairy3S_*.npz"))
    if len(probabilities) != args.expected:
        raise RuntimeError(
            f"External probability export incomplete: {len(probabilities)}/{args.expected}"
        )
    print(f"External five-fold probabilities ready: {len(probabilities)} in {output}")


if __name__ == "__main__":
    main()
