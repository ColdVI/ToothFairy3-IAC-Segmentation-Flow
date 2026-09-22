"""Optional frozen nnU-Net OOF probability export.

This module intentionally delegates preprocessing and export to nnU-Net v2 so
the saved probability geometry matches the original images.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from .manifest import load_splits


def export_oof_probabilities(config: dict[str, Any], *, fold: int | None = None, overwrite: bool = False) -> Path:
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as error:
        raise RuntimeError("Install nnunetv2 before exporting OOF probabilities") from error

    paths = config["paths"]
    options = config.get("nnunet_oof", {})
    dataset_root = Path(paths["dataset_root"]).expanduser().resolve()
    model_folder = Path(options["model_folder"]).expanduser().resolve()
    output_root = Path(paths["oof_probability_root"]).expanduser().resolve()
    splits = load_splits(paths["splits_file"])
    folds = [int(fold)] if fold is not None else list(range(len(splits)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("OOF export is a 3-D nnU-Net job; select a CUDA runtime")

    for fold_index in folds:
        output_dir = output_root / f"fold_{fold_index}"
        output_dir.mkdir(parents=True, exist_ok=True)
        case_ids = splits[fold_index]["val"]
        inputs, outputs = [], []
        for case_id in case_ids:
            image = dataset_root / "imagesTr" / f"{case_id}_0000.nii.gz"
            output_prefix = output_dir / case_id
            if output_prefix.with_suffix(".npz").exists() and not overwrite:
                continue
            if not image.exists():
                raise FileNotFoundError(image)
            inputs.append([str(image)])
            outputs.append(str(output_prefix))
        if not inputs:
            continue
        predictor_kwargs = {
            "tile_step_size": float(options.get("tile_step_size", 0.5)),
            "use_gaussian": True,
            "use_mirroring": bool(options.get("use_mirroring", False)),
            "perform_everything_on_device": True,
            "device": device,
            "verbose": False,
            "verbose_preprocessing": False,
            "allow_tqdm": True,
        }
        constructor_parameters = inspect.signature(nnUNetPredictor).parameters
        predictor = nnUNetPredictor(
            **{key: value for key, value in predictor_kwargs.items() if key in constructor_parameters}
        )
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            use_folds=(fold_index,),
            checkpoint_name=str(options.get("checkpoint_name", "checkpoint_final.pth")),
        )
        signature = inspect.signature(predictor.predict_from_files)
        kwargs = {
            "save_probabilities": True,
            "overwrite": overwrite,
        }
        if "num_processes_preprocessing" in signature.parameters:
            kwargs["num_processes_preprocessing"] = int(options.get("preprocess_workers", 2))
        if "num_processes_segmentation_export" in signature.parameters:
            kwargs["num_processes_segmentation_export"] = int(options.get("export_workers", 2))
        predictor.predict_from_files(inputs, outputs, **kwargs)
    return output_root
