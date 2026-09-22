"""Export frozen nnU-Net priors for the 52 ToothFairy3S cases.

The existing 480 development cases already have true out-of-fold probabilities in
``canalmanifold_oof_softmax/fold_0..fold_4``.  The 52 S-cases were excluded from
all five nnU-Net training folds, so *any* fold is unseen for them.

To keep the prior distribution comparable to the 480 OOF cases (one network per
case rather than a 5-model ensemble), this script deterministically assigns each
S-case to exactly one frozen fold and exports ``save_probabilities=True`` outputs to
``<prob_root>/external_singlefold``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path


def _load_external(splits: Path) -> list[str]:
    obj = json.loads(splits.read_text())
    ext = list(obj.get("external_test", []))
    if not ext:
        raise ValueError(f"{splits}: no external_test cases")
    return ext


def _install_custom_trainer(trainer_dir: Path | None):
    if trainer_dir is None:
        return
    trainer_dir = Path(trainer_dir)
    src = trainer_dir / "nnUNetTrainerIAC.py"
    if not src.exists():
        raise FileNotFoundError(f"Custom trainer file not found: {src}")
    import nnunetv2
    dst_dir = Path(nnunetv2.__file__).resolve().parent / "training" / "nnUNetTrainer"
    dst = dst_dir / src.name
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        shutil.copy2(src, dst)
        importlib.invalidate_caches()
        print(f"[trainer] installed {src} -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_tf3", required=True, help="full 532-case ToothFairy3 root")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--model_folder", required=True)
    ap.add_argument("--prob_root", required=True, help="canalmanifold_oof_softmax root")
    ap.add_argument("--trainer_dir", default=None, help="folder containing nnUNetTrainerIAC.py")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preprocess_workers", type=int, default=2)
    ap.add_argument("--export_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    _install_custom_trainer(Path(a.trainer_dir) if a.trainer_dir else None)

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    raw = Path(a.raw_tf3)
    splits = Path(a.splits)
    model = Path(a.model_folder)
    out = Path(a.prob_root) / "external_singlefold"
    out.mkdir(parents=True, exist_ok=True)
    ext = _load_external(splits)

    # Stable, transparent assignment: sorted case i -> fold (i mod 5).
    assignment = {case: i % 5 for i, case in enumerate(sorted(ext))}
    manifest = {
        "policy": "sorted external_test cases assigned round-robin to one unseen frozen nnU-Net fold",
        "why_single_fold": "matches the one-model-per-case distribution of the 480 development OOF priors",
        "cases": assignment,
    }
    (out / "source_fold_manifest.json").write_text(json.dumps(manifest, indent=2))

    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This is a 3D nnU-Net export job; use a CUDA Colab runtime")

    for fold in range(5):
        cases = [c for c in sorted(ext) if assignment[c] == fold]
        missing = [c for c in cases if a.overwrite or not (out / f"{c}.npz").exists()]
        print(f"[fold {fold}] assigned={len(cases)} missing={len(missing)}")
        if not missing:
            continue

        inputs, outputs = [], []
        for case in missing:
            image = raw / "imagesTr" / f"{case}_0000.nii.gz"
            if not image.exists():
                raise FileNotFoundError(image)
            inputs.append([str(image)])
            outputs.append(str(out / case))

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(model),
            use_folds=(fold,),
            checkpoint_name="checkpoint_final.pth",
        )
        predictor.predict_from_files(
            inputs,
            outputs,
            save_probabilities=True,
            overwrite=a.overwrite,
            num_processes_preprocessing=a.preprocess_workers,
            num_processes_segmentation_export=a.export_workers,
        )

    ready = [c for c in ext if (out / f"{c}.npz").exists()]
    print(f"[done] external priors ready: {len(ready)}/{len(ext)} -> {out}")
    if len(ready) != len(ext):
        raise RuntimeError(f"External prior export incomplete: {len(ready)}/{len(ext)}")


if __name__ == "__main__":
    main()
