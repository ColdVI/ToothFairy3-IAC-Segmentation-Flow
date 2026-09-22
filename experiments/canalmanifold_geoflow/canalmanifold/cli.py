"""Command-line entry points used by the Colab notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .dense_sdf import evaluate_dense_sdf, precompute_dense_sdf, train_dense_sdf
from .evaluate import evaluate
from .geoflow_train import train_geoflow
from .manifest import write_manifest
from .oof_predict import export_oof_probabilities
from .precompute import precompute
from .surface_train import train_surface
from .train import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="canalmanifold")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    oof = subparsers.add_parser("oof")
    oof.add_argument("--fold", type=int)
    oof.add_argument("--overwrite", action="store_true")
    cache = subparsers.add_parser("precompute")
    cache.add_argument("--limit", type=int)
    cache.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("gate")
    trainer = subparsers.add_parser("train")
    trainer.add_argument("--mode", choices=("direct", "linear", "staged"), required=True)
    trainer.add_argument("--fold", type=int, default=0)
    trainer.add_argument("--no-resume", action="store_true")
    surface = subparsers.add_parser("train-surface")
    surface.add_argument("--fold", type=int, default=0)
    surface.add_argument("--no-resume", action="store_true")
    geoflow = subparsers.add_parser("train-geoflow")
    geoflow.add_argument("--fold", type=int, default=0)
    geoflow.add_argument("--no-resume", action="store_true")
    dense_cache = subparsers.add_parser("dense-precompute")
    dense_cache.add_argument("--limit", type=int)
    dense_cache.add_argument("--overwrite", action="store_true")
    dense_train = subparsers.add_parser("train-dense")
    dense_train.add_argument("--fold", type=int, default=0)
    dense_train.add_argument("--no-resume", action="store_true")
    dense_eval = subparsers.add_parser("evaluate-dense")
    dense_eval.add_argument("--checkpoint", required=True)
    dense_eval.add_argument("--fold", type=int, default=0)
    dense_eval.add_argument("--output", required=True)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--checkpoint", required=True)
    evaluator.add_argument("--fold", type=int, default=0)
    evaluator.add_argument("--save-predictions", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "manifest":
        print(write_manifest(config))
    elif args.command == "oof":
        print(export_oof_probabilities(config, fold=args.fold, overwrite=args.overwrite))
    elif args.command == "precompute":
        print(precompute(config, limit=args.limit, overwrite=args.overwrite))
    elif args.command == "gate":
        report = Path(config["paths"]["cache_dir"]) / "precompute_report.csv"
        frame = pd.read_csv(report)
        valid = frame[frame.status.isin(["ok", "cached"])]
        if valid.empty:
            raise RuntimeError(f"No successful shards in {report}")
        table = pd.DataFrame(
            [
                {
                    "representation": "analytic tube m=2..8",
                    "fit_dice_mean": valid.q1_ceiling_dice.mean(),
                    "fit_dice_std": valid.q1_ceiling_dice.std(),
                    "hd95_mm_mean": valid.q1_ceiling_hd95_mm.replace([float("inf")], pd.NA).mean(),
                    "fallback_fraction": (
                        valid.training_excluded.astype(float).mean()
                        if "training_excluded" in valid
                        else valid.fallback.astype(float).mean()
                    ),
                },
                {
                    "representation": "free radial surface h(s,theta)",
                    "fit_dice_mean": valid.q1_free_ceiling_dice.mean(),
                    "fit_dice_std": valid.q1_free_ceiling_dice.std(),
                    "hd95_mm_mean": valid.q1_free_ceiling_hd95_mm.replace(
                        [float("inf")], pd.NA
                    ).mean(),
                    "fallback_fraction": (
                        valid.training_excluded.astype(float).mean()
                        if "training_excluded" in valid
                        else valid.fallback.astype(float).mean()
                    ),
                },
                {
                    "representation": "raw OOF nnU-Net prior",
                    "fit_dice_mean": pd.NA,
                    "fit_dice_std": pd.NA,
                    "hd95_mm_mean": pd.NA,
                    "fallback_fraction": 0.0,
                },
            ]
        )
        output = report.parent / "representation_gate.csv"
        table.to_csv(output, index=False)
        print(table.to_string(index=False))
        print(output)
    elif args.command == "train":
        print(train(config, mode=args.mode, refinement_fold=args.fold, resume=not args.no_resume))
    elif args.command == "train-surface":
        print(train_surface(config, refinement_fold=args.fold, resume=not args.no_resume))
    elif args.command == "train-geoflow":
        print(
            train_geoflow(
                config,
                refinement_fold=args.fold,
                resume=not args.no_resume,
            )
        )
    elif args.command == "dense-precompute":
        print(precompute_dense_sdf(config, overwrite=args.overwrite, limit=args.limit))
    elif args.command == "train-dense":
        print(train_dense_sdf(config, refinement_fold=args.fold, resume=not args.no_resume))
    elif args.command == "evaluate-dense":
        print(
            evaluate_dense_sdf(
                config,
                checkpoint_path=args.checkpoint,
                refinement_fold=args.fold,
                output_dir=args.output,
            )
        )
    elif args.command == "evaluate":
        print(
            evaluate(
                config,
                checkpoint_path=args.checkpoint,
                refinement_fold=args.fold,
                save_predictions=args.save_predictions,
            )
        )


if __name__ == "__main__":
    main()
