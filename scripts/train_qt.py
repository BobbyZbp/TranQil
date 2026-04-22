#!/usr/bin/env python
"""Train the first QT milestone model from a YAML config."""

from __future__ import annotations

import argparse

from tranqil.cli_overrides import apply_train_overrides
from tranqil.config import load_experiment_config
from tranqil.anchor_trainer import AnchorQTTrainer
from tranqil.trainer import QTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first TranQil QT milestone.")
    parser.add_argument("--config", required=True, help="Path to the YAML experiment config.")
    parser.add_argument("--steps", type=int, default=None, help="Optional override for training steps.")
    parser.add_argument("--run-name", default=None, help="Optional override for the output run name.")
    parser.add_argument("--base-dir", default=None, help="Optional override for outputs.base_dir.")
    parser.add_argument("--device", default=None, help="Optional override for the training device.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for the run seed.")
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help="Optional override for evaluation target return during training.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional path to a checkpoint (latest.pt) to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_train_overrides(
        load_experiment_config(args.config),
        steps=args.steps,
        run_name=args.run_name,
        base_dir=args.base_dir,
        device=args.device,
        seed=args.seed,
        target_return=args.target_return,
    )

    is_anchor = bool(config.get("anchor", {}).get("enabled", False))
    if is_anchor:
        trainer = AnchorQTTrainer(config, resume_from=args.resume_from)
    else:
        if args.resume_from is not None:
            raise SystemExit("--resume-from is currently only supported for anchor-mode training.")
        trainer = QTTrainer(config)
    summary = trainer.train()
    print("=== QT Training Complete ===")
    print(f"run_dir: {summary['run_dir']}")
    print(f"checkpoint_path: {summary['checkpoint_path']}")
    print(f"best_checkpoint_path: {summary['best_checkpoint_path']}")
    print(f"metrics_path: {summary['metrics_path']}")
    print(f"final_step: {summary['final_step']}")
    if summary["best_evaluation"] is not None:
        print(f"best_mean_return: {summary['best_evaluation']['mean_return']:.4f}")


if __name__ == "__main__":
    main()
