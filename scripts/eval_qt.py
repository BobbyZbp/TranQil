#!/usr/bin/env python
"""Evaluate a saved QT checkpoint in the live environment."""

from __future__ import annotations

import argparse

from tranqil.evaluation import build_evaluation_config, evaluate_policy, load_evaluation_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TranQil QT checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the YAML experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path produced by training.")
    parser.add_argument("--episodes", type=int, default=None, help="Optional override for evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional override for max env steps.")
    parser.add_argument("--output-path", default=None, help="Optional JSON summary path.")
    parser.add_argument("--device", default=None, help="Optional override for the evaluation device.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for the evaluation seed.")
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help="Optional override for the evaluation target return.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = load_evaluation_session(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        target_return=args.target_return,
    )
    summary = evaluate_policy(
        actor=session.actor,
        critic=session.critic,
        task_spec=session.task_spec,
        stats=session.stats,
        evaluation_config=build_evaluation_config(session.config),
        device=session.device,
        seed=int(session.config["seed"]),
        output_path=args.output_path,
        checkpoint_path=args.checkpoint,
        checkpoint_step=session.checkpoint_step,
        state_mean=session.state_mean,
        state_std=session.state_std,
    )
    print("=== QT Evaluation Complete ===")
    print(f"checkpoint_path: {args.checkpoint}")
    print(f"episodes: {summary['episodes']}")
    print(f"mean_return: {summary['mean_return']:.4f}")
    if "mean_normalized_score" in summary:
        print(f"mean_normalized_score: {summary['mean_normalized_score']:.4f}")
    print(f"mean_episode_length: {summary['mean_episode_length']:.2f}")


if __name__ == "__main__":
    main()
