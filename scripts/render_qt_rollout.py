#!/usr/bin/env python
"""Render a learned QT policy rollout from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tranqil.evaluation import (
    build_evaluation_config,
    create_gym_env,
    load_evaluation_session,
    normalize_episode_returns,
    rollout_anchor_policy_episode,
    rollout_policy_episode,
)
from tranqil.rendering import (
    capture_frame,
    mp4_is_supported,
    resolve_qt_rollout_output_path,
    resolve_summary_path,
    write_preview,
)
from tranqil.runtime import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a learned QT checkpoint rollout.")
    parser.add_argument("--config", required=True, help="Path to the YAML experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to render.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum rollout length.")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for the rollout seed.")
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help="Optional override for the target return conditioning value.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output video fps.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Keep every N-th rendered frame to control file size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to the run-local rollouts directory.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "mp4", "gif"),
        default="auto",
        help="Preferred export format.",
    )
    parser.add_argument("--device", default=None, help="Optional device override.")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    session = load_evaluation_session(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_steps=args.max_steps,
        seed=args.seed,
        target_return=args.target_return,
    )

    evaluation_config = build_evaluation_config(session.config)
    rollout_seed = int(session.config["seed"])

    mp4_supported = mp4_is_supported()
    output_path = resolve_qt_rollout_output_path(
        checkpoint_path=session.checkpoint_path,
        env_name=session.task_spec.env_name,
        seed=rollout_seed,
        output=args.output,
        output_format=args.format,
        mp4_supported=mp4_supported,
    )
    if args.format == "mp4" and output_path.suffix.lower() != ".mp4":
        raise RuntimeError("mp4 was requested but imageio-ffmpeg is not available.")

    env = create_gym_env(session.task_spec.env_name, action_seed=rollout_seed)

    was_training = session.actor.training
    session.actor.eval()
    try:
        if session.mode == "anchor":
            rollout = rollout_anchor_policy_episode(
                actor=session.actor,
                critic=session.critic,
                env=env,
                task_spec=session.task_spec,
                state_mean=session.state_mean,
                state_std=session.state_std,
                evaluation_config=evaluation_config,
                device=session.device,
                seed=rollout_seed,
                capture_frame_fn=capture_frame,
                frame_skip=int(args.frame_skip),
            )
        else:
            rollout = rollout_policy_episode(
                actor=session.actor,
                env=env,
                task_spec=session.task_spec,
                stats=session.stats,
                evaluation_config=evaluation_config,
                device=session.device,
                seed=rollout_seed,
                capture_frame_fn=capture_frame,
                frame_skip=int(args.frame_skip),
            )
        write_preview(output_path, rollout["frames"], fps=int(args.fps))
        normalized_scores = normalize_episode_returns(
            session.task_spec.env_name,
            [float(rollout["total_reward"])],
        )
        summary = {
            "env_name": session.task_spec.env_name,
            "checkpoint_path": str(session.checkpoint_path),
            "checkpoint_step": session.checkpoint_step,
            "seed": rollout_seed,
            "target_return": float(rollout["target_return"]),
            "steps_requested": int(args.max_steps),
            "steps_executed": int(rollout["episode_length"]),
            "frames_written": len(rollout["frames"]),
            "fps": int(args.fps),
            "frame_skip": int(args.frame_skip),
            "total_reward": float(rollout["total_reward"]),
            "normalized_score": normalized_scores[0] if normalized_scores else None,
            "output": str(output_path.resolve()),
            "format": output_path.suffix.lower().lstrip("."),
            "mp4_supported": mp4_supported,
            "done_reached": bool(rollout["done_reached"]),
            "observation_shape": list(np.asarray(rollout["final_observation"]).shape),
            "candidate_target_returns": rollout.get("candidate_target_returns", []),
        }
    finally:
        if was_training:
            session.actor.train()
        if hasattr(env, "close"):
            env.close()

    summary_path = resolve_summary_path(output_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
