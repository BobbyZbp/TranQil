#!/usr/bin/env python

from __future__ import annotations

"""Render a short random-policy preview rollout for one Gym/D4RL task.

Purpose:
    Validate the final runtime layer after the smoke test: the environment must
    be able to reset, accept actions, advance with `step(...)`, render
    `rgb_array` frames, and write a preview artifact.

Pipeline role:
    Phase 5: `preview rollout validation`

Functionality implemented here:
    - parse preview configuration from CLI flags
    - create the requested Gym/D4RL env
    - seed the action space when possible
    - sample actions online via `env.action_space.sample()`
    - execute the live environment with `env.step(action)`
    - capture rendered RGB frames
    - continue across episodes unless `--stop-on-done` is set
    - export a `.mp4` when ffmpeg support is available, otherwise a `.gif`
    - write a neighboring JSON summary with run metadata

Important behavior:
    - this script generates a random-policy rollout from the live environment
    - it does not replay a stored trajectory from the D4RL `.hdf5` dataset
    - one preview file may contain multiple episodes when `--stop-on-done`
      remains false and the env terminates before the step budget is exhausted

Output convention:
    Default outputs are written under `results/previews/` using an env-derived
    filename such as `walker2d_medium_replay_v2_random.mp4`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tranqil.evaluation import create_gym_env, reset_env, unwrap_step_output
from tranqil.rendering import (
    capture_frame,
    mp4_is_supported,
    resolve_random_rollout_output_path,
    resolve_summary_path,
    write_preview,
)
from tranqil.runtime import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a short random-policy rollout preview.")
    parser.add_argument("--env", required=True, help="Gym/D4RL environment id.")
    parser.add_argument("--steps", type=int, default=200, help="Maximum rollout length.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for env and action sampling.")
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
        help="Output path. Defaults to results/previews/<env>_random.mp4 when possible, else .gif.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "mp4", "gif"),
        default="auto",
        help="Preferred export format.",
    )
    parser.add_argument(
        "--stop-on-done",
        action="store_true",
        help="Stop at the first terminal state instead of continuing across episodes.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    mp4_supported = mp4_is_supported()
    output_path = resolve_random_rollout_output_path(
        env_name=args.env,
        output=args.output,
        output_format=args.format,
        mp4_supported=mp4_supported,
    )

    if args.format == "mp4" and output_path.suffix.lower() != ".mp4":
        raise RuntimeError("mp4 was requested but imageio-ffmpeg is not available.")

    env = create_gym_env(args.env, action_seed=args.seed)

    try:
        episode_count = 1
        observation = reset_env(env, args.seed)
        total_reward = 0.0
        frames = [capture_frame(env)]
        steps_executed = 0

        for step_idx in range(args.steps):
            action = env.action_space.sample()
            observation, reward, done, _ = unwrap_step_output(env.step(action))
            total_reward += float(reward)
            steps_executed += 1

            if step_idx % args.frame_skip == 0 or done:
                frames.append(capture_frame(env))

            if done:
                if args.stop_on_done:
                    break
                episode_count += 1
                observation = reset_env(env, args.seed + episode_count - 1)
                frames.append(capture_frame(env))

        write_preview(output_path, frames, fps=args.fps)

        summary = {
            "env": args.env,
            "seed": args.seed,
            "steps_requested": args.steps,
            "steps_executed": steps_executed,
            "frames_written": len(frames),
            "fps": args.fps,
            "frame_skip": args.frame_skip,
            "total_reward": total_reward,
            "output": str(output_path),
            "format": output_path.suffix.lower().lstrip("."),
            "mp4_supported": mp4_supported,
            "episodes_visited": episode_count,
            "stop_on_done": args.stop_on_done,
            "observation_shape": list(np.asarray(observation).shape),
        }
    finally:
        env.close()

    summary_path = resolve_summary_path(output_path)
    write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
