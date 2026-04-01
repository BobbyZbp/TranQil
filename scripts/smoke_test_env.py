#!/usr/bin/env python

from __future__ import annotations

"""Validate the scoped D4RL tasks end to end.

Purpose:
    Confirm that the runtime produced by the previous pipeline stages is
    actually usable for the three scoped benchmark tasks.

Pipeline role:
    Phase 4: `smoke validation`

Functionality implemented here:
    - reads the target env list from repeated `--env` flags or from the default
      benchmark trio
    - prints runtime context such as Python executable and key environment vars
    - imports `gym`, `torch`, and `d4rl`
    - creates each env with `gym.make(...)`
    - calls `env.get_dataset()` for each task
    - reports the shapes of the core dataset arrays
    - raises an error if observations, actions, rewards, or terminals are empty

What this script is and is not:
    - it is a runtime validation step
    - it is not a rollout renderer
    - it is not a dataset-trajectory replay
    - it is not a QT training run

Outputs:
    Console validation only. If D4RL datasets are missing locally, the first
    run may populate `data/d4rl`.
"""

import argparse
import os
import sys
from pathlib import Path


TARGET_ENVS = [
    "walker2d-medium-replay-v2",
    "hopper-medium-replay-v2",
    "maze2d-medium-v1",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TranQil D4RL environment smoke tests.")
    parser.add_argument(
        "--env",
        dest="env_names",
        action="append",
        help="Specific environment ID to test. Repeat the flag to test multiple envs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_names = args.env_names or TARGET_ENVS

    print("=== TranQil Environment Smoke Test ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"D4RL_DATASET_DIR: {os.environ.get('D4RL_DATASET_DIR', '<unset>')}")
    print(f"MUJOCO_PY_MUJOCO_PATH: {os.environ.get('MUJOCO_PY_MUJOCO_PATH', '<unset>')}")
    print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', '<unset>')}")

    dataset_dir = os.environ.get("D4RL_DATASET_DIR")
    require(dataset_dir is not None, "D4RL_DATASET_DIR must be set before running the smoke test.")
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    import gym  # noqa: WPS433
    import torch  # noqa: WPS433
    import d4rl  # noqa: F401,WPS433

    print(f"gym version: {gym.__version__}")
    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    for env_name in env_names:
        print(f"\n--- Checking {env_name} ---")
        env = gym.make(env_name)
        dataset = env.get_dataset()
        observation_shape = dataset["observations"].shape
        action_shape = dataset["actions"].shape
        reward_shape = dataset["rewards"].shape
        terminal_shape = dataset["terminals"].shape

        print(f"observations: {observation_shape}")
        print(f"actions: {action_shape}")
        print(f"rewards: {reward_shape}")
        print(f"terminals: {terminal_shape}")

        require(observation_shape[0] > 0, f"{env_name} observations are empty.")
        require(action_shape[0] > 0, f"{env_name} actions are empty.")
        require(reward_shape[0] > 0, f"{env_name} rewards are empty.")
        require(terminal_shape[0] > 0, f"{env_name} terminals are empty.")

        if hasattr(env, "close"):
            env.close()

    print("\nSmoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
