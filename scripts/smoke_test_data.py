#!/usr/bin/env python

from __future__ import annotations

"""Validate the scoped QT data pipeline end to end.

Purpose:
    Confirm that the QT dataset loader, preprocessing stack, metadata cache,
    and sequence sampler all work for the three scoped benchmark tasks.

Outputs:
    Console validation only. The run may write metadata caches under
    `data/qt_cache`.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from tranqil.data import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_DISCOUNT,
    SUPPORTED_ENVS,
    build_qt_dataset,
    make_sequence_dataloader,
)


EXPECTED_SAMPLE_KEYS = {
    "observations",
    "actions",
    "rewards",
    "returns_to_go",
    "next_observations",
    "terminals",
    "timeouts",
    "bootstrap_mask",
    "timesteps",
    "attention_mask",
    "episode_id",
    "env_name",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TranQil QT data smoke tests.")
    parser.add_argument(
        "--env",
        dest="env_names",
        action="append",
        help="Specific environment ID to test. Repeat the flag to test multiple envs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the sequence dataloader preview.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length for QT sequence windows.",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=DEFAULT_DISCOUNT,
        help="Discount metadata tracked for downstream critic training.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache override for metadata `.npz` files.",
    )
    return parser.parse_args()


def resolve_cache_dir(args: argparse.Namespace) -> Path:
    """Resolve the cache directory used by this smoke test run."""

    repo_root = Path(os.environ.get("TRANQIL_REPO_ROOT", Path(__file__).resolve().parents[1]))
    return args.cache_dir or (repo_root / "data" / "qt_cache" / "smoke")


def print_runtime_context(cache_dir: Path) -> None:
    """Print runtime information for the smoke test session."""

    print("=== TranQil QT Data Smoke Test ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '<unset>')}")
    print(f"D4RL_DATASET_DIR: {os.environ.get('D4RL_DATASET_DIR', '<unset>')}")
    print(f"Cache directory: {cache_dir}")


def build_dataset_for_env(
    env_name: str,
    args: argparse.Namespace,
    cache_dir: Path,
):
    """Build one QT dataset for smoke validation."""

    return build_qt_dataset(
        env_name=env_name,
        context_length=args.context_length,
        discount=args.discount,
        cache_dir=cache_dir,
    )


def validate_dataset_structure(env_name: str, dataset, context_length: int):
    """Validate a single dataset instance and return one sample."""

    require(len(dataset) > 0, f"{env_name} produced an empty QT dataset.")
    require(
        dataset.task_spec.observation_dim == dataset.observations.shape[1],
        f"{env_name} observation dim mismatch in QT dataset.",
    )
    require(
        dataset.task_spec.action_dim == dataset.actions.shape[1],
        f"{env_name} action dim mismatch in QT dataset.",
    )

    sample = dataset[0]
    require(EXPECTED_SAMPLE_KEYS.issubset(sample.keys()), f"{env_name} sample is missing QT keys.")
    require(
        tuple(sample["observations"].shape) == (context_length, dataset.task_spec.observation_dim),
        f"{env_name} sample observations have an unexpected shape.",
    )
    require(sample["env_name"] == env_name, f"{env_name} sample env_name is incorrect.")
    return sample


def validate_batch(env_name: str, batch, dataset, context_length: int) -> None:
    """Validate one dataloader batch from the QT sequence dataset."""

    require(batch["observations"].shape[0] > 0, f"{env_name} dataloader emitted an empty batch.")
    require(
        tuple(batch["observations"].shape[1:]) == (context_length, dataset.task_spec.observation_dim),
        f"{env_name} batch observations have an unexpected shape.",
    )
    require(
        tuple(batch["actions"].shape[1:]) == (context_length, dataset.task_spec.action_dim),
        f"{env_name} batch actions have an unexpected shape.",
    )


def print_env_summary(dataset, batch) -> None:
    """Print the standard per-env smoke-test summary lines."""

    print(f"samples: {len(dataset)}")
    print(f"episodes: {dataset.metadata['episode_count']}")
    print(f"cache hit on first build: {dataset.metadata['cache_hit']}")
    print(f"batch observations: {tuple(batch['observations'].shape)}")
    print(f"batch actions: {tuple(batch['actions'].shape)}")


def validate_cache_roundtrip(env_name: str, args: argparse.Namespace, cache_dir: Path, dataset) -> None:
    """Rebuild a dataset and confirm that metadata/stat caches roundtrip cleanly."""

    dataset_cached = build_dataset_for_env(env_name=env_name, args=args, cache_dir=cache_dir)
    require(bool(dataset_cached.metadata["cache_hit"]), f"{env_name} failed cache roundtrip validation.")
    require(
        np.array_equal(
            dataset.metadata["episode_start_indices"],
            dataset_cached.metadata["episode_start_indices"],
        ),
        f"{env_name} episode starts changed after cache reload.",
    )
    require(
        np.array_equal(
            dataset.metadata["valid_sample_indices"],
            dataset_cached.metadata["valid_sample_indices"],
        ),
        f"{env_name} valid sample indices changed after cache reload.",
    )
    require(
        np.allclose(dataset.stats.obs_mean, dataset_cached.stats.obs_mean),
        f"{env_name} observation means changed after cache reload.",
    )
    require(
        np.allclose(dataset.stats.obs_std, dataset_cached.stats.obs_std),
        f"{env_name} observation std changed after cache reload.",
    )


def validate_maze2d_specifics(dataset) -> None:
    """Validate Maze2D-specific timeout and next-observation behavior."""

    timeout_indices = np.flatnonzero(dataset.timeouts)
    require(timeout_indices.size > 0, "maze2d-medium-v1 should contain timeout boundaries.")
    require(int(dataset.terminals.sum()) == 0, "maze2d-medium-v1 should not contain terminal flags.")
    require(
        np.allclose(dataset.next_observations[timeout_indices], dataset.observations[timeout_indices]),
        "maze2d-medium-v1 derived next_observations leaked across episode resets.",
    )
    print(f"maze2d timeout transitions checked: {timeout_indices.size}")


def validate_env(env_name: str, args: argparse.Namespace, cache_dir: Path) -> None:
    """Run the full smoke-test flow for one environment."""

    print(f"\n--- Checking {env_name} ---")
    dataset = build_dataset_for_env(env_name=env_name, args=args, cache_dir=cache_dir)
    validate_dataset_structure(env_name=env_name, dataset=dataset, context_length=args.context_length)

    dataloader = make_sequence_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    validate_batch(env_name=env_name, batch=batch, dataset=dataset, context_length=args.context_length)
    print_env_summary(dataset, batch)
    validate_cache_roundtrip(env_name=env_name, args=args, cache_dir=cache_dir, dataset=dataset)

    if env_name == "maze2d-medium-v1":
        validate_maze2d_specifics(dataset)


def main() -> int:
    args = parse_args()
    env_names = args.env_names or list(SUPPORTED_ENVS)
    cache_dir = resolve_cache_dir(args)

    print_runtime_context(cache_dir)
    for env_name in env_names:
        validate_env(env_name=env_name, args=args, cache_dir=cache_dir)

    print("\nData smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
