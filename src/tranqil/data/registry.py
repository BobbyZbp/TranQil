"""Registry utilities for the scoped QT benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONTEXT_LENGTH = 20
DEFAULT_DISCOUNT = 0.99


@dataclass(frozen=True)
class TaskSpec:
    """Static metadata for a supported D4RL task."""

    env_name: str
    observation_dim: int
    action_dim: int
    dataset_alias: str
    default_context_length: int = DEFAULT_CONTEXT_LENGTH
    default_discount: float = DEFAULT_DISCOUNT


TASK_SPECS = {
    "walker2d-medium-replay-v2": TaskSpec(
        env_name="walker2d-medium-replay-v2",
        observation_dim=17,
        action_dim=6,
        dataset_alias="walker2d_medium_replay-v2.hdf5",
    ),
    "hopper-medium-replay-v2": TaskSpec(
        env_name="hopper-medium-replay-v2",
        observation_dim=11,
        action_dim=3,
        dataset_alias="hopper_medium_replay-v2.hdf5",
    ),
    "maze2d-medium-v1": TaskSpec(
        env_name="maze2d-medium-v1",
        observation_dim=4,
        action_dim=2,
        dataset_alias="maze2d-medium-sparse-v1.hdf5",
    ),
}

SUPPORTED_ENVS = tuple(TASK_SPECS)


def get_task_spec(env_name: str) -> TaskSpec:
    """Return the task spec for a supported env id."""

    try:
        return TASK_SPECS[env_name]
    except KeyError as error:
        supported = ", ".join(SUPPORTED_ENVS)
        raise KeyError(f"Unsupported env '{env_name}'. Supported envs: {supported}.") from error


def get_repo_root() -> Path:
    """Resolve the repository root from the package location."""

    return Path(__file__).resolve().parents[3]


def get_default_cache_dir() -> Path:
    """Return the local cache directory used by the QT data pipeline."""

    return get_repo_root() / "data" / "qt_cache"


def sanitize_env_name(env_name: str) -> str:
    """Create a filesystem-safe cache stem from an env id."""

    return env_name.replace("/", "_")
