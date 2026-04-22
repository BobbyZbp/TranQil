"""Original-QT anchor metadata for the scoped replication tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data.registry import get_repo_root


@dataclass(frozen=True)
class AnchorTaskSpec:
    """Original-repo task metadata used by anchor-mode training and eval."""

    env_name: str
    env_family: str
    dataset_name: str
    version: int
    max_ep_len: int
    scale: float
    test_scale: float
    env_targets: tuple[float, ...]
    eta: float
    grad_norm: float
    max_iters: int
    num_steps_per_iter: int
    early_epoch: int

    @property
    def trajectory_export_path(self) -> Path:
        return get_repo_root() / "data" / "D4RL" / f"{self.env_name}.pkl"


SCOPED_ANCHOR_TASKS = {
    "walker2d-medium-replay-v2": AnchorTaskSpec(
        env_name="walker2d-medium-replay-v2",
        env_family="walker2d",
        dataset_name="medium-replay",
        version=2,
        max_ep_len=1000,
        scale=1000.0,
        test_scale=1000.0,
        env_targets=(5000.0, 4000.0, 2500.0),
        eta=2.0,
        grad_norm=5.0,
        max_iters=500,
        num_steps_per_iter=1000,
        early_epoch=100,
    ),
    "hopper-medium-replay-v2": AnchorTaskSpec(
        env_name="hopper-medium-replay-v2",
        env_family="hopper",
        dataset_name="medium-replay",
        version=2,
        max_ep_len=1000,
        scale=1000.0,
        test_scale=1000.0,
        env_targets=(3600.0, 1800.0),
        eta=3.0,
        grad_norm=9.0,
        max_iters=500,
        num_steps_per_iter=1000,
        early_epoch=100,
    ),
    "maze2d-medium-v1": AnchorTaskSpec(
        env_name="maze2d-medium-v1",
        env_family="maze2d",
        dataset_name="medium",
        version=1,
        max_ep_len=1000,
        scale=10.0,
        test_scale=10.0,
        env_targets=(300.0, 200.0, 150.0, 100.0, 50.0, 20.0),
        eta=5.0,
        grad_norm=9.0,
        max_iters=100,
        num_steps_per_iter=1000,
        early_epoch=50,
    ),
}


def get_anchor_task_spec(env_name: str) -> AnchorTaskSpec:
    """Return anchor metadata for one supported env id."""

    try:
        return SCOPED_ANCHOR_TASKS[env_name]
    except KeyError as error:
        supported = ", ".join(sorted(SCOPED_ANCHOR_TASKS))
        raise KeyError(
            f"Unsupported anchor env '{env_name}'. Supported envs: {supported}."
        ) from error
