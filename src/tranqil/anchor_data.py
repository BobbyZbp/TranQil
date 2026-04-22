"""Original-QT trajectory export and batch sampling utilities."""

from __future__ import annotations

import collections
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from .anchor_registry import AnchorTaskSpec, get_anchor_task_spec
from .data.preprocessing import validate_d4rl_dataset


def _load_raw_dataset_from_hdf5(dataset_path: str | Path) -> dict[str, np.ndarray]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    dataset: dict[str, np.ndarray] = {}
    with h5py.File(dataset_path, "r") as handle:
        for key in (
            "observations",
            "actions",
            "rewards",
            "terminals",
            "timeouts",
            "next_observations",
        ):
            if key in handle:
                dataset[key] = handle[key][()]
    return dataset


def _load_raw_dataset_from_env(env_name: str) -> Mapping[str, Any]:
    import d4rl  # noqa: F401
    import gym

    env = gym.make(env_name)
    try:
        return env.get_dataset()
    finally:
        if hasattr(env, "close"):
            env.close()


def load_raw_anchor_dataset(
    env_name: str,
    *,
    dataset_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Load one raw D4RL dataset for anchor-mode processing."""

    if dataset_path is not None:
        return _load_raw_dataset_from_hdf5(dataset_path)
    return _load_raw_dataset_from_env(env_name)


@dataclass(frozen=True)
class AnchorTrajectoryDataset:
    """Trajectory-level dataset statistics used by the original QT trainer."""

    env_name: str
    trajectories: tuple[dict[str, np.ndarray], ...]
    state_mean: np.ndarray
    state_std: np.ndarray
    traj_lens: np.ndarray
    returns: np.ndarray
    sorted_inds: np.ndarray
    p_sample: np.ndarray
    num_trajectories: int
    state_dim: int
    action_dim: int
    max_ep_len: int


def discount_cumsum(values: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted cumulative sums exactly as the original repo does."""

    values = np.asarray(values, dtype=np.float32)
    discounted = np.zeros_like(values)
    discounted[-1] = values[-1]
    for index in reversed(range(values.shape[0] - 1)):
        discounted[index] = values[index] + gamma * discounted[index + 1]
    return discounted


def export_anchor_trajectories(
    env_name: str,
    *,
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, np.ndarray]]:
    """Export repo-compatible trajectory pickles from raw D4RL transitions."""

    task_spec = get_anchor_task_spec(env_name)
    raw_source = load_raw_anchor_dataset(env_name, dataset_path=dataset_path)
    dataset = validate_d4rl_dataset(raw_source)

    transition_count = int(dataset["rewards"].shape[0])
    episode_step = 0
    use_timeouts = "timeouts" in raw_source
    buckets: collections.defaultdict[str, list[np.ndarray]] = collections.defaultdict(list)
    paths: list[dict[str, np.ndarray]] = []

    for index in range(transition_count):
        done_bool = bool(dataset["terminals"][index])
        if use_timeouts:
            final_timestep = bool(np.asarray(raw_source["timeouts"])[index])
        else:
            final_timestep = episode_step == task_spec.max_ep_len - 1

        for key in ("observations", "actions", "rewards", "terminals"):
            buckets[key].append(dataset[key][index])

        if done_bool or final_timestep:
            episode = {
                key: np.asarray(values, dtype=np.float32 if key != "terminals" else np.bool_)
                for key, values in buckets.items()
            }
            paths.append(episode)
            buckets = collections.defaultdict(list)
            episode_step = 0
        else:
            episode_step += 1

    if buckets:
        episode = {
            key: np.asarray(values, dtype=np.float32 if key != "terminals" else np.bool_)
            for key, values in buckets.items()
        }
        paths.append(episode)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(paths, handle)

    return paths


def load_or_create_anchor_trajectories(
    env_name: str,
    *,
    dataset_path: str | Path | None = None,
    export_path: str | Path | None = None,
    save_export: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Load an existing trajectory pickle or export one from raw D4RL data."""

    path = Path(export_path) if export_path is not None else get_anchor_task_spec(env_name).trajectory_export_path
    if path.exists():
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        return [
            {
                "observations": np.asarray(traj["observations"], dtype=np.float32),
                "actions": np.asarray(traj["actions"], dtype=np.float32),
                "rewards": np.asarray(traj["rewards"], dtype=np.float32),
                "terminals": np.asarray(traj["terminals"], dtype=np.bool_),
            }
            for traj in loaded
        ]

    return export_anchor_trajectories(
        env_name,
        dataset_path=dataset_path,
        output_path=path if save_export else None,
    )


def build_anchor_trajectory_dataset(
    env_name: str,
    *,
    dataset_path: str | Path | None = None,
    export_path: str | Path | None = None,
    save_export: bool = False,
    pct_traj: float = 1.0,
) -> AnchorTrajectoryDataset:
    """Build trajectory-level stats and sampling weights for anchor-mode QT."""

    task_spec = get_anchor_task_spec(env_name)
    trajectories = load_or_create_anchor_trajectories(
        env_name,
        dataset_path=dataset_path,
        export_path=export_path,
        save_export=save_export,
    )

    if not trajectories:
        raise ValueError("Anchor trajectory export produced no trajectories.")

    states = [traj["observations"] for traj in trajectories]
    traj_lens = np.asarray([traj["observations"].shape[0] for traj in trajectories], dtype=np.int64)
    returns = np.asarray([traj["rewards"].sum() for traj in trajectories], dtype=np.float32)
    state_array = np.concatenate(states, axis=0).astype(np.float32)
    state_mean = state_array.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_std = np.maximum(
        state_array.std(axis=0, dtype=np.float64).astype(np.float32),
        1e-6,
    ).astype(np.float32)

    total_timesteps = max(int(traj_lens.sum() * float(pct_traj)), 1)
    sorted_inds = np.argsort(returns)
    timesteps = int(traj_lens[sorted_inds[-1]])
    num_trajectories = 1
    index = len(trajectories) - 2
    while index >= 0 and timesteps + int(traj_lens[sorted_inds[index]]) <= total_timesteps:
        timesteps += int(traj_lens[sorted_inds[index]])
        num_trajectories += 1
        index -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    p_sample = traj_lens[sorted_inds].astype(np.float64)
    p_sample = (p_sample / p_sample.sum()).astype(np.float64)

    return AnchorTrajectoryDataset(
        env_name=env_name,
        trajectories=tuple(trajectories),
        state_mean=state_mean,
        state_std=state_std,
        traj_lens=traj_lens,
        returns=returns,
        sorted_inds=sorted_inds.astype(np.int64),
        p_sample=p_sample,
        num_trajectories=int(num_trajectories),
        state_dim=int(state_array.shape[1]),
        action_dim=int(np.asarray(trajectories[0]["actions"]).shape[1]),
        max_ep_len=task_spec.max_ep_len,
    )


class AnchorBatchSampler:
    """Original-QT compatible episodic batch sampler."""

    def __init__(
        self,
        *,
        dataset: AnchorTrajectoryDataset,
        task_spec: AnchorTaskSpec,
        context_length: int,
        batch_size: int,
        device: torch.device,
        scale: float,
        reward_tune: str = "no",
    ) -> None:
        self.dataset = dataset
        self.task_spec = task_spec
        self.context_length = int(context_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.scale = float(scale)
        self.reward_tune = str(reward_tune)

    def _trajectory_start_index(self, traj: Mapping[str, np.ndarray]) -> int:
        traj_len = int(traj["rewards"].shape[0])
        if "hopper-medium" in self.task_spec.env_name and traj_len > self.context_length + 1:
            upper = traj_len - self.context_length - 1
        else:
            upper = traj_len - 1
        return random.randint(0, max(upper, 0))

    def _trajectory_rewards(self, traj: Mapping[str, np.ndarray]) -> np.ndarray:
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        if self.reward_tune == "cql_antmaze":
            return (rewards - 0.5) * 4.0
        return rewards

    def sample_batch(
        self,
        batch_size: int | None = None,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Sample one anchor-compatible batch on the configured device."""

        batch_size = int(batch_size or self.batch_size)
        max_len = int(max_len or self.context_length)

        batch_inds = np.random.choice(
            np.arange(self.dataset.num_trajectories),
            size=batch_size,
            replace=True,
            p=self.dataset.p_sample,
        )

        state_batch: list[np.ndarray] = []
        action_batch: list[np.ndarray] = []
        reward_batch: list[np.ndarray] = []
        target_action_batch: list[np.ndarray] = []
        done_batch: list[np.ndarray] = []
        rtg_batch: list[np.ndarray] = []
        timestep_batch: list[np.ndarray] = []
        mask_batch: list[np.ndarray] = []

        for batch_index in range(batch_size):
            traj = self.dataset.trajectories[int(self.dataset.sorted_inds[batch_inds[batch_index]])]
            start_index = self._trajectory_start_index(traj)
            states = np.asarray(
                traj["observations"][start_index : start_index + max_len],
                dtype=np.float32,
            ).reshape(1, -1, self.dataset.state_dim)
            actions = np.asarray(
                traj["actions"][start_index : start_index + max_len],
                dtype=np.float32,
            ).reshape(1, -1, self.dataset.action_dim)
            rewards = self._trajectory_rewards(traj)[start_index : start_index + max_len].reshape(1, -1, 1)
            terminals = np.asarray(
                traj["terminals"][start_index : start_index + max_len],
                dtype=np.float32,
            ).reshape(1, -1, 1)
            timesteps = np.arange(start_index, start_index + states.shape[1], dtype=np.int64).reshape(1, -1)
            timesteps[timesteps >= self.dataset.max_ep_len] = self.dataset.max_ep_len - 1
            rtg = discount_cumsum(self._trajectory_rewards(traj)[start_index:], gamma=1.0)[
                : states.shape[1] + 1
            ].reshape(1, -1, 1)
            if rtg.shape[1] <= states.shape[1]:
                rtg = np.concatenate([rtg, np.zeros((1, 1, 1), dtype=np.float32)], axis=1)

            seq_len = states.shape[1]
            pad = max_len - seq_len
            state_batch.append(
                np.concatenate([np.zeros((1, pad, self.dataset.state_dim), dtype=np.float32), states], axis=1)
            )
            action_batch.append(
                np.concatenate([np.zeros((1, pad, self.dataset.action_dim), dtype=np.float32), actions], axis=1)
            )
            reward_batch.append(np.concatenate([np.zeros((1, pad, 1), dtype=np.float32), rewards], axis=1))
            target_action_batch.append(
                np.concatenate([np.zeros((1, pad, self.dataset.action_dim), dtype=np.float32), actions], axis=1)
            )
            done_batch.append(np.concatenate([np.ones((1, pad, 1), dtype=np.float32), terminals], axis=1))
            rtg_batch.append(
                np.concatenate([np.zeros((1, pad, 1), dtype=np.float32), rtg], axis=1) / self.scale
            )
            timestep_batch.append(np.concatenate([np.zeros((1, pad), dtype=np.int64), timesteps], axis=1))
            mask_batch.append(np.concatenate([np.zeros((1, pad), dtype=np.float32), np.ones((1, seq_len), dtype=np.float32)], axis=1))

        states_tensor = torch.from_numpy(np.concatenate(state_batch, axis=0)).to(dtype=torch.float32, device=self.device)
        actions_tensor = torch.from_numpy(np.concatenate(action_batch, axis=0)).to(dtype=torch.float32, device=self.device)
        rewards_tensor = torch.from_numpy(np.concatenate(reward_batch, axis=0)).to(dtype=torch.float32, device=self.device)
        target_actions_tensor = torch.from_numpy(np.concatenate(target_action_batch, axis=0)).to(dtype=torch.float32, device=self.device)
        dones_tensor = torch.from_numpy(np.concatenate(done_batch, axis=0)).to(dtype=torch.float32, device=self.device)
        rtg_tensor = torch.from_numpy(np.concatenate(rtg_batch, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps_tensor = torch.from_numpy(np.concatenate(timestep_batch, axis=0)).to(dtype=torch.long, device=self.device)
        mask_tensor = torch.from_numpy(np.concatenate(mask_batch, axis=0)).to(device=self.device)

        states_tensor = (states_tensor - torch.from_numpy(self.dataset.state_mean).to(self.device)) / torch.from_numpy(
            self.dataset.state_std
        ).to(self.device)
        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            target_actions_tensor,
            dones_tensor,
            rtg_tensor,
            timesteps_tensor,
            mask_tensor,
        )


def resolve_anchor_eval_targets(
    task_spec: AnchorTaskSpec,
    *,
    target_return: float | None = None,
    test_scale: float | None = None,
) -> list[float]:
    """Resolve the candidate RTG list used during anchor-mode evaluation."""

    if target_return is not None:
        scale = float(test_scale if test_scale is not None else task_spec.test_scale)
        return [float(target_return) / scale]
    scale = float(test_scale if test_scale is not None else task_spec.test_scale)
    return [float(value) / scale for value in task_spec.env_targets]
