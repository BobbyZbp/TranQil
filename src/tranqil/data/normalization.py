"""Observation normalization utilities for QT datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


OBS_STD_EPSILON = 1e-6


def _scalar_value(value: np.ndarray | float | int) -> float | int:
    """Convert numpy scalars and length-1 arrays to plain Python values."""

    if isinstance(value, np.ndarray):
        return value.item()
    return value


@dataclass(frozen=True)
class NormalizationStats:
    """Normalization tensors and dataset-level summary statistics."""

    obs_mean: np.ndarray
    obs_std: np.ndarray
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    return_mean: float
    return_std: float
    return_min: float
    return_max: float
    episode_count: int
    transition_count: int
    obs_std_epsilon: float = OBS_STD_EPSILON

    def normalize_observations(self, observations: np.ndarray) -> np.ndarray:
        """Apply z-score observation normalization."""

        normalized = (np.asarray(observations, dtype=np.float32) - self.obs_mean) / self.obs_std
        return normalized.astype(np.float32, copy=False)

    def to_cache_dict(self) -> dict[str, np.ndarray]:
        """Serialize stats into cache-friendly numpy values."""

        return {
            "obs_mean": self.obs_mean.astype(np.float32, copy=False),
            "obs_std": self.obs_std.astype(np.float32, copy=False),
            "reward_mean": np.array(self.reward_mean, dtype=np.float32),
            "reward_std": np.array(self.reward_std, dtype=np.float32),
            "reward_min": np.array(self.reward_min, dtype=np.float32),
            "reward_max": np.array(self.reward_max, dtype=np.float32),
            "return_mean": np.array(self.return_mean, dtype=np.float32),
            "return_std": np.array(self.return_std, dtype=np.float32),
            "return_min": np.array(self.return_min, dtype=np.float32),
            "return_max": np.array(self.return_max, dtype=np.float32),
            "episode_count": np.array(self.episode_count, dtype=np.int64),
            "transition_count": np.array(self.transition_count, dtype=np.int64),
            "obs_std_epsilon": np.array(self.obs_std_epsilon, dtype=np.float32),
        }

    @classmethod
    def from_cache_dict(cls, cache: Mapping[str, np.ndarray]) -> "NormalizationStats":
        """Deserialize stats from an `.npz` cache payload."""

        return cls(
            obs_mean=np.asarray(cache["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(cache["obs_std"], dtype=np.float32),
            reward_mean=float(_scalar_value(cache["reward_mean"])),
            reward_std=float(_scalar_value(cache["reward_std"])),
            reward_min=float(_scalar_value(cache["reward_min"])),
            reward_max=float(_scalar_value(cache["reward_max"])),
            return_mean=float(_scalar_value(cache["return_mean"])),
            return_std=float(_scalar_value(cache["return_std"])),
            return_min=float(_scalar_value(cache["return_min"])),
            return_max=float(_scalar_value(cache["return_max"])),
            episode_count=int(_scalar_value(cache["episode_count"])),
            transition_count=int(_scalar_value(cache["transition_count"])),
            obs_std_epsilon=float(_scalar_value(cache["obs_std_epsilon"])),
        )


def compute_normalization_stats(
    observations: np.ndarray,
    rewards: np.ndarray,
    episode_returns: np.ndarray,
    epsilon: float = OBS_STD_EPSILON,
) -> NormalizationStats:
    """Compute observation normalization tensors and dataset summaries."""

    observations = np.asarray(observations, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    episode_returns = np.asarray(episode_returns, dtype=np.float32)

    if observations.ndim != 2:
        raise ValueError("observations must be a rank-2 array.")
    if rewards.ndim != 1:
        raise ValueError("rewards must be a rank-1 array.")
    if episode_returns.ndim != 1:
        raise ValueError("episode_returns must be a rank-1 array.")
    if observations.shape[0] != rewards.shape[0]:
        raise ValueError("observations and rewards must have the same number of transitions.")
    if episode_returns.size == 0:
        raise ValueError("episode_returns must contain at least one episode.")

    obs_mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    obs_std = observations.std(axis=0, dtype=np.float64).astype(np.float32)
    obs_std = np.maximum(obs_std, epsilon).astype(np.float32)

    return NormalizationStats(
        obs_mean=obs_mean,
        obs_std=obs_std,
        reward_mean=float(rewards.mean(dtype=np.float64)),
        reward_std=float(rewards.std(dtype=np.float64)),
        reward_min=float(rewards.min()),
        reward_max=float(rewards.max()),
        return_mean=float(episode_returns.mean(dtype=np.float64)),
        return_std=float(episode_returns.std(dtype=np.float64)),
        return_min=float(episode_returns.min()),
        return_max=float(episode_returns.max()),
        episode_count=int(episode_returns.shape[0]),
        transition_count=int(observations.shape[0]),
        obs_std_epsilon=float(epsilon),
    )
