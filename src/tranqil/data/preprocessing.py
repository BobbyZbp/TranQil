"""Dataset preprocessing for QT actor and critic inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


REQUIRED_DATASET_KEYS = ("observations", "actions", "rewards", "terminals")


@dataclass
class PreprocessedDataset:
    """Transition-level arrays derived from a raw D4RL dataset."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    next_observations: np.ndarray
    returns_to_go: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    bootstrap_mask: np.ndarray
    episode_returns: np.ndarray
    episode_start_indices: np.ndarray
    episode_end_indices: np.ndarray
    valid_sample_indices: np.ndarray
    metadata: dict[str, Any]

    @property
    def transition_count(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def episode_count(self) -> int:
        return int(self.episode_returns.shape[0])


def validate_d4rl_dataset(raw_dataset: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate and coerce a raw dataset mapping into numpy arrays."""

    missing = [key for key in REQUIRED_DATASET_KEYS if key not in raw_dataset]
    if missing:
        formatted = ", ".join(missing)
        raise KeyError(f"Dataset is missing required keys: {formatted}.")

    observations = np.asarray(raw_dataset["observations"], dtype=np.float32)
    actions = np.asarray(raw_dataset["actions"], dtype=np.float32)
    rewards = np.asarray(raw_dataset["rewards"], dtype=np.float32)
    terminals = np.asarray(raw_dataset["terminals"], dtype=np.bool_)
    timeouts = np.asarray(raw_dataset.get("timeouts", np.zeros_like(terminals)), dtype=np.bool_)

    if observations.ndim != 2:
        raise ValueError("observations must be a rank-2 array.")
    if actions.ndim != 2:
        raise ValueError("actions must be a rank-2 array.")
    if rewards.ndim != 1:
        raise ValueError("rewards must be a rank-1 array.")
    if terminals.ndim != 1:
        raise ValueError("terminals must be a rank-1 array.")
    if timeouts.ndim != 1:
        raise ValueError("timeouts must be a rank-1 array.")

    transition_count = rewards.shape[0]
    if transition_count == 0:
        raise ValueError("Dataset must contain at least one transition.")

    if observations.shape[0] != transition_count:
        raise ValueError("observations and rewards must have the same number of transitions.")
    if actions.shape[0] != transition_count:
        raise ValueError("actions and rewards must have the same number of transitions.")
    if terminals.shape[0] != transition_count:
        raise ValueError("terminals and rewards must have the same number of transitions.")
    if timeouts.shape[0] != transition_count:
        raise ValueError("timeouts and rewards must have the same number of transitions.")

    dataset = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }

    if "next_observations" in raw_dataset:
        next_observations = np.asarray(raw_dataset["next_observations"], dtype=np.float32)
        if next_observations.shape != observations.shape:
            raise ValueError("next_observations must have the same shape as observations.")
        dataset["next_observations"] = next_observations

    return dataset


def build_episode_boundary_mask(
    terminals: np.ndarray,
    timeouts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the final episode-boundary mask and track forced terminal markers."""

    terminals = np.asarray(terminals, dtype=np.bool_)
    timeouts = np.asarray(timeouts, dtype=np.bool_)
    transition_count = terminals.shape[0]
    boundary_mask = terminals | timeouts

    forced_episode_end_mask = np.zeros(transition_count, dtype=np.bool_)
    if not boundary_mask[-1]:
        boundary_mask = boundary_mask.copy()
        boundary_mask[-1] = True
        forced_episode_end_mask[-1] = True

    return boundary_mask, forced_episode_end_mask


def _episode_index_bounds(boundary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a boundary mask into half-open episode start/end indices."""

    boundary_mask = np.asarray(boundary_mask, dtype=np.bool_)
    episode_end_positions = np.flatnonzero(boundary_mask)
    episode_end_indices = (episode_end_positions + 1).astype(np.int64, copy=False)
    episode_start_indices = np.concatenate(
        [np.array([0], dtype=np.int64), episode_end_indices[:-1]]
    ).astype(np.int64, copy=False)
    return episode_start_indices, episode_end_indices


def compute_episode_boundaries(
    terminals: np.ndarray,
    timeouts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split transitions into episodes using terminal or timeout flags."""

    boundary_mask, forced_episode_end_mask = build_episode_boundary_mask(terminals, timeouts)
    episode_start_indices, episode_end_indices = _episode_index_bounds(boundary_mask)
    return episode_start_indices, episode_end_indices, forced_episode_end_mask


def build_next_observations(
    dataset: Mapping[str, np.ndarray],
    episode_start_indices: np.ndarray,
    episode_end_indices: np.ndarray,
) -> np.ndarray:
    """Return next observations without leaking across episode boundaries."""

    if "next_observations" in dataset:
        return np.asarray(dataset["next_observations"], dtype=np.float32).copy()

    observations = np.asarray(dataset["observations"], dtype=np.float32)
    next_observations = observations.copy()
    for start, end in zip(episode_start_indices, episode_end_indices):
        start_index = int(start)
        end_index = int(end)
        if end_index - start_index > 1:
            next_observations[start_index : end_index - 1] = observations[start_index + 1 : end_index]
        next_observations[end_index - 1] = observations[end_index - 1]
    return next_observations


def build_episode_features(
    rewards: np.ndarray,
    episode_start_indices: np.ndarray,
    episode_end_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute episode ids, timesteps, returns, and return-to-go tokens."""

    transition_count = rewards.shape[0]
    episode_count = episode_start_indices.shape[0]
    returns_to_go = np.zeros(transition_count, dtype=np.float32)
    episode_ids = np.zeros(transition_count, dtype=np.int64)
    timesteps = np.zeros(transition_count, dtype=np.int64)
    episode_returns = np.zeros(episode_count, dtype=np.float32)

    for episode_id, (start, end) in enumerate(zip(episode_start_indices, episode_end_indices)):
        start_index = int(start)
        end_index = int(end)
        episode_slice = slice(start_index, end_index)
        episode_rewards = rewards[episode_slice].astype(np.float64, copy=False)

        episode_returns[episode_id] = float(episode_rewards.sum())
        episode_ids[episode_slice] = episode_id
        timesteps[episode_slice] = np.arange(end_index - start_index, dtype=np.int64)
        # QT follows DT-style current-inclusive RTG tokens. The critic discount
        # is tracked in metadata for later training code instead of changing RTG semantics here.
        returns_to_go[episode_slice] = np.cumsum(episode_rewards[::-1], dtype=np.float64)[::-1].astype(
            np.float32
        )

    return returns_to_go, episode_ids, timesteps, episode_returns


def preprocess_d4rl_dataset(
    raw_dataset: Mapping[str, Any],
    discount: float = 0.99,
) -> PreprocessedDataset:
    """Preprocess a raw dataset into per-transition QT training arrays."""

    dataset = validate_d4rl_dataset(raw_dataset)
    if not 0.0 < float(discount) <= 1.0:
        raise ValueError("discount must be in the interval (0, 1].")

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]
    timeouts = dataset["timeouts"]

    boundary_mask, forced_episode_end_mask = build_episode_boundary_mask(
        terminals=terminals,
        timeouts=timeouts,
    )
    episode_start_indices, episode_end_indices = _episode_index_bounds(boundary_mask)
    next_observations = build_next_observations(
        dataset=dataset,
        episode_start_indices=episode_start_indices,
        episode_end_indices=episode_end_indices,
    )

    returns_to_go, episode_ids, timesteps, episode_returns = build_episode_features(
        rewards=rewards,
        episode_start_indices=episode_start_indices,
        episode_end_indices=episode_end_indices,
    )
    bootstrap_mask = (~boundary_mask).astype(np.float32, copy=False)
    valid_sample_indices = np.arange(rewards.shape[0], dtype=np.int64)

    metadata = {
        "discount": float(discount),
        "source_has_next_observations": "next_observations" in dataset,
        "forced_episode_end_mask": forced_episode_end_mask,
        "transition_count": int(rewards.shape[0]),
        "episode_count": int(episode_start_indices.shape[0]),
        "terminal_count": int(terminals.sum()),
        "timeout_count": int(timeouts.sum()),
    }

    return PreprocessedDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        next_observations=next_observations,
        returns_to_go=returns_to_go,
        episode_ids=episode_ids,
        timesteps=timesteps,
        bootstrap_mask=bootstrap_mask,
        episode_returns=episode_returns,
        episode_start_indices=episode_start_indices,
        episode_end_indices=episode_end_indices,
        valid_sample_indices=valid_sample_indices,
        metadata=metadata,
    )
