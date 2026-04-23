"""Sequence window sampling for QT training batches."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import NormalizationStats
from .preprocessing import PreprocessedDataset
from .registry import TaskSpec


WINDOW_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "returns_to_go",
    "next_observations",
    "terminals",
    "timeouts",
    "bootstrap_mask",
    "timesteps",
)


class QTSequenceDataset(Dataset):
    """PyTorch dataset that returns left-padded fixed-length QT windows."""

    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        stats: NormalizationStats,
        preprocessed: PreprocessedDataset,
        context_length: int,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive.")

        self.task_spec = task_spec
        self.stats = stats
        self.context_length = int(context_length)

        self.observations = stats.normalize_observations(preprocessed.observations)
        self.next_observations = stats.normalize_observations(preprocessed.next_observations)
        self.actions = preprocessed.actions.astype(np.float32, copy=False)
        self.rewards = preprocessed.rewards.astype(np.float32, copy=False)
        self.returns_to_go = preprocessed.returns_to_go.astype(np.float32, copy=False)
        self.terminals = preprocessed.terminals.astype(np.bool_, copy=False)
        self.timeouts = preprocessed.timeouts.astype(np.bool_, copy=False)
        self.bootstrap_mask = preprocessed.bootstrap_mask.astype(np.float32, copy=False)
        self.timesteps = preprocessed.timesteps.astype(np.int64, copy=False)
        self.episode_ids = preprocessed.episode_ids.astype(np.int64, copy=False)
        self.episode_start_indices = preprocessed.episode_start_indices.astype(np.int64, copy=False)
        self.episode_end_indices = preprocessed.episode_end_indices.astype(np.int64, copy=False)
        self.valid_sample_indices = preprocessed.valid_sample_indices.astype(np.int64, copy=False)

        self.metadata = {
            "discount": float(preprocessed.metadata["discount"]),
            "source_has_next_observations": bool(preprocessed.metadata["source_has_next_observations"]),
            "forced_episode_end_mask": np.asarray(
                preprocessed.metadata["forced_episode_end_mask"],
                dtype=np.bool_,
            ),
            "transition_count": int(preprocessed.metadata["transition_count"]),
            "episode_count": int(preprocessed.metadata["episode_count"]),
            "terminal_count": int(preprocessed.metadata["terminal_count"]),
            "timeout_count": int(preprocessed.metadata["timeout_count"]),
            "episode_start_indices": self.episode_start_indices,
            "episode_end_indices": self.episode_end_indices,
            "valid_sample_indices": self.valid_sample_indices,
        }

    def __len__(self) -> int:
        return int(self.valid_sample_indices.shape[0])

    def _resolve_window(self, index: int) -> tuple[int, int, slice, slice]:
        """Resolve the source and target slices for one sampled transition."""

        transition_index = int(self.valid_sample_indices[index])
        episode_id = int(self.episode_ids[transition_index])
        episode_start = int(self.episode_start_indices[episode_id])
        sequence_start = max(episode_start, transition_index - self.context_length + 1)
        sequence_stop = transition_index + 1
        pad_length = self.context_length - (sequence_stop - sequence_start)
        return transition_index, episode_id, slice(sequence_start, sequence_stop), slice(
            pad_length,
            self.context_length,
        )

    def _allocate_window(self) -> dict[str, np.ndarray]:
        """Allocate zero-padded arrays for one sequence window."""

        observation_shape = (self.context_length, self.task_spec.observation_dim)
        action_shape = (self.context_length, self.task_spec.action_dim)

        return {
            "observations": np.zeros(observation_shape, dtype=np.float32),
            "actions": np.zeros(action_shape, dtype=np.float32),
            "rewards": np.zeros(self.context_length, dtype=np.float32),
            "returns_to_go": np.zeros(self.context_length, dtype=np.float32),
            "next_observations": np.zeros(observation_shape, dtype=np.float32),
            "terminals": np.zeros(self.context_length, dtype=np.bool_),
            "timeouts": np.zeros(self.context_length, dtype=np.bool_),
            "bootstrap_mask": np.zeros(self.context_length, dtype=np.float32),
            "timesteps": np.zeros(self.context_length, dtype=np.int64),
            "attention_mask": np.zeros(self.context_length, dtype=np.float32),
        }

    def _populate_window(
        self,
        window: dict[str, np.ndarray],
        source_slice: slice,
        target_slice: slice,
    ) -> None:
        """Copy one sequence window from stored arrays into padded buffers."""

        for field_name in WINDOW_FIELDS:
            window[field_name][target_slice] = getattr(self, field_name)[source_slice]
        window["attention_mask"][target_slice] = 1.0

    def _to_torch_sample(
        self,
        window: dict[str, np.ndarray],
        episode_id: int,
    ) -> dict[str, torch.Tensor | str]:
        """Convert one padded window into the public sample payload."""

        return {
            "observations": torch.from_numpy(window["observations"]),
            "actions": torch.from_numpy(window["actions"]),
            "rewards": torch.from_numpy(window["rewards"]),
            "returns_to_go": torch.from_numpy(window["returns_to_go"]),
            "next_observations": torch.from_numpy(window["next_observations"]),
            "terminals": torch.from_numpy(window["terminals"]),
            "timeouts": torch.from_numpy(window["timeouts"]),
            "bootstrap_mask": torch.from_numpy(window["bootstrap_mask"]),
            "timesteps": torch.from_numpy(window["timesteps"]),
            "attention_mask": torch.from_numpy(window["attention_mask"]),
            "episode_id": torch.tensor(episode_id, dtype=torch.long),
            "env_name": self.task_spec.env_name,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        _, episode_id, source_slice, target_slice = self._resolve_window(index)
        window = self._allocate_window()
        self._populate_window(window, source_slice, target_slice)
        return self._to_torch_sample(window, episode_id)
