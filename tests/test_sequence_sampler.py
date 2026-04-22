from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.data.normalization import NormalizationStats
from tranqil.data.preprocessing import preprocess_d4rl_dataset
from tranqil.data.registry import TaskSpec
from tranqil.data.sequence_sampler import QTSequenceDataset


def make_synthetic_dataset() -> dict[str, np.ndarray]:
    return {
        "observations": np.array([[10.0], [11.0], [20.0], [21.0], [30.0]], dtype=np.float32),
        "actions": np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32),
        "rewards": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "terminals": np.array([False, True, False, False, False], dtype=np.bool_),
        "timeouts": np.array([False, False, False, True, False], dtype=np.bool_),
    }


def make_identity_stats() -> NormalizationStats:
    return NormalizationStats(
        obs_mean=np.array([0.0], dtype=np.float32),
        obs_std=np.array([1.0], dtype=np.float32),
        reward_mean=3.0,
        reward_std=1.4142135,
        reward_min=1.0,
        reward_max=5.0,
        return_mean=5.0,
        return_std=1.6329932,
        return_min=3.0,
        return_max=7.0,
        episode_count=3,
        transition_count=5,
    )


class SequenceSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        preprocessed = preprocess_d4rl_dataset(make_synthetic_dataset(), discount=0.99)
        task_spec = TaskSpec(
            env_name="unit-test-env",
            observation_dim=1,
            action_dim=1,
            dataset_alias="unit-test.hdf5",
        )
        self.dataset = QTSequenceDataset(
            task_spec=task_spec,
            stats=make_identity_stats(),
            preprocessed=preprocessed,
            context_length=3,
        )

    def test_dataset_length_matches_transition_count(self) -> None:
        self.assertEqual(len(self.dataset), 5)

    def test_left_padding_does_not_cross_episode_boundary(self) -> None:
        sample = self.dataset[3]

        self.assertEqual(sample["env_name"], "unit-test-env")
        self.assertEqual(int(sample["episode_id"]), 1)
        self.assertEqual(sample["observations"].squeeze(-1).tolist(), [0.0, 20.0, 21.0])
        self.assertEqual(sample["actions"].squeeze(-1).tolist(), [0.0, 0.30000001192092896, 0.4000000059604645])
        self.assertEqual(sample["timesteps"].tolist(), [0, 0, 1])
        self.assertEqual(sample["attention_mask"].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(sample["timeouts"].tolist(), [False, False, True])
        self.assertEqual(sample["bootstrap_mask"].tolist(), [0.0, 1.0, 0.0])

    def test_first_step_window_is_left_padded(self) -> None:
        sample = self.dataset[2]

        self.assertEqual(sample["observations"].squeeze(-1).tolist(), [0.0, 0.0, 20.0])
        self.assertEqual(sample["attention_mask"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(sample["timesteps"].tolist(), [0, 0, 0])
        self.assertEqual(sample["returns_to_go"].tolist(), [0.0, 0.0, 7.0])
        self.assertEqual(sample["terminals"].tolist(), [False, False, False])


if __name__ == "__main__":
    unittest.main()
