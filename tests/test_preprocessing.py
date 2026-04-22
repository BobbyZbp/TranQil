from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.data.preprocessing import preprocess_d4rl_dataset, validate_d4rl_dataset


def make_synthetic_dataset(include_timeouts: bool = True) -> dict[str, np.ndarray]:
    dataset = {
        "observations": np.array([[10.0], [11.0], [20.0], [21.0], [30.0]], dtype=np.float32),
        "actions": np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float32),
        "rewards": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "terminals": np.array([False, True, False, False, False], dtype=np.bool_),
    }
    if include_timeouts:
        dataset["timeouts"] = np.array([False, False, False, True, False], dtype=np.bool_)
    return dataset


class PreprocessingTests(unittest.TestCase):
    def test_validate_adds_zero_timeouts_when_missing(self) -> None:
        validated = validate_d4rl_dataset(make_synthetic_dataset(include_timeouts=False))
        self.assertIn("timeouts", validated)
        self.assertTrue(np.array_equal(validated["timeouts"], np.zeros(5, dtype=np.bool_)))

    def test_preprocess_splits_episodes_and_computes_rtg(self) -> None:
        processed = preprocess_d4rl_dataset(make_synthetic_dataset(), discount=0.99)

        self.assertTrue(np.array_equal(processed.episode_start_indices, np.array([0, 2, 4], dtype=np.int64)))
        self.assertTrue(np.array_equal(processed.episode_end_indices, np.array([2, 4, 5], dtype=np.int64)))
        self.assertTrue(np.array_equal(processed.episode_ids, np.array([0, 0, 1, 1, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(processed.timesteps, np.array([0, 1, 0, 1, 0], dtype=np.int64)))
        self.assertTrue(
            np.allclose(processed.returns_to_go, np.array([3.0, 2.0, 7.0, 4.0, 5.0], dtype=np.float32))
        )
        self.assertTrue(np.allclose(processed.episode_returns, np.array([3.0, 7.0, 5.0], dtype=np.float32)))
        self.assertTrue(
            np.allclose(processed.bootstrap_mask, np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        )
        self.assertTrue(np.array_equal(processed.valid_sample_indices, np.arange(5, dtype=np.int64)))

    def test_preprocess_derives_next_observations_without_episode_leakage(self) -> None:
        processed = preprocess_d4rl_dataset(make_synthetic_dataset(), discount=0.99)

        expected_next = np.array([[11.0], [11.0], [21.0], [21.0], [30.0]], dtype=np.float32)
        self.assertTrue(np.allclose(processed.next_observations, expected_next))
        self.assertEqual(int(processed.terminals.sum()), 1)
        self.assertEqual(int(processed.timeouts.sum()), 1)


if __name__ == "__main__":
    unittest.main()
