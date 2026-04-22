from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.anchor_data import (
    AnchorBatchSampler,
    build_anchor_trajectory_dataset,
    export_anchor_trajectories,
)
from tranqil.anchor_registry import get_anchor_task_spec


def make_walker_fixture_dataset() -> dict[str, np.ndarray]:
    return {
        "observations": np.array(
            [
                np.arange(17, dtype=np.float32),
                np.arange(17, dtype=np.float32) + 1,
                np.arange(17, dtype=np.float32) + 10,
                np.arange(17, dtype=np.float32) + 11,
                np.arange(17, dtype=np.float32) + 20,
            ],
            dtype=np.float32,
        ),
        "actions": np.array(
            [
                np.arange(6, dtype=np.float32) * 0.1,
                np.arange(6, dtype=np.float32) * 0.2,
                np.arange(6, dtype=np.float32) * 0.3,
                np.arange(6, dtype=np.float32) * 0.4,
                np.arange(6, dtype=np.float32) * 0.5,
            ],
            dtype=np.float32,
        ),
        "rewards": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "terminals": np.array([False, True, False, False, False], dtype=np.bool_),
        "timeouts": np.array([False, False, False, True, False], dtype=np.bool_),
    }


def write_hdf5_fixture(path: Path, dataset: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        for key, value in dataset.items():
            handle.create_dataset(key, data=value)


class AnchorDataTests(unittest.TestCase):
    def test_export_and_sample_anchor_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "walker_fixture.hdf5"
            export_path = tmp_path / "walker_fixture.pkl"
            write_hdf5_fixture(dataset_path, make_walker_fixture_dataset())

            paths = export_anchor_trajectories(
                "walker2d-medium-replay-v2",
                dataset_path=dataset_path,
                output_path=export_path,
            )
            self.assertEqual(len(paths), 3)
            self.assertTrue(export_path.exists())

            trajectory_dataset = build_anchor_trajectory_dataset(
                "walker2d-medium-replay-v2",
                dataset_path=dataset_path,
                export_path=export_path,
                save_export=False,
                pct_traj=1.0,
            )
            self.assertEqual(trajectory_dataset.num_trajectories, 3)
            self.assertEqual(tuple(trajectory_dataset.state_mean.shape), (17,))

            sampler = AnchorBatchSampler(
                dataset=trajectory_dataset,
                task_spec=get_anchor_task_spec("walker2d-medium-replay-v2"),
                context_length=3,
                batch_size=2,
                device=torch.device("cpu"),
                scale=1000.0,
            )
            states, actions, rewards, target_actions, dones, rtg, timesteps, mask = sampler.sample_batch()

            self.assertEqual(tuple(states.shape), (2, 3, 17))
            self.assertEqual(tuple(actions.shape), (2, 3, 6))
            self.assertEqual(tuple(rewards.shape), (2, 3, 1))
            self.assertEqual(tuple(target_actions.shape), (2, 3, 6))
            self.assertEqual(tuple(dones.shape), (2, 3, 1))
            self.assertEqual(tuple(rtg.shape), (2, 4, 1))
            self.assertEqual(tuple(timesteps.shape), (2, 3))
            self.assertEqual(tuple(mask.shape), (2, 3))
            self.assertTrue(torch.isfinite(states).all())
            self.assertTrue(torch.isfinite(rtg).all())
