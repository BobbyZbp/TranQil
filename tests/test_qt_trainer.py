from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.checkpoints import load_checkpoint
from tranqil.evaluation import restore_actor_from_checkpoint
from tranqil.trainer import QTTrainer


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


class QTTrainerTests(unittest.TestCase):
    def test_trainer_runs_finite_steps_and_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "walker_fixture.hdf5"
            write_hdf5_fixture(dataset_path, make_walker_fixture_dataset())

            config = {
                "run_name": "qt_unit_test",
                "seed": 0,
                "device": "cpu",
                "env": {
                    "name": "walker2d-medium-replay-v2",
                },
                "data": {
                    "context_length": 3,
                    "discount": 0.99,
                    "batch_size": 2,
                    "cache_dir": str(tmp_path / "cache"),
                    "dataset_path": str(dataset_path),
                    "num_workers": 0,
                    "pin_memory": False,
                    "drop_last": False,
                    "shuffle": False,
                },
                "model": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dropout": 0.0,
                    "ffn_multiplier": 2,
                    "max_timestep": 32,
                },
                "critic": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                },
                "optimization": {
                    "actor_lr": 1e-3,
                    "critic_lr": 1e-3,
                    "weight_decay": 0.0,
                },
                "training": {
                    "steps": 2,
                    "gamma": 0.99,
                    "eta": 0.5,
                    "grad_clip_norm": 1.0,
                    "target_tau": 0.1,
                    "actor_update_every": 1,
                    "log_interval": 1,
                    "checkpoint_interval": 2,
                    "eval_interval": 0,
                },
                "evaluation": {
                    "episodes": 1,
                    "max_steps": 5,
                    "target_return": 0.0,
                },
                "outputs": {
                    "base_dir": str(tmp_path / "results"),
                },
            }

            summary = QTTrainer(config).train()
            self.assertEqual(summary["final_step"], 2)
            self.assertIsNotNone(summary["checkpoint_path"])
            self.assertTrue(Path(summary["checkpoint_path"]).exists())
            self.assertTrue(Path(summary["metrics_path"]).exists())
            self.assertTrue(Path(summary["run_dir"]).joinpath("config_resolved.yaml").exists())

            latest_metrics = summary["latest_train_metrics"]
            for metric_name in ("critic_loss", "actor_loss", "bc_loss", "policy_q_mean"):
                self.assertIn(metric_name, latest_metrics)
                self.assertTrue(np.isfinite(latest_metrics[metric_name]))

            checkpoint = load_checkpoint(summary["checkpoint_path"])
            self.assertEqual(int(checkpoint["step"]), 2)

            actor, task_spec, stats, checkpoint_config = restore_actor_from_checkpoint(
                summary["checkpoint_path"],
                device="cpu",
            )
            self.assertEqual(task_spec.env_name, "walker2d-medium-replay-v2")
            self.assertEqual(task_spec.observation_dim, 17)
            self.assertEqual(stats.transition_count, 5)
            self.assertEqual(checkpoint_config["run_name"], "qt_unit_test")
            self.assertTrue(any(parameter.numel() > 0 for parameter in actor.parameters()))


if __name__ == "__main__":
    unittest.main()
