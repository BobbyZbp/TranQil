from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.anchor_trainer import AnchorQTTrainer
from tranqil.checkpoints import load_checkpoint
from tranqil.evaluation import load_evaluation_session


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


class AnchorTrainerTests(unittest.TestCase):
    def test_anchor_trainer_respects_iteration_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "walker_fixture.hdf5"
            export_path = tmp_path / "walker_fixture.pkl"
            write_hdf5_fixture(dataset_path, make_walker_fixture_dataset())

            config = {
                "run_name": "qt_anchor_unit_test",
                "seed": 0,
                "device": "cpu",
                "env": {"name": "walker2d-medium-replay-v2"},
                "data": {
                    "dataset_path": str(dataset_path),
                },
                "evaluation": {
                    "episodes": 1,
                    "max_steps": 5,
                    "target_return": None,
                },
                "outputs": {
                    "base_dir": str(tmp_path / "results"),
                },
                "anchor": {
                    "enabled": True,
                    "K": 3,
                    "pct_traj": 1.0,
                    "batch_size": 2,
                    "embed_dim": 32,
                    "n_layer": 2,
                    "n_head": 4,
                    "activation_function": "relu",
                    "dropout": 0.0,
                    "learning_rate": 1e-3,
                    "lr_min": 0.0,
                    "weight_decay": 0.0,
                    "num_eval_episodes": 1,
                    "max_iters": 1,
                    "num_steps_per_iter": 2,
                    "discount": 0.99,
                    "tau": 0.1,
                    "eta": 1.0,
                    "eta2": 1.0,
                    "lambda": 1.0,
                    "max_q_backup": False,
                    "lr_decay": False,
                    "grad_norm": 1.0,
                    "early_stop": False,
                    "early_epoch": 100,
                    "k_rewards": True,
                    "use_discount": True,
                    "sar": False,
                    "reward_tune": "no",
                    "scale": 1000.0,
                    "test_scale": 1000.0,
                    "env_targets": [5000.0, 4000.0, 2500.0],
                    "rtg_no_q": False,
                    "infer_no_q": False,
                    "ema_decay": 0.995,
                    "ema_start_step": 1000,
                    "ema_update_every": 5,
                    "trajectory_export_path": str(export_path),
                    "save_trajectory_pkl": True,
                },
            }

            fake_eval = {
                "env_name": "walker2d-medium-replay-v2",
                "episodes": 1,
                "seed": 0,
                "checkpoint_path": None,
                "checkpoint_step": 2,
                "target_return": 5000.0,
                "candidate_target_returns": [5000.0, 4000.0, 2500.0],
                "max_steps": 5,
                "mean_return": 123.0,
                "std_return": 0.0,
                "mean_episode_length": 5.0,
                "episode_returns": [123.0],
                "episode_lengths": [5],
                "normalized_scores": [95.0],
                "mean_normalized_score": 95.0,
                "std_normalized_score": 0.0,
            }

            with mock.patch("tranqil.anchor_trainer.evaluate_policy", return_value=fake_eval):
                summary = AnchorQTTrainer(config).train()

            self.assertEqual(summary["final_step"], 2)
            self.assertEqual(summary["final_iteration"], 1)
            self.assertTrue(Path(summary["checkpoint_path"]).exists())
            self.assertTrue(Path(summary["best_checkpoint_path"]).exists())
            self.assertTrue(Path(summary["metrics_path"]).exists())
            self.assertTrue(Path(summary["run_dir"]).joinpath("config_resolved.yaml").exists())

            checkpoint = load_checkpoint(summary["best_checkpoint_path"])
            self.assertEqual(int(checkpoint["step"]), 2)
            self.assertEqual(checkpoint["kind"], "anchor")

    def test_fresh_anchor_checkpoint_restores_through_evaluation_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "walker_fixture.hdf5"
            export_path = tmp_path / "walker_fixture.pkl"
            config_path = tmp_path / "anchor_config.yaml"
            write_hdf5_fixture(dataset_path, make_walker_fixture_dataset())

            config = {
                "run_name": "qt_anchor_restore_test",
                "seed": 0,
                "device": "cpu",
                "env": {"name": "walker2d-medium-replay-v2"},
                "data": {
                    "dataset_path": str(dataset_path),
                },
                "evaluation": {
                    "episodes": 1,
                    "max_steps": 5,
                    "target_return": None,
                },
                "outputs": {
                    "base_dir": str(tmp_path / "results"),
                },
                "anchor": {
                    "enabled": True,
                    "K": 3,
                    "pct_traj": 1.0,
                    "batch_size": 2,
                    "embed_dim": 32,
                    "n_layer": 2,
                    "n_head": 4,
                    "activation_function": "relu",
                    "dropout": 0.0,
                    "learning_rate": 1e-3,
                    "lr_min": 0.0,
                    "weight_decay": 0.0,
                    "num_eval_episodes": 1,
                    "max_iters": 1,
                    "num_steps_per_iter": 2,
                    "discount": 0.99,
                    "tau": 0.1,
                    "eta": 1.0,
                    "eta2": 1.0,
                    "lambda": 1.0,
                    "max_q_backup": False,
                    "lr_decay": False,
                    "grad_norm": 1.0,
                    "early_stop": False,
                    "early_epoch": 100,
                    "k_rewards": True,
                    "use_discount": True,
                    "sar": False,
                    "reward_tune": "no",
                    "scale": 1000.0,
                    "test_scale": 1000.0,
                    "env_targets": [5000.0, 4000.0, 2500.0],
                    "rtg_no_q": False,
                    "infer_no_q": False,
                    "ema_decay": 0.995,
                    "ema_start_step": 1000,
                    "ema_update_every": 5,
                    "trajectory_export_path": str(export_path),
                    "save_trajectory_pkl": True,
                },
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            fake_eval = {
                "env_name": "walker2d-medium-replay-v2",
                "episodes": 1,
                "seed": 0,
                "checkpoint_path": None,
                "checkpoint_step": 2,
                "target_return": 5000.0,
                "candidate_target_returns": [5000.0, 4000.0, 2500.0],
                "max_steps": 5,
                "mean_return": 123.0,
                "std_return": 0.0,
                "mean_episode_length": 5.0,
                "episode_returns": [123.0],
                "episode_lengths": [5],
                "normalized_scores": [95.0],
                "mean_normalized_score": 95.0,
                "std_normalized_score": 0.0,
            }

            with mock.patch("tranqil.anchor_trainer.evaluate_policy", return_value=fake_eval):
                summary = AnchorQTTrainer(config).train()

            session = load_evaluation_session(
                config_path=config_path,
                checkpoint_path=summary["best_checkpoint_path"],
                device="cpu",
                episodes=1,
                max_steps=5,
                seed=7,
            )

            self.assertEqual(session.mode, "anchor")
            self.assertIsNotNone(session.critic)
            self.assertEqual(session.task_spec.env_name, "walker2d-medium-replay-v2")
            self.assertEqual(session.config["seed"], 7)
            self.assertEqual(session.config["evaluation"]["episodes"], 1)
            self.assertEqual(session.config["evaluation"]["max_steps"], 5)
            self.assertEqual(session.checkpoint_step, 2)
            self.assertEqual(session.state_mean.shape[0], 17)
            self.assertEqual(session.state_std.shape[0], 17)

    def test_early_stop_matches_original_qt_loop_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "walker_fixture.hdf5"
            export_path = tmp_path / "walker_fixture.pkl"
            write_hdf5_fixture(dataset_path, make_walker_fixture_dataset())

            config = {
                "run_name": "qt_anchor_early_stop_test",
                "seed": 0,
                "device": "cpu",
                "env": {"name": "walker2d-medium-replay-v2"},
                "data": {"dataset_path": str(dataset_path)},
                "evaluation": {
                    "episodes": 1,
                    "max_steps": 5,
                    "target_return": None,
                },
                "outputs": {"base_dir": str(tmp_path / "results")},
                "anchor": {
                    "enabled": True,
                    "K": 3,
                    "pct_traj": 1.0,
                    "batch_size": 2,
                    "embed_dim": 32,
                    "n_layer": 2,
                    "n_head": 4,
                    "activation_function": "relu",
                    "dropout": 0.0,
                    "learning_rate": 1e-3,
                    "lr_min": 0.0,
                    "weight_decay": 0.0,
                    "num_eval_episodes": 1,
                    "max_iters": 5,
                    "num_steps_per_iter": 1,
                    "discount": 0.99,
                    "tau": 0.1,
                    "eta": 1.0,
                    "eta2": 1.0,
                    "lambda": 1.0,
                    "max_q_backup": False,
                    "lr_decay": False,
                    "grad_norm": 1.0,
                    "early_stop": True,
                    "early_epoch": 2,
                    "k_rewards": True,
                    "use_discount": True,
                    "sar": False,
                    "reward_tune": "no",
                    "scale": 1000.0,
                    "test_scale": 1000.0,
                    "env_targets": [5000.0, 4000.0, 2500.0],
                    "rtg_no_q": False,
                    "infer_no_q": False,
                    "ema_decay": 0.995,
                    "ema_start_step": 1000,
                    "ema_update_every": 5,
                    "trajectory_export_path": str(export_path),
                    "save_trajectory_pkl": True,
                },
            }

            fake_eval = {
                "env_name": "walker2d-medium-replay-v2",
                "episodes": 1,
                "seed": 0,
                "checkpoint_path": None,
                "checkpoint_step": 1,
                "target_return": 5000.0,
                "candidate_target_returns": [5000.0, 4000.0, 2500.0],
                "max_steps": 5,
                "mean_return": 123.0,
                "std_return": 0.0,
                "mean_episode_length": 5.0,
                "episode_returns": [123.0],
                "episode_lengths": [5],
                "normalized_scores": [95.0],
                "mean_normalized_score": 95.0,
                "std_normalized_score": 0.0,
            }

            with mock.patch("tranqil.anchor_trainer.evaluate_policy", return_value=fake_eval):
                summary = AnchorQTTrainer(config).train()

            self.assertEqual(summary["final_iteration"], 3)
            self.assertEqual(summary["final_step"], 3)

    def test_train_step_runs_without_shape_error(self) -> None:
        """Direct train_step() over P1+P2 stack: confirms shapes flow through
        the HF GPT2 backbone and actor+critic coupling without error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "walker_fixture.hdf5"
            export_path = tmp_path / "walker_fixture.pkl"
            write_hdf5_fixture(dataset_path, make_walker_fixture_dataset())

            config = {
                "run_name": "qt_anchor_train_step_test",
                "seed": 0,
                "device": "cpu",
                "env": {"name": "walker2d-medium-replay-v2"},
                "data": {"dataset_path": str(dataset_path)},
                "evaluation": {
                    "episodes": 1,
                    "max_steps": 5,
                    "target_return": None,
                },
                "outputs": {"base_dir": str(tmp_path / "results")},
                "anchor": {
                    "enabled": True,
                    "K": 3,
                    "pct_traj": 1.0,
                    "batch_size": 4,
                    "embed_dim": 32,
                    "n_layer": 2,
                    "n_head": 4,
                    "activation_function": "relu",
                    "dropout": 0.0,
                    "learning_rate": 1e-3,
                    "lr_min": 0.0,
                    "weight_decay": 0.0,
                    "num_eval_episodes": 1,
                    "max_iters": 1,
                    "num_steps_per_iter": 1,
                    "discount": 0.99,
                    "tau": 0.1,
                    "eta": 1.0,
                    "eta2": 1.0,
                    "lambda": 1.0,
                    "max_q_backup": False,
                    "lr_decay": False,
                    "grad_norm": 1.0,
                    "early_stop": False,
                    "early_epoch": 100,
                    "k_rewards": True,
                    "use_discount": True,
                    "sar": False,
                    "reward_tune": "no",
                    "scale": 1000.0,
                    "test_scale": 1000.0,
                    "env_targets": [5000.0, 4000.0, 2500.0],
                    "rtg_no_q": False,
                    "infer_no_q": False,
                    "ema_decay": 0.995,
                    "ema_start_step": 1000,
                    "ema_update_every": 5,
                    "trajectory_export_path": str(export_path),
                    "save_trajectory_pkl": True,
                },
            }

            trainer = AnchorQTTrainer(config)
            metrics = trainer.train_step()

            for key in ("bc_loss", "ql_loss", "actor_loss", "critic_loss", "target_q_mean"):
                self.assertIn(key, metrics)
                self.assertTrue(np.isfinite(metrics[key]), f"{key} not finite: {metrics[key]}")
            self.assertEqual(trainer.global_step, 1)

    def test_resume_from_checkpoint_continues_training(self) -> None:
        """Train 1 iter → save latest.pt → resume_from that → train 1 more iter.
        Confirms global_step/iteration continue and metrics.jsonl is appended
        rather than clobbered."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "walker_fixture.hdf5"
            export_path = tmp_path / "walker_fixture.pkl"
            write_hdf5_fixture(dataset_path, make_walker_fixture_dataset())

            base_config = {
                "run_name": "qt_anchor_resume_test",
                "seed": 0,
                "device": "cpu",
                "env": {"name": "walker2d-medium-replay-v2"},
                "data": {"dataset_path": str(dataset_path)},
                "evaluation": {"episodes": 1, "max_steps": 5, "target_return": None},
                "outputs": {"base_dir": str(tmp_path / "results")},
                "anchor": {
                    "enabled": True,
                    "K": 3,
                    "pct_traj": 1.0,
                    "batch_size": 2,
                    "embed_dim": 32,
                    "n_layer": 2,
                    "n_head": 4,
                    "activation_function": "relu",
                    "dropout": 0.0,
                    "learning_rate": 1e-3,
                    "lr_min": 0.0,
                    "weight_decay": 0.0,
                    "num_eval_episodes": 1,
                    "max_iters": 1,
                    "num_steps_per_iter": 2,
                    "discount": 0.99,
                    "tau": 0.1,
                    "eta": 1.0,
                    "eta2": 1.0,
                    "lambda": 1.0,
                    "max_q_backup": False,
                    "lr_decay": False,
                    "grad_norm": 1.0,
                    "early_stop": False,
                    "early_epoch": 100,
                    "k_rewards": True,
                    "use_discount": True,
                    "sar": False,
                    "reward_tune": "no",
                    "scale": 1000.0,
                    "test_scale": 1000.0,
                    "env_targets": [5000.0, 4000.0, 2500.0],
                    "rtg_no_q": False,
                    "infer_no_q": False,
                    "ema_decay": 0.995,
                    "ema_start_step": 1000,
                    "ema_update_every": 5,
                    "trajectory_export_path": str(export_path),
                    "save_trajectory_pkl": True,
                },
            }

            fake_eval = {
                "env_name": "walker2d-medium-replay-v2",
                "episodes": 1,
                "seed": 0,
                "checkpoint_path": None,
                "checkpoint_step": 2,
                "target_return": 5000.0,
                "candidate_target_returns": [5000.0, 4000.0, 2500.0],
                "max_steps": 5,
                "mean_return": 50.0,
                "std_return": 0.0,
                "mean_episode_length": 5.0,
                "episode_returns": [50.0],
                "episode_lengths": [5],
                "normalized_scores": [10.0],
                "mean_normalized_score": 10.0,
                "std_normalized_score": 0.0,
            }

            with mock.patch("tranqil.anchor_trainer.evaluate_policy", return_value=fake_eval):
                summary_1 = AnchorQTTrainer(base_config).train()

            run_dir = Path(summary_1["run_dir"])
            latest_path = run_dir / "checkpoints" / "latest.pt"
            metrics_path = run_dir / "metrics.jsonl"
            self.assertTrue(latest_path.exists())
            iter1_lines = metrics_path.read_text().splitlines()
            self.assertEqual(len(iter1_lines), 1)

            # Resume config: extend max_iters so iteration 2 runs.
            resume_config = {**base_config, "anchor": {**base_config["anchor"], "max_iters": 2}}
            with mock.patch("tranqil.anchor_trainer.evaluate_policy", return_value=fake_eval):
                summary_2 = AnchorQTTrainer(resume_config, resume_from=latest_path).train()

            self.assertEqual(summary_2["final_step"], 4)
            self.assertEqual(summary_2["final_iteration"], 2)

            appended_lines = metrics_path.read_text().splitlines()
            self.assertEqual(len(appended_lines), 2)
            self.assertEqual(appended_lines[0], iter1_lines[0])

            resumed_ckpt = load_checkpoint(latest_path)
            self.assertEqual(int(resumed_ckpt["step"]), 4)
            self.assertEqual(int(resumed_ckpt["iteration"]), 2)
            self.assertIn("rng_python", resumed_ckpt)
            self.assertIn("rng_numpy", resumed_ckpt)
            self.assertIn("rng_torch", resumed_ckpt)
            self.assertIn("eta2", resumed_ckpt)
