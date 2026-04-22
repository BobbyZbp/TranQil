from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.cli_overrides import apply_eval_overrides, apply_train_overrides


class CLIOverrideTests(unittest.TestCase):
    def test_apply_train_overrides_updates_seed_and_target_return(self) -> None:
        config = {
            "run_name": "base",
            "seed": 0,
            "device": "cpu",
            "training": {"steps": 10},
            "evaluation": {"target_return": None},
            "outputs": {"base_dir": "results/default"},
            "anchor": {"enabled": False},
        }

        updated = apply_train_overrides(
            config,
            steps=25,
            run_name="override",
            base_dir="results/custom",
            device="cuda",
            seed=3,
            target_return=300.0,
        )

        self.assertEqual(updated["training"]["steps"], 25)
        self.assertEqual(updated["run_name"], "override")
        self.assertEqual(updated["outputs"]["base_dir"], "results/custom")
        self.assertEqual(updated["device"], "cuda")
        self.assertEqual(updated["seed"], 3)
        self.assertEqual(updated["evaluation"]["target_return"], 300.0)
        self.assertEqual(config["seed"], 0)

    def test_apply_train_overrides_retargets_anchor_steps(self) -> None:
        config = {
            "run_name": "anchor",
            "seed": 0,
            "device": "cpu",
            "training": {"steps": 10},
            "evaluation": {"target_return": None},
            "anchor": {
                "enabled": True,
                "max_iters": 500,
                "num_steps_per_iter": 1000,
            },
        }

        updated = apply_train_overrides(config, steps=8)

        self.assertEqual(updated["training"]["steps"], 8)
        self.assertEqual(updated["anchor"]["max_iters"], 1)
        self.assertEqual(updated["anchor"]["num_steps_per_iter"], 8)
        self.assertEqual(config["anchor"]["max_iters"], 500)

    def test_apply_eval_overrides_updates_seed_and_target_return(self) -> None:
        config = {
            "seed": 0,
            "device": "cpu",
            "evaluation": {"episodes": 2, "max_steps": 100, "target_return": None},
        }

        updated = apply_eval_overrides(
            config,
            device="cuda",
            episodes=5,
            max_steps=200,
            seed=2,
            target_return=150.0,
        )

        self.assertEqual(updated["device"], "cuda")
        self.assertEqual(updated["seed"], 2)
        self.assertEqual(updated["evaluation"]["episodes"], 5)
        self.assertEqual(updated["evaluation"]["max_steps"], 200)
        self.assertEqual(updated["evaluation"]["target_return"], 150.0)
        self.assertIsNone(config["evaluation"]["target_return"])


if __name__ == "__main__":
    unittest.main()
