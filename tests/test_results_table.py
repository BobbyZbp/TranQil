from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.results_table import build_results_package, load_run_artifact_summary, select_canonical_run


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_checkpoint(path: Path, *, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": int(step)}, path)


class ResultsTableTests(unittest.TestCase):
    def test_results_table_builds_finalized_package_from_synthetic_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dirs = []
            for seed, best_mean, best_std, latest_mean in (
                (0, 10.0, 2.0, 9.0),
                (1, 12.0, 1.0, 10.0),
                (2, 11.0, 0.5, 10.5),
            ):
                run_dir = tmp_path / f"seed_{seed}"
                run_dirs.append(run_dir)

                write_checkpoint(run_dir / "checkpoints" / "best.pt", step=100 + seed)
                write_checkpoint(run_dir / "checkpoints" / "latest.pt", step=200 + seed)
                write_json(
                    run_dir / "summary.json",
                    {
                        "run_dir": str(run_dir),
                        "checkpoint_path": str(run_dir / "checkpoints" / "latest.pt"),
                        "best_checkpoint_path": str(run_dir / "checkpoints" / "best.pt"),
                        "best_evaluation": {
                            "env_name": "walker2d-medium-replay-v2",
                            "seed": seed,
                            "mean_return": best_mean,
                            "std_return": best_std,
                            "episodes": 10,
                            "target_return": 300.0,
                        },
                        "latest_evaluation": {
                            "env_name": "walker2d-medium-replay-v2",
                            "seed": seed,
                            "mean_return": latest_mean,
                            "std_return": 2.5,
                            "episodes": 10,
                            "target_return": 300.0,
                        },
                    },
                )
                write_json(
                    run_dir / "best_eval.json",
                    {
                        "env_name": "walker2d-medium-replay-v2",
                        "seed": seed,
                        "mean_return": best_mean,
                        "std_return": best_std,
                        "mean_normalized_score": 50.0 + seed,
                        "std_normalized_score": 1.0 + seed,
                        "episodes": 10,
                        "target_return": 300.0,
                    },
                )
                write_json(
                    run_dir / "latest_eval.json",
                    {
                        "env_name": "walker2d-medium-replay-v2",
                        "seed": seed,
                        "mean_return": latest_mean,
                        "std_return": 2.5,
                        "mean_normalized_score": 45.0 + seed,
                        "std_normalized_score": 1.5,
                        "episodes": 10,
                        "target_return": 300.0,
                    },
                )

            rollout_path = tmp_path / "canonical.mp4"
            rollout_path.write_bytes(b"video")
            (tmp_path / "canonical.mp4.json").write_text("{}", encoding="utf-8")

            reference_path = tmp_path / "reference.json"
            write_json(
                reference_path,
                {
                    "env_name": "walker2d-medium-replay-v2",
                    "entries": [
                        {
                            "label": "QT paper",
                            "source_name": "paper",
                            "source_note": "reference row",
                            "source_url": "https://example.com",
                            "mean_normalized_score": 98.5,
                            "std_normalized_score": 1.1,
                        }
                    ],
                },
            )

            run_summaries = [load_run_artifact_summary(path) for path in run_dirs]
            canonical = select_canonical_run(run_summaries)
            self.assertEqual(canonical.seed, 1)

            finalized_dir = tmp_path / "finalized"
            manifest = build_results_package(
                run_dirs=run_dirs,
                reference_results_path=reference_path,
                finalized_dir=finalized_dir,
                canonical_rollout_path=rollout_path,
            )

            self.assertEqual(manifest["canonical_seed"], 1)
            self.assertTrue((finalized_dir / "manifest.json").exists())
            self.assertTrue((finalized_dir / "results_table.csv").exists())
            self.assertTrue((finalized_dir / "results_table.md").exists())
            self.assertTrue((finalized_dir / "best.pt").exists())
            self.assertTrue((finalized_dir / "best_eval.json").exists())
            self.assertTrue((finalized_dir / "canonical_rollout.mp4").exists())
            self.assertTrue((finalized_dir / "canonical_rollout.mp4.json").exists())


if __name__ == "__main__":
    unittest.main()
