from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.rendering import resolve_qt_rollout_output_path


class RenderingTests(unittest.TestCase):
    def test_qt_rollout_output_path_uses_run_local_rollouts_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            checkpoint_path = tmp_path / "results" / "qt_runs" / "sample_run" / "checkpoints" / "best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")

            output_path = resolve_qt_rollout_output_path(
                checkpoint_path=checkpoint_path,
                env_name="walker2d-medium-replay-v2",
                seed=2,
                output=None,
                output_format="auto",
                mp4_supported=True,
            )

            self.assertEqual(output_path.parent, checkpoint_path.parent.parent / "rollouts")
            self.assertEqual(output_path.suffix, ".mp4")
            self.assertTrue(output_path.name.startswith("walker2d_medium_replay_v2_best_seed2"))


if __name__ == "__main__":
    unittest.main()
