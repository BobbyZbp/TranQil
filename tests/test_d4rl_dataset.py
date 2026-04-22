from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.data.d4rl_dataset import build_qt_dataset


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


class D4RLDatasetBuilderTests(unittest.TestCase):
    def test_build_qt_dataset_from_hdf5_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "walker_fixture.hdf5"
            cache_dir = tmp_path / "cache"
            write_hdf5_fixture(dataset_path, make_walker_fixture_dataset())

            dataset = build_qt_dataset(
                env_name="walker2d-medium-replay-v2",
                context_length=3,
                discount=0.99,
                cache_dir=cache_dir,
                dataset_path=dataset_path,
            )

            self.assertEqual(len(dataset), 5)
            self.assertFalse(dataset.metadata["cache_hit"])
            self.assertEqual(dataset.metadata["dataset_path"], str(dataset_path))
            self.assertEqual(dataset.metadata["cache_path"], str(cache_dir / "walker2d-medium-replay-v2.npz"))
            self.assertEqual(tuple(dataset[0]["observations"].shape), (3, 17))

    def test_build_qt_dataset_cache_hit_and_discount_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "walker_fixture.hdf5"
            cache_dir = tmp_path / "cache"
            write_hdf5_fixture(dataset_path, make_walker_fixture_dataset())

            dataset_first = build_qt_dataset(
                env_name="walker2d-medium-replay-v2",
                context_length=3,
                discount=0.99,
                cache_dir=cache_dir,
                dataset_path=dataset_path,
            )
            self.assertFalse(dataset_first.metadata["cache_hit"])

            cache_path = cache_dir / "walker2d-medium-replay-v2.npz"
            with np.load(cache_path, allow_pickle=False) as cache_payload:
                self.assertEqual(cache_payload["discount"].dtype, np.float32)

            dataset_second = build_qt_dataset(
                env_name="walker2d-medium-replay-v2",
                context_length=3,
                discount=0.99,
                cache_dir=cache_dir,
                dataset_path=dataset_path,
            )
            self.assertTrue(dataset_second.metadata["cache_hit"])

            dataset_third = build_qt_dataset(
                env_name="walker2d-medium-replay-v2",
                context_length=3,
                discount=0.95,
                cache_dir=cache_dir,
                dataset_path=dataset_path,
            )
            self.assertFalse(dataset_third.metadata["cache_hit"])


if __name__ == "__main__":
    unittest.main()
