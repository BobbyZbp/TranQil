"""High-level builders for the QT D4RL data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from torch.utils.data import DataLoader

from .normalization import NormalizationStats, compute_normalization_stats
from .preprocessing import (
    PreprocessedDataset,
    build_next_observations,
    preprocess_d4rl_dataset,
    validate_d4rl_dataset,
)
from .registry import TaskSpec, get_default_cache_dir, get_task_spec, sanitize_env_name
from .sequence_sampler import QTSequenceDataset


RAW_DATASET_KEYS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "timeouts",
    "next_observations",
)

CACHE_VERSION = 1
CACHE_REQUIRED_KEYS = (
    "cache_version",
    "env_name",
    "discount",
    "transition_count",
    "observation_dim",
    "action_dim",
    "terminal_count",
    "timeout_count",
    "dataset_alias",
    "source_has_next_observations",
    "forced_episode_end_mask",
    "episode_start_indices",
    "episode_end_indices",
    "episode_ids",
    "timesteps",
    "returns_to_go",
    "bootstrap_mask",
    "episode_returns",
    "valid_sample_indices",
    "obs_mean",
    "obs_std",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_max",
    "return_mean",
    "return_std",
    "return_min",
    "return_max",
    "episode_count",
    "obs_std_epsilon",
)


@dataclass(frozen=True)
class DatasetBuildRequest:
    """Inputs needed to build one QT dataset instance."""

    env_name: str
    task_spec: TaskSpec
    context_length: int
    discount: float
    validated_dataset: dict[str, np.ndarray]
    cache_path: Path
    dataset_path: str | None


@dataclass(frozen=True)
class CacheSignature:
    """Stable identity fields used for cache compatibility checks."""

    env_name: str
    dataset_alias: str
    discount: float
    transition_count: int
    observation_dim: int
    action_dim: int
    terminal_count: int
    timeout_count: int

    @classmethod
    def from_request(cls, request: DatasetBuildRequest) -> "CacheSignature":
        dataset = request.validated_dataset
        return cls(
            env_name=request.env_name,
            dataset_alias=request.task_spec.dataset_alias,
            discount=request.discount,
            transition_count=int(dataset["rewards"].shape[0]),
            observation_dim=int(dataset["observations"].shape[1]),
            action_dim=int(dataset["actions"].shape[1]),
            terminal_count=int(dataset["terminals"].sum()),
            timeout_count=int(dataset["timeouts"].sum()),
        )

    @classmethod
    def from_cache_payload(cls, cache: Mapping[str, np.ndarray]) -> "CacheSignature":
        return cls(
            env_name=str(_coerce_scalar(cache["env_name"])),
            dataset_alias=str(_coerce_scalar(cache["dataset_alias"])),
            discount=float(_coerce_scalar(cache["discount"])),
            transition_count=int(_coerce_scalar(cache["transition_count"])),
            observation_dim=int(_coerce_scalar(cache["observation_dim"])),
            action_dim=int(_coerce_scalar(cache["action_dim"])),
            terminal_count=int(_coerce_scalar(cache["terminal_count"])),
            timeout_count=int(_coerce_scalar(cache["timeout_count"])),
        )

    def matches(self, other: "CacheSignature") -> bool:
        """Check whether two cache signatures describe the same dataset build."""

        return (
            self.env_name == other.env_name
            and self.dataset_alias == other.dataset_alias
            and bool(np.isclose(self.discount, other.discount))
            and self.transition_count == other.transition_count
            and self.observation_dim == other.observation_dim
            and self.action_dim == other.action_dim
            and self.terminal_count == other.terminal_count
            and self.timeout_count == other.timeout_count
        )


@dataclass(frozen=True)
class MaterializedDataset:
    """Preprocessed arrays and normalization stats for one dataset build."""

    preprocessed: PreprocessedDataset
    stats: NormalizationStats
    cache_hit: bool


def _coerce_scalar(value: np.ndarray | float | int | str | bool) -> Any:
    """Convert numpy scalars to plain Python values."""

    if isinstance(value, np.ndarray):
        return value.item()
    return value


def _resolve_cache_path(env_name: str, cache_dir: str | Path | None) -> Path:
    """Resolve the `.npz` cache path for an env id."""

    base_dir = Path(cache_dir) if cache_dir is not None else get_default_cache_dir()
    return base_dir / f"{sanitize_env_name(env_name)}.npz"


def _load_raw_dataset_from_hdf5(dataset_path: str | Path) -> dict[str, np.ndarray]:
    """Load a raw dataset mapping directly from an HDF5 file."""

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    dataset: dict[str, np.ndarray] = {}
    with h5py.File(dataset_path, "r") as handle:
        for key in RAW_DATASET_KEYS:
            if key in handle:
                dataset[key] = handle[key][()]
    return dataset


def _load_raw_dataset_from_env(env_name: str) -> Mapping[str, Any]:
    """Load a raw dataset from the installed Gym + D4RL runtime."""

    import d4rl  # noqa: F401
    import gym

    env = gym.make(env_name)
    try:
        return env.get_dataset()
    finally:
        if hasattr(env, "close"):
            env.close()


def _load_raw_dataset(env_name: str, dataset_path: str | Path | None) -> Mapping[str, Any]:
    """Load a raw dataset from either D4RL or an explicit HDF5 file."""

    if dataset_path is not None:
        return _load_raw_dataset_from_hdf5(dataset_path)
    return _load_raw_dataset_from_env(env_name)


def _validate_dataset_dimensions(dataset: Mapping[str, np.ndarray], task_spec: TaskSpec) -> None:
    """Confirm that the loaded dataset matches the task registry metadata."""

    observation_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]

    if observation_dim != task_spec.observation_dim:
        raise ValueError(
            f"{task_spec.env_name} observation dim mismatch: expected {task_spec.observation_dim}, "
            f"got {observation_dim}."
        )
    if action_dim != task_spec.action_dim:
        raise ValueError(
            f"{task_spec.env_name} action dim mismatch: expected {task_spec.action_dim}, "
            f"got {action_dim}."
        )


def _prepare_build_request(
    *,
    env_name: str,
    context_length: int,
    discount: float,
    cache_dir: str | Path | None,
    dataset_path: str | Path | None,
) -> DatasetBuildRequest:
    """Phase 1: load and validate the raw dataset inputs."""

    task_spec = get_task_spec(env_name)
    context_length = int(context_length)
    discount = float(discount)

    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in the interval (0, 1].")

    validated_dataset = validate_d4rl_dataset(
        _load_raw_dataset(env_name=env_name, dataset_path=dataset_path)
    )
    _validate_dataset_dimensions(validated_dataset, task_spec)

    resolved_dataset_path = str(dataset_path) if dataset_path is not None else None
    return DatasetBuildRequest(
        env_name=env_name,
        task_spec=task_spec,
        context_length=context_length,
        discount=discount,
        validated_dataset=validated_dataset,
        cache_path=_resolve_cache_path(env_name=env_name, cache_dir=cache_dir),
        dataset_path=resolved_dataset_path,
    )


def _has_required_cache_keys(cache: Mapping[str, np.ndarray]) -> bool:
    """Check whether a cache payload contains the fields required for restore."""

    return set(CACHE_REQUIRED_KEYS).issubset(cache.keys())


def _load_cache_payload(cache_path: str | Path) -> dict[str, np.ndarray] | None:
    """Load a cache payload into plain numpy arrays."""

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as cache_file:
        return {key: cache_file[key] for key in cache_file.files}


def _is_cache_compatible(cache: Mapping[str, np.ndarray], request: DatasetBuildRequest) -> bool:
    """Check whether a cache payload matches the current build request."""

    if not _has_required_cache_keys(cache):
        return False

    try:
        cache_version = int(_coerce_scalar(cache["cache_version"]))
        cache_signature = CacheSignature.from_cache_payload(cache)
    except (KeyError, TypeError, ValueError):
        return False

    return cache_version == CACHE_VERSION and cache_signature.matches(CacheSignature.from_request(request))


def _restore_preprocessed_from_cache(
    request: DatasetBuildRequest,
    cache: Mapping[str, np.ndarray],
) -> PreprocessedDataset:
    """Rebuild a preprocessed dataset view from cached metadata and raw arrays."""

    episode_start_indices = np.asarray(cache["episode_start_indices"], dtype=np.int64)
    episode_end_indices = np.asarray(cache["episode_end_indices"], dtype=np.int64)
    next_observations = build_next_observations(
        dataset=request.validated_dataset,
        episode_start_indices=episode_start_indices,
        episode_end_indices=episode_end_indices,
    )

    return PreprocessedDataset(
        observations=np.asarray(request.validated_dataset["observations"], dtype=np.float32),
        actions=np.asarray(request.validated_dataset["actions"], dtype=np.float32),
        rewards=np.asarray(request.validated_dataset["rewards"], dtype=np.float32),
        terminals=np.asarray(request.validated_dataset["terminals"], dtype=np.bool_),
        timeouts=np.asarray(request.validated_dataset["timeouts"], dtype=np.bool_),
        next_observations=next_observations,
        returns_to_go=np.asarray(cache["returns_to_go"], dtype=np.float32),
        episode_ids=np.asarray(cache["episode_ids"], dtype=np.int64),
        timesteps=np.asarray(cache["timesteps"], dtype=np.int64),
        bootstrap_mask=np.asarray(cache["bootstrap_mask"], dtype=np.float32),
        episode_returns=np.asarray(cache["episode_returns"], dtype=np.float32),
        episode_start_indices=episode_start_indices,
        episode_end_indices=episode_end_indices,
        valid_sample_indices=np.asarray(cache["valid_sample_indices"], dtype=np.int64),
        metadata={
            "discount": float(_coerce_scalar(cache["discount"])),
            "source_has_next_observations": bool(_coerce_scalar(cache["source_has_next_observations"])),
            "forced_episode_end_mask": np.asarray(cache["forced_episode_end_mask"], dtype=np.bool_),
            "transition_count": int(_coerce_scalar(cache["transition_count"])),
            "episode_count": int(_coerce_scalar(cache["episode_count"])),
            "terminal_count": int(_coerce_scalar(cache["terminal_count"])),
            "timeout_count": int(_coerce_scalar(cache["timeout_count"])),
        },
    )


def _build_cache_payload(
    request: DatasetBuildRequest,
    preprocessed: PreprocessedDataset,
    stats: NormalizationStats,
) -> dict[str, np.ndarray]:
    """Serialize preprocessed metadata and normalization stats to cache."""

    payload = {
        "cache_version": np.array(CACHE_VERSION, dtype=np.int64),
        "env_name": np.array(request.env_name),
        "dataset_alias": np.array(request.task_spec.dataset_alias),
        "context_length": np.array(request.context_length, dtype=np.int64),
        "discount": np.array(preprocessed.metadata["discount"], dtype=np.float32),
        "transition_count": np.array(preprocessed.transition_count, dtype=np.int64),
        "episode_count": np.array(preprocessed.episode_count, dtype=np.int64),
        "observation_dim": np.array(request.task_spec.observation_dim, dtype=np.int64),
        "action_dim": np.array(request.task_spec.action_dim, dtype=np.int64),
        "terminal_count": np.array(preprocessed.metadata["terminal_count"], dtype=np.int64),
        "timeout_count": np.array(preprocessed.metadata["timeout_count"], dtype=np.int64),
        "source_has_next_observations": np.array(
            preprocessed.metadata["source_has_next_observations"],
            dtype=np.bool_,
        ),
        "forced_episode_end_mask": np.asarray(
            preprocessed.metadata["forced_episode_end_mask"],
            dtype=np.bool_,
        ),
        "episode_start_indices": preprocessed.episode_start_indices.astype(np.int64, copy=False),
        "episode_end_indices": preprocessed.episode_end_indices.astype(np.int64, copy=False),
        "episode_ids": preprocessed.episode_ids.astype(np.int64, copy=False),
        "timesteps": preprocessed.timesteps.astype(np.int64, copy=False),
        "returns_to_go": preprocessed.returns_to_go.astype(np.float32, copy=False),
        "bootstrap_mask": preprocessed.bootstrap_mask.astype(np.float32, copy=False),
        "episode_returns": preprocessed.episode_returns.astype(np.float32, copy=False),
        "valid_sample_indices": preprocessed.valid_sample_indices.astype(np.int64, copy=False),
    }
    payload.update(stats.to_cache_dict())
    return payload


def _write_cache_payload(
    request: DatasetBuildRequest,
    preprocessed: PreprocessedDataset,
    stats: NormalizationStats,
) -> None:
    """Persist a cache payload for future dataset builds."""

    request.cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        request.cache_path,
        **_build_cache_payload(request=request, preprocessed=preprocessed, stats=stats),
    )


def _materialize_dataset(request: DatasetBuildRequest) -> MaterializedDataset:
    """Phase 2: restore from cache or preprocess and cache the dataset."""

    cache_payload = _load_cache_payload(request.cache_path)
    if cache_payload is not None and _is_cache_compatible(cache_payload, request):
        return MaterializedDataset(
            preprocessed=_restore_preprocessed_from_cache(request, cache_payload),
            stats=NormalizationStats.from_cache_dict(cache_payload),
            cache_hit=True,
        )

    preprocessed = preprocess_d4rl_dataset(
        raw_dataset=request.validated_dataset,
        discount=request.discount,
    )
    stats = compute_normalization_stats(
        observations=preprocessed.observations,
        rewards=preprocessed.rewards,
        episode_returns=preprocessed.episode_returns,
    )
    _write_cache_payload(request=request, preprocessed=preprocessed, stats=stats)
    return MaterializedDataset(preprocessed=preprocessed, stats=stats, cache_hit=False)


def _finalize_dataset(
    request: DatasetBuildRequest,
    materialized: MaterializedDataset,
) -> QTSequenceDataset:
    """Phase 3: build the public sequence dataset object."""

    dataset = QTSequenceDataset(
        task_spec=request.task_spec,
        stats=materialized.stats,
        preprocessed=materialized.preprocessed,
        context_length=request.context_length,
    )
    dataset.metadata["cache_hit"] = materialized.cache_hit
    dataset.metadata["cache_path"] = str(request.cache_path)
    dataset.metadata["dataset_path"] = request.dataset_path
    return dataset


def build_qt_dataset(
    env_name: str,
    context_length: int = 20,
    discount: float = 0.99,
    cache_dir: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> QTSequenceDataset:
    """Build the QT sequence dataset for one supported D4RL task."""

    request = _prepare_build_request(
        env_name=env_name,
        context_length=context_length,
        discount=discount,
        cache_dir=cache_dir,
        dataset_path=dataset_path,
    )
    materialized = _materialize_dataset(request)
    return _finalize_dataset(request, materialized)


def make_sequence_dataloader(
    env_name: str | None = None,
    *,
    dataset: QTSequenceDataset | None = None,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    context_length: int = 20,
    discount: float = 0.99,
    cache_dir: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> DataLoader:
    """Create a PyTorch dataloader for the QT sequence dataset."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if dataset is None:
        if env_name is None:
            raise ValueError("env_name must be provided when dataset is not supplied.")
        dataset = build_qt_dataset(
            env_name=env_name,
            context_length=context_length,
            discount=discount,
            cache_dir=cache_dir,
            dataset_path=dataset_path,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
