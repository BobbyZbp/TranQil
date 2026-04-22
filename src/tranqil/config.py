"""Config loading and run-directory helpers for QT experiments."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "qt_run",
    "seed": 0,
    "device": "cpu",
    "env": {
        "name": "walker2d-medium-replay-v2",
    },
    "data": {
        "context_length": 20,
        "discount": 0.99,
        "batch_size": 64,
        "cache_dir": None,
        "dataset_path": None,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": True,
        "shuffle": True,
    },
    "model": {
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "ffn_multiplier": 4,
        "max_timestep": 2048,
    },
    "critic": {
        "hidden_dim": 256,
        "num_layers": 3,
    },
    "optimization": {
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "weight_decay": 0.0,
    },
    "training": {
        "steps": 1000,
        "gamma": 0.99,
        "eta": 1.0,
        "grad_clip_norm": 1.0,
        "target_tau": 0.005,
        "actor_update_every": 1,
        "log_interval": 50,
        "checkpoint_interval": 250,
        "eval_interval": 250,
    },
    "evaluation": {
        "episodes": 3,
        "max_steps": 1000,
        "target_return": None,
    },
    "outputs": {
        "base_dir": "results/qt_runs",
    },
    "anchor": {
        "enabled": False,
        "env_family": None,
        "dataset_name": None,
        "version": None,
        "K": 20,
        "pct_traj": 1.0,
        "batch_size": 256,
        "embed_dim": 256,
        "n_layer": 4,
        "n_head": 4,
        "activation_function": "relu",
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "lr_min": 0.0,
        "weight_decay": 1e-4,
        "num_eval_episodes": 10,
        "max_iters": 500,
        "num_steps_per_iter": 1000,
        "discount": 0.99,
        "tau": 0.005,
        "eta": 1.0,
        "eta2": 1.0,
        "lambda": 1.0,
        "max_q_backup": False,
        "lr_decay": False,
        "grad_norm": 1.0,
        "early_stop": False,
        "early_epoch": 100,
        "k_rewards": False,
        "use_discount": False,
        "sar": False,
        "reward_tune": "no",
        "scale": None,
        "test_scale": None,
        "env_targets": [],
        "rtg_no_q": False,
        "infer_no_q": False,
        "ema_decay": 0.995,
        "ema_start_step": 1000,
        "ema_update_every": 5,
        "trajectory_export_path": None,
        "save_trajectory_pkl": False,
    },
}


def get_repo_root() -> Path:
    """Resolve the repository root from the package location."""

    return Path(__file__).resolve().parents[2]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two nested config mappings."""

    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require_config_keys(config: Mapping[str, Any]) -> None:
    """Validate the minimum required experiment settings."""

    env_name = config.get("env", {}).get("name")
    if not env_name:
        raise ValueError("Config must define env.name.")

    batch_size = int(config.get("data", {}).get("batch_size", 0))
    context_length = int(config.get("data", {}).get("context_length", 0))
    steps = int(config.get("training", {}).get("steps", 0))
    anchor_enabled = bool(config.get("anchor", {}).get("enabled", False))

    if anchor_enabled:
        anchor = config.get("anchor", {})
        anchor_batch_size = int(anchor.get("batch_size", 0))
        max_iters = int(anchor.get("max_iters", 0))
        steps_per_iter = int(anchor.get("num_steps_per_iter", 0))
        if anchor_batch_size <= 0:
            raise ValueError("anchor.batch_size must be positive.")
        if int(anchor.get("K", 0)) <= 0:
            raise ValueError("anchor.K must be positive.")
        if max_iters <= 0:
            raise ValueError("anchor.max_iters must be positive.")
        if steps_per_iter <= 0:
            raise ValueError("anchor.num_steps_per_iter must be positive.")
        return

    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")
    if context_length <= 0:
        raise ValueError("data.context_length must be positive.")
    if steps <= 0:
        raise ValueError("training.steps must be positive.")


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """Load one YAML config and merge it onto repository defaults."""

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    config = _deep_merge(DEFAULT_CONFIG, loaded)
    config["_config_path"] = str(config_path.resolve())
    _require_config_keys(config)
    return config


def resolve_repo_path(path_value: str | Path | None) -> Path | None:
    """Resolve a config path relative to the repository root."""

    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_repo_root() / path


def resolve_run_dir(config: Mapping[str, Any]) -> Path:
    """Return the experiment output directory for one run."""

    base_dir = resolve_repo_path(config["outputs"]["base_dir"])
    if base_dir is None:
        raise ValueError("outputs.base_dir must resolve to a real path.")
    return base_dir / str(config["run_name"])


def save_resolved_config(config: Mapping[str, Any], output_path: str | Path) -> None:
    """Persist a resolved config for reproducibility."""

    serializable = {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)
