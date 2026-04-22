"""Checkpoint helpers for QT training and evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from .data.normalization import NormalizationStats
from .data.registry import TaskSpec


def build_checkpoint_payload(
    *,
    config: Mapping[str, Any],
    step: int,
    task_spec: TaskSpec,
    stats: NormalizationStats,
    actor_state_dict: Mapping[str, Any],
    critic_state_dict: Mapping[str, Any],
    target_actor_state_dict: Mapping[str, Any],
    target_critic_state_dict: Mapping[str, Any],
    actor_optimizer_state_dict: Mapping[str, Any],
    critic_optimizer_state_dict: Mapping[str, Any],
    latest_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one portable training checkpoint payload."""

    return {
        "config": dict(config),
        "step": int(step),
        "task_spec": asdict(task_spec),
        "normalization_stats": stats.to_cache_dict(),
        "actor_state_dict": actor_state_dict,
        "critic_state_dict": critic_state_dict,
        "target_actor_state_dict": target_actor_state_dict,
        "target_critic_state_dict": target_critic_state_dict,
        "actor_optimizer_state_dict": actor_optimizer_state_dict,
        "critic_optimizer_state_dict": critic_optimizer_state_dict,
        "latest_metrics": dict(latest_metrics),
    }


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Persist one checkpoint with `torch.save`."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load one checkpoint into memory."""

    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)
