"""Model construction helpers for QT experiments."""

from __future__ import annotations

from typing import Any, Mapping

from .data.registry import TaskSpec
from .models import DoubleQCritic, QTActor


def build_actor_from_config(config: Mapping[str, Any], task_spec: TaskSpec) -> QTActor:
    """Construct the QT actor described by one experiment config."""

    model_config = config["model"]
    return QTActor(
        observation_dim=task_spec.observation_dim,
        action_dim=task_spec.action_dim,
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        dropout=float(model_config["dropout"]),
        ffn_multiplier=int(model_config["ffn_multiplier"]),
        max_timestep=int(model_config["max_timestep"]),
    )


def build_critic_from_config(config: Mapping[str, Any], task_spec: TaskSpec) -> DoubleQCritic:
    """Construct the double-Q critic described by one experiment config."""

    critic_config = config["critic"]
    return DoubleQCritic(
        observation_dim=task_spec.observation_dim,
        action_dim=task_spec.action_dim,
        hidden_dim=int(critic_config["hidden_dim"]),
        num_layers=int(critic_config["num_layers"]),
    )
