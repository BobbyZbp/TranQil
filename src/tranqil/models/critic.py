"""Double-Q critic modules for QT training."""

from __future__ import annotations

import torch
from torch import nn


class _QNetwork(nn.Module):
    """Single MLP Q-function over normalized observations and actions."""

    def __init__(self, *, observation_dim: int, action_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        input_dim = int(observation_dim) + int(action_dim)
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        return self.network(inputs)


class DoubleQCritic(nn.Module):
    """Pair of independent Q-functions with shared public helpers."""

    def __init__(self, *, observation_dim: int, action_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.q1 = _QNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.q2 = _QNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observations, actions), self.q2(observations, actions)

    def min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(observations, actions)
        return torch.minimum(q1, q2)
