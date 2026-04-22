"""Anchor-mode critic matching the original QT MLP structure."""

from __future__ import annotations

import torch
from torch import nn


class _AnchorQNetwork(nn.Module):
    def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(state_dim) + int(action_dim), hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([states, actions], dim=-1))


class AnchorDoubleQCritic(nn.Module):
    """Double-Q critic used by the anchor-mode QT trainer."""

    def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q1_model = _AnchorQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        self.q2_model = _AnchorQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1_model(states, actions), self.q2_model(states, actions)

    def q1(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q1_model(states, actions)

    def q_min(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(states, actions)
        return torch.minimum(q1, q2)
