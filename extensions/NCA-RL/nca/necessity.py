"""NecessityHead, NecessityEnsemble, and the offline Q-margin proxy.

Paper §3.2 (proxy):
    N^off(s, a) := Q_φ(s, a) - E_{ã ~ κ_ε}[Q_φ(s, ã)]

Paper §4.1 (head): the parametric head N_ψ regresses against N^off (paper
§4.2, eq. 5). Two parallel heads support an ensemble disagreement penalty
(§4.4 bias-breaking mechanism C).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_hidden: int, out_dim: int):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NecessityHead(nn.Module):
    """N_ψ : (s, a) → R. Raw output; rectification (ReLU / rank-norm) at use-time."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = _MLP(state_dim + action_dim, hidden_dim, n_hidden, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class NecessityEnsemble(nn.Module):
    """Two parallel heads. Disagreement is used as an epistemic-uncertainty
    signal and as a bias-breaking penalty (paper §4.4)."""

    def __init__(self, state_dim: int, action_dim: int, **kw):
        super().__init__()
        self.head1 = NecessityHead(state_dim, action_dim, **kw)
        self.head2 = NecessityHead(state_dim, action_dim, **kw)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head1(state, action), self.head2(state, action)

    def mean(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        n1, n2 = self(state, action)
        return 0.5 * (n1 + n2)

    def disagreement(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        n1, n2 = self(state, action)
        return (n1 - n2).abs()


@torch.no_grad()
def compute_offline_proxy(
    q_network: nn.Module,
    sample_action,
    kernel,
    states: torch.Tensor,        # (B, S)
    actions: torch.Tensor,       # (B, A)
    n_samples: int = 8,
):
    """Estimate N^off(s,a) = Q(s,a) - E[Q(s, ã)] via Monte Carlo, ã ~ κ.

    Returns (proxy, violation_rate) where violation_rate is the fraction of
    final ã samples still inside the ε-ball after rejection sampling. Useful
    as a diagnostic when the actor stddev is too small relative to ε.

    `q_network` should be a single-output Q. For CORL's TwinQ this is the
    pessimistic min(Q1, Q2) returned by `__call__`.
    """
    q_sa = q_network(states, actions)  # (B,)

    if hasattr(kernel, "sample_with_violation"):
        tilde, violation = kernel.sample_with_violation(
            sample_action, states, actions, n_samples
        )
    else:
        tilde = kernel.sample(sample_action, states, actions, n_samples)
        violation = 0.0
    B, L, A = tilde.shape

    state_rep = states.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
    tilde_flat = tilde.reshape(B * L, A)
    q_alt = q_network(state_rep, tilde_flat).view(B, L)

    return q_sa - q_alt.mean(dim=1), violation


def rank_normalise(x: torch.Tensor) -> torch.Tensor:
    """Map necessity scores to [0, 1] by within-batch rank percentile.

    Paper §4.2 default for continuous control: scale-invariant to Q-magnitude
    and avoids negative weights when raw N is negative (§4.4 pitfall #4).
    """
    ranks = x.argsort().argsort().float()
    return ranks / max(len(x) - 1, 1)
