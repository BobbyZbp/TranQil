"""Counterfactual intervention kernels κ_ε(·|s,a).

Paper §3.1: necessity is defined relative to a kernel that specifies *which*
intervention is being tested. The same kernel is used in the offline proxy
(§3.2) and the online drop test (§4.3) so that all three estimate the same
quantity.

Default: policy-supported perturbation — ã ~ π(·|s) with rejection enforcing
||ã - a|| > ε in normalised action space.
"""
from __future__ import annotations

from typing import Callable

import torch


SampleAction = Callable[[torch.Tensor], torch.Tensor]
"""Stateless action sampler. Given a state tensor (B, S), returns (B, A)."""


class PolicySupportedKernel:
    """ã ~ π(·|s) subject to ||ã - a|| > ε (rejection sampling).

    Args:
        eps: minimum L2 margin in (normalised) action space.
        max_iters: max rejection iterations before giving up.
    """

    def __init__(self, eps: float = 0.05, max_iters: int = 20):
        self.eps = eps
        self.max_iters = max_iters

    def sample(
        self,
        sample_action: SampleAction,
        state: torch.Tensor,    # (B, S)
        action: torch.Tensor,   # (B, A)
        n_samples: int,
    ) -> torch.Tensor:
        """Returns (B, n_samples, A) tensor of counterfactual actions.

        Vectorised over B*L: we sample n_samples alternatives per (s, a) by
        broadcasting state and resampling rejected entries up to max_iters.
        """
        tilde, _ = self.sample_with_violation(sample_action, state, action, n_samples)
        return tilde

    def sample_with_violation(
        self,
        sample_action: SampleAction,
        state: torch.Tensor,
        action: torch.Tensor,
        n_samples: int,
    ):
        """Same as `sample` but also returns final margin-violation rate (scalar)."""
        B, A = action.shape
        L = n_samples
        state_rep = state.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
        action_rep = action.unsqueeze(1).expand(B, L, -1).reshape(B * L, A)

        tilde = sample_action(state_rep)  # (B*L, A)
        for _ in range(self.max_iters):
            bad = (tilde - action_rep).norm(dim=-1) <= self.eps  # (B*L,)
            if not bad.any():
                break
            resamp = sample_action(state_rep)
            tilde = torch.where(bad.unsqueeze(-1), resamp, tilde)

        violation = ((tilde - action_rep).norm(dim=-1) <= self.eps).float().mean().item()
        return tilde.view(B, L, A), violation
