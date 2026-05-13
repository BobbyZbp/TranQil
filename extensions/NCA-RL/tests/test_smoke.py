"""Smoke tests for the nca/ library and a syntax check for nca_iql.py.

Runs without d4rl / mujoco / gym — uses synthetic tensors for everything.
The full pipeline still has to be exercised on a GPU host with the D4RL
stack installed (Tier 1 plan in experiments/checkpoint_probing/README.md).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch
import torch.nn as nn

# allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nca import (  # noqa: E402
    NecessityEnsemble,
    NecessityHead,
    PolicySupportedKernel,
    compute_offline_proxy,
    rank_normalise,
)


# --------------------------------------------------------------------------- #
# Synthetic networks that mimic CORL's interface                              #
# --------------------------------------------------------------------------- #
class FakeTwinQ(nn.Module):
    """Mimics CORL's TwinQ.__call__ which returns min(q1, q2)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.f(torch.cat([s, a], dim=-1)).squeeze(-1)


class FakeGaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.net = nn.Linear(state_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.max_action = max_action

    def forward(self, s: torch.Tensor):
        mean = torch.tanh(self.net(s)) * self.max_action
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_kernel_shape_and_margin() -> None:
    torch.manual_seed(0)
    actor = FakeGaussianActor(state_dim=4, action_dim=3, max_action=1.0)
    kernel = PolicySupportedKernel(eps=0.05, max_iters=20)

    def sample_action(s: torch.Tensor) -> torch.Tensor:
        return actor(s).sample().clamp(-1.0, 1.0)

    B, L = 16, 8
    s = torch.randn(B, 4)
    a = torch.randn(B, 3).clamp(-1.0, 1.0)

    tilde = kernel.sample(sample_action, s, a, n_samples=L)
    assert tilde.shape == (B, L, 3), f"bad shape {tilde.shape}"

    diffs = (tilde - a.unsqueeze(1)).norm(dim=-1)  # (B, L)
    # rejection is best-effort up to max_iters; with stochastic Gaussian
    # actor this should virtually always pass on margin=0.05
    assert (diffs > 0.05).float().mean() > 0.95, (
        f"too many alternatives violate margin: {(diffs > 0.05).float().mean():.3f}"
    )
    print(f"[ok] kernel: {tilde.shape}, mean ||ã-a||={diffs.mean():.3f}")


def test_necessity_head_forward_backward() -> None:
    head = NecessityHead(state_dim=4, action_dim=3)
    s = torch.randn(8, 4)
    a = torch.randn(8, 3)
    out = head(s, a)
    assert out.shape == (8,), f"bad shape {out.shape}"
    out.sum().backward()
    assert head.net.net[0].weight.grad is not None
    print(f"[ok] head: out.shape={out.shape}, grads non-None")


def test_ensemble_disagreement() -> None:
    ens = NecessityEnsemble(state_dim=4, action_dim=3)
    s = torch.randn(8, 4)
    a = torch.randn(8, 3)
    n1, n2 = ens(s, a)
    assert n1.shape == (8,) and n2.shape == (8,)
    dis = ens.disagreement(s, a)
    assert dis.shape == (8,) and (dis >= 0).all()
    mean = ens.mean(s, a)
    assert torch.allclose(mean, 0.5 * (n1 + n2))
    print(f"[ok] ensemble: disagreement mean={dis.mean():.4f}")


def test_offline_proxy() -> None:
    torch.manual_seed(0)
    qf = FakeTwinQ(state_dim=4, action_dim=3)
    actor = FakeGaussianActor(state_dim=4, action_dim=3)
    kernel = PolicySupportedKernel(eps=0.05)

    def sample_action(s: torch.Tensor) -> torch.Tensor:
        return actor(s).sample().clamp(-1.0, 1.0)

    s = torch.randn(32, 4)
    a = torch.randn(32, 3).clamp(-1.0, 1.0)
    n_off, violation = compute_offline_proxy(qf, sample_action, kernel, s, a, n_samples=8)
    assert n_off.shape == (32,)
    assert 0.0 <= violation <= 1.0
    assert torch.isfinite(n_off).all()
    # under random init, mean N^off should be near zero (no systematic bias)
    print(
        f"[ok] N^off: shape={n_off.shape}, "
        f"mean={n_off.mean():+.4f}, std={n_off.std():.4f}"
    )


def test_stage2_update_step() -> None:
    """Standalone Stage 2 step: regress N_ψ to N^off on synthetic data.
    Mimics what NCAImplicitQLearning.update_necessity does — without
    importing nca_iql.py (which depends on d4rl/gym)."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    state_dim, action_dim = 4, 3
    qf = FakeTwinQ(state_dim, action_dim)
    actor = FakeGaussianActor(state_dim, action_dim)
    kernel = PolicySupportedKernel(eps=0.05)
    ens = NecessityEnsemble(state_dim, action_dim, hidden_dim=64, n_hidden=2)
    opt = torch.optim.Adam(ens.parameters(), lr=3e-4)

    def sample_action(s: torch.Tensor) -> torch.Tensor:
        return actor(s).sample().clamp(-1.0, 1.0)

    losses = []
    for _ in range(50):
        s = torch.randn(64, state_dim)
        a = torch.randn(64, action_dim).clamp(-1.0, 1.0)
        target, _ = compute_offline_proxy(qf, sample_action, kernel, s, a, n_samples=8)
        p1, p2 = ens(s, a)
        loss = F.mse_loss(p1, target) + F.mse_loss(p2, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # loss should drop noticeably across 50 steps on a fixed Q
    assert losses[-1] < losses[0], f"no learning: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"[ok] stage2: loss {losses[0]:.4f} -> {losses[-1]:.4f} over 50 steps")


def test_rank_normalise() -> None:
    x = torch.tensor([3.0, 1.0, 4.0, 1.5, 9.0])
    r = rank_normalise(x)
    assert r.shape == x.shape
    assert (r >= 0).all() and (r <= 1).all()
    # max value of x maps to 1.0, min to 0.0
    assert r[x.argmax()].item() == 1.0
    assert r[x.argmin()].item() == 0.0
    print(f"[ok] rank_normalise: {r.tolist()}")


def test_d4rl_scripts_parse() -> None:
    """The three scripts in experiments/checkpoint_probing/ all import d4rl
    which we don't have locally; AST-parse them instead so syntax breakage
    is caught before we deploy to the GPU host."""
    cp = ROOT / "experiments" / "checkpoint_probing"
    for name in ("nca_iql.py", "compute_proxy.py", "drop_test.py"):
        p = cp / name
        src = p.read_text()
        try:
            ast.parse(src, filename=str(p))
        except SyntaxError as e:
            raise AssertionError(f"{name} syntax error: {e}") from e
        print(f"[ok] {name} parses ({len(src.splitlines())} lines)")


if __name__ == "__main__":
    tests = [
        test_kernel_shape_and_margin,
        test_necessity_head_forward_backward,
        test_ensemble_disagreement,
        test_offline_proxy,
        test_stage2_update_step,
        test_rank_normalise,
        test_d4rl_scripts_parse,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} smoke tests passed.")
