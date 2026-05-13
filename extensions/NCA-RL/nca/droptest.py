"""Counterfactual drop test (paper §4.3).

Given a pretrained (π, Q, N_ψ), the drop test samples states from rollouts,
*restores simulator state* to each, and runs paired rollouts:
    G_base : starting from s_t, follow π_θ to termination
    G_cf   : starting from s_t, take ã ~ κ_ε once, then follow π_θ
ΔG(s) = G_base - G_cf is a finite-sample interventional estimate of
N^π_κ(s, π_θ(s)). Aggregated by N_ψ decile and reported as Spearman ρ_S
with bootstrap CI.

State restoration requires a resettable simulator. For D4RL antmaze /
maze2d we use `env.sim.set_state(qpos, qvel)` (mujoco-py); for new mujoco
the API is `env.set_state(qpos, qvel)`. We keep this module sim-agnostic
by accepting a user-supplied `restore_fn` callback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class DropTestRecord:
    """One drop-test datum. Becomes a row in D_val (paper §4.3)."""
    state: np.ndarray
    action: np.ndarray
    necessity_pred: float       # N_ψ(s, a)
    G_base: float
    G_cf: float

    @property
    def delta_G(self) -> float:
        return self.G_base - self.G_cf


@dataclass
class DropTestResult:
    records: List[DropTestRecord] = field(default_factory=list)

    def deltas(self) -> np.ndarray:
        return np.array([r.delta_G for r in self.records])

    def predictions(self) -> np.ndarray:
        return np.array([r.necessity_pred for r in self.records])

    def spearman(self) -> float:
        from scipy.stats import spearmanr
        return float(spearmanr(self.predictions(), self.deltas()).correlation)

    def spearman_bootstrap_ci(self, n_boot: int = 1000, alpha: float = 0.05):
        """Returns (point, lo, hi) with (1-alpha) CI."""
        from scipy.stats import spearmanr
        p, d = self.predictions(), self.deltas()
        rng = np.random.default_rng(0)
        boots = []
        n = len(p)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boots.append(spearmanr(p[idx], d[idx]).correlation)
        boots = np.array(boots)
        return (
            float(spearmanr(p, d).correlation),
            float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)),
        )

    def decile_means(self) -> np.ndarray:
        """Mean ΔG per N_ψ decile. Returns (10,) array, lowest decile first."""
        p, d = self.predictions(), self.deltas()
        order = np.argsort(p)
        chunks = np.array_split(d[order], 10)
        return np.array([c.mean() for c in chunks])


def paired_rollout(
    env,
    actor_act: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    counterfactual_action: np.ndarray,
    restore_fn: Callable[[object, np.ndarray], None],
    max_steps: int = 1000,
    discount: float = 1.0,
) -> tuple[float, float]:
    """Run one paired rollout from `state`. Returns (G_base, G_cf).

    Both branches share the same starting state; the cf branch deviates only
    on the first action. Subsequent stochasticity comes from the env / actor.
    """
    # baseline
    restore_fn(env, state)
    obs = state
    G_base = 0.0
    g = 1.0
    for _ in range(max_steps):
        a = actor_act(obs)
        obs, r, done, _ = env.step(a)
        G_base += g * r
        g *= discount
        if done:
            break

    # counterfactual: first action overridden
    restore_fn(env, state)
    obs, r, done, _ = env.step(counterfactual_action)
    G_cf = r
    g = discount
    if not done:
        for _ in range(max_steps - 1):
            a = actor_act(obs)
            obs, r, done, _ = env.step(a)
            G_cf += g * r
            g *= discount
            if done:
                break

    return float(G_base), float(G_cf)
