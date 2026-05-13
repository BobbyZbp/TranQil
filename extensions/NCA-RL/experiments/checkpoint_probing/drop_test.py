"""Counterfactual drop-test runner (paper §4.3).

For each (s, a) probed by `compute_proxy.py`:
  1. Restore simulator state to s
  2. Roll out π_θ to termination → G_base
  3. Restore simulator state to s, take ã ~ κ_ε once, then π_θ → G_cf
  4. Record ΔG = G_base - G_cf

Output: list of `DropTestRecord`, saved as .npz, consumed by `plots.py`.

State-restoration notes
-----------------------
For D4RL antmaze (mujoco-py based) and maze2d, state restoration goes
through `env.unwrapped.set_state(qpos, qvel)`. We capture (qpos, qvel)
on rollout and replay it before each paired branch.

For env types we haven't tried yet: extend `_capture_state` and
`_restore_state` below.

Run on a GPU host with d4rl + mujoco_py installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import d4rl  # noqa: F401
import gym
import numpy as np
import pyrallis
import torch

from nca import (
    DropTestRecord,
    DropTestResult,
    NecessityEnsemble,
    PolicySupportedKernel,
)
from nca_iql import (
    DeterministicPolicy,
    GaussianPolicy,
    TwinQ,
    compute_mean_std,
    normalize_states,
    wrap_env,
)


@dataclass
class DropTestConfig:
    checkpoint: str = ""
    probe_npz: str = "probe_results.npz"
    env: str = "antmaze-medium-diverse-v2"
    n_states: int = 200       # subset of probed states; full 1000 is expensive
    n_episodes_per: int = 1   # paired rollouts per state (paper uses 1 baseline + 1 cf)
    max_episode_steps: int = 1000
    kernel_eps: float = 0.05
    actor_jitter: float = 0.1
    iql_deterministic: bool = False
    output: str = "droptest_results.npz"
    device: str = "cuda"
    seed: int = 0


# --------------------------------------------------------------------------- #
# Sim-state capture/restore. mujoco-py path.                                  #
# --------------------------------------------------------------------------- #
def _capture_state(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
    sim = env.unwrapped.sim  # mujoco-py: MjSim
    return sim.data.qpos.copy(), sim.data.qvel.copy()


def _restore_state(env: gym.Env, qpos: np.ndarray, qvel: np.ndarray) -> None:
    env.unwrapped.set_state(qpos, qvel)


# --------------------------------------------------------------------------- #
# Paired rollout                                                              #
# --------------------------------------------------------------------------- #
def _act_unnorm(actor: torch.nn.Module, obs: np.ndarray, max_action: float,
                state_mean: np.ndarray, state_std: np.ndarray, device: str,
                stochastic: bool) -> np.ndarray:
    """Run actor on a *raw* env observation (un-normalised) — we apply the
    same normalisation Stage 1 used."""
    s = (obs - state_mean) / state_std
    s_t = torch.as_tensor(s.reshape(1, -1), dtype=torch.float32, device=device)
    out = actor(s_t)
    if isinstance(out, torch.distributions.Distribution):
        a = out.sample() if stochastic else out.mean
    else:
        a = out
    a = (a * max_action).clamp(-max_action, max_action)
    return a.detach().cpu().numpy().flatten()


def paired_rollout(
    env: gym.Env,
    actor: torch.nn.Module,
    qpos: np.ndarray,
    qvel: np.ndarray,
    cf_action: np.ndarray,
    max_action: float,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: str,
    stochastic: bool,
    max_steps: int,
) -> Tuple[float, float]:
    """Returns (G_base, G_cf). Both branches start from the same (qpos, qvel)."""
    # baseline
    env.reset()
    _restore_state(env, qpos, qvel)
    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else env.reset()
    G_base = 0.0
    for _ in range(max_steps):
        a = _act_unnorm(actor, obs, max_action, state_mean, state_std, device, stochastic)
        obs, r, done, _ = env.step(a)
        G_base += r
        if done:
            break

    # counterfactual
    env.reset()
    _restore_state(env, qpos, qvel)
    obs, r, done, _ = env.step(cf_action)
    G_cf = float(r)
    if not done:
        for _ in range(max_steps - 1):
            a = _act_unnorm(actor, obs, max_action, state_mean, state_std, device, stochastic)
            obs, r, done, _ = env.step(a)
            G_cf += r
            if done:
                break

    return float(G_base), float(G_cf)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
@pyrallis.wrap()
def main(config: DropTestConfig):
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # State normalisation (must match Stage 1)
    dataset = d4rl.qlearning_dataset(env)
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)

    # Networks
    qf = TwinQ(state_dim, action_dim).to(config.device)
    actor = (
        DeterministicPolicy(state_dim, action_dim, max_action)
        if config.iql_deterministic
        else GaussianPolicy(state_dim, action_dim, max_action)
    ).to(config.device)
    nec = NecessityEnsemble(state_dim, action_dim).to(config.device)

    sd = torch.load(config.checkpoint, map_location=config.device)
    qf.load_state_dict(sd["qf"])
    actor.load_state_dict(sd["actor"])
    nec.load_state_dict(sd["necessity_ensemble"])
    qf.eval()
    actor.eval()
    nec.eval()

    # Probed states from compute_proxy.py
    probe = np.load(config.probe_npz, allow_pickle=True)
    probe_idx = probe["idx"]
    n = min(config.n_states, len(probe_idx))
    chosen = np.random.default_rng(config.seed).choice(len(probe_idx), n, replace=False)

    # We need (qpos, qvel) per state — but probe.npz only has obs (post-normalisation).
    # For state restoration we need the raw qpos/qvel from the dataset, which D4RL
    # qlearning_dataset DOES NOT expose. We re-roll a short trajectory from each
    # probed observation by walking forward from the env reset and matching obs.
    #
    # SIMPLER: we replace the probe sampling here with a *fresh on-policy rollout*
    # under π_θ that records (s, a, qpos, qvel) at every step. We then re-evaluate
    # N_ψ on those states. This avoids the dataset-→-mujoco gap entirely.
    raise NotImplementedError(
        "Replace this with the on-policy state-collection loop. "
        "TODO: roll out π_θ for K episodes; at each step record (obs, qpos, qvel, "
        "actor_action); subsample n_states uniformly; query nec(s, a); then run "
        "paired_rollout for each. See §4.3 of paper."
    )


if __name__ == "__main__":
    main()
