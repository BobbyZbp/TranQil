"""Drop test runner for maze2d-large-dense-v2 via gymnasium + minari stack.

For each of N (s, a) pairs (collected from on-policy rollouts so we have
(qpos, qvel) for state restoration):
  1. Restore env to (qpos, qvel), roll π_θ to termination → G_base
  2. Restore env, take ã ~ κ_ε once, then π_θ → G_cf
  3. ΔG = G_base - G_cf

Output: probe_results.npz augmented with delta_g[].
Then computes Spearman ρ(N_ψ, ΔG) and ρ(Q, ΔG) — paper §4.3.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
from nca import NecessityEnsemble, PolicySupportedKernel  # noqa: E402
from nca_iql import (  # noqa: E402
    DeterministicPolicy,
    GaussianPolicy,
    TwinQ,
    compute_mean_std,
    normalize_states,
)
from nca_iql_maze2d import (  # noqa: E402
    DATASET_MAP,
    GYM_ENV_MAP,
    flatten_obs,
    load_minari_as_d4rl,
)


@dataclass
class DropTestConfig:
    checkpoint: str
    env: str = "maze2d-large-dense-v2"
    n_states: int = 200
    n_cf_samples: int = 5
    max_episode_steps: int = 300
    kernel_eps: float = 0.3
    actor_jitter: float = 0.1
    iql_deterministic: bool = False
    output: str = "droptest_results.npz"
    device: str = "cuda"
    seed: int = 0


def get_qpos_qvel(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
    """Try several MuJoCo-state extraction APIs used by PointMaze variants."""
    inner = env.unwrapped
    # PointMaze in gymnasium-robotics wraps a PointEnv internally
    if hasattr(inner, "point_env"):
        data = inner.point_env.data
        return data.qpos.copy(), data.qvel.copy()
    if hasattr(inner, "data"):
        return inner.data.qpos.copy(), inner.data.qvel.copy()
    if hasattr(inner, "sim"):
        return inner.sim.data.qpos.copy(), inner.sim.data.qvel.copy()
    raise RuntimeError("don't know how to read qpos/qvel from this env")


def set_qpos_qvel(env: gym.Env, qpos: np.ndarray, qvel: np.ndarray) -> None:
    inner = env.unwrapped
    if hasattr(inner, "point_env"):
        inner.point_env.set_state(qpos, qvel)
        return
    if hasattr(inner, "set_state"):
        inner.set_state(qpos, qvel)
        return
    raise RuntimeError("don't know how to set state on this env")


def get_obs_from_state(env: gym.Env, qpos: np.ndarray, qvel: np.ndarray, desired_goal: np.ndarray) -> dict:
    """Manually construct the Dict obs that matches PointMaze's format.
    PointMaze's observation is {observation: [pos, vel] (4,), achieved_goal: pos (2,), desired_goal: (2,)}.
    """
    return {
        "observation": np.concatenate([qpos, qvel]).astype(np.float64),
        "achieved_goal": qpos.copy().astype(np.float64),
        "desired_goal": desired_goal.copy().astype(np.float64),
    }


def actor_step(actor, obs_flat, state_mean, state_std, max_action, device, stochastic):
    s = (obs_flat - state_mean) / state_std
    s_t = torch.as_tensor(s.reshape(1, -1), dtype=torch.float32, device=device)
    out = actor(s_t)
    if isinstance(out, torch.distributions.Distribution):
        a = out.sample() if stochastic else out.mean
    else:
        a = out
    a = a.clamp(-max_action, max_action)
    return a.detach().cpu().numpy().flatten()


def rollout_from_obs(env, actor, current_obs, state_mean, state_std, max_action, device, max_steps,
                     initial_action=None, stochastic=False) -> float:
    """Single-rollout return starting from a known obs (env already at the right state)."""
    flat = flatten_obs(current_obs)
    G = 0.0
    steps = 0
    if initial_action is not None:
        new_obs, r, term, trunc, _ = env.step(initial_action)
        G += r
        steps += 1
        if term or trunc:
            return G
        flat = flatten_obs(new_obs)
    for _ in range(max_steps - steps):
        a = actor_step(actor, flat, state_mean, state_std, max_action, device, stochastic)
        new_obs, r, term, trunc, _ = env.step(a)
        G += r
        if term or trunc:
            break
        flat = flatten_obs(new_obs)
    return float(G)


def main(config: DropTestConfig):
    # 1. Build env
    env = gym.make(GYM_ENV_MAP[config.env], max_episode_steps=config.max_episode_steps)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 2. Load dataset for normalisation stats
    dataset = load_minari_as_d4rl(DATASET_MAP[config.env])
    state_dim = dataset["observations"].shape[1]
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)

    # 3. Load networks
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

    @torch.no_grad()
    def sample_action(state_t: torch.Tensor) -> torch.Tensor:
        out = actor(state_t)
        if isinstance(out, torch.distributions.Distribution):
            return out.sample().clamp(-max_action, max_action)
        return (out + config.actor_jitter * torch.randn_like(out)).clamp(
            -max_action, max_action
        )

    kernel = PolicySupportedKernel(eps=config.kernel_eps)
    rng = np.random.default_rng(config.seed)

    # 4. Collect (obs_dict, action, qpos, qvel, desired_goal) by rolling out the actor
    print(f"[droptest] collecting {config.n_states} on-policy states...")
    collected = []  # (obs_dict, action, qpos, qvel)
    seed_iter = config.seed
    while len(collected) < config.n_states * 4:
        cur_obs, _info = env.reset(seed=seed_iter)
        seed_iter += 1
        for t in range(config.max_episode_steps):
            flat = flatten_obs(cur_obs)
            qpos, qvel = get_qpos_qvel(env)
            a = actor_step(actor, flat, state_mean, state_std, max_action, config.device, stochastic=False)
            collected.append((dict(cur_obs), a.copy(), qpos.copy(), qvel.copy()))
            cur_obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if len(collected) >= config.n_states * 4:
            break

    # subsample uniformly
    pick = rng.choice(len(collected), config.n_states, replace=False)
    chosen = [collected[i] for i in pick]
    print(f"[droptest] picked {config.n_states} states from {len(collected)} on-policy steps")

    # 5. For each (s, a), compute N_ψ, Q, paired ΔG
    n_psi_list, q_list, delta_g_list = [], [], []
    for i, (obs_dict, a, qpos, qvel) in enumerate(chosen):
        flat = flatten_obs(obs_dict)
        s_norm = (flat - state_mean) / state_std
        s_t = torch.as_tensor(s_norm.reshape(1, -1), dtype=torch.float32, device=config.device)
        a_t = torch.as_tensor(a.reshape(1, -1), dtype=torch.float32, device=config.device)
        with torch.no_grad():
            n_psi = nec.mean(s_t, a_t).item()
            q = qf(s_t, a_t).item()
            tilde = kernel.sample(sample_action, s_t, a_t, n_samples=config.n_cf_samples)
            # tilde: (1, n_cf_samples, action_dim)
            cf_actions = tilde[0].cpu().numpy().clip(-max_action, max_action)

        # baseline: restore state, take a, then actor (deterministic)
        env.reset()
        set_qpos_qvel(env, qpos, qvel)
        G_base = rollout_from_obs(env, actor, obs_dict, state_mean, state_std, max_action,
                                  config.device, config.max_episode_steps, initial_action=a)

        # counterfactual: average over n_cf_samples rollouts
        G_cf_list = []
        for k in range(config.n_cf_samples):
            env.reset()
            set_qpos_qvel(env, qpos, qvel)
            G_cf_k = rollout_from_obs(env, actor, obs_dict, state_mean, state_std, max_action,
                                      config.device, config.max_episode_steps,
                                      initial_action=cf_actions[k])
            G_cf_list.append(G_cf_k)
        G_cf = float(np.mean(G_cf_list))

        delta_g = G_base - G_cf
        n_psi_list.append(n_psi)
        q_list.append(q)
        delta_g_list.append(delta_g)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{config.n_states}] G_base={G_base:.2f} G_cf={G_cf:.2f} ΔG={delta_g:+.3f}")

    n_psi_arr = np.array(n_psi_list)
    q_arr = np.array(q_list)
    delta_g_arr = np.array(delta_g_list)

    np.savez(config.output, n_psi=n_psi_arr, q_sa=q_arr, delta_g=delta_g_arr,
             env=config.env, kernel_eps=config.kernel_eps)
    print(f"[droptest] saved {config.output}")

    # 6. Spearman ρ
    rho_npsi, p_npsi = spearmanr(n_psi_arr, delta_g_arr)
    rho_q, p_q = spearmanr(q_arr, delta_g_arr)
    pcc_npsi = pearsonr(n_psi_arr, delta_g_arr)[0]
    pcc_q = pearsonr(q_arr, delta_g_arr)[0]
    print()
    print("=" * 60)
    print(f"  N_ψ vs ΔG : spearman = {rho_npsi:+.4f} (p={p_npsi:.2e})  pearson = {pcc_npsi:+.4f}")
    print(f"  Q_φ vs ΔG : spearman = {rho_q:+.4f} (p={p_q:.2e})  pearson = {pcc_q:+.4f}")
    print(f"  ΔG mean = {delta_g_arr.mean():+.3f}, std = {delta_g_arr.std():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--env", type=str, default="maze2d-large-dense-v2")
    p.add_argument("--n_states", type=int, default=200)
    p.add_argument("--n_cf_samples", type=int, default=5)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--kernel_eps", type=float, default=0.3)
    p.add_argument("--actor_jitter", type=float, default=0.1)
    p.add_argument("--iql_deterministic", action="store_true")
    p.add_argument("--output", type=str, default="droptest_results.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(DropTestConfig(**vars(args)))
