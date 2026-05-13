"""Drop test for maze2d with 8-direction effect-grouping counterfactuals.

Difference vs `drop_test_maze2d.py`:
  - Each cf candidate is simulated for one env step from the current
    (qpos, qvel). The resulting positional displacement is binned into 8
    angular sectors. We keep only candidates whose bin differs from the
    bin of the original (dataset/policy) action. This filters cf actions by
    *behavioural effect* rather than action-space distance.

The N_ψ network and Q_φ network are NOT modified or retrained. Only the
counterfactual distribution at evaluation changes. Comparing ρ(N_ψ, ΔG_eff)
to ρ(N_ψ, ΔG_action_far) directly tests the hypothesis that effect-space
counterfactuals correlate better with ground-truth necessity.
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
)
from nca_iql_maze2d import (  # noqa: E402
    DATASET_MAP,
    GYM_ENV_MAP,
    flatten_obs,
    load_minari_as_d4rl,
)


@dataclass
class DropTestEffectConfig:
    checkpoint: str
    env: str = "maze2d-large-dense-v2"
    n_states: int = 200
    n_cf_samples: int = 5
    oversample_factor: int = 8           # how many extra candidates to draw before filtering
    max_episode_steps: int = 300
    kernel_eps: float = 0.05
    actor_jitter: float = 0.1
    iql_deterministic: bool = False
    output: str = "droptest_effect_results.npz"
    device: str = "cuda"
    seed: int = 0


# ---------------- env state helpers (same as drop_test_maze2d.py) ----------------

def get_qpos_qvel(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
    inner = env.unwrapped
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


# ---------------- effect-bin helper ----------------

def effect_bin_8(env: gym.Env, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray,
                 dead_eps: float = 1e-4) -> int:
    """Set env state, simulate one step, return a discretized direction bin.

    Returns an integer in [0..7] for one of 8 angular sectors of the (Δx, Δy)
    displacement, or -1 if the move is too small to assign a direction
    (used as a 'no-effect' sentinel).
    """
    set_qpos_qvel(env, qpos, qvel)
    obs_next, _r, _term, _trunc, _info = env.step(action)
    pos_next = np.asarray(obs_next["observation"][:2], dtype=np.float64)
    delta = pos_next - np.asarray(qpos, dtype=np.float64)
    if float(np.linalg.norm(delta)) < dead_eps:
        return -1
    angle = np.arctan2(delta[1], delta[0])  # in [-π, π]
    # Map angle to bin 0..7
    bin_idx = int(np.floor(((angle + np.pi) / (2.0 * np.pi)) * 8.0)) % 8
    return bin_idx


# ---------------- actor / rollout helpers ----------------

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


def rollout_from_obs(env, actor, current_obs, state_mean, state_std, max_action,
                    device, max_steps, initial_action=None, stochastic=False) -> float:
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


# ---------------- main ----------------

def main(config: DropTestEffectConfig):
    env = gym.make(GYM_ENV_MAP[config.env], max_episode_steps=config.max_episode_steps)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = load_minari_as_d4rl(DATASET_MAP[config.env])
    state_dim = dataset["observations"].shape[1]
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)

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

    # Collect on-policy states
    print(f"[droptest-effect] collecting {config.n_states} on-policy states...")
    collected = []
    seed_iter = config.seed
    while len(collected) < config.n_states * 4:
        cur_obs, _info = env.reset(seed=seed_iter)
        seed_iter += 1
        for t in range(config.max_episode_steps):
            qpos, qvel = get_qpos_qvel(env)
            flat = flatten_obs(cur_obs)
            a = actor_step(actor, flat, state_mean, state_std, max_action, config.device, stochastic=False)
            collected.append((dict(cur_obs), a.copy(), qpos.copy(), qvel.copy()))
            cur_obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if len(collected) >= config.n_states * 4:
            break

    pick = rng.choice(len(collected), config.n_states, replace=False)
    chosen = [collected[i] for i in pick]
    print(f"[droptest-effect] picked {config.n_states} from {len(collected)} on-policy steps")

    # Main loop
    n_psi_list, q_list, delta_g_list = [], [], []
    n_eff_far_list = []      # how many effect-different cf were found per state
    a_bin_dist = []          # original-action bin distribution

    oversample = config.n_cf_samples * config.oversample_factor

    for i, (obs_dict, a, qpos, qvel) in enumerate(chosen):
        flat = flatten_obs(obs_dict)
        s_norm = (flat - state_mean) / state_std
        s_t = torch.as_tensor(s_norm.reshape(1, -1), dtype=torch.float32, device=config.device)
        a_t = torch.as_tensor(a.reshape(1, -1), dtype=torch.float32, device=config.device)

        with torch.no_grad():
            n_psi = nec.mean(s_t, a_t).item()
            q = qf(s_t, a_t).item()
            tilde = kernel.sample(sample_action, s_t, a_t, n_samples=oversample)
            cf_candidates = tilde[0].cpu().numpy().clip(-max_action, max_action)

        # Bin of the original action
        a_bin = effect_bin_8(env, qpos, qvel, a)
        a_bin_dist.append(a_bin)

        # Filter cf candidates to effect-different bins
        filtered = []
        for cand in cf_candidates:
            cand_bin = effect_bin_8(env, qpos, qvel, cand)
            if cand_bin != a_bin and cand_bin != -1:
                filtered.append(cand)
            if len(filtered) >= config.n_cf_samples:
                break

        n_eff_far = len(filtered)
        n_eff_far_list.append(n_eff_far)
        if n_eff_far == 0:
            # fall back so we don't crash
            filtered = [cf_candidates[k] for k in range(config.n_cf_samples)]

        # Baseline rollout — restore state again before stepping
        env.reset()
        set_qpos_qvel(env, qpos, qvel)
        G_base = rollout_from_obs(env, actor, obs_dict, state_mean, state_std, max_action,
                                  config.device, config.max_episode_steps, initial_action=a)

        # Counterfactual rollouts on the filtered candidates
        G_cf_list = []
        for cand in filtered:
            env.reset()
            set_qpos_qvel(env, qpos, qvel)
            G_cf_k = rollout_from_obs(env, actor, obs_dict, state_mean, state_std, max_action,
                                      config.device, config.max_episode_steps, initial_action=cand)
            G_cf_list.append(G_cf_k)
        G_cf = float(np.mean(G_cf_list))

        delta_g = G_base - G_cf
        n_psi_list.append(n_psi)
        q_list.append(q)
        delta_g_list.append(delta_g)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{config.n_states}] G_base={G_base:.2f} G_cf={G_cf:.2f} "
                  f"ΔG={delta_g:+.3f}  n_eff_far={n_eff_far}/{config.n_cf_samples}")

    n_psi_arr = np.array(n_psi_list)
    q_arr = np.array(q_list)
    delta_g_arr = np.array(delta_g_list)
    n_eff_far_arr = np.array(n_eff_far_list)
    a_bin_arr = np.array(a_bin_dist)

    np.savez(config.output,
             n_psi=n_psi_arr, q_sa=q_arr, delta_g=delta_g_arr,
             n_eff_far=n_eff_far_arr, a_bin=a_bin_arr,
             env=config.env, kernel_eps=config.kernel_eps,
             n_cf_samples=config.n_cf_samples, oversample_factor=config.oversample_factor)
    print(f"[droptest-effect] saved {config.output}")

    # Diagnostics
    rho_npsi, p_npsi = spearmanr(n_psi_arr, delta_g_arr)
    rho_q, p_q = spearmanr(q_arr, delta_g_arr)
    pcc_npsi = pearsonr(n_psi_arr, delta_g_arr)[0]
    pcc_q = pearsonr(q_arr, delta_g_arr)[0]

    print()
    print("=" * 60)
    print(f"  N_ψ vs ΔG : spearman = {rho_npsi:+.4f} (p={p_npsi:.2e})  pearson = {pcc_npsi:+.4f}")
    print(f"  Q_φ vs ΔG : spearman = {rho_q:+.4f} (p={p_q:.2e})  pearson = {pcc_q:+.4f}")
    print(f"  ΔG mean = {delta_g_arr.mean():+.3f}, std = {delta_g_arr.std():.3f}")
    print(f"  n_eff_far per state: mean={n_eff_far_arr.mean():.2f}/{config.n_cf_samples}, "
          f"min={n_eff_far_arr.min()}, frac_zero={(n_eff_far_arr == 0).mean():.2%}")
    print(f"  a_bin distribution: " + ", ".join(
        f"{b}={int((a_bin_arr == b).sum())}" for b in range(-1, 8)
    ))
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--env", type=str, default="maze2d-large-dense-v2")
    p.add_argument("--n_states", type=int, default=200)
    p.add_argument("--n_cf_samples", type=int, default=5)
    p.add_argument("--oversample_factor", type=int, default=8)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--kernel_eps", type=float, default=0.05)
    p.add_argument("--actor_jitter", type=float, default=0.1)
    p.add_argument("--iql_deterministic", action="store_true")
    p.add_argument("--output", type=str, default="droptest_effect_results.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(DropTestEffectConfig(**vars(args)))
