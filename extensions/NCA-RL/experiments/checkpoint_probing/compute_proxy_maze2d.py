"""Compute necessity scores on N sampled (s, a) from a minari pointmaze dataset.

Mirror of `compute_proxy.py` adapted for the gymnasium + minari stack used by
`nca_iql_maze2d.py`.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import minari
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from nca import (  # noqa: E402
    NecessityEnsemble,
    PolicySupportedKernel,
    compute_offline_proxy,
)
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
class ProbeConfig:
    checkpoint: str
    env: str = "maze2d-large-dense-v2"
    n_states: int = 1000
    kernel_eps: float = 0.05
    kernel_L: int = 8
    actor_jitter: float = 0.1
    iql_deterministic: bool = False
    output: str = "probe_results.npz"
    device: str = "cuda"
    seed: int = 0


def main(config: ProbeConfig):
    gym_env_name = GYM_ENV_MAP[config.env]
    env = gym.make(gym_env_name, max_episode_steps=300)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    dataset = load_minari_as_d4rl(DATASET_MAP[config.env])
    state_dim = dataset["observations"].shape[1]

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    obs_norm = normalize_states(dataset["observations"], state_mean, state_std)
    actions_np = dataset["actions"]

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

    rng = np.random.default_rng(config.seed)
    idx = rng.choice(len(obs_norm), config.n_states, replace=False)
    s = torch.as_tensor(obs_norm[idx], dtype=torch.float32, device=config.device)
    a = torch.as_tensor(actions_np[idx], dtype=torch.float32, device=config.device)

    @torch.no_grad()
    def sample_action(state: torch.Tensor) -> torch.Tensor:
        out = actor(state)
        if isinstance(out, torch.distributions.Distribution):
            return out.sample().clamp(-max_action, max_action)
        return (out + config.actor_jitter * torch.randn_like(out)).clamp(
            -max_action, max_action
        )

    kernel = PolicySupportedKernel(eps=config.kernel_eps)

    with torch.no_grad():
        n_off, _violation = compute_offline_proxy(
            qf, sample_action, kernel, s, a, n_samples=config.kernel_L
        )
        n_psi = nec.mean(s, a)
        q_sa = qf(s, a)

    np.savez(
        config.output,
        states=s.cpu().numpy(),
        actions=a.cpu().numpy(),
        idx=idx,
        n_off=n_off.cpu().numpy(),
        n_psi=n_psi.cpu().numpy(),
        q_sa=q_sa.cpu().numpy(),
        env=config.env,
        kernel_eps=config.kernel_eps,
        kernel_L=config.kernel_L,
    )
    print(f"[probe] saved {config.output}")
    print(
        f"  N^off  mean={n_off.mean():+.4f}  std={n_off.std():.4f}\n"
        f"  N_ψ    mean={n_psi.mean():+.4f}  std={n_psi.std():.4f}\n"
        f"  Q_φ    mean={q_sa.mean():+.4f}  std={q_sa.std():.4f}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--env", type=str, default="maze2d-large-dense-v2")
    p.add_argument("--n_states", type=int, default=1000)
    p.add_argument("--kernel_eps", type=float, default=0.05)
    p.add_argument("--kernel_L", type=int, default=8)
    p.add_argument("--actor_jitter", type=float, default=0.1)
    p.add_argument("--iql_deterministic", action="store_true")
    p.add_argument("--output", type=str, default="probe_results.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = ProbeConfig(**vars(args))
    main(cfg)
