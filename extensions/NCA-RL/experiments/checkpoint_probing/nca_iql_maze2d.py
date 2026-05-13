"""Stage 1+2 NCA-IQL training on D4RL pointmaze datasets via minari + gymnasium.

Adapter for `nca_iql.py` that:
  1. Loads dataset via minari (D4RL/pointmaze/large-dense-v2 style)
  2. Flattens Dict obs (achieved_goal/desired_goal/observation) into a single vector
  3. Wraps gymnasium env to expose gym-style API expected by ImplicitQLearning

Eval is done by rolling out a fresh gymnasium env; we don't compute D4RL
normalized scores (they don't apply to dense pointmaze), only mean return.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers PointMaze envs)
import minari
import numpy as np
import torch
import wandb

# Import everything from the existing nca_iql module so we reuse architectures
import sys
sys.path.insert(0, os.path.dirname(__file__))
from nca_iql import (  # noqa: E402
    DeterministicPolicy,
    GaussianPolicy,
    ImplicitQLearning,
    ReplayBuffer,
    TwinQ,
    ValueFunction,
    compute_mean_std,
    normalize_states,
    set_seed,
    wandb_init,
)
from nca import NecessityEnsemble, PolicySupportedKernel  # noqa: E402


# Map env names → minari datasets
DATASET_MAP = {
    "maze2d-large-dense-v2": "D4RL/pointmaze/large-dense-v2",
    "maze2d-medium-dense-v2": "D4RL/pointmaze/medium-dense-v2",
    "maze2d-large-v2": "D4RL/pointmaze/large-v2",
    "maze2d-medium-v2": "D4RL/pointmaze/medium-v2",
}
GYM_ENV_MAP = {
    "maze2d-large-dense-v2": "PointMaze_LargeDense-v3",
    "maze2d-medium-dense-v2": "PointMaze_MediumDense-v3",
    "maze2d-large-v2": "PointMaze_Large-v3",
    "maze2d-medium-v2": "PointMaze_Medium-v3",
}


def flatten_obs(obs_dict) -> np.ndarray:
    """Concatenate observation + desired_goal into a flat vector."""
    return np.concatenate(
        [obs_dict["observation"], obs_dict["desired_goal"]], axis=-1
    )


def load_minari_as_d4rl(dataset_name: str) -> dict:
    """Convert a minari dataset to the d4rl qlearning_dataset format.

    Returns dict with keys: observations, actions, rewards, next_observations,
    terminals.
    """
    print(f"[minari] loading {dataset_name}")
    ds = minari.load_dataset(dataset_name, download=True)

    obs_list, act_list, rew_list, next_obs_list, term_list = [], [], [], [], []
    for ep in ds.iterate_episodes():
        obs = flatten_obs(ep.observations)  # (T+1, D)
        actions = ep.actions  # (T, A)
        rewards = ep.rewards  # (T,)
        terminations = ep.terminations  # (T,)
        truncations = ep.truncations  # (T,)

        T = len(actions)
        obs_list.append(obs[:T])
        next_obs_list.append(obs[1:T + 1])
        act_list.append(actions)
        rew_list.append(rewards)
        # pointmaze datasets only have truncations (no terminations). Treat
        # truncations as terminals so Q-learning bootstrap is bounded.
        done = np.logical_or(terminations, truncations).astype(np.float32)
        term_list.append(done)

    out = {
        "observations": np.concatenate(obs_list).astype(np.float32),
        "actions": np.concatenate(act_list).astype(np.float32),
        "rewards": np.concatenate(rew_list).astype(np.float32),
        "next_observations": np.concatenate(next_obs_list).astype(np.float32),
        "terminals": np.concatenate(term_list).astype(np.float32),
    }

    # Reward normalisation for dense pointmaze: scale per-traj returns into
    # a comparable range to antmaze sparse, so IQL hyperparameters (beta, tau)
    # don't blow up Q values. Following CORL's locomotion convention:
    # divide by (max_ret - min_ret), multiply by typical episode length.
    # CORL-style reward normalisation for locomotion-like dense reward:
    # divide rewards by (max_traj_return - min_traj_return), then multiply by
    # max_episode_steps (1000). Keeps per-step rewards O(1/T) so cumulative
    # discounted return is bounded.
    traj_returns = []
    cur_ret = 0.0
    for r, term in zip(out["rewards"], out["terminals"]):
        cur_ret += r
        if term > 0.5:
            traj_returns.append(cur_ret)
            cur_ret = 0.0
    if cur_ret != 0.0:
        traj_returns.append(cur_ret)
    if len(traj_returns) > 1:
        max_ret, min_ret = max(traj_returns), min(traj_returns)
        scale = (max_ret - min_ret) if (max_ret - min_ret) > 1e-3 else 1.0
        out["rewards"] /= scale
        out["rewards"] *= 1000.0
        print(f"[minari] reward scaled: traj_return [{min_ret:.2f}, {max_ret:.2f}] → /{scale:.3f}*1000")

    print(
        f"[minari] dataset: obs={out['observations'].shape}, "
        f"act={out['actions'].shape}, reward mean={out['rewards'].mean():.3f}, "
        f"terminals frac={out['terminals'].mean():.4f}"
    )
    return out


class GymnasiumToGymAdapter:
    """Tiny shim so eval rollouts use the gym-style step() returning 4-tuple."""

    def __init__(self, env, state_mean, state_std):
        self.env = env
        self.state_mean = state_mean
        self.state_std = state_std

    def reset(self):
        obs, _info = self.env.reset()
        flat = flatten_obs(obs)
        return (flat - self.state_mean) / self.state_std

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        flat = flatten_obs(obs)
        norm = (flat - self.state_mean) / self.state_std
        done = terminated or truncated
        info["TimeLimit.truncated"] = truncated and not terminated
        return norm, reward, done, info

    def seed(self, seed):
        self.env.reset(seed=seed)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space


def eval_actor_pointmaze(env, actor, device, n_episodes, seed, max_steps=300):
    actor.eval()
    returns = []
    for i in range(n_episodes):
        env.seed(seed + i)
        obs = env.reset()
        ep_ret = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                s = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=device)
                out = actor(s)
                if isinstance(out, torch.distributions.Distribution):
                    a = out.mean
                else:
                    a = out
                a = a.cpu().numpy().flatten()
            obs, r, done, _ = env.step(a)
            ep_ret += r
            if done:
                break
        returns.append(ep_ret)
    actor.train()
    return np.array(returns), 0.0


@dataclass
class TrainConfig:
    env: str = "maze2d-large-dense-v2"
    seed: int = 0
    eval_seed: int = 0
    eval_freq: int = int(5e4)
    n_episodes: int = 20
    offline_iterations: int = int(1e6)
    necessity_iterations: int = int(2e5)
    necessity_lr: float = 3e-4
    kernel_eps: float = 0.30
    kernel_L: int = 16
    necessity_hidden_dim: int = 256
    necessity_n_hidden: int = 2
    actor_jitter: float = 0.1
    checkpoints_path: Optional[str] = None
    actor_dropout: float = 0.0
    buffer_size: int = 2_000_000
    batch_size: int = 256
    discount: float = 0.99  # CORL convention; rely on reward scaling to bound Q
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7
    iql_deterministic: bool = False
    normalize: bool = True
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    device: str = "cuda"
    project: str = "NCA-T"
    group: str = "NCA-IQL-pointmaze"
    name: str = "NCA-IQL"

    def __post_init__(self):
        import uuid
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            import glob as _glob
            has_stage1 = (
                _glob.glob(os.path.join(self.checkpoints_path, "stage1_step*.pt"))
                or os.path.exists(os.path.join(self.checkpoints_path, "stage1.pt"))
            )
            if not has_stage1:
                self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def train(config: TrainConfig):
    # 1. Load minari dataset
    dataset_name = DATASET_MAP[config.env]
    gym_env_name = GYM_ENV_MAP[config.env]
    dataset = load_minari_as_d4rl(dataset_name)

    state_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    eval_env_raw = gym.make(gym_env_name, max_episode_steps=300)
    eval_env = GymnasiumToGymAdapter(eval_env_raw, state_mean, state_std)
    max_action = float(eval_env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)

    set_seed(config.seed)
    eval_env.seed(config.eval_seed)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
        if config.iql_deterministic
        else GaussianPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    trainer = ImplicitQLearning(
        max_action=max_action, actor=actor, actor_optimizer=actor_optimizer,
        q_network=q_network, q_optimizer=q_optimizer,
        v_network=v_network, v_optimizer=v_optimizer,
        discount=config.discount, tau=config.tau, device=config.device,
        beta=config.beta, iql_tau=config.iql_tau,
        max_steps=config.offline_iterations,
    )

    from dataclasses import asdict
    wandb_init(asdict(config))

    # ============================================================
    # Stage 1
    # ============================================================
    stage1_start = 0
    if config.checkpoints_path is not None and os.path.isdir(config.checkpoints_path):
        s1_final = os.path.join(config.checkpoints_path, "stage1.pt")
        if os.path.exists(s1_final):
            trainer.load_state_dict(torch.load(s1_final, map_location=config.device))
            actor = trainer.actor
            stage1_start = int(config.offline_iterations)
            print("[resume] stage1.pt found, skipping Stage 1")
        else:
            ckpts = sorted(Path(config.checkpoints_path).glob("stage1_step*.pt"))
            if ckpts:
                latest = ckpts[-1]
                sd = torch.load(latest, map_location=config.device)
                trainer.load_state_dict(sd["trainer"])
                actor = trainer.actor
                stage1_start = sd["step"]
                print(f"[resume] loaded {latest.name}, continuing from step {stage1_start}")

    print("---------------------------------------")
    print(f"STAGE 1: IQL offline pretrain  ({int(config.offline_iterations)} steps)")
    print("---------------------------------------")
    for t in range(stage1_start, int(config.offline_iterations)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        log_dict["stage1_iter"] = t
        wandb.log(log_dict, step=trainer.total_it)

        if (t + 1) % config.eval_freq == 0:
            returns, _ = eval_actor_pointmaze(
                eval_env, actor, config.device, config.n_episodes, config.seed
            )
            print(f"[stage1 step {t+1}] eval_return mean={returns.mean():.2f} std={returns.std():.2f}")
            wandb.log({"eval/return": returns.mean()}, step=trainer.total_it)
            if config.checkpoints_path is not None:
                ckpt = os.path.join(config.checkpoints_path, f"stage1_step{t+1}.pt")
                torch.save({"trainer": trainer.state_dict(), "step": t + 1}, ckpt)

    if config.checkpoints_path is not None:
        torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, "stage1.pt"))
        print("[stage1] saved stage1.pt")

    # ============================================================
    # Stage 2
    # ============================================================
    print("---------------------------------------")
    print(f"STAGE 2: NecessityHead regression ({int(config.necessity_iterations)} steps, "
          f"L={config.kernel_L}, eps={config.kernel_eps})")
    print("---------------------------------------")

    necessity_ensemble = NecessityEnsemble(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=config.necessity_hidden_dim, n_hidden=config.necessity_n_hidden,
    ).to(config.device)
    necessity_optimizer = torch.optim.Adam(necessity_ensemble.parameters(), lr=config.necessity_lr)
    kernel = PolicySupportedKernel(eps=config.kernel_eps)

    trainer.necessity_ensemble = necessity_ensemble
    trainer.necessity_optimizer = necessity_optimizer
    trainer.kernel = kernel
    trainer.kernel_L = config.kernel_L
    trainer.actor_jitter = config.actor_jitter

    log_every = max(1, config.eval_freq // 5)
    stage2_start = 0
    if config.checkpoints_path is not None and os.path.isdir(config.checkpoints_path):
        s2_ckpts = sorted(Path(config.checkpoints_path).glob("stage2_step*.pt"))
        if s2_ckpts:
            latest = s2_ckpts[-1]
            sd = torch.load(latest, map_location=config.device)
            trainer.load_state_dict(sd["trainer"])
            stage2_start = sd["step"]
            print(f"[resume] loaded {latest.name}, continuing stage2 from step {stage2_start}")

    for t in range(stage2_start, int(config.necessity_iterations)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        observations, actions = batch[0], batch[1]
        log_dict = {}
        trainer.update_necessity(observations, actions, log_dict)
        log_dict["stage2_iter"] = t
        wandb.log(log_dict, step=trainer.total_it + t + 1)

        if (t + 1) % log_every == 0:
            print(
                f"[stage2 step {t+1}] loss={log_dict['nec_loss']:.4f}  "
                f"target_mean={log_dict['nec_target_mean']:+.4f}  "
                f"target_std={log_dict['nec_target_std']:.4f}  "
                f"disagree={log_dict['nec_disagreement']:.4f}"
            )
            if config.checkpoints_path is not None:
                ckpt = os.path.join(config.checkpoints_path, f"stage2_step{t+1}.pt")
                torch.save({"trainer": trainer.state_dict(), "step": t + 1}, ckpt)

    if config.checkpoints_path is not None:
        torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, "stage2.pt"))
        print("[stage2] saved stage2.pt")

    # Save state_mean/std for later use by compute_proxy
    if config.checkpoints_path is not None:
        np.savez(
            os.path.join(config.checkpoints_path, "norm.npz"),
            state_mean=state_mean, state_std=state_std,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="maze2d-large-dense-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offline_iterations", type=int, default=int(1e6))
    p.add_argument("--necessity_iterations", type=int, default=int(2e5))
    p.add_argument("--checkpoints_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--kernel_eps", type=float, default=0.30)
    p.add_argument("--kernel_L", type=int, default=16)
    p.add_argument("--actor_jitter", type=float, default=0.1)
    args = p.parse_args()

    cfg = TrainConfig(
        env=args.env,
        seed=args.seed,
        offline_iterations=args.offline_iterations,
        necessity_iterations=args.necessity_iterations,
        checkpoints_path=args.checkpoints_path,
        device=args.device,
        kernel_eps=args.kernel_eps,
        kernel_L=args.kernel_L,
        actor_jitter=args.actor_jitter,
    )
    train(cfg)
