# NCA-IQL: forked from CORL/algorithms/finetune/iql.py
# Adds Stage 2 (NecessityHead training) on top of standard IQL pretrain.
# Original source: https://github.com/gwthomas/IQL-PyTorch
# IQL paper:       https://arxiv.org/pdf/2110.06169.pdf
# NCA-T paper:     paper/ncat.tex (this repo)
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

from nca import (
    NecessityEnsemble,
    PolicySupportedKernel,
    compute_offline_proxy,
)

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "antmaze-umaze-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Eval environment seed
    eval_freq: int = int(5e4)  # How often (time steps) we evaluate
    n_episodes: int = 100  # How many episodes run during evaluation
    offline_iterations: int = int(1e6)  # Stage 1: standard IQL pretrain
    online_iterations: int = int(0)  # Tier 2 (future work) — kept at 0 here
    # NCA-T (paper §4.1, §4.2)
    necessity_iterations: int = int(2e5)  # Stage 2: train N_ψ against N^off proxy
    necessity_lr: float = 3e-4            # ψ optimiser lr
    kernel_eps: float = 0.05              # κ_ε margin in normalised action space
    kernel_L: int = 8                     # MC samples per (s,a) for proxy
    necessity_hidden_dim: int = 256
    necessity_n_hidden: int = 2
    actor_jitter: float = 0.1             # σ for deterministic-actor sampling in κ
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    actor_dropout: float = 0.0  # Dropout in actor network
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    expl_noise: float = 0.03  # Std of Gaussian exploration noise
    noise_clip: float = 0.5  # Range to clip noise
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    # Wandb logging
    project: str = "NCA-T"
    group: str = "NCA-IQL-D4RL"
    name: str = "NCA-IQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            # Only append name subdir if path doesn't already contain stage1 checkpoints
            # (i.e. not a resume path pointing directly at the checkpoint dir)
            import glob as _glob
            if not _glob.glob(os.path.join(self.checkpoints_path, "stage1_step*.pt")):
                self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)
        # raise NotImplementedError


def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save(wandb.run.dir + "/*")


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        # NCA-T extensions (set after Stage 1, used in Stage 2)
        necessity_ensemble: Optional[nn.Module] = None,
        necessity_optimizer: Optional[torch.optim.Optimizer] = None,
        kernel: Optional[object] = None,
        kernel_L: int = 8,
        actor_jitter: float = 0.1,
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        # NCA-T
        self.necessity_ensemble = necessity_ensemble
        self.necessity_optimizer = necessity_optimizer
        self.kernel = kernel
        self.kernel_L = kernel_L
        self.actor_jitter = actor_jitter

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    @torch.no_grad()
    def _sample_actor(self, state: torch.Tensor) -> torch.Tensor:
        """Sample one action per state from π_θ(·|s). For deterministic
        actors we add Gaussian jitter so the rejection kernel has support."""
        out = self.actor(state)
        if isinstance(out, torch.distributions.Distribution):
            action = out.sample()
        else:
            action = out + self.actor_jitter * torch.randn_like(out)
        return action.clamp(-self.max_action, self.max_action)

    def update_necessity(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        """Stage 2: regress N_ψ against the offline Q-margin proxy
        (paper §4.2, eq. 5). Q_φ is frozen in this stage — gradients only
        flow into ψ because compute_offline_proxy is decorated @torch.no_grad."""
        assert self.necessity_ensemble is not None, "necessity_ensemble not attached"
        assert self.necessity_optimizer is not None, "necessity_optimizer not attached"
        assert self.kernel is not None, "kernel not attached"

        target, kernel_violation = compute_offline_proxy(
            self.qf,
            sample_action=self._sample_actor,
            kernel=self.kernel,
            states=observations,
            actions=actions,
            n_samples=self.kernel_L,
        )

        pred1, pred2 = self.necessity_ensemble(observations, actions)
        loss = F.mse_loss(pred1, target) + F.mse_loss(pred2, target)

        self.necessity_optimizer.zero_grad()
        loss.backward()
        self.necessity_optimizer.step()

        log_dict["nec_loss"] = loss.item()
        log_dict["nec_target_mean"] = target.mean().item()
        log_dict["nec_target_std"] = target.std().item()
        log_dict["nec_disagreement"] = (pred1 - pred2).abs().mean().item()
        log_dict["nec_kernel_violation"] = kernel_violation

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        sd = {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }
        if self.necessity_ensemble is not None:
            sd["necessity_ensemble"] = self.necessity_ensemble.state_dict()
        if self.necessity_optimizer is not None:
            sd["necessity_optimizer"] = self.necessity_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]

        if self.necessity_ensemble is not None and "necessity_ensemble" in state_dict:
            self.necessity_ensemble.load_state_dict(state_dict["necessity_ensemble"])
        if self.necessity_optimizer is not None and "necessity_optimizer" in state_dict:
            self.necessity_optimizer.load_state_dict(state_dict["necessity_optimizer"])


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)
    eval_env = gym.make(config.env)

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)

    max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env)

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
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.offline_iterations,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations: List[float] = []

    # =====================================================================
    # Stage 1 — standard IQL offline pretrain (paper §4.2)
    # =====================================================================
    # Resume from latest periodic checkpoint if available; skip Stage 1 if fully done
    stage1_start = 0
    if config.checkpoints_path is not None and os.path.isdir(config.checkpoints_path):
        stage1_final = os.path.join(config.checkpoints_path, "stage1.pt")
        if os.path.exists(stage1_final):
            trainer.load_state_dict(torch.load(stage1_final, map_location=config.device))
            actor = trainer.actor
            stage1_start = int(config.offline_iterations)
            print(f"[resume] stage1.pt found, skipping Stage 1")
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
            eval_scores, success_rate = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = eval_env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            eval_log: Dict[str, float] = {
                "eval/d4rl_normalized_score": normalized_eval_score,
            }
            if is_env_with_goal:
                eval_log["eval/success_rate"] = success_rate
            print(
                f"[stage1 step {t+1}] eval raw={eval_score:.3f}  "
                f"D4RL={normalized_eval_score:.3f}"
                + (f"  success={success_rate:.3f}" if is_env_with_goal else "")
            )
            wandb.log(eval_log, step=trainer.total_it)
            if config.checkpoints_path is not None:
                ckpt = os.path.join(config.checkpoints_path, f"stage1_step{t+1}.pt")
                torch.save({"trainer": trainer.state_dict(), "step": t + 1}, ckpt)

    if config.checkpoints_path is not None:
        ckpt_path = os.path.join(config.checkpoints_path, "stage1.pt")
        torch.save(trainer.state_dict(), ckpt_path)
        print(f"[stage1] saved {ckpt_path}")

    # =====================================================================
    # Stage 2 — train NecessityHead via Q-margin proxy regression
    # (paper §4.2 eq. 5; Q_φ frozen via @torch.no_grad inside proxy fn)
    # =====================================================================
    print("---------------------------------------")
    print(
        f"STAGE 2: NecessityHead regression  "
        f"({int(config.necessity_iterations)} steps, L={config.kernel_L}, "
        f"eps={config.kernel_eps})"
    )
    print("---------------------------------------")

    necessity_ensemble = NecessityEnsemble(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.necessity_hidden_dim,
        n_hidden=config.necessity_n_hidden,
    ).to(config.device)
    necessity_optimizer = torch.optim.Adam(
        necessity_ensemble.parameters(), lr=config.necessity_lr
    )
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
                f"[stage2 step {t+1}] "
                f"loss={log_dict['nec_loss']:.4f}  "
                f"target_mean={log_dict['nec_target_mean']:+.4f}  "
                f"target_std={log_dict['nec_target_std']:.4f}  "
                f"disagree={log_dict['nec_disagreement']:.4f}"
            )
            if config.checkpoints_path is not None:
                ckpt = os.path.join(config.checkpoints_path, f"stage2_step{t+1}.pt")
                torch.save({"trainer": trainer.state_dict(), "step": t + 1}, ckpt)

    if config.checkpoints_path is not None:
        ckpt_path = os.path.join(config.checkpoints_path, "stage2.pt")
        torch.save(trainer.state_dict(), ckpt_path)
        print(f"[stage2] saved {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    train()
