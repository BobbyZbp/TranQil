"""Evaluation helpers for QT policy checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .anchor_data import resolve_anchor_eval_targets
from .anchor_registry import get_anchor_task_spec
from .checkpoints import load_checkpoint
from .cli_overrides import apply_eval_overrides
from .config import load_experiment_config
from .data.normalization import NormalizationStats
from .data.registry import TaskSpec
from .models import AnchorDecisionTransformer, AnchorDoubleQCritic
from .model_factory import build_actor_from_config
from .runtime import resolve_device, write_json


@dataclass(frozen=True)
class EvaluationSession:
    """Loaded checkpoint state plus the resolved evaluation config."""

    actor: torch.nn.Module
    task_spec: TaskSpec
    stats: NormalizationStats
    config: dict[str, Any]
    device: torch.device
    checkpoint_step: int
    checkpoint_path: Path
    critic: torch.nn.Module | None = None
    state_mean: np.ndarray | None = None
    state_std: np.ndarray | None = None
    mode: str = "standard"


@dataclass
class _RolloutBuffers:
    """Rolling sequence buffers used for autoregressive policy rollouts."""

    observations: torch.Tensor
    returns_to_go: torch.Tensor
    timesteps: torch.Tensor
    attention_mask: torch.Tensor


def normalize_episode_returns(env_name: str, episode_returns: list[float]) -> list[float]:
    """Convert raw returns into D4RL normalized scores on a 0-100 scale."""

    if not episode_returns:
        return []

    env = create_gym_env(env_name)
    try:
        if not hasattr(env, "get_normalized_score"):
            return []
        return [float(env.get_normalized_score(ret) * 100.0) for ret in episode_returns]
    finally:
        if hasattr(env, "close"):
            env.close()


def unwrap_reset_output(reset_output):
    """Normalize Gym reset outputs across API versions."""

    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def unwrap_step_output(step_output):
    """Normalize Gym step outputs across API versions."""

    if len(step_output) == 5:
        observation, reward, terminated, truncated, info = step_output
        return observation, reward, bool(terminated or truncated), info
    observation, reward, done, info = step_output
    return observation, reward, bool(done), info


def create_gym_env(env_name: str, *, action_seed: int | None = None):
    """Create one live Gym/D4RL environment with optional action-space seeding."""

    import d4rl  # noqa: F401
    import gym

    env = gym.make(env_name)
    if action_seed is not None and hasattr(env.action_space, "seed"):
        env.action_space.seed(action_seed)
    return env


def reset_env(env, seed: int):
    """Reset one environment with a stable seed across Gym API versions."""

    try:
        reset_output = env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        reset_output = env.reset()
    return unwrap_reset_output(reset_output)


def build_evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the evaluation config shape shared by trainer and CLI scripts."""

    if bool(config.get("anchor", {}).get("enabled", False)):
        anchor_config = dict(config["anchor"])
        task_spec = get_anchor_task_spec(config["env"]["name"])
        scale = anchor_config.get("scale")
        if scale is None:
            scale = task_spec.scale
        test_scale = anchor_config.get("test_scale")
        if test_scale is None:
            test_scale = task_spec.test_scale
        env_targets = anchor_config.get("env_targets") or list(task_spec.env_targets)
        return {
            "anchor_enabled": True,
            "context_length": int(anchor_config["K"]),
            "max_timestep": int(task_spec.max_ep_len),
            "episodes_max_steps": int(config["evaluation"]["max_steps"]),
            "episodes": int(config["evaluation"]["episodes"]),
            "target_return": config["evaluation"]["target_return"],
            "env_targets": [float(item) for item in env_targets],
            "scale": float(scale),
            "test_scale": float(test_scale),
            "max_ep_len": int(task_spec.max_ep_len),
        }

    evaluation_config = config["evaluation"]
    return {
        "anchor_enabled": False,
        "context_length": int(config["data"]["context_length"]),
        "max_timestep": int(config["model"]["max_timestep"]),
        "episodes_max_steps": int(evaluation_config["max_steps"]),
        "episodes": int(evaluation_config["episodes"]),
        "target_return": evaluation_config["target_return"],
    }


def resolve_target_return(stats: NormalizationStats, evaluation_config: Mapping[str, Any]) -> float:
    """Resolve the rollout target return, falling back to the dataset maximum."""

    target_return = evaluation_config["target_return"]
    if target_return is None:
        return float(stats.return_max)
    return float(target_return)


def restore_actor_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    checkpoint: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, TaskSpec, NormalizationStats, dict[str, Any]]:
    """Restore the saved actor plus its task metadata and normalization stats."""

    checkpoint_payload = checkpoint
    if checkpoint_payload is None:
        checkpoint_payload = load_checkpoint(checkpoint_path, map_location=device)

    config = dict(checkpoint_payload["config"])
    task_spec = TaskSpec(**checkpoint_payload["task_spec"])
    stats = NormalizationStats.from_cache_dict(checkpoint_payload["normalization_stats"])

    actor = build_actor_from_config(config, task_spec).to(device)
    actor.load_state_dict(checkpoint_payload["actor_state_dict"])
    actor.eval()
    return actor, task_spec, stats, config


def _restore_anchor_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    checkpoint: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module, TaskSpec, dict[str, Any], np.ndarray, np.ndarray]:
    """Restore the anchor-mode actor, critic, and state stats from checkpoint."""

    checkpoint_payload = checkpoint
    if checkpoint_payload is None:
        checkpoint_payload = load_checkpoint(checkpoint_path, map_location=device)

    config = dict(checkpoint_payload["config"])
    task_spec = TaskSpec(**checkpoint_payload["task_spec"])
    anchor_config = config["anchor"]
    anchor_task = get_anchor_task_spec(task_spec.env_name)

    actor = AnchorDecisionTransformer(
        state_dim=task_spec.observation_dim,
        action_dim=task_spec.action_dim,
        max_length=int(anchor_config["K"]),
        max_ep_len=anchor_task.max_ep_len,
        hidden_size=int(anchor_config["embed_dim"]),
        n_layer=int(anchor_config["n_layer"]),
        n_head=int(anchor_config["n_head"]),
        dropout=float(anchor_config["dropout"]),
        activation_function=str(anchor_config["activation_function"]),
        scale=float(anchor_config.get("scale") or anchor_task.scale),
        sar=bool(anchor_config.get("sar", False)),
        rtg_no_q=bool(anchor_config.get("rtg_no_q", False)),
        infer_no_q=bool(anchor_config.get("infer_no_q", False)),
    ).to(device)
    actor.load_state_dict(checkpoint_payload["actor_state_dict"])
    actor.eval()

    critic = AnchorDoubleQCritic(
        state_dim=task_spec.observation_dim,
        action_dim=task_spec.action_dim,
        hidden_dim=int(anchor_config["embed_dim"]),
    ).to(device)
    critic_state = checkpoint_payload.get("critic_target_state_dict", checkpoint_payload["critic_state_dict"])
    critic.load_state_dict(critic_state)
    critic.eval()

    return (
        actor,
        critic,
        task_spec,
        config,
        np.asarray(checkpoint_payload["state_mean"], dtype=np.float32),
        np.asarray(checkpoint_payload["state_std"], dtype=np.float32),
    )


def load_evaluation_session(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | None = None,
    episodes: int | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
    target_return: float | None = None,
) -> EvaluationSession:
    """Load the shared evaluation session used by eval and render flows."""

    file_config = load_experiment_config(config_path)
    resolved_device = resolve_device(device or str(file_config["device"]))
    checkpoint_payload = load_checkpoint(checkpoint_path, map_location=resolved_device)
    checkpoint_config = checkpoint_payload.get("config", file_config)
    if bool(checkpoint_config.get("anchor", {}).get("enabled", False)) or checkpoint_payload.get("kind") == "anchor":
        actor, critic, task_spec, restored_config, state_mean, state_std = _restore_anchor_from_checkpoint(
            checkpoint_path,
            device=resolved_device,
            checkpoint=checkpoint_payload,
        )
        config = apply_eval_overrides(
            restored_config if isinstance(restored_config, dict) else file_config,
            device=device,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            target_return=target_return,
        )
        return EvaluationSession(
            actor=actor,
            critic=critic,
            task_spec=task_spec,
            stats=None,  # type: ignore[arg-type]
            config=config,
            device=resolved_device,
            checkpoint_step=int(checkpoint_payload["step"]),
            checkpoint_path=Path(checkpoint_path).resolve(),
            state_mean=state_mean,
            state_std=state_std,
            mode="anchor",
        )

    actor, task_spec, stats, checkpoint_config = restore_actor_from_checkpoint(
        checkpoint_path,
        device=resolved_device,
        checkpoint=checkpoint_payload,
    )
    config = apply_eval_overrides(
        checkpoint_config if isinstance(checkpoint_config, dict) else file_config,
        device=device,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        target_return=target_return,
    )
    return EvaluationSession(
        actor=actor,
        task_spec=task_spec,
        stats=stats,
        config=config,
        device=resolved_device,
        checkpoint_step=int(checkpoint_payload["step"]),
        checkpoint_path=Path(checkpoint_path).resolve(),
        mode="standard",
    )


def _initialize_rollout_buffers(
    *,
    context_length: int,
    observation_dim: int,
    device: torch.device,
) -> _RolloutBuffers:
    """Allocate empty sequence buffers for one rollout."""

    return _RolloutBuffers(
        observations=torch.zeros(
            1,
            context_length,
            observation_dim,
            dtype=torch.float32,
            device=device,
        ),
        returns_to_go=torch.zeros(1, context_length, dtype=torch.float32, device=device),
        timesteps=torch.zeros(1, context_length, dtype=torch.long, device=device),
        attention_mask=torch.zeros(1, context_length, dtype=torch.float32, device=device),
    )


def _append_policy_context(
    *,
    buffers: _RolloutBuffers,
    normalized_observation: np.ndarray,
    remaining_return: float,
    step_index: int,
    max_timestep: int,
    device: torch.device,
) -> None:
    """Roll the context window and append the newest observation token."""

    buffers.observations = torch.roll(buffers.observations, shifts=-1, dims=1)
    buffers.returns_to_go = torch.roll(buffers.returns_to_go, shifts=-1, dims=1)
    buffers.timesteps = torch.roll(buffers.timesteps, shifts=-1, dims=1)
    buffers.attention_mask = torch.roll(buffers.attention_mask, shifts=-1, dims=1)

    buffers.observations[:, -1, :] = torch.from_numpy(normalized_observation).to(device)
    buffers.returns_to_go[:, -1] = float(remaining_return)
    buffers.timesteps[:, -1] = min(step_index, max_timestep - 1)
    buffers.attention_mask[:, -1] = 1.0


def _predict_rollout_action(
    *,
    actor: torch.nn.Module,
    buffers: _RolloutBuffers,
    env,
) -> np.ndarray:
    """Predict and clip one environment action from the current context window."""

    with torch.no_grad():
        action = actor.predict_last_action(
            observations=buffers.observations,
            returns_to_go=buffers.returns_to_go,
            timesteps=buffers.timesteps,
            attention_mask=buffers.attention_mask,
        )

    action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
    return action_np


def rollout_policy_episode(
    *,
    actor: torch.nn.Module,
    env,
    task_spec: TaskSpec,
    stats: NormalizationStats,
    evaluation_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    capture_frame_fn: Callable[[Any], np.ndarray] | None = None,
    frame_skip: int = 1,
) -> dict[str, Any]:
    """Run one live policy rollout, optionally capturing rendered frames."""

    context_length = int(evaluation_config["context_length"])
    max_timestep = int(evaluation_config["max_timestep"])
    max_steps = int(evaluation_config["episodes_max_steps"])
    target_return = resolve_target_return(stats, evaluation_config)

    observation = reset_env(env, seed)
    remaining_return = float(target_return)
    buffers = _initialize_rollout_buffers(
        context_length=context_length,
        observation_dim=task_spec.observation_dim,
        device=device,
    )

    frames = [capture_frame_fn(env)] if capture_frame_fn is not None else []
    total_reward = 0.0
    steps_executed = 0
    done = False

    for step_index in range(max_steps):
        normalized_observation = stats.normalize_observations(
            np.asarray(observation, dtype=np.float32)[None, :]
        )[0]
        _append_policy_context(
            buffers=buffers,
            normalized_observation=normalized_observation,
            remaining_return=remaining_return,
            step_index=step_index,
            max_timestep=max_timestep,
            device=device,
        )

        step_output = env.step(
            _predict_rollout_action(
                actor=actor,
                buffers=buffers,
                env=env,
            )
        )
        observation, reward, done, _ = unwrap_step_output(step_output)
        total_reward += float(reward)
        remaining_return -= float(reward)
        steps_executed += 1

        if capture_frame_fn is not None and (step_index % int(frame_skip) == 0 or done):
            frames.append(capture_frame_fn(env))
        if done:
            break

    return {
        "target_return": float(target_return),
        "total_reward": float(total_reward),
        "episode_length": int(steps_executed),
        "done_reached": bool(done),
        "final_observation": np.asarray(observation, dtype=np.float32),
        "frames": frames,
    }


def rollout_anchor_policy_episode(
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    env,
    task_spec: TaskSpec,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    evaluation_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    capture_frame_fn: Callable[[Any], np.ndarray] | None = None,
    frame_skip: int = 1,
) -> dict[str, Any]:
    """Run one original-QT style rollout with multi-target RTG conditioning."""

    max_steps = int(evaluation_config["episodes_max_steps"])
    test_scale = float(evaluation_config["test_scale"])
    candidate_target_returns = resolve_anchor_eval_targets(
        get_anchor_task_spec(task_spec.env_name),
        target_return=(
            float(evaluation_config["target_return"])
            if evaluation_config["target_return"] is not None
            else None
        ),
        test_scale=test_scale,
    )

    state_mean_tensor = torch.from_numpy(np.asarray(state_mean, dtype=np.float32)).to(device=device)
    state_std_tensor = torch.from_numpy(np.asarray(state_std, dtype=np.float32)).to(device=device)
    observation = reset_env(env, seed)
    if isinstance(observation, tuple):
        observation = observation[0]

    states = torch.from_numpy(np.asarray(observation, dtype=np.float32)).reshape(1, task_spec.observation_dim).to(
        device=device,
        dtype=torch.float32,
    )
    actions = [torch.zeros((1, task_spec.action_dim), device=device, dtype=torch.float32)]
    rewards = [torch.zeros((1, 1), device=device, dtype=torch.float32)]
    target_return = torch.tensor(candidate_target_returns, device=device, dtype=torch.float32).unsqueeze(-1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    frames = [capture_frame_fn(env)] if capture_frame_fn is not None else []
    total_reward = 0.0
    steps_executed = 0
    done = False

    actor.eval()
    critic.eval()
    with torch.no_grad():
        for step_index in range(max_steps):
            normalized_states = (states.to(dtype=torch.float32) - state_mean_tensor) / state_std_tensor
            action = actor.get_action(
                critic,
                normalized_states,
                torch.cat(actions, dim=0).to(dtype=torch.float32),
                torch.cat(rewards, dim=1).to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            action_np = action.detach().cpu().numpy().flatten().astype(np.float32)
            observation, reward, done, _ = unwrap_step_output(env.step(action_np))

            actions.insert(-1, torch.from_numpy(action_np).reshape(1, task_spec.action_dim).to(device=device))
            rewards.insert(-1, torch.tensor(reward, device=device, dtype=torch.float32).reshape(1).unsqueeze(0))
            current_state = torch.from_numpy(np.asarray(observation, dtype=np.float32)).reshape(1, task_spec.observation_dim)
            states = torch.cat([states, current_state.to(device=device)], dim=0)
            predicted_return = target_return[:, -1:] - (float(reward) / test_scale)
            target_return = torch.cat([target_return, predicted_return], dim=1)
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long) * (step_index + 1),
                ],
                dim=1,
            )

            total_reward += float(reward)
            steps_executed += 1

            if capture_frame_fn is not None and (step_index % int(frame_skip) == 0 or done):
                frames.append(capture_frame_fn(env))
            if done:
                break

    return {
        "target_return": (
            float(evaluation_config["target_return"])
            if evaluation_config["target_return"] is not None
            else float(candidate_target_returns[0] * test_scale)
        ),
        "candidate_target_returns": [float(value) * test_scale for value in candidate_target_returns],
        "total_reward": float(total_reward),
        "episode_length": int(steps_executed),
        "done_reached": bool(done),
        "final_observation": np.asarray(observation, dtype=np.float32),
        "frames": frames,
    }


def evaluate_anchor_policy(
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    task_spec: TaskSpec,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    evaluation_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_step: int | None = None,
    on_episode_end: Callable[[int, float, int], None] | None = None,
) -> dict[str, Any]:
    """Run original-QT style evaluation with anchor candidate targets."""

    num_episodes = int(evaluation_config["episodes"])
    env = create_gym_env(task_spec.env_name, action_seed=seed)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    candidate_targets: list[float] | None = None
    was_training = actor.training
    actor.eval()
    try:
        for episode_index in range(num_episodes):
            rollout = rollout_anchor_policy_episode(
                actor=actor,
                critic=critic,
                env=env,
                task_spec=task_spec,
                state_mean=state_mean,
                state_std=state_std,
                evaluation_config=evaluation_config,
                device=device,
                seed=int(seed + episode_index),
            )
            candidate_targets = [float(value) for value in rollout["candidate_target_returns"]]
            episode_returns.append(float(rollout["total_reward"]))
            episode_lengths.append(int(rollout["episode_length"]))
            if on_episode_end is not None:
                on_episode_end(
                    episode_index + 1,
                    float(rollout["total_reward"]),
                    int(rollout["episode_length"]),
                )
    finally:
        if was_training:
            actor.train()
        if hasattr(env, "close"):
            env.close()

    normalized_scores = normalize_episode_returns(task_spec.env_name, episode_returns)
    summary = {
        "env_name": task_spec.env_name,
        "episodes": num_episodes,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_step": int(checkpoint_step) if checkpoint_step is not None else None,
        "target_return": (
            float(evaluation_config["target_return"])
            if evaluation_config["target_return"] is not None
            else (candidate_targets[0] if candidate_targets else None)
        ),
        "candidate_target_returns": candidate_targets or [],
        "max_steps": int(evaluation_config["episodes_max_steps"]),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }
    if normalized_scores:
        summary["normalized_scores"] = normalized_scores
        summary["mean_normalized_score"] = float(np.mean(normalized_scores))
        summary["std_normalized_score"] = float(np.std(normalized_scores))
    if output_path is not None:
        write_json(output_path, summary)
    return summary


def evaluate_policy(
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module | None,
    task_spec: TaskSpec,
    stats: NormalizationStats | None,
    evaluation_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_step: int | None = None,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    on_episode_end: Callable[[int, float, int], None] | None = None,
) -> dict[str, Any]:
    """Run policy rollouts in the live Gym/D4RL environment."""

    if bool(evaluation_config.get("anchor_enabled", False)):
        if critic is None:
            raise ValueError("Anchor evaluation requires a critic.")
        if state_mean is None or state_std is None:
            raise ValueError("Anchor evaluation requires state_mean and state_std.")
        return evaluate_anchor_policy(
            actor=actor,
            critic=critic,
            task_spec=task_spec,
            state_mean=state_mean,
            state_std=state_std,
            evaluation_config=evaluation_config,
            device=device,
            seed=seed,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            on_episode_end=on_episode_end,
        )

    num_episodes = int(evaluation_config["episodes"])
    target_return = resolve_target_return(stats, evaluation_config)
    resolved_evaluation_config = dict(evaluation_config)
    resolved_evaluation_config["target_return"] = target_return

    env = create_gym_env(task_spec.env_name, action_seed=seed)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    was_training = actor.training
    actor.eval()
    try:
        for episode_index in range(num_episodes):
            rollout = rollout_policy_episode(
                actor=actor,
                env=env,
                task_spec=task_spec,
                stats=stats,
                evaluation_config=resolved_evaluation_config,
                device=device,
                seed=int(seed + episode_index),
            )
            episode_returns.append(float(rollout["total_reward"]))
            episode_lengths.append(int(rollout["episode_length"]))
    finally:
        if was_training:
            actor.train()
        if hasattr(env, "close"):
            env.close()

    normalized_scores = normalize_episode_returns(task_spec.env_name, episode_returns)
    summary = {
        "env_name": task_spec.env_name,
        "episodes": num_episodes,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_step": int(checkpoint_step) if checkpoint_step is not None else None,
        "target_return": float(target_return),
        "max_steps": int(resolved_evaluation_config["episodes_max_steps"]),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }
    if normalized_scores:
        summary["normalized_scores"] = normalized_scores
        summary["mean_normalized_score"] = float(np.mean(normalized_scores))
        summary["std_normalized_score"] = float(np.std(normalized_scores))
    if output_path is not None:
        write_json(output_path, summary)
    return summary
