"""Minimal trainer for the first end-to-end QT milestone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .checkpoints import build_checkpoint_payload, save_checkpoint
from .config import resolve_repo_path, resolve_run_dir, save_resolved_config
from .data import build_qt_dataset, make_sequence_dataloader
from .evaluation import build_evaluation_config, evaluate_policy
from .model_factory import build_actor_from_config, build_critic_from_config
from .runtime import (
    append_jsonl,
    ensure_directory,
    move_batch_to_device,
    resolve_device,
    seed_everything,
    write_json,
)


@dataclass(frozen=True)
class TrainingPaths:
    """Filesystem layout for one QT run."""

    run_dir: Path
    checkpoints_dir: Path
    evaluations_dir: Path
    metrics_path: Path
    evaluation_metrics_path: Path
    summary_path: Path
    resolved_config_path: Path


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    """Enable or disable gradients for one module."""

    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Exponential moving-average target update."""

    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def _masked_action_mse(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE only over non-padded action positions."""

    mask = attention_mask.unsqueeze(-1)
    squared_error = (predicted_actions - target_actions) ** 2
    denominator = mask.sum().clamp(min=1.0) * predicted_actions.shape[-1]
    return (squared_error * mask).sum() / denominator


def _build_next_actor_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Shift a right-aligned sequence window to the next transition context."""

    next_observations = torch.roll(batch["observations"], shifts=-1, dims=1).clone()
    next_returns_to_go = torch.roll(batch["returns_to_go"], shifts=-1, dims=1).clone()
    next_timesteps = torch.roll(batch["timesteps"], shifts=-1, dims=1).clone()
    next_attention_mask = torch.roll(batch["attention_mask"], shifts=-1, dims=1).clone()

    next_observations[:, -1, :] = batch["next_observations"][:, -1, :]
    next_returns_to_go[:, -1] = torch.where(
        batch["bootstrap_mask"][:, -1] > 0.0,
        batch["returns_to_go"][:, -1] - batch["rewards"][:, -1],
        torch.zeros_like(batch["returns_to_go"][:, -1]),
    )
    next_timesteps[:, -1] = batch["timesteps"][:, -1] + 1
    next_attention_mask[:, -1] = 1.0

    return {
        "observations": next_observations,
        "returns_to_go": next_returns_to_go,
        "timesteps": next_timesteps,
        "attention_mask": next_attention_mask,
    }


class QTTrainer:
    """Training loop wrapper for the first QT experiment milestone."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.seed = int(self.config["seed"])
        seed_everything(self.seed)

        self.device = resolve_device(str(self.config["device"]))
        data_config = self.config["data"]

        cache_dir = resolve_repo_path(data_config.get("cache_dir"))
        dataset_path = resolve_repo_path(data_config.get("dataset_path"))
        self.dataset = build_qt_dataset(
            env_name=self.config["env"]["name"],
            context_length=int(data_config["context_length"]),
            discount=float(data_config["discount"]),
            cache_dir=cache_dir,
            dataset_path=dataset_path,
        )
        self.dataloader = make_sequence_dataloader(
            dataset=self.dataset,
            batch_size=int(data_config["batch_size"]),
            shuffle=bool(data_config["shuffle"]),
            num_workers=int(data_config["num_workers"]),
            pin_memory=bool(data_config["pin_memory"]),
            drop_last=bool(data_config["drop_last"]),
        )
        self._dataloader_iterator = iter(self.dataloader)

        self.actor = build_actor_from_config(self.config, self.dataset.task_spec).to(self.device)
        self.critic = build_critic_from_config(self.config, self.dataset.task_spec).to(self.device)
        self.target_actor = build_actor_from_config(self.config, self.dataset.task_spec).to(self.device)
        self.target_critic = build_critic_from_config(self.config, self.dataset.task_spec).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        optimization_config = self.config["optimization"]
        self.actor_optimizer = AdamW(
            self.actor.parameters(),
            lr=float(optimization_config["actor_lr"]),
            weight_decay=float(optimization_config["weight_decay"]),
        )
        self.critic_optimizer = AdamW(
            self.critic.parameters(),
            lr=float(optimization_config["critic_lr"]),
            weight_decay=float(optimization_config["weight_decay"]),
        )

        training_config = self.config["training"]
        self.gamma = float(training_config["gamma"])
        self.eta = float(training_config["eta"])
        self.grad_clip_norm = float(training_config["grad_clip_norm"])
        self.target_tau = float(training_config["target_tau"])
        self.actor_update_every = int(training_config["actor_update_every"])

        run_dir = ensure_directory(resolve_run_dir(self.config))
        self.paths = TrainingPaths(
            run_dir=run_dir,
            checkpoints_dir=ensure_directory(run_dir / "checkpoints"),
            evaluations_dir=ensure_directory(run_dir / "evaluations"),
            metrics_path=run_dir / "metrics.jsonl",
            evaluation_metrics_path=run_dir / "evaluations.jsonl",
            summary_path=run_dir / "summary.json",
            resolved_config_path=run_dir / "config_resolved.yaml",
        )
        if self.paths.metrics_path.exists():
            self.paths.metrics_path.unlink()
        if self.paths.evaluation_metrics_path.exists():
            self.paths.evaluation_metrics_path.unlink()
        save_resolved_config(self.config, self.paths.resolved_config_path)

        self.latest_train_metrics: dict[str, Any] = {}
        self.latest_evaluation_metrics: dict[str, Any] | None = None
        self.best_evaluation_metrics: dict[str, Any] | None = None
        self.latest_checkpoint_path: Path | None = None
        self.best_checkpoint_path: Path | None = None

    def _next_batch(self) -> dict[str, Any]:
        """Yield the next training batch, recycling the dataloader iterator as needed."""

        try:
            batch = next(self._dataloader_iterator)
        except StopIteration:
            self._dataloader_iterator = iter(self.dataloader)
            batch = next(self._dataloader_iterator)
        return move_batch_to_device(batch, self.device)

    def _compute_critic_loss(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the current critic TD loss and metrics."""

        current_observations = batch["observations"][:, -1, :]
        current_actions = batch["actions"][:, -1, :]
        rewards = batch["rewards"][:, -1]
        next_observations = batch["next_observations"][:, -1, :]
        bootstrap_mask = batch["bootstrap_mask"][:, -1]

        with torch.no_grad():
            next_actor_inputs = _build_next_actor_inputs(batch)
            next_actions = self.target_actor.predict_last_action(**next_actor_inputs)
            target_q = self.target_critic.min(next_observations, next_actions).squeeze(-1)
            td_target = rewards + self.gamma * bootstrap_mask * target_q

        q1, q2 = self.critic(current_observations, current_actions)
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        metrics = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "q1_mean": float(q1.detach().mean().cpu().item()),
            "q2_mean": float(q2.detach().mean().cpu().item()),
            "target_q_mean": float(target_q.detach().mean().cpu().item()),
            "td_target_mean": float(td_target.detach().mean().cpu().item()),
            "reward_mean": float(rewards.detach().mean().cpu().item()),
        }
        return critic_loss, metrics

    def _compute_actor_loss(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the actor objective and metrics."""

        predicted_actions = self.actor(
            observations=batch["observations"],
            returns_to_go=batch["returns_to_go"],
            timesteps=batch["timesteps"],
            attention_mask=batch["attention_mask"],
        )
        bc_loss = _masked_action_mse(
            predicted_actions=predicted_actions,
            target_actions=batch["actions"],
            attention_mask=batch["attention_mask"],
        )

        current_observations = batch["observations"][:, -1, :]
        current_policy_actions = predicted_actions[:, -1, :]
        q_values = self.critic.min(current_observations, current_policy_actions).squeeze(-1)
        q_scale = q_values.detach().abs().mean().clamp(min=1.0)
        q_regularization = -(q_values / q_scale).mean()
        actor_loss = bc_loss + self.eta * q_regularization

        metrics = {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "bc_loss": float(bc_loss.detach().cpu().item()),
            "policy_q_mean": float(q_values.detach().mean().cpu().item()),
            "policy_q_abs_mean": float(q_values.detach().abs().mean().cpu().item()),
            "q_regularization": float(q_regularization.detach().cpu().item()),
        }
        return actor_loss, metrics

    def _save_checkpoint(self, step: int) -> Path:
        """Write a step checkpoint plus a `latest.pt` pointer."""

        payload = self._build_checkpoint_payload(step)
        step_path = self.paths.checkpoints_dir / f"step_{step:07d}.pt"
        save_checkpoint(step_path, payload)
        return self._save_checkpoint_alias(step, alias_name="latest.pt", payload=payload)

    def _save_best_checkpoint(self, step: int) -> Path:
        """Persist the current model state as the run's best checkpoint."""

        return self._save_checkpoint_alias(step, alias_name="best.pt")

    def _build_checkpoint_payload(self, step: int) -> dict[str, Any]:
        """Assemble the current run state into one checkpoint payload."""

        return build_checkpoint_payload(
            config=self.config,
            step=step,
            task_spec=self.dataset.task_spec,
            stats=self.dataset.stats,
            actor_state_dict=self.actor.state_dict(),
            critic_state_dict=self.critic.state_dict(),
            target_actor_state_dict=self.target_actor.state_dict(),
            target_critic_state_dict=self.target_critic.state_dict(),
            actor_optimizer_state_dict=self.actor_optimizer.state_dict(),
            critic_optimizer_state_dict=self.critic_optimizer.state_dict(),
            latest_metrics={
                "train": self.latest_train_metrics,
                "evaluation": self.latest_evaluation_metrics,
            },
        )

    def _save_checkpoint_alias(
        self,
        step: int,
        *,
        alias_name: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save the current payload to one stable checkpoint alias."""

        alias_path = self.paths.checkpoints_dir / alias_name
        save_checkpoint(alias_path, payload or self._build_checkpoint_payload(step))
        if alias_name == "latest.pt":
            self.latest_checkpoint_path = alias_path
        elif alias_name == "best.pt":
            self.best_checkpoint_path = alias_path
        return alias_path

    def _evaluate(self, step: int) -> dict[str, Any]:
        """Run a live-environment evaluation and persist the summary."""

        output_path = self.paths.evaluations_dir / f"step_{step:07d}.json"
        summary = evaluate_policy(
            actor=self.actor,
            critic=None,
            task_spec=self.dataset.task_spec,
            stats=self.dataset.stats,
            evaluation_config=build_evaluation_config(self.config),
            device=self.device,
            seed=self.seed,
            output_path=output_path,
            checkpoint_path=self.latest_checkpoint_path,
            checkpoint_step=step,
            state_mean=None,
            state_std=None,
        )
        self.latest_evaluation_metrics = summary
        append_jsonl(self.paths.evaluation_metrics_path, {"step": step, **summary})
        if (
            self.best_evaluation_metrics is None
            or float(summary["mean_return"]) > float(self.best_evaluation_metrics["mean_return"])
        ):
            self.best_evaluation_metrics = dict(summary)
            best_path = self._save_best_checkpoint(step)
            self.best_evaluation_metrics["best_checkpoint_path"] = str(best_path)
        return summary

    def train(self) -> dict[str, Any]:
        """Run the configured QT training loop."""

        total_steps = int(self.config["training"]["steps"])
        log_interval = int(self.config["training"]["log_interval"])
        checkpoint_interval = int(self.config["training"]["checkpoint_interval"])
        eval_interval = int(self.config["training"]["eval_interval"])

        for step in range(1, total_steps + 1):
            batch = self._next_batch()

            critic_loss, critic_metrics = self._compute_critic_loss(batch)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
            self.critic_optimizer.step()

            actor_metrics: dict[str, float] = {}
            if step % self.actor_update_every == 0:
                _set_requires_grad(self.critic, False)
                actor_loss, actor_metrics = self._compute_actor_loss(batch)
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                self.actor_optimizer.step()
                _set_requires_grad(self.critic, True)

            _soft_update(self.target_actor, self.actor, self.target_tau)
            _soft_update(self.target_critic, self.critic, self.target_tau)

            train_metrics = {
                "step": step,
                **critic_metrics,
                **actor_metrics,
            }
            self.latest_train_metrics = train_metrics

            if step == 1 or step % log_interval == 0 or step == total_steps:
                append_jsonl(self.paths.metrics_path, train_metrics)

            if step % checkpoint_interval == 0 or step == total_steps:
                self._save_checkpoint(step)

            if eval_interval > 0 and (step % eval_interval == 0 or step == total_steps):
                self._evaluate(step)

        summary = {
            "run_dir": str(self.paths.run_dir),
            "metrics_path": str(self.paths.metrics_path),
            "checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path is not None else None,
            "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path is not None else None,
            "final_step": total_steps,
            "latest_train_metrics": self.latest_train_metrics,
            "latest_evaluation": self.latest_evaluation_metrics,
            "best_evaluation": self.best_evaluation_metrics,
        }
        write_json(self.paths.summary_path, summary)
        return summary
