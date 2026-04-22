"""Original-QT anchor-mode trainer for scoped replication."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .anchor_data import AnchorBatchSampler, build_anchor_trajectory_dataset
from .anchor_registry import get_anchor_task_spec
from .checkpoints import load_checkpoint, save_checkpoint
from .config import resolve_repo_path, resolve_run_dir, save_resolved_config
from .data.registry import get_task_spec
from .evaluation import build_evaluation_config, evaluate_policy
from .models import AnchorDecisionTransformer, AnchorDoubleQCritic
from .runtime import append_jsonl, ensure_directory, resolve_device, seed_everything, write_json


@dataclass(frozen=True)
class AnchorTrainingPaths:
    run_dir: Path
    checkpoints_dir: Path
    evaluations_dir: Path
    metrics_path: Path
    evaluation_metrics_path: Path
    summary_path: Path
    resolved_config_path: Path


class EMA:
    """Empirical moving average helper from the original QT trainer."""

    def __init__(self, beta: float) -> None:
        self.beta = float(beta)

    def update_model_average(self, moving_average_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, moving_average_params in zip(current_model.parameters(), moving_average_model.parameters()):
            moving_average_params.data = self.update_average(
                moving_average_params.data,
                current_params.data,
            )

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        return old * self.beta + (1.0 - self.beta) * new


class AnchorQTTrainer:
    """Paper-faithful QT trainer for the scoped anchor tasks."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        resume_from: str | Path | None = None,
    ) -> None:
        self.config = dict(config)
        self.anchor_config = dict(self.config["anchor"])
        if not bool(self.anchor_config.get("enabled", False)):
            raise ValueError("AnchorQTTrainer requires anchor.enabled=true.")

        self.seed = int(self.config["seed"])
        seed_everything(self.seed)
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self.device = resolve_device(str(self.config["device"]))
        self.anchor_task = get_anchor_task_spec(self.config["env"]["name"])
        self.task_spec = get_task_spec(self.config["env"]["name"])

        dataset_path = resolve_repo_path(self.config["data"].get("dataset_path"))
        export_path = resolve_repo_path(self.anchor_config.get("trajectory_export_path"))
        self.trajectory_dataset = build_anchor_trajectory_dataset(
            self.config["env"]["name"],
            dataset_path=dataset_path,
            export_path=export_path,
            save_export=bool(self.anchor_config.get("save_trajectory_pkl", False)),
            pct_traj=float(self.anchor_config["pct_traj"]),
        )
        scale = self.anchor_config.get("scale")
        if scale is None:
            scale = self.anchor_task.scale
            self.anchor_config["scale"] = scale
        if self.anchor_config.get("test_scale") is None:
            self.anchor_config["test_scale"] = self.anchor_task.test_scale
        if not self.anchor_config.get("env_targets"):
            self.anchor_config["env_targets"] = list(self.anchor_task.env_targets)

        self.batch_sampler = AnchorBatchSampler(
            dataset=self.trajectory_dataset,
            task_spec=self.anchor_task,
            context_length=int(self.anchor_config["K"]),
            batch_size=int(self.anchor_config["batch_size"]),
            device=self.device,
            scale=float(scale),
            reward_tune=str(self.anchor_config.get("reward_tune", "no")),
        )

        self.actor = AnchorDecisionTransformer(
            state_dim=self.task_spec.observation_dim,
            action_dim=self.task_spec.action_dim,
            max_length=int(self.anchor_config["K"]),
            max_ep_len=self.anchor_task.max_ep_len,
            hidden_size=int(self.anchor_config["embed_dim"]),
            n_layer=int(self.anchor_config["n_layer"]),
            n_head=int(self.anchor_config["n_head"]),
            dropout=float(self.anchor_config["dropout"]),
            activation_function=str(self.anchor_config["activation_function"]),
            scale=float(self.anchor_config["scale"]),
            sar=bool(self.anchor_config.get("sar", False)),
            rtg_no_q=bool(self.anchor_config.get("rtg_no_q", False)),
            infer_no_q=bool(self.anchor_config.get("infer_no_q", False)),
        ).to(self.device)
        self.ema_actor = copy.deepcopy(self.actor)
        self.ema = EMA(float(self.anchor_config["ema_decay"]))
        self.critic = AnchorDoubleQCritic(
            state_dim=self.task_spec.observation_dim,
            action_dim=self.task_spec.action_dim,
            hidden_dim=int(self.anchor_config["embed_dim"]),
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=float(self.anchor_config["learning_rate"]),
            weight_decay=float(self.anchor_config["weight_decay"]),
        )
        self.critic_optimizer = Adam(self.critic.parameters(), lr=3e-4)
        self.lr_decay = bool(self.anchor_config["lr_decay"])
        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer,
                T_max=int(self.anchor_config["max_iters"]),
                eta_min=float(self.anchor_config["lr_min"]),
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer,
                T_max=int(self.anchor_config["max_iters"]),
                eta_min=float(self.anchor_config["lr_min"]),
            )

        self.tau = float(self.anchor_config["tau"])
        self.discount = float(self.anchor_config["discount"])
        self.eta = float(self.anchor_config["eta"])
        self.eta2 = float(self.anchor_config["eta2"])
        self.grad_norm = float(self.anchor_config["grad_norm"])
        self.max_q_backup = bool(self.anchor_config["max_q_backup"])
        self.scale = float(self.anchor_config["scale"])
        self.k_rewards = bool(self.anchor_config["k_rewards"])
        self.use_discount = bool(self.anchor_config["use_discount"])
        self.ema_start_step = int(self.anchor_config["ema_start_step"])
        self.ema_update_every = int(self.anchor_config["ema_update_every"])

        run_dir = ensure_directory(resolve_run_dir(self.config))
        self.paths = AnchorTrainingPaths(
            run_dir=run_dir,
            checkpoints_dir=ensure_directory(run_dir / "checkpoints"),
            evaluations_dir=ensure_directory(run_dir / "evaluations"),
            metrics_path=run_dir / "metrics.jsonl",
            evaluation_metrics_path=run_dir / "evaluations.jsonl",
            summary_path=run_dir / "summary.json",
            resolved_config_path=run_dir / "config_resolved.yaml",
        )
        if self.resume_from is None:
            if self.paths.metrics_path.exists():
                self.paths.metrics_path.unlink()
            if self.paths.evaluation_metrics_path.exists():
                self.paths.evaluation_metrics_path.unlink()

        self.config["anchor"] = dict(self.anchor_config)
        save_resolved_config(self.config, self.paths.resolved_config_path)

        self.latest_train_metrics: dict[str, Any] = {}
        self.latest_evaluation_metrics: dict[str, Any] | None = None
        self.best_evaluation_metrics: dict[str, Any] | None = None
        self.latest_checkpoint_path: Path | None = None
        self.best_checkpoint_path: Path | None = None
        self.global_step = 0
        self.start_iteration = 1

        if self.resume_from is not None:
            self._resume_from_checkpoint(self.resume_from)

    def _step_ema(self) -> None:
        if self.global_step > self.ema_start_step and self.global_step % self.ema_update_every == 0:
            self.ema.update_model_average(self.ema_actor, self.actor)

    def _log(self, message: str) -> None:
        print(f"[anchor][{self.config['run_name']}] {message}", flush=True)

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None or not np.isfinite(seconds):
            return "unknown"
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:d}h{minutes:02d}m{secs:02d}s"
        if minutes > 0:
            return f"{minutes:d}m{secs:02d}s"
        return f"{secs:d}s"

    def _save_checkpoint_payload(self, *, step: int, iteration: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": dict(self.config),
            "step": int(step),
            "iteration": int(iteration),
            "kind": "anchor",
            "task_spec": asdict(self.task_spec),
            "state_mean": self.trajectory_dataset.state_mean.astype(np.float32),
            "state_std": self.trajectory_dataset.state_std.astype(np.float32),
            "actor_state_dict": self.actor.state_dict(),
            "ema_actor_state_dict": self.ema_actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "latest_metrics": {
                "train": self.latest_train_metrics,
                "evaluation": self.latest_evaluation_metrics,
            },
            "eta2": float(self.eta2),
            "best_evaluation_metrics": dict(self.best_evaluation_metrics) if self.best_evaluation_metrics is not None else None,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
        }
        if self.lr_decay:
            payload["actor_lr_scheduler_state_dict"] = self.actor_lr_scheduler.state_dict()
            payload["critic_lr_scheduler_state_dict"] = self.critic_lr_scheduler.state_dict()
        if torch.cuda.is_available():
            payload["rng_torch_cuda"] = torch.cuda.get_rng_state_all()
        return payload

    def _resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Restore trainer state for continuing an interrupted run."""

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        # Load to CPU first so RNG state tensors keep their uint8 dtype.
        # load_state_dict below moves model tensors to the correct device.
        payload = load_checkpoint(path, map_location="cpu")

        self.actor.load_state_dict(payload["actor_state_dict"])
        self.ema_actor.load_state_dict(payload["ema_actor_state_dict"])
        self.critic.load_state_dict(payload["critic_state_dict"])
        self.critic_target.load_state_dict(payload["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        if self.lr_decay:
            if "actor_lr_scheduler_state_dict" in payload:
                self.actor_lr_scheduler.load_state_dict(payload["actor_lr_scheduler_state_dict"])
            if "critic_lr_scheduler_state_dict" in payload:
                self.critic_lr_scheduler.load_state_dict(payload["critic_lr_scheduler_state_dict"])

        self.global_step = int(payload["step"])
        resumed_iteration = int(payload.get("iteration", 0))
        self.start_iteration = resumed_iteration + 1
        if "eta2" in payload:
            self.eta2 = float(payload["eta2"])
        if payload.get("best_evaluation_metrics") is not None:
            self.best_evaluation_metrics = dict(payload["best_evaluation_metrics"])
        latest_metrics = payload.get("latest_metrics", {})
        self.latest_train_metrics = dict(latest_metrics.get("train") or {})
        self.latest_evaluation_metrics = (
            dict(latest_metrics["evaluation"]) if latest_metrics.get("evaluation") else None
        )
        self.latest_checkpoint_path = path
        best_path = self.paths.checkpoints_dir / "best.pt"
        if best_path.exists():
            self.best_checkpoint_path = best_path

        if "rng_python" in payload:
            random.setstate(payload["rng_python"])
        if "rng_numpy" in payload:
            np.random.set_state(payload["rng_numpy"])
        if "rng_torch" in payload:
            rng_cpu = payload["rng_torch"].cpu().to(dtype=torch.uint8)
            torch.set_rng_state(rng_cpu)
        if torch.cuda.is_available() and "rng_torch_cuda" in payload:
            cuda_states = [s.cpu().to(dtype=torch.uint8) for s in payload["rng_torch_cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)

        self._log(
            f"resumed from {path} global_step={self.global_step} "
            f"iteration={resumed_iteration} eta2={self.eta2:.6f}"
        )

    def _save_checkpoint(self, *, iteration: int, alias_name: str) -> Path:
        payload = self._save_checkpoint_payload(step=self.global_step, iteration=iteration)
        path = self.paths.checkpoints_dir / alias_name
        save_checkpoint(path, payload)
        if alias_name == "latest.pt":
            self.latest_checkpoint_path = path
        if alias_name == "best.pt":
            self.best_checkpoint_path = path
        return path

    def _step_target_network(self) -> None:
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _compute_target_q(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target_actions: torch.Tensor,
        dones: torch.Tensor,
        rtg: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = states.shape[0], states.shape[1]
        device = states.device
        repeat_num = 10

        if self.max_q_backup:
            states_rpt = torch.repeat_interleave(states, repeats=repeat_num, dim=0)
            actions_rpt = torch.repeat_interleave(actions, repeats=repeat_num, dim=0)
            rewards_rpt = torch.repeat_interleave(rewards, repeats=repeat_num, dim=0)
            rtg_rpt = torch.repeat_interleave(rtg, repeats=repeat_num, dim=0)
            timesteps_rpt = torch.repeat_interleave(timesteps, repeats=repeat_num, dim=0)
            attention_mask_rpt = torch.repeat_interleave(attention_mask, repeats=repeat_num, dim=0)
            noise = torch.cat(
                [
                    torch.zeros(1, 1, 1, device=device),
                    torch.randn(repeat_num - 1, 1, 1, device=device),
                ],
                dim=0,
            ).repeat(batch_size, 1, 1)
            rtg_rpt[:, -2:-1] = rtg_rpt[:, -2:-1] + noise * 0.1
            _, next_action, _ = self.ema_actor(
                states_rpt,
                actions_rpt,
                rewards_rpt,
                None,
                rtg_rpt[:, :-1],
                timesteps_rpt,
                attention_mask=attention_mask_rpt,
            )
        else:
            _, next_action, _ = self.ema_actor(
                states,
                actions,
                rewards,
                target_actions,
                rtg[:, :-1],
                timesteps,
                attention_mask=attention_mask,
            )

        if self.k_rewards:
            if self.max_q_backup:
                critic_next_states = states_rpt[:, -1]
                next_action = next_action[:, -1]
                target_q1, target_q2 = self.critic_target(critic_next_states, next_action)
                target_q1 = target_q1.view(batch_size, repeat_num).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, repeat_num).max(dim=1, keepdim=True)[0]
            else:
                critic_next_states = states[:, -1]
                next_action = next_action[:, -1]
                target_q1, target_q2 = self.critic_target(critic_next_states, next_action)

            target_q = torch.minimum(target_q1, target_q2)
            not_done = 1.0 - dones[:, -1]
            if self.use_discount:
                rewards = rewards.clone()
                rewards[:, -1] = 0.0
                lengths = attention_mask.sum(dim=1).detach().cpu().to(dtype=torch.long)
                reverse_discount = [length - 1 - torch.arange(length) for length in lengths]
                reverse_discount = torch.stack(
                    [torch.cat([item, torch.zeros(sequence_length - len(item))], dim=0) for item in reverse_discount],
                    dim=0,
                )
                reverse_discount = (self.discount ** reverse_discount).unsqueeze(-1).to(device)
                k_rewards = torch.cumsum(rewards.flip(dims=[1]) * reverse_discount, dim=1).flip(dims=[1])

                forward_discount = [torch.arange(length) for length in lengths]
                forward_discount = torch.stack(
                    [torch.cat([torch.zeros(sequence_length - len(item)), item], dim=0) for item in forward_discount],
                    dim=0,
                )
                forward_discount = (self.discount ** forward_discount).unsqueeze(-1).to(device)
                k_rewards = k_rewards / forward_discount

                tail_discount = [length - 1 - torch.arange(length) for length in lengths]
                tail_discount = torch.stack(
                    [torch.cat([torch.zeros(sequence_length - len(item)), item], dim=0) for item in tail_discount],
                    dim=0,
                )
                tail_discount = (self.discount ** tail_discount).to(device)
                target_q = (k_rewards + (not_done * tail_discount * target_q).unsqueeze(-1)).detach()
            else:
                k_rewards = (rtg[:, :-1] - rtg[:, -2:-1]) * self.scale
                target_q = (k_rewards + (not_done * target_q).unsqueeze(-1)).detach()
        else:
            if self.max_q_backup:
                target_q1, target_q2 = self.critic_target(states_rpt, next_action)
                target_q1 = target_q1.view(batch_size, repeat_num, sequence_length, 1).max(dim=1)[0]
                target_q2 = target_q2.view(batch_size, repeat_num, sequence_length, 1).max(dim=1)[0]
            else:
                target_q1, target_q2 = self.critic_target(states, next_action)
            target_q = torch.minimum(target_q1, target_q2)
            target_q = rewards[:, :-1] + self.discount * target_q[:, 1:]
            target_q = torch.cat([target_q, torch.zeros(batch_size, 1, 1, device=device)], dim=1)

        return target_q

    def train_step(self) -> dict[str, float]:
        states, actions, rewards, target_actions, dones, rtg, timesteps, attention_mask = self.batch_sampler.sample_batch()
        batch_size = states.shape[0]
        action_dim = actions.shape[-1]
        state_dim = states.shape[-1]

        current_q1, current_q2 = self.critic(states, actions)
        target_q = self._compute_target_q(
            states=states,
            actions=actions,
            rewards=rewards,
            target_actions=target_actions,
            dones=dones,
            rtg=rtg,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        critic_loss = F.mse_loss(current_q1[:, :-1][attention_mask[:, :-1] > 0], target_q[:, :-1][attention_mask[:, :-1] > 0])
        critic_loss = critic_loss + F.mse_loss(
            current_q2[:, :-1][attention_mask[:, :-1] > 0],
            target_q[:, :-1][attention_mask[:, :-1] > 0],
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = None
        if self.grad_norm > 0:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        state_preds, action_preds, reward_preds = self.actor(
            states,
            actions,
            rewards,
            target_actions,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        action_preds_flat = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_targets_flat = target_actions.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        state_preds = state_preds[:, :-1]
        state_targets = states[:, 1:]
        states_loss = ((state_preds - state_targets) ** 2)[attention_mask[:, :-1] > 0].mean()
        if reward_preds is not None:
            reward_preds_flat = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_targets_flat = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0] / self.scale
            rewards_loss = F.mse_loss(reward_preds_flat, reward_targets_flat)
        else:
            rewards_loss = torch.tensor(0.0, device=self.device)
        bc_loss = F.mse_loss(action_preds_flat, action_targets_flat) + states_loss + rewards_loss

        actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        q1_new_action, q2_new_action = self.critic(actor_states, action_preds_flat)
        if np.random.uniform() > 0.5:
            q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()
        actor_loss = self.eta2 * bc_loss + self.eta * q_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = None
        if self.grad_norm > 0:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        self._step_ema()
        self._step_target_network()
        self.global_step += 1

        return {
            "bc_loss": float(bc_loss.detach().cpu().item()),
            "ql_loss": float(q_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "target_q_mean": float(target_q.mean().detach().cpu().item()),
            "action_error": float(torch.mean((action_preds - target_actions) ** 2).detach().cpu().item()),
            "actor_grad_norm": float(actor_grad_norm.max().item()) if actor_grad_norm is not None else 0.0,
            "critic_grad_norm": float(critic_grad_norm.max().item()) if critic_grad_norm is not None else 0.0,
        }

    def _evaluate_iteration(self, iteration: int) -> dict[str, Any]:
        output_path = self.paths.evaluations_dir / f"iteration_{iteration:05d}.json"

        def _log_episode(ep_idx: int, ep_return: float, ep_length: int) -> None:
            self._log(
                f"iteration {iteration} eval episode {ep_idx} "
                f"return={ep_return:.4f} length={ep_length}"
            )

        summary = evaluate_policy(
            actor=self.actor,
            critic=self.critic_target,
            task_spec=self.task_spec,
            stats=None,
            evaluation_config=build_evaluation_config(self.config),
            device=self.device,
            seed=self.seed,
            output_path=output_path,
            checkpoint_path=self.latest_checkpoint_path,
            checkpoint_step=self.global_step,
            state_mean=self.trajectory_dataset.state_mean,
            state_std=self.trajectory_dataset.state_std,
            on_episode_end=_log_episode,
        )
        self.latest_evaluation_metrics = summary
        append_jsonl(
            self.paths.evaluation_metrics_path,
            {"iteration": iteration, "step": self.global_step, **summary},
        )
        metric_name = "mean_return"
        if self.best_evaluation_metrics is None or float(summary[metric_name]) > float(self.best_evaluation_metrics[metric_name]):
            self.best_evaluation_metrics = dict(summary)
            best_path = self._save_checkpoint(iteration=iteration, alias_name="best.pt")
            self.best_evaluation_metrics["best_checkpoint_path"] = str(best_path)
        return summary

    def train(self) -> dict[str, Any]:
        max_iters = int(self.anchor_config["max_iters"])
        steps_per_iter = int(self.anchor_config["num_steps_per_iter"])
        progress_interval = min(50, steps_per_iter)
        best_return = -float("inf")
        best_normalized = -float("inf")
        best_iteration = -1
        if self.best_evaluation_metrics is not None:
            best_return = float(self.best_evaluation_metrics.get("mean_return", best_return))
            best_normalized = float(self.best_evaluation_metrics.get("mean_normalized_score", best_normalized))
        train_start = time.perf_counter()

        if self.start_iteration > max_iters:
            self._log(
                f"resume target already reached: start_iteration={self.start_iteration} "
                f"> max_iters={max_iters}; skipping training loop"
            )
            iteration = self.start_iteration - 1
            summary = {
                "run_dir": str(self.paths.run_dir),
                "metrics_path": str(self.paths.metrics_path),
                "checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path is not None else None,
                "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path is not None else None,
                "final_step": int(self.global_step),
                "final_iteration": int(iteration),
                "best_iteration": int(best_iteration),
                "best_return": float(best_return),
                "best_normalized_score": float(best_normalized),
                "latest_train_metrics": self.latest_train_metrics,
                "latest_evaluation": self.latest_evaluation_metrics,
                "best_evaluation": self.best_evaluation_metrics,
            }
            write_json(self.paths.summary_path, summary)
            return summary

        self._log(
            "starting training "
            f"env={self.task_spec.env_name} seed={self.seed} device={self.device} "
            f"run_dir={self.paths.run_dir} start_iteration={self.start_iteration} "
            f"max_iters={max_iters} steps_per_iter={steps_per_iter} "
            f"eval_episodes={self.config['evaluation']['episodes']}"
        )

        for iteration in range(self.start_iteration, max_iters + 1):
            iteration_start = time.perf_counter()
            loss_metrics = {
                "bc_loss": [],
                "ql_loss": [],
                "actor_loss": [],
                "critic_loss": [],
                "target_q_mean": [],
                "action_error": [],
            }
            self._log(f"iteration {iteration}/{max_iters} started")
            for step_in_iter in range(1, steps_per_iter + 1):
                step_metrics = self.train_step()
                for key in loss_metrics:
                    loss_metrics[key].append(step_metrics[key])
                if step_in_iter % progress_interval == 0 or step_in_iter == steps_per_iter:
                    completed_fraction = (iteration - 1) + (step_in_iter / steps_per_iter)
                    total_elapsed = time.perf_counter() - train_start
                    estimated_total = (total_elapsed / completed_fraction) * max_iters if completed_fraction > 0 else None
                    eta_seconds = None if estimated_total is None else estimated_total - total_elapsed
                    self._log(
                        f"iteration {iteration}/{max_iters} step {step_in_iter}/{steps_per_iter} "
                        f"global_step={self.global_step} "
                        f"actor_loss={np.mean(loss_metrics['actor_loss']):.4f} "
                        f"critic_loss={np.mean(loss_metrics['critic_loss']):.4f} "
                        f"ql_loss={np.mean(loss_metrics['ql_loss']):.4f} "
                        f"elapsed={self._format_duration(total_elapsed)} "
                        f"eta={self._format_duration(eta_seconds)}"
                    )

            if self.lr_decay:
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()

            aggregated = {
                "iteration": iteration,
                "step": self.global_step,
                "bc_loss": float(np.mean(loss_metrics["bc_loss"])),
                "ql_loss": float(np.mean(loss_metrics["ql_loss"])),
                "actor_loss": float(np.mean(loss_metrics["actor_loss"])),
                "critic_loss": float(np.mean(loss_metrics["critic_loss"])),
                "target_q_mean": float(np.mean(loss_metrics["target_q_mean"])),
                "action_error": float(np.mean(loss_metrics["action_error"])),
                "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
                "critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
            }
            self.latest_train_metrics = aggregated
            append_jsonl(self.paths.metrics_path, aggregated)

            self._save_checkpoint(iteration=iteration, alias_name="latest.pt")
            self._log(
                f"iteration {iteration}/{max_iters} checkpointed latest.pt "
                f"(iteration_time={self._format_duration(time.perf_counter() - iteration_start)})"
            )
            self._log(
                f"iteration {iteration}/{max_iters} evaluating "
                f"episodes={self.config['evaluation']['episodes']} "
                f"max_steps={self.config['evaluation']['max_steps']}"
            )
            evaluation = self._evaluate_iteration(iteration)
            self.eta2 = self.eta2 / float(self.anchor_config["lambda"])

            best_return = max(best_return, float(evaluation["mean_return"]))
            best_normalized = max(best_normalized, float(evaluation.get("mean_normalized_score", -float("inf"))))
            if best_return == float(evaluation["mean_return"]):
                best_iteration = iteration

            normalized = evaluation.get("mean_normalized_score")
            normalized_text = "n/a" if normalized is None else f"{float(normalized):.4f}"
            self._log(
                f"iteration {iteration}/{max_iters} eval "
                f"mean_return={float(evaluation['mean_return']):.4f} "
                f"mean_normalized_score={normalized_text} "
                f"best_return={best_return:.4f} best_iteration={best_iteration}"
            )

            # Match the original QT repo's zero-based loop semantics:
            # break after completing the iteration whose zero-based index
            # reaches early_epoch.
            if bool(self.anchor_config["early_stop"]) and iteration > int(self.anchor_config["early_epoch"]):
                self._log(
                    f"early_stop triggered at iteration {iteration} "
                    f"(early_epoch={int(self.anchor_config['early_epoch'])})"
                )
                break

        summary = {
            "run_dir": str(self.paths.run_dir),
            "metrics_path": str(self.paths.metrics_path),
            "checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path is not None else None,
            "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path is not None else None,
            "final_step": int(self.global_step),
            "final_iteration": int(iteration),
            "best_iteration": int(best_iteration),
            "best_return": float(best_return),
            "best_normalized_score": float(best_normalized),
            "latest_train_metrics": self.latest_train_metrics,
            "latest_evaluation": self.latest_evaluation_metrics,
            "best_evaluation": self.best_evaluation_metrics,
        }
        write_json(self.paths.summary_path, summary)
        total_time = time.perf_counter() - train_start
        self._log(
            f"training complete final_iteration={iteration} final_step={self.global_step} "
            f"best_return={best_return:.4f} best_normalized_score={best_normalized:.4f} "
            f"duration={self._format_duration(total_time)}"
        )
        return summary
