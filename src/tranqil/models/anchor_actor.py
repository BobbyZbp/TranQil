"""Anchor-mode Decision Transformer actor for QT replication."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model


class AnchorDecisionTransformer(nn.Module):
    """Interleaved RTG/state/action transformer matching QT token semantics."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        max_length: int,
        max_ep_len: int,
        n_layer: int,
        n_head: int,
        dropout: float,
        activation_function: str = "relu",
        action_tanh: bool = True,
        sar: bool = False,
        scale: float = 1.0,
        rtg_no_q: bool = False,
        infer_no_q: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.max_length = int(max_length)
        self.max_ep_len = int(max_ep_len)
        self.n_head = int(n_head)
        self.sar = bool(sar)
        self.scale = float(scale)
        self.rtg_no_q = bool(rtg_no_q)
        self.infer_no_q = bool(infer_no_q)

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_reward = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(self.act_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # Match the original QT repo, which vendors a GPT2Model variant with the
        # positional-embedding add removed (see trajectory_gpt2.py). Zeroing wpe
        # and freezing it is mathematically identical: we supply our own timestep
        # embeddings already, so GPT-2's position channel must contribute nothing.
        gpt2_config = GPT2Config(
            vocab_size=1,
            n_embd=self.hidden_size,
            n_layer=int(n_layer),
            n_head=self.n_head,
            n_inner=4 * self.hidden_size,
            activation_function=activation_function,
            n_positions=1024,
            resid_pdrop=float(dropout),
            attn_pdrop=float(dropout),
            embd_pdrop=float(dropout),
        )
        self.transformer = GPT2Model(gpt2_config)
        self.transformer.wpe.weight.data.zero_()
        self.transformer.wpe.weight.requires_grad = False

        self.predict_state = nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_reward = nn.Linear(self.hidden_size, 1)

    def _stack_inputs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        if self.sar:
            if rewards is None:
                raise ValueError("rewards must be provided when sar=True.")
            reward_embeddings = self.embed_reward(rewards / self.scale) + time_embeddings
            stacked = torch.stack((state_embeddings, action_embeddings, reward_embeddings), dim=1)
        else:
            return_embeddings = return_embeddings + time_embeddings
            stacked = torch.stack((return_embeddings, state_embeddings, action_embeddings), dim=1)
        stacked = stacked.permute(0, 2, 1, 3).reshape(states.shape[0], states.shape[1] * 3, self.hidden_size)
        return self.embed_ln(stacked)

    def _stack_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(attention_mask.shape[0], attention_mask.shape[1] * 3)
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        returns_to_go: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        del targets
        if returns_to_go is None or timesteps is None:
            raise ValueError("returns_to_go and timesteps are required.")
        if attention_mask is None:
            attention_mask = torch.ones(
                (states.shape[0], states.shape[1]),
                dtype=torch.long,
                device=states.device,
            )

        stacked_inputs = self._stack_inputs(states, actions, rewards, returns_to_go, timesteps)
        stacked_attention_mask = self._stack_attention_mask(attention_mask.to(dtype=torch.long))
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        hidden = transformer_outputs.last_hidden_state
        hidden = hidden.reshape(states.shape[0], states.shape[1], 3, self.hidden_size).permute(0, 2, 1, 3)

        if self.sar:
            action_preds = self.predict_action(hidden[:, 0])
            reward_preds = self.predict_reward(hidden[:, 1])
            state_preds = self.predict_state(hidden[:, 2])
        else:
            action_preds = self.predict_action(hidden[:, 1])
            state_preds = self.predict_state(hidden[:, 2])
            reward_preds = None
        return state_preds, action_preds, reward_preds

    def get_action(
        self,
        critic: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None = None,
        returns_to_go: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if rewards is None or returns_to_go is None or timesteps is None:
            raise ValueError("rewards, returns_to_go, and timesteps are required for anchor rollout.")

        states = states.reshape(1, -1, self.state_dim).repeat_interleave(repeats=50, dim=0)
        actions = actions.reshape(1, -1, self.act_dim).repeat_interleave(repeats=50, dim=0)
        rewards = rewards.reshape(1, -1, 1).repeat_interleave(repeats=50, dim=0)
        timesteps = timesteps.reshape(1, -1).repeat_interleave(repeats=50, dim=0)

        bs = int(returns_to_go.shape[0])
        returns_to_go = returns_to_go.reshape(bs, -1, 1).repeat_interleave(repeats=50 // bs, dim=0)
        returns_to_go = torch.cat(
            [
                returns_to_go,
                torch.randn(
                    (50 - returns_to_go.shape[0], returns_to_go.shape[1], 1),
                    device=returns_to_go.device,
                ),
            ],
            dim=0,
        )

        states = states[:, -self.max_length :]
        actions = actions[:, -self.max_length :]
        returns_to_go = returns_to_go[:, -self.max_length :]
        rewards = rewards[:, -self.max_length :]
        timesteps = timesteps[:, -self.max_length :]

        pad = self.max_length - states.shape[1]
        attention_mask = torch.cat(
            [
                torch.zeros(pad, device=states.device, dtype=torch.long),
                torch.ones(states.shape[1], device=states.device, dtype=torch.long),
            ],
            dim=0,
        ).reshape(1, -1).repeat_interleave(repeats=50, dim=0)

        states = torch.cat(
            [torch.zeros((states.shape[0], pad, self.state_dim), device=states.device), states],
            dim=1,
        ).to(dtype=torch.float32)
        actions = torch.cat(
            [torch.zeros((actions.shape[0], pad, self.act_dim), device=actions.device), actions],
            dim=1,
        ).to(dtype=torch.float32)
        rewards = torch.cat(
            [torch.zeros((rewards.shape[0], pad, 1), device=rewards.device), rewards],
            dim=1,
        ).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [torch.zeros((returns_to_go.shape[0], pad, 1), device=returns_to_go.device), returns_to_go],
            dim=1,
        ).to(dtype=torch.float32)
        timesteps = torch.cat(
            [torch.zeros((timesteps.shape[0], pad), device=timesteps.device, dtype=torch.long), timesteps],
            dim=1,
        ).to(dtype=torch.long)

        returns_to_go[bs:, -1] = returns_to_go[bs:, -1] + torch.randn_like(returns_to_go[bs:, -1]) * 0.1
        if not self.rtg_no_q:
            returns_to_go[-1, -1] = critic.q_min(states[-1:, -2], actions[-1:, -2]).flatten() - rewards[-1, -2] / self.scale

        _, action_preds, _ = self.forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        state_rpt = states[:, -1, :]
        action_preds = action_preds[:, -1, :]
        q_values = critic.q_min(state_rpt, action_preds).flatten()
        idx = torch.multinomial(F.softmax(q_values, dim=-1), 1)

        if not self.infer_no_q:
            return action_preds[idx]
        return action_preds[0]
