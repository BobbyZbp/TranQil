"""Transformer actor used by the QT training pipeline."""

from __future__ import annotations

import torch
from torch import nn


class QTActor(nn.Module):
    """Causal Transformer policy over observation/RTG/timestep tokens."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: int,
        max_timestep: int,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if max_timestep <= 0:
            raise ValueError("max_timestep must be positive.")

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_timestep = int(max_timestep)
        self.num_heads = int(num_heads)

        self.observation_embed = nn.Linear(self.observation_dim, self.hidden_dim)
        self.return_embed = nn.Linear(1, self.hidden_dim)
        self.timestep_embed = nn.Embedding(self.max_timestep, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

    def _attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return a per-sample causal mask that is safe for left-padded windows."""

        batch_size, sequence_length = attention_mask.shape
        device = attention_mask.device

        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        padded_tokens = attention_mask <= 0.0
        padded_key_mask = padded_tokens.unsqueeze(1).expand(-1, sequence_length, -1)
        padded_query_mask = padded_tokens.unsqueeze(2).expand(-1, -1, sequence_length)
        self_only_mask = padded_query_mask.clone()

        diagonal = torch.eye(sequence_length, device=device, dtype=torch.bool).unsqueeze(0)
        self_only_mask &= ~diagonal

        combined_mask = causal_mask | padded_key_mask | self_only_mask
        combined_mask[:, torch.arange(sequence_length), torch.arange(sequence_length)] &= ~padded_tokens
        return combined_mask.repeat_interleave(self.num_heads, dim=0)

    def encode(
        self,
        observations: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one sequence batch into causal hidden states."""

        timesteps = timesteps.clamp(min=0, max=self.max_timestep - 1)
        hidden = (
            self.observation_embed(observations)
            + self.return_embed(returns_to_go.unsqueeze(-1))
            + self.timestep_embed(timesteps)
        )
        token_mask = attention_mask.unsqueeze(-1)
        hidden = hidden * token_mask
        hidden = self.input_dropout(self.input_norm(hidden))

        attention_bias = self._attention_mask(attention_mask)
        hidden = self.transformer(
            hidden,
            mask=attention_bias,
        )
        hidden = self.output_norm(hidden)
        hidden = torch.nan_to_num(hidden)
        return hidden * token_mask

    def forward(
        self,
        observations: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one continuous action per sequence position."""

        hidden = self.encode(
            observations=observations,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        return torch.tanh(self.action_head(hidden))

    def predict_last_action(
        self,
        observations: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper for the right-aligned transition action."""

        return self(
            observations=observations,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )[:, -1, :]
