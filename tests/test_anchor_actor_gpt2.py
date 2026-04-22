from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.models.anchor_actor import AnchorDecisionTransformer


def _make_actor(**overrides) -> AnchorDecisionTransformer:
    kwargs = dict(
        state_dim=8,
        action_dim=3,
        hidden_size=32,
        max_length=5,
        max_ep_len=1000,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        scale=1000.0,
    )
    kwargs.update(overrides)
    return AnchorDecisionTransformer(**kwargs)


class GPT2BackboneTests(unittest.TestCase):
    def test_wpe_is_zero_and_frozen(self) -> None:
        actor = _make_actor()
        wpe = actor.transformer.wpe.weight
        self.assertTrue(torch.all(wpe == 0).item())
        self.assertFalse(wpe.requires_grad)

    def test_forward_output_shapes(self) -> None:
        actor = _make_actor()
        actor.eval()
        batch, seq = 4, 5
        states = torch.randn(batch, seq, 8)
        actions = torch.randn(batch, seq, 3)
        rewards = torch.randn(batch, seq, 1)
        rtg = torch.randn(batch, seq, 1)
        timesteps = torch.arange(seq).unsqueeze(0).expand(batch, -1).long()

        with torch.no_grad():
            state_preds, action_preds, reward_preds = actor(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=rtg,
                timesteps=timesteps,
            )

        self.assertEqual(state_preds.shape, (batch, seq, 8))
        self.assertEqual(action_preds.shape, (batch, seq, 3))
        self.assertIsNone(reward_preds)

    def test_sar_mode_produces_reward_preds(self) -> None:
        actor = _make_actor(sar=True)
        actor.eval()
        batch, seq = 2, 5
        states = torch.randn(batch, seq, 8)
        actions = torch.randn(batch, seq, 3)
        rewards = torch.randn(batch, seq, 1)
        rtg = torch.randn(batch, seq, 1)
        timesteps = torch.arange(seq).unsqueeze(0).expand(batch, -1).long()

        with torch.no_grad():
            state_preds, action_preds, reward_preds = actor(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=rtg,
                timesteps=timesteps,
            )

        self.assertEqual(state_preds.shape, (batch, seq, 8))
        self.assertEqual(action_preds.shape, (batch, seq, 3))
        self.assertEqual(reward_preds.shape, (batch, seq, 1))

    def test_padded_positions_are_finite(self) -> None:
        actor = _make_actor()
        actor.eval()
        batch, seq = 2, 5
        states = torch.randn(batch, seq, 8)
        actions = torch.randn(batch, seq, 3)
        rewards = torch.randn(batch, seq, 1)
        rtg = torch.randn(batch, seq, 1)
        timesteps = torch.arange(seq).unsqueeze(0).expand(batch, -1).long()
        attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long)

        with torch.no_grad():
            state_preds, action_preds, _ = actor(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=rtg,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )

        self.assertTrue(torch.isfinite(state_preds).all().item())
        self.assertTrue(torch.isfinite(action_preds).all().item())

    def test_padded_values_do_not_affect_unpadded_outputs(self) -> None:
        """Causal mask + attention mask must prevent padded positions from
        influencing the later unpadded positions."""
        actor = _make_actor()
        actor.eval()
        batch, seq = 1, 5
        torch.manual_seed(0)
        states = torch.randn(batch, seq, 8)
        actions = torch.randn(batch, seq, 3)
        rewards = torch.randn(batch, seq, 1)
        rtg = torch.randn(batch, seq, 1)
        timesteps = torch.arange(seq).unsqueeze(0).long()
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)

        with torch.no_grad():
            _, action_preds_a, _ = actor(
                states=states.clone(),
                actions=actions.clone(),
                rewards=rewards.clone(),
                returns_to_go=rtg.clone(),
                timesteps=timesteps.clone(),
                attention_mask=attention_mask,
            )

        # Scramble only padded positions (indices 0, 1) — unpadded outputs
        # should not change.
        states_b = states.clone()
        actions_b = actions.clone()
        rewards_b = rewards.clone()
        rtg_b = rtg.clone()
        states_b[:, :2] = torch.randn_like(states_b[:, :2]) * 1000.0
        actions_b[:, :2] = torch.randn_like(actions_b[:, :2]) * 1000.0
        rewards_b[:, :2] = torch.randn_like(rewards_b[:, :2]) * 1000.0
        rtg_b[:, :2] = torch.randn_like(rtg_b[:, :2]) * 1000.0

        with torch.no_grad():
            _, action_preds_b, _ = actor(
                states=states_b,
                actions=actions_b,
                rewards=rewards_b,
                returns_to_go=rtg_b,
                timesteps=timesteps.clone(),
                attention_mask=attention_mask,
            )

        # Compare unpadded positions (indices 2, 3, 4).
        self.assertTrue(
            torch.allclose(action_preds_a[:, 2:], action_preds_b[:, 2:], atol=1e-5)
        )

    def test_gpt2_inner_and_activation_defaults(self) -> None:
        actor = _make_actor(hidden_size=64)
        config = actor.transformer.config
        self.assertEqual(config.n_inner, 4 * 64)
        self.assertEqual(config.activation_function, "relu")


if __name__ == "__main__":
    unittest.main()
