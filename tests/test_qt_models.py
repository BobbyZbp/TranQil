from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.models.actor import QTActor
from tranqil.models.critic import DoubleQCritic


class QTModelTests(unittest.TestCase):
    def test_actor_forward_shapes(self) -> None:
        actor = QTActor(
            observation_dim=17,
            action_dim=6,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            ffn_multiplier=2,
            max_timestep=32,
        )
        actor.eval()

        observations = torch.randn(2, 5, 17)
        returns_to_go = torch.randn(2, 5)
        timesteps = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 2]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.float32)

        actions = actor(
            observations=observations,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        self.assertEqual(tuple(actions.shape), (2, 5, 6))
        self.assertTrue(torch.isfinite(actions).all())

    def test_actor_ignores_masked_prefix_tokens(self) -> None:
        actor = QTActor(
            observation_dim=4,
            action_dim=2,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            ffn_multiplier=2,
            max_timestep=16,
        )
        actor.eval()

        shared_valid_observations = torch.tensor(
            [[[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]],
            dtype=torch.float32,
        )
        altered_prefix_observations = shared_valid_observations.clone()
        altered_prefix_observations[:, :2, :] = 123.0

        returns_to_go = torch.tensor([[0.0, 0.0, 4.0, 2.0]], dtype=torch.float32)
        timesteps = torch.tensor([[0, 0, 0, 1]], dtype=torch.long)
        attention_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)

        action_a = actor.predict_last_action(
            observations=shared_valid_observations,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        action_b = actor.predict_last_action(
            observations=altered_prefix_observations,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )

        self.assertTrue(torch.allclose(action_a, action_b, atol=1e-6, rtol=1e-6))

    def test_double_q_critic_shapes(self) -> None:
        critic = DoubleQCritic(
            observation_dim=17,
            action_dim=6,
            hidden_dim=32,
            num_layers=2,
        )

        observations = torch.randn(4, 17)
        actions = torch.randn(4, 6)
        q1, q2 = critic(observations, actions)
        q_min = critic.min(observations, actions)

        self.assertEqual(tuple(q1.shape), (4, 1))
        self.assertEqual(tuple(q2.shape), (4, 1))
        self.assertEqual(tuple(q_min.shape), (4, 1))
        self.assertTrue(torch.isfinite(q1).all())
        self.assertTrue(torch.isfinite(q2).all())
        self.assertTrue(torch.isfinite(q_min).all())


if __name__ == "__main__":
    unittest.main()
