from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tranqil.models.anchor_actor import AnchorDecisionTransformer
from tranqil.models.anchor_critic import AnchorDoubleQCritic


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


def _make_critic() -> AnchorDoubleQCritic:
    return AnchorDoubleQCritic(state_dim=8, action_dim=3, hidden_dim=32)


def _make_inputs(context_len: int, num_targets: int) -> dict[str, torch.Tensor]:
    return {
        "states": torch.randn(1, context_len, 8),
        "actions": torch.randn(1, context_len, 3),
        "rewards": torch.randn(1, context_len, 1),
        # Matches rollout shape: target_return grows with timesteps → (bs, K).
        "returns_to_go": torch.randn(num_targets, context_len),
        "timesteps": torch.arange(context_len).unsqueeze(0).long(),
    }


class GetActionTests(unittest.TestCase):
    def test_returns_finite_action_tensor(self) -> None:
        actor = _make_actor()
        critic = _make_critic()
        actor.eval()
        critic.eval()

        inputs = _make_inputs(context_len=3, num_targets=3)
        with torch.no_grad():
            action = actor.get_action(critic, **inputs)

        self.assertEqual(action.numel(), 3)
        self.assertTrue(torch.isfinite(action).all().item())

    def test_forward_always_sees_fifty_candidates(self) -> None:
        actor = _make_actor()
        critic = _make_critic()
        actor.eval()
        critic.eval()

        seen_batch_sizes: list[int] = []
        real_forward = actor.forward

        def capturing_forward(*args, **kwargs):
            states_arg = kwargs.get("states") if "states" in kwargs else args[0]
            seen_batch_sizes.append(int(states_arg.shape[0]))
            return real_forward(*args, **kwargs)

        for num_targets in (1, 2, 3, 6):
            with mock.patch.object(actor, "forward", side_effect=capturing_forward):
                inputs = _make_inputs(context_len=3, num_targets=num_targets)
                with torch.no_grad():
                    actor.get_action(critic, **inputs)

        self.assertEqual(seen_batch_sizes, [50, 50, 50, 50])

    def test_infer_no_q_returns_primary_candidate_deterministically(self) -> None:
        actor = _make_actor(infer_no_q=True)
        critic = _make_critic()
        actor.eval()
        critic.eval()

        torch.manual_seed(0)
        inputs = _make_inputs(context_len=3, num_targets=3)
        with torch.no_grad():
            action_a = actor.get_action(critic, **inputs)
            action_b = actor.get_action(critic, **{k: v.clone() for k, v in inputs.items()})

        self.assertTrue(torch.allclose(action_a, action_b))

    def test_rtg_no_q_skips_bootstrap(self) -> None:
        actor = _make_actor(rtg_no_q=True)
        critic = _make_critic()
        actor.eval()
        critic.eval()

        captured_rtg: list[torch.Tensor] = []
        real_forward = actor.forward

        def capturing_forward(*args, **kwargs):
            captured_rtg.append(kwargs["returns_to_go"].detach().clone())
            return real_forward(*args, **kwargs)

        inputs = _make_inputs(context_len=3, num_targets=3)
        with mock.patch.object(actor, "forward", side_effect=capturing_forward):
            with torch.no_grad():
                actor.get_action(critic, **inputs)

        self.assertEqual(len(captured_rtg), 1)
        rtg = captured_rtg[0]
        bs = int(inputs["returns_to_go"].shape[0])
        repeated_targets = inputs["returns_to_go"].reshape(bs, -1, 1).repeat_interleave(
            repeats=50 // bs,
            dim=0,
        )
        # Original QT jitters `returns_to_go[bs:, -1]`, so only the first
        # `bs` rows retain the untouched pre-jitter layout from
        # `repeat_interleave`, which for bs=3 are repeated copies of the first
        # target rather than one row per target.
        for row_idx in range(bs):
            self.assertAlmostEqual(
                float(rtg[row_idx, -1, 0]),
                float(repeated_targets[row_idx, -1, 0]),
                places=5,
            )

    def test_q_bootstrap_applied_to_last_candidate(self) -> None:
        actor = _make_actor(rtg_no_q=False)
        critic = _make_critic()
        actor.eval()
        critic.eval()

        captured_rtg: list[torch.Tensor] = []
        captured_states: list[torch.Tensor] = []
        captured_actions: list[torch.Tensor] = []
        captured_rewards: list[torch.Tensor] = []
        real_forward = actor.forward

        def capturing_forward(*args, **kwargs):
            captured_rtg.append(kwargs["returns_to_go"].detach().clone())
            captured_states.append(kwargs["states"].detach().clone())
            captured_actions.append(kwargs["actions"].detach().clone())
            captured_rewards.append(kwargs["rewards"].detach().clone())
            return real_forward(*args, **kwargs)

        inputs = _make_inputs(context_len=3, num_targets=3)
        with mock.patch.object(actor, "forward", side_effect=capturing_forward):
            with torch.no_grad():
                actor.get_action(critic, **inputs)

        rtg = captured_rtg[0]
        s = captured_states[0]
        a = captured_actions[0]
        r = captured_rewards[0]
        with torch.no_grad():
            expected = (
                critic.q_min(s[-1:, -2], a[-1:, -2]).flatten() - r[-1, -2] / actor.scale
            )
        self.assertAlmostEqual(float(rtg[-1, -1, 0]), float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
