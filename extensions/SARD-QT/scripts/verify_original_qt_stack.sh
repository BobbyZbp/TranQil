#!/usr/bin/env bash

set -euo pipefail

python - <<'PY'
import importlib.util

mods = ["torch", "gym", "d4rl", "mujoco_py", "mjrl", "transformers", "tensorboard"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing imports: {missing}")

import gym
import d4rl
import torch
import transformers
from decision_transformer.models.ql_DT import Critic, DecisionTransformer

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
print(f"DecisionTransformer={DecisionTransformer.__name__}, Critic={Critic.__name__}")

for env_name in ["antmaze-large-diverse-v0", "antmaze-large-diverse-v2"]:
    env = gym.make(env_name)
    print(f"{env_name}: obs={env.observation_space.shape}, action={env.action_space.shape}")
PY
