# TranQil: Re-implementing Q-value Regularized Transformer for Offline RL

TranQil is a project repository for reproducing and analyzing **Q-value Regularized Transformer (QT)** on a focused subset of D4RL benchmark tasks. The project follows a paper-first approach: establish a reliable runtime for the legacy D4RL + MuJoCo stack, validate the benchmark tasks end to end, and build a clean re-implementation that supports reproduction, ablation, and stability analysis.

## Abstract

Q-value Regularized Transformer (QT) combines a sequence-model policy with a learned double-Q critic for offline reinforcement learning. The policy is trained with both an imitation-style objective and a Q-guided regularization term that biases action selection toward higher-value actions while remaining close to the behavior policy. This repository is organized around a scoped, reproducible implementation of QT, with particular attention to environment reliability, evaluation hygiene, and experimental clarity.

## Target Paper

- Paper: [Q-value Regularized Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2405.17098)
- Venue: ICML 2024
- Reference implementation: [charleshsc/QT](https://github.com/charleshsc/QT)

## Project Scope

The current benchmark scope is intentionally compact:

- `walker2d-medium-replay-v2`
- `hopper-medium-replay-v2`
- `maze2d-medium-v1`

The broader project goals are:

- a faithful QT reproduction on the scoped tasks
- a compact ablation study
- a focused stability analysis around critic regularization and training behavior

## Repository Overview

The repository is organized around a small runtime pipeline:

`env spec -> activation/runtime vars -> install/repair stack -> smoke validation -> preview rollout validation -> QT implementation`

The main files for that pipeline are:

- [environment.yml](./environment.yml)
- [activate_env.sh](./scripts/activate_env.sh)
- [env_vars.sh](./scripts/env_vars.sh)
- [install_d4rl_stack.sh](./scripts/install_d4rl_stack.sh)
- [patch_mujoco_py_builder.py](./scripts/patch_mujoco_py_builder.py)
- [run_smoke_test.sh](./scripts/run_smoke_test.sh)
- [smoke_test_env.py](./scripts/smoke_test_env.py)
- [run_rollout_preview.sh](./scripts/run_rollout_preview.sh)
- [rollout_preview.py](./scripts/rollout_preview.py)
- [run_benchmark_rollouts.sh](./scripts/run_benchmark_rollouts.sh)

Detailed technical behavior and file-level specifications are documented directly at the top of those files.

## Getting Started

Run these commands from the repository root.

Prerequisite: `micromamba` must already be installed and available on `PATH`.

```bash
command -v micromamba
```

If that command prints nothing, install `micromamba` first and then continue.

Clone the repository and enter it:

```bash
git clone git@github.com:BobbyZbp/TranQil.git
cd TranQil
```

Create the base repo-local environment:

```bash
export MAMBA_ROOT_PREFIX="$PWD/.micromamba/root"
micromamba create -y -f environment.yml
```

Activate the repo-local environment:

```bash
source scripts/activate_env.sh
```

Install or repair the D4RL + MuJoCo stack:

```bash
bash scripts/install_d4rl_stack.sh
```

Run the scoped smoke test:

```bash
bash scripts/run_smoke_test.sh
```

Render a single preview rollout:

```bash
bash scripts/run_rollout_preview.sh \
  --env walker2d-medium-replay-v2 \
  --steps 150 \
  --fps 20 \
  --frame-skip 2
```

Render previews for all scoped tasks:

```bash
bash scripts/run_benchmark_rollouts.sh
```

## Runtime Notes

- The environment setup intentionally follows the legacy `gym` + `d4rl` + `mujoco-py` ecosystem because that stack is the most reproduction-relevant for QT.
- The runtime depends on repo-local environment variables exported by [env_vars.sh](./scripts/env_vars.sh), especially for the MuJoCo path, cache locations, and WSL2 library resolution.
- Preview artifacts are random-policy rollouts generated from live environment interaction. They are not trajectory replays sampled from the D4RL `.hdf5` datasets.

## Repository Layout

- [scripts/](./scripts): environment setup, validation, and preview entrypoints
- [src/tranqil/](./src/tranqil): implementation package root
- [data/](./data): local dataset cache
- [results/](./results): generated previews and future evaluation artifacts

## Citation / Reference

If you use this repository as a project reference, cite the original paper:

```text
Hu, S., Fan, Z., Huang, C., Shen, L., Zhang, Y., Wang, Y., and Tao, D.
Q-value Regularized Transformer for Offline Reinforcement Learning.
Proceedings of the 41st International Conference on Machine Learning, 2024.
```
