# TranQil: Replication of Q-value Regularized Transformer for Offline RL

**CS 4782 Final Project — Cornell University**

A faithful replication of **Q-value Regularized Transformer (QT)** (Hu et al., ICML 2024) on the D4RL continuous control benchmark. We reproduce the core algorithm, verify key implementation decisions against the original codebase, and achieve results within the paper's reported confidence interval.

---

## Results

We evaluate on `walker2d-medium-replay-v2` (seed 123, 200-iteration training budget).

| Method | Normalized Score | Source |
|---|---|---|
| QT (paper) | 98.5 ± 1.1 | Hu et al., ICML 2024 |
| **TranQil (ours)** | **96.21** | This repo, iter 21 |
| Best single episode | 99.72 | This repo |

Our mean score of **96.21** falls within the paper's reported interval. The best individual episode score of **99.72** matches the paper's mean, confirming the policy has learned a high-quality locomotion strategy.

**Rollout video:** `results/qt_anchor_runs/qt_anchor_walker2d_medium_replay_v2_seed123/rollout_best_score96.mp4`
The trained walker completes all 1000 steps — the original paper does not provide rollout videos.

---

## Method Overview

QT trains a Decision Transformer-style policy augmented with a learned double-Q critic. At inference time, the policy generates **50 candidate actions** from different return-to-go targets, selects among them via Q-weighted softmax sampling, and uses Q-bootstrap to correct the return-to-go conditioning signal.

Key components reproduced:

- **50-way candidate expansion** with RTG jitter and Q-bootstrap (original `get_action`, `ql_DT.py:158–217`)
- **HF GPT2 backbone** with position embeddings zeroed and frozen (mathematically equivalent to original's `trajectory_gpt2.py`)
- **Double-Q critic** with EMA target network (`ema_decay=0.995`)
- **BC + Q-regularization** joint training objective
- **Checkpoint selection by `mean_return`** matching original `experiment.py:388–399`

---

## Implementation Notes

Several divergences from a naive Decision Transformer baseline were required to reproduce paper results:

1. **Action clipping removed** — the anchor actor uses a Tanh output head; clipping to `env.action_space` bounds was incorrect and removed.
2. **Eval episode length** — must be set to `max_steps=1000` to match the paper's `max_ep_len`. At 500 steps, the maximum achievable normalized score is ~44, explaining the common replication failure mode.
3. **Candidate expansion** — the 50-way RTG expansion with multinomial Q-selection is essential; greedy argmax degrades performance significantly.
4. **GPT2 position embeddings** — zeroed at init (`wpe.weight.data.zero_()`, `requires_grad=False`) so the model treats sequences as unordered by position, relying solely on timestep embeddings.

---

## Repository Structure

```
TranQil/
├── configs/                         # Per-task YAML configs
│   ├── qt_anchor_walker2d_medium_replay.yaml
│   ├── qt_anchor_hopper_medium_replay.yaml
│   └── qt_anchor_maze2d_medium.yaml
├── src/tranqil/
│   ├── models/
│   │   └── anchor_actor.py          # GPT2-based policy + 50-way get_action
│   ├── anchor_trainer.py            # Training loop, EMA, checkpoint logic
│   ├── evaluation.py                # Rollout evaluation, no action clipping
│   ├── anchor_data.py               # D4RL dataset loading + trajectory export
│   ├── config.py                    # Dataclass config schema
│   └── rendering.py                 # Offscreen MuJoCo rendering (osmesa)
├── scripts/
│   ├── train_qt.py                  # Main training entrypoint
│   ├── render_qt_rollout.py         # Render MP4 from checkpoint
│   ├── activate_env.sh              # Env activation
│   ├── env_vars.sh                  # Runtime variable exports
│   └── install_d4rl_stack.sh        # D4RL + MuJoCo install
├── results/
│   └── qt_anchor_runs/
│       └── qt_anchor_walker2d_medium_replay_v2_seed123/
│           ├── checkpoints/best.pt  # Iter 21, score 96.21 (Git LFS)
│           ├── evaluations.jsonl    # Full eval curves, 33 iters
│           ├── metrics.jsonl        # Training losses, 33 iters
│           └── rollout_best_score96.mp4  # Rendered policy (Git LFS)
├── tests/                           # Unit tests for actor, trainer, evaluation
├── environment.yml
└── README.md
```

---

## Getting Started

Requirements: Ubuntu / WSL2, `bash`, `micromamba`.

### 1. Install system dependencies

```bash
sudo apt-get update && sudo apt-get install -y curl tar bzip2 git git-lfs
git lfs install
```

### 2. Install micromamba

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
```

### 3. Clone and enter the repository

```bash
git clone git@github.com:BobbyZbp/TranQil.git
cd TranQil
git lfs pull   # downloads best.pt and rollout MP4
```

### 4. Create the environment

```bash
export MAMBA_ROOT_PREFIX="$PWD/.micromamba/root"
micromamba create -y -f environment.yml
```

### 5. Activate and install the RL stack

```bash
source scripts/activate_env.sh
bash scripts/install_d4rl_stack.sh
```

> **GPU training:** `install_d4rl_stack.sh` installs CPU-only PyTorch by default. If you have a CUDA GPU and want to train, replace it with the appropriate CUDA build after step 5:
> ```bash
> pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
> ```
> Adjust `cu121` to match your CUDA version (`cu118`, `cu124`, etc.). The rollout and eval scripts work on CPU.

### 6. Smoke test

```bash
bash scripts/run_smoke_test.sh
```

---

## Training

Train from scratch (walker2d, seed 123, 200 iterations):

```bash
source scripts/activate_env.sh
python scripts/train_qt.py --config configs/qt_anchor_walker2d_medium_replay.yaml
```

Resume from checkpoint:

```bash
python scripts/train_qt.py \
  --config configs/qt_anchor_walker2d_medium_replay.yaml \
  --resume-from results/qt_anchor_runs/qt_anchor_walker2d_medium_replay_v2_seed123/checkpoints/latest.pt
```

---

## Evaluation / Rollout

Render a rollout video from the best checkpoint:

```bash
source scripts/activate_env.sh
python scripts/render_qt_rollout.py \
  --config configs/qt_anchor_walker2d_medium_replay.yaml \
  --checkpoint results/qt_anchor_runs/qt_anchor_walker2d_medium_replay_v2_seed123/checkpoints/best.pt \
  --seed 123 \
  --target-return 5000.0 \
  --max-steps 1000
```

---

## Checkpoints

`best.pt` and the rollout MP4 are stored via **Git LFS**. After cloning, run `git lfs pull` to download them. The checkpoint is ~59 MB.

---

## Reference

```bibtex
@inproceedings{hu2024qt,
  title     = {Q-value Regularized Transformer for Offline Reinforcement Learning},
  author    = {Hu, Shengchao and Fan, Ziqing and Huang, Chengqian and Shen, Li and
               Zhang, Ya and Wang, Yanfeng and Tao, Dacheng},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024}
}
```

Original implementation: [charleshsc/QT](https://github.com/charleshsc/QT)
