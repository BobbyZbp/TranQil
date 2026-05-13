<p align="center" width="100%">
</p>

<div id="top" align="center">

# TranQil / QSC-QT

### Q-Spectral Candidate Generation for Q-Transformer

<img src="https://img.shields.io/badge/Status-Research%20Extension-orange.svg" alt="Status">
<img src="https://img.shields.io/badge/Based%20on-QT%20(ICML%202024)-blue.svg" alt="Based on QT">
<img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">

<h4>
| <a href="https://arxiv.org/abs/2405.17098">Original QT Paper</a> |
<a href="https://github.com/charleshsc/QT">Original QT Repo</a> |
</h4>

</div>

---

## Overview

This repository is a research extension of **Q-value Regularized Transformer (QT)** for offline reinforcement learning.

The original QT method combines conditional sequence modeling with Q-guided candidate reranking. At evaluation time, QT generates multiple candidate actions from a return-conditioned Transformer and uses learned Q-values to select among them.

Our work studies a fundamental bottleneck in QT-style reranking:

> When candidates are generated from a single return-to-go (RTG) anchor with small perturbations, they cluster in a narrow region of action space. In this regime, Q-values across candidates become near-uniform, causing multinomial selection to degenerate toward random sampling — a failure mode we call **Q-spread collapse**.

We propose **QSC-QT** (Q-Spectral Candidate generation for Q-Transformer):

> A test-time candidate generation strategy that exploits the **local geometric structure of the Q-function** — specifically, the eigendecomposition of its action-space Hessian — to generate candidates along directions where Q is most discriminative. QSC-QT requires no additional training and provides a theoretical guarantee of second-order Q-spread maximization for fixed candidate budgets.

This project is under active development.

---

## Research Question

**Can the local spectral structure of the Q-function — specifically the eigendecomposition of its action-space Hessian — be exploited at inference time to generate provably Q-discriminable candidates for reranking in offline reinforcement learning?**

This question decomposes into three sub-questions:

1. **Diagnostic**: Does QT exhibit Q-spread collapse on sparse-reward tasks, and is candidate clustering (rather than Q-function inaccuracy) the root cause?
2. **Method**: Can the eigenvectors of Q's action-space Hessian provide structurally meaningful directions for candidate diversification?
3. **Validation**: Does spectral diversification yield empirical improvement over both vanilla QT and behavior-only baselines (DT) across reward regimes?

---

## Relationship to Original QT

This repository is forked from the original QT implementation:

- Original paper: [Q-value Regularized Transformer for Offline Reinforcement Learning](https://arxiv.org/abs/2405.17098)
- Original repository: [charleshsc/QT](https://github.com/charleshsc/QT)

The original QT paper was published at ICML 2024.

Our work does **not** replace QT. Instead, it modifies only the candidate generation step at inference time, leaving QT's training pipeline (transformer + critic + Q-learning loss) entirely unchanged.

---

## Method Summary

### Baseline QT (unchanged)

Standard QT generates candidates from a single RTG anchor with small Gaussian noise:

```python
candidates = transformer(state, history, rtg=rtg_target + noise)  # [50, action_dim]
q_values = critic(state, candidates)
selected = multinomial(softmax(q_values))
return candidates[selected]
```

When candidates cluster in action space, `q_values` become near-uniform and selection degenerates.

### QSC-QT (our method)

QSC-QT replaces the candidate generation step with spectral diversification:

```python
# Step 1: Get the transformer's central prediction
a_center = transformer(state, history, rtg=rtg_target)

# Step 2: Compute Q's action-space Hessian at (state, a_center)
# H = d^2 Q / d a^2, shape [action_dim, action_dim]
H = compute_action_hessian(critic, state, a_center)
H_reg = H + epsilon * eye(action_dim)  # numerical regularization

# Step 3: Eigendecomposition - identify Q-discriminative directions
eigenvalues, eigenvectors = linalg.eigh(H_reg)

# Step 4: Generate candidates along top-K eigenvectors of |H|
top_k_idx = argsort(abs(eigenvalues), descending=True)[:K]
spectral_directions = eigenvectors[:, top_k_idx]  # [action_dim, K]

candidates = [a_center]
for k in range(K):
    direction = spectral_directions[:, k]
    for scale in linspace(-2.0, 2.0, n_per_dir) * sigma:
        candidates.append(a_center + scale * direction)
candidates = clip(stack(candidates), action_low, action_high)  # [50, action_dim]

# Step 5: Standard Q-rerank (unchanged from QT)
q_values = critic(state, candidates)
selected = multinomial(softmax(q_values))
return candidates[selected]
```

### Theoretical Property

For Q twice-differentiable in action and a candidate budget of N, candidates sampled along the top-K Hessian eigenvectors **maximize the second-order Q-spread** of the candidate set, by Taylor expansion of Q around `a_center`. This is a mathematical guarantee, not an empirical hope.

When the Hessian degenerates (all eigenvalues near zero, indicating Q is locally flat), the regularization term ensures eigenvectors remain well-defined and the method falls back gracefully toward vanilla QT behavior.

---

## Current Experimental Focus

We prioritize tasks where the action-space Hessian is numerically well-behaved, then validate generalization to harder regimes.

### Primary tasks (numerically safe)

| Task | Action dim | Reward | Role |
|---|---|---|---|
| `maze2d-large-v1` | 2 | sparse | Main showcase: low-dim action, Hessian highly stable |
| `walker2d-medium-v2` | 6 | dense | Dense-reward sanity check |
| `hopper-medium-v2` | 3 | dense | Small action dim, narrow data distribution |

### Stretch task (if time permits)

| Task | Action dim | Reward | Role |
|---|---|---|---|
| `antmaze-medium-diverse-v0` | 8 | sparse | Hard test, may require Hessian fallback |

### Methods compared

| Method | Purpose |
|---|---|
| **DT** (no-Q) | Reference: pure behavior-conditioned generation, no Q signal |
| **QT** (vanilla) | Original baseline: single-RTG candidates + Q-rerank |
| **QSC-QT** (ours) | Spectral candidate generation + Q-rerank |

### Key diagnostic metrics

- **Q-spread**: standard deviation of Q-values across the 50 candidates per timestep
- **Hessian eigenvalue distribution**: tracked across all evaluation states to validate spectral structure assumptions
- **Candidate Q-spread before vs. after QSC**: direct measurement of method's mechanism
- **Success rate / normalized return**: standard performance metrics

### Critical ablations

| Ablation | Tests |
|---|---|
| QSC with **random orthogonal eigenvectors** (replace Hessian eigenvectors with random directions) | Sanity check: is the Hessian actually informative, or does any diversification work? |
| QSC **without Hessian regularization** | Robustness to ill-conditioning |
| **K sweep** (K = 3, 5, 7) | How many spectral directions to use |

The **random eigenvector** ablation is the single most important comparison: it isolates the contribution of Hessian-based directions versus generic diversification.

---

## Installation

Clone the repository:

```bash
git clone git@github.com:BobbyZbp/TranQil.git
cd TranQil
```

Create and activate a conda environment:

```bash
conda create -n tranqil python=3.10 -y
conda activate tranqil
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the appropriate PyTorch build for your CUDA version from the official PyTorch website if needed.

---

## Quick Start

### Original QT-style command (baseline)

```bash
python experiment.py --seed 123 \
    --env hopper --dataset medium \
    --eta 1.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount
```

### TranQil / QSC-QT-style command

For configuration-based runs in this repository:

```bash
# Vanilla QT baseline
python scripts/train_qt.py \
    --config configs/qt_maze2d_large_v1.yaml

# QSC-QT (our method)
python scripts/train_qt.py \
    --config configs/qsc_maze2d_large_v1.yaml
```

For resuming from a checkpoint:

```bash
python scripts/train_qt.py \
    --config configs/qsc_maze2d_large_v1.yaml \
    --resume-from results/qsc_runs/qsc_maze2d_large_v1/checkpoints/latest.pt
```

For long-running experiments inside `tmux`:

```bash
tmux new-session -d -s qsc_maze2d \
  "bash -lc 'cd /path/to/TranQil && source scripts/activate_env.sh && python scripts/train_qt.py --config configs/qsc_maze2d_large_v1.yaml'"
```

---

## Suggested Experiment Workflow

### 1. Reproduce QT baseline

Confirm this fork reproduces the original QT baseline on the chosen task before any modification.

```bash
python scripts/train_qt.py --config configs/qt_maze2d_large_v1.yaml
```

### 2. Log Q-spread diagnostics

During evaluation, record per-step Q-value distributions across the 50 candidates. Use these to characterize the Q-spread collapse hypothesis empirically before testing QSC-QT.

```text
q_spread_t = std(Q(s_t, c_i) for i in 1..50)
```

Plot histograms across all evaluation steps for both dense (walker, hopper) and sparse (maze2d, antmaze) tasks. Validate that sparse-reward tasks show systematically narrower Q-spread distributions.

### 3. Implement QSC-QT and validate Hessian structure

Before running full experiments:

- Confirm Hessian eigenvalues are non-degenerate on the chosen task (most states should have at least one eigenvalue with magnitude above the regularization floor).
- Confirm that candidates generated along Hessian eigenvectors achieve substantially higher Q-spread than vanilla candidates.

These two checks validate the method's mechanism before committing GPU time to full experiments.

### 4. Run main experiments

Three methods x three tasks x three seeds (27 runs).

### 5. Run ablations

The most critical ablation is **QSC with random orthogonal directions** — this isolates whether the Hessian structure itself is the source of improvement.

---

## Repository Structure

```text
.
├── configs/                 # YAML configs (per method, per task)
├── scripts/                 # Training, evaluation, analysis scripts
├── src/
│   └── tranqil/             # Core implementation
│       ├── methods/
│       │   └── qsc.py       # QSC-QT spectral candidate generation
│       └── models/
├── tests/                   # Unit tests (Hessian, eigendecomposition)
├── results/                 # Experiment outputs and checkpoints
├── analysis/                # Figure generation scripts
├── run.sh                   # Original QT-style launcher
├── experiment.py            # Original QT-style entry point
└── README.md
```

---

## Development Principles

1. **Keep the original QT baseline reproducible.** All modifications occur strictly at inference time; QT's training pipeline is unchanged.
2. **Single, well-isolated method.** QSC-QT introduces one mechanism (spectral candidate generation). The codebase explicitly avoids combining auxiliary components such as reward-shaping, behavior-cloning fallbacks, or counterfactual scorers.
3. **Track diagnostic metrics alongside performance.** Q-spread and Hessian eigenvalue distributions are recorded by default to validate the method's mechanism, not only its end-to-end results.
4. **Maintain clear attribution.** All training infrastructure derives from the original QT implementation.
5. **Numerical safety first.** Hessian computation includes regularization; tasks are prioritized by action-space dimensionality and reward density.

The current main research line is:

```text
QT baseline reproduction
-> Q-spread collapse diagnostic on sparse-reward tasks
-> QSC-QT implementation and Hessian validation
-> Main experiments on maze2d-large, walker2d-medium, hopper-medium
-> Critical ablations (random eigenvectors, K sweep, regularization)
-> Stretch validation on antmaze-medium-diverse
```

Components such as reward-neighborhood shaping, bridge scoring, support gating, reliability gating, trust gating (TQT), or RTG-anchor diversification (SARD/DRCG) are **not** part of the current mainline method. Such components may have been explored in prior branches but are explicitly excluded from the QSC-QT contribution to maintain a focused workshop submission.

---

## Original QT

This project builds on the official implementation of:

> **Q-value Regularized Transformer for Offline Reinforcement Learning**
> Shengchao Hu, Ziqing Fan, Chaoqin Huang, Li Shen, Ya Zhang, Yanfeng Wang, Dacheng Tao
> ICML 2024

QT combines conditional sequence modeling with Q-value regularization to improve offline reinforcement learning. It uses value estimates to guide action selection while staying close to the behavior policy learned from offline data.

Please refer to the original repository for the baseline implementation and original experiment setup:

[https://github.com/charleshsc/QT](https://github.com/charleshsc/QT)

---

## Citation

If you use this repository, please cite the original QT paper:

```bibtex
@inproceedings{QT,
    title={Q-value Regularized Transformer for Offline Reinforcement Learning},
    author={Hu, Shengchao and Fan, Ziqing and Huang, Chaoqin and Shen, Li and Zhang, Ya and Wang, Yanfeng and Tao, Dacheng},
    booktitle={International Conference on Machine Learning},
    year={2024},
}
```

If you use the QSC-QT extension, please also cite our project once available.

---

## Acknowledgements

This repository is based on the official implementation of **Q-value Regularized Transformer (QT)**. We thank the QT authors for releasing their code.

The original QT repository also acknowledges Decision Transformer and Diffusion-QL, whose codebases influenced the original implementation.

---

## License

This repository follows the license of the original QT repository unless otherwise specified.

The original QT implementation is released under the Apache 2.0 License.
