# TranQil

TranQil is a course-project workspace for re-implementing and analyzing the stability of **Q-value Regularized Transformer (QT)** for offline reinforcement learning.

## Goal

We are targeting a high-quality, reproducible replication of the paper on a scoped subset of D4RL tasks:

- `walker2d-medium-replay-v2`
- `hopper-medium-replay-v2`
- `maze2d-medium-v1`

The intended deliverables are:

- a working QT training/evaluation pipeline
- a small ablation suite
- a focused stability analysis around critic quality and the regularization weight `eta`

## Workspace Layout

- `CS4782_QT_Project_Proposal.pdf`: original proposal
- `docs/`: plans, notes, and reports
- `configs/`: experiment configs
- `scripts/`: runnable entrypoints and utilities
- `src/tranqil/`: core implementation
- `experiments/`: experiment manifests and logs
- `results/`: figures, tables, and exported artifacts

## Near-Term Priorities

1. Stand up a reproducible environment and data-loading path for the selected D4RL tasks.
2. Implement the baseline QT actor, double-Q critic, and training loop.
3. Reproduce main returns on one task first, then scale to the full scoped set.
4. Add ablations and stability instrumentation only after the base pipeline is trustworthy.
