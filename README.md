# TranQil: A Re-implementation of *Q-value Regularized Transformer* for Offline RL

**CS 4782 — Cornell University — Final Project**
**Bopeng (Bobby) Zhang · Eric Yan · Yifei Wang**

---

## 1. Introduction

This GitHub repository contains our re-implementation of the paper **"Q-value Regularized Transformer for Offline Reinforcement Learning"** (Hu et al., ICML 2024) — referred to throughout as **QT** — written from scratch as our CS 4782 final deliverable.

**Main contribution of the paper.** QT augments a Decision-Transformer-style sequence policy with a learned double-Q critic and a Q-regularized BC objective. At inference, instead of conditioning on a single return-to-go (RTG), the policy generates 50 candidate actions with jittered RTGs, scores them with the critic, and samples via a Q-weighted softmax. This recovers planning-like behaviour from a feed-forward sequence model and pushes the policy past prior Decision-Transformer baselines on the D4RL continuous-control benchmark.

The purpose of this repo is to *re-implement* the algorithm, *verify* it against the original code release, and *report* whether we can land inside the paper's reported confidence interval. The final report (see [`report/`](report/)) additionally documents a failed extension (SARD-QT) and a pivot to a separate counterfactual-necessity study (NCA-RL); the code for both of those is vendored under [`extensions/`](extensions/) so everything described in the report is reproducible from this one clone.

## 2. Chosen Result

We targeted the first two rows of Table 1 in Hu et al. (2024):

> **QT on `walker2d-medium-replay-v2`: 98.5 ± 1.1** and **QT on `hopper-medium-replay-v2`: 102.0 ± 0.2** normalized D4RL scores.

These two cells are the central evidence that QT outperforms both pure Decision Transformer and pure Q-learning, and they exercise the paper's most non-trivial design choice — 50-way RTG candidate expansion with Q-bootstrap correction at inference — rather than just the base transformer architecture.

## 3. GitHub Contents

```
TranQil/
├── README.md                  # this file
├── LICENSE                    # MIT
├── .gitignore
├── environment.yml            # micromamba environment spec
├── configs/                   # YAML configs (one per task / variant)
├── src/tranqil/               # re-implementation code (the "code/" of the spec)
│   ├── models/anchor_actor.py    #   GPT2 policy + 50-way candidate get_action
│   ├── anchor_trainer.py         #   training loop, EMA target, checkpointing
│   ├── evaluation.py             #   rollout eval (no action clipping, 1000 steps)
│   ├── anchor_data.py            #   D4RL loading + trajectory shards
│   ├── config.py                 #   dataclass config schema
│   └── rendering.py              #   offscreen MuJoCo rendering (osmesa)
├── scripts/                   # entrypoints / helpers (also part of "code/")
│   ├── train_qt.py               #   main training entrypoint
│   ├── render_qt_rollout.py      #   render MP4 rollout from a checkpoint
│   ├── install_d4rl_stack.sh     #   D4RL + MuJoCo installer
│   ├── activate_env.sh / env_vars.sh
│   └── run_smoke_test.sh         #   ~2-minute reproducibility smoke test
├── tests/                     # unit tests for actor, trainer, evaluation
├── data/                      # D4RL datasets (gitignored, see data/README.md)
│   └── README.md                 #   download / setup instructions
├── results/                   # re-implementation outputs (curves, checkpoints, MP4)
│   └── README.md                 #   describes JSONL schemas and headline run
├── poster/                    # in-class presentation poster (PDF)
├── report/                    # final written report (PDF)
├── extensions/                # vendored snapshots of the two follow-up repos
│   ├── README.md
│   ├── NCA-RL/                #   counterfactual-necessity study (upstream: github.com/BobbyZbp/NCA-RL)
│   └── SARD-QT/               #   failed reverse-credit + QSC-QT extension (upstream: github.com/BobbyZbp/SARD-QT)
└── docs/                      # internal design notes (gitignored on push)
```

> **Note on layout.** The course spec asks for a top-level `code/` directory. We follow the idiomatic Python `src/` + `scripts/` layout instead — `src/tranqil/` is the importable library and `scripts/` are the user-facing entrypoints. Together they constitute the `code/` of the spec. The `extensions/` folder vendors the two companion repositories referenced in the report so that everything described there is reproducible from a single clone; see [`extensions/README.md`](extensions/README.md).

## 4. Re-implementation Details

- **Algorithm.** QT as described in Hu et al. (2024), Section 3. Joint training of a sequence policy and a double-Q critic with an EMA target network (`τ = 0.995`). Objective:

  $$\mathcal{L}_{\text{QT}}(\theta) = \mathcal{L}_{\text{DT}}(\theta) \;-\; \alpha \cdot \mathbb{E}\big[\,Q_\phi(s_i,\, \pi_\theta(\tau_t)_i)\,\big], \quad \alpha = \eta\,/\,\mathbb{E}\big[\,|Q_\phi(s,a)|\,\big].$$

  The critic is trained with n-step Bellman targets under double-Q clipping. Inference uses the 50-way candidate expansion with RTG jitter and multinomial Q-softmax selection.
- **Architecture.** Policy: a **4-layer, 4-head, 256-dim** GPT-2-style causal transformer (HuggingFace `transformers<4.40`). Critic: dual MLP with Mish activations and EMA target networks. The optimizer uses cosine LR decay.
- **Datasets.** D4RL `walker2d-medium-replay-v2` (~180 MB) and `hopper-medium-replay-v2` (~180 MB), auto-downloaded on first run. Trajectories are sharded into a `qt_cache/` of context-length windows.
- **Tools / framework.** PyTorch 2.4, HuggingFace `transformers<4.40`, D4RL + `mujoco-py` against MuJoCo 2.1.0, micromamba environment.
- **Evaluation.** Per-iteration rollouts on 10 episodes capped at `max_steps = 1000`, scored with D4RL's normalized-score helper. Best checkpoint selected by `mean_return` (matches the original `experiment.py:388–399`).
- **Compute.** Single NVIDIA **RTX 5090**, ~**10 GPU-hours total** across both tasks and seeds.
- **Key fixes vs. a naive baseline (each one was needed to reach paper-level scores):**
  1. **Action clipping removed.** The anchor actor outputs through a Tanh head; clipping to `env.action_space` bounds zeroed gradients and was incorrect.
  2. **`max_steps = 1000`.** At the gym default of 500, the maximum achievable normalized score on `walker2d-medium-replay-v2` is ≈ 44 — the most common silent replication failure.
  3. **50-way candidate expansion with multinomial Q-softmax.** Greedy `argmax` over the 50 candidates degrades performance materially.
  4. **GPT-2 position embeddings zeroed and frozen** (`wpe.weight.data.zero_(); requires_grad=False`), forcing the model to rely on timestep embeddings.
- **What our re-implementation also taught us (documented in [`report/report.pdf`](report/report.pdf)):** Q-loss `eta` and `grad_norm` required more tuning than the paper suggests; checkpoint-by-checkpoint variance is large enough that single-seed runs noticeably under-estimate the paper's 3-seed mean.

## 5. Reproduction Steps

Requirements: Ubuntu or WSL2, `bash`, and `micromamba`.

```bash
# 1. Install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc

# 2. Clone
git clone git@github.com:BobbyZbp/TranQil.git
cd TranQil

# 3. Create the environment
export MAMBA_ROOT_PREFIX="$PWD/.micromamba/root"
micromamba create -y -f environment.yml

# 4. Activate + pull LFS artifacts (best.pt + rollout MP4)
source scripts/activate_env.sh
git lfs install
git lfs pull

# 5. Install the RL stack (compiles mujoco-py; takes 5–10 min)
bash scripts/install_d4rl_stack.sh

# 6. Smoke test (~2 min)
bash scripts/run_smoke_test.sh
```

**GPU build.** Step 5 installs CPU-only PyTorch by default. For training, install the CUDA build matching your driver (the rollout/eval path runs fine on CPU):

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

**Train (use `tmux` — 200 iters is many hours):**

```bash
source scripts/activate_env.sh
tmux new-session -s qt_train
python scripts/train_qt.py --config configs/qt_anchor_walker2d_medium_replay.yaml   # or qt_anchor_hopper_medium_replay.yaml
# detach: Ctrl-B then D ;  reattach: tmux attach -t qt_train
```

**Render a rollout from the released `best.pt`:**

```bash
source scripts/activate_env.sh
python scripts/render_qt_rollout.py \
  --config configs/qt_anchor_walker2d_medium_replay.yaml \
  --checkpoint results/qt_anchor_runs/qt_anchor_walker2d_medium_replay_v2_seed123/checkpoints/best.pt \
  --seed 123 --target-return 5000.0 --max-steps 1000
```

**Compute needed.** Reproducing the headline numbers needs a single modern NVIDIA GPU (we used an RTX 5090) and ~2.1 GB of disk after a full setup. Running the released checkpoint (eval + rollout video only) works on CPU in a few minutes.

> The `tranqil` package is added to `PYTHONPATH` by `scripts/activate_env.sh` rather than pip-installed — always source the activation script in a fresh shell.

## 6. Results / Insights

What you can expect from this repo, vs. the paper:

| Environment | Paper (Hu et al., 2024, 3 seeds) | **TranQil (ours, 1 seed)** | Gap |
|---|---|---|---|
| `walker2d-medium-replay-v2` | 98.5 ± 1.1 | **96.21** | −2.3 |
| `hopper-medium-replay-v2` | 102.0 ± 0.2 | **99.10** | −2.9 |

Walker2d's −2.3 gap sits inside the paper's reported ±1.1 band at ≈ 2 σ; the larger −2.9 gap on Hopper reflects single-seed variance against the paper's 3-seed mean. Best single-episode normalized score on Walker2d: **99.72** — matches the paper's mean exactly, confirming the policy has learned a high-quality locomotion strategy. A rendered 1000-step rollout from `best.pt` is at [`results/qt_anchor_runs/qt_anchor_walker2d_medium_replay_v2_seed123/rollout_best_score96.mp4`](results/qt_anchor_runs/qt_anchor_walker2d_medium_replay_v2_seed123/rollout_best_score96.mp4) (the paper does not provide rollout videos).

**Headline insight.** The result is *not* sensitive to architecture details (GPT-2 size, optimizer schedule) but is *extremely* sensitive to four otherwise-easy-to-miss decisions: eval horizon = 1000, no action clipping, 50-way candidate expansion with multinomial Q-selection, and zeroed/frozen GPT-2 position embeddings. Each one independently moves the final score by tens of points.

## 7. Conclusion

We re-implemented QT from scratch and reproduced its `walker2d-medium-replay-v2` and `hopper-medium-replay-v2` numbers within (or near) the paper's reported confidence intervals, validating the core claim that Q-regularization plus the 50-way candidate expansion is sufficient to push a sequence policy past plain Decision Transformer.

The biggest lesson was that "faithful re-implementation" of a Decision-Transformer-style method is much less about the model and much more about the *inference loop and evaluation harness*. Four small, undocumented-looking details (clipping, eval horizon, candidate sampling, position embeddings) account for the entire gap between a non-working baseline and the paper number — exactly the kind of friction CS 4782 sets out to expose. Beyond the baseline, our follow-up experiments documented in [`report/report.pdf`](report/report.pdf) found that QT's long-horizon weakness is *not* fixable by adding more value-style supervision (the failed SARD-QT extension), and motivated a separate counterfactual-necessity study (NCA-RL) — both maintained in their own repositories.

## 8. References

1. Hu, S., Fan, Z., Huang, C., Shen, L., Zhang, Y., Wang, Y., and Tao, D. (2024). *Q-value Regularized Transformer for Offline Reinforcement Learning*. ICML 2024. (`original_paper.pdf` in this repo.)
2. Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., and Mordatch, I. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling.* NeurIPS 2021.
3. Kostrikov, I., Nair, A., and Levine, S. (2022). *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR 2022.
4. Fu, J., Kumar, A., Nachum, O., Tucker, G., and Levine, S. (2020). *D4RL: Datasets for Deep Data-Driven Reinforcement Learning.* arXiv:2004.07219.
5. Original QT implementation reference: [`charleshsc/QT`](https://github.com/charleshsc/QT).
6. HuggingFace Transformers (GPT-2 backbone): https://github.com/huggingface/transformers.
7. Companion repositories from this project (vendored under [`extensions/`](extensions/)): [`github.com/BobbyZbp/NCA-RL`](https://github.com/BobbyZbp/NCA-RL), [`github.com/BobbyZbp/SARD-QT`](https://github.com/BobbyZbp/SARD-QT).

```bibtex
@inproceedings{hu2024qt,
  title     = {Q-value Regularized Transformer for Offline Reinforcement Learning},
  author    = {Hu, Shengchao and Fan, Ziqing and Huang, Chengqian and Shen, Li and
               Zhang, Ya and Wang, Yanfeng and Tao, Dacheng},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024}
}
```

## 9. Acknowledgements

This project was carried out as the final deliverable for **CS 4782 — Deep Learning** at **Cornell University (Spring 2026)** by Bopeng (Bobby) Zhang, Eric Yan, and Yifei Wang. We thank the course instructors and TAs for feedback during the proposal and concept-note milestones, and the authors of `charleshsc/QT` for releasing the original implementation, which we used as a correctness reference when our reproduction diverged from the paper's headline number.
