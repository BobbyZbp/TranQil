# Tier 1 Results

## Summary

| Environment | reward | N^off vs N_ψ ρ | Q vs N^off ρ | N_ψ vs ΔG ρ (drop test) |
|---|---|---|---|---|
| antmaze-medium-diverse-v2 | sparse | 0.326 | 0.375 | — (not run) |
| maze2d-large-dense-v2 | dense | **0.797** | **−0.033** | **+0.164** (p=0.02) |

ρ = Spearman rank correlation.

---

## antmaze-medium-diverse-v2 (sparse reward)

Stage 1 (IQL, 1M steps): eval D4RL = 0, success = 0 (sparse reward, no solve).
Stage 2 (NCA, 200K steps): loss → 0, target_std ≈ 0.005, disagree ≈ 3e-4.

Probe (N=1000):
- N^off: mean=+0.0003, std=0.0051
- N_ψ:    mean=+0.0004, std=0.0037
- Q_φ:    mean=+0.0295, std=0.0732

Correlations:
- N^off vs N_ψ : pearson=0.672, spearman=0.326
- Q_φ vs N^off : spearman=+0.375
- Q_φ vs N_ψ   : spearman=+0.321

Drop test: not executed.

Conclusion: proxy signal too weak (std=0.005 ≈ noise) to draw any conclusion. Sparse reward → Q values nearly degenerate.

---

## maze2d-large-dense-v2 (dense reward)

Stage 1 (IQL, 1M steps): eval_return mean ≈ 75 (std ≈ 45).
Stage 2 (NCA, 200K steps): loss 7→4, target_std ≈ 2.5, disagree 0.1→0.5 (rises).

Probe (N=1000):
- N^off: mean=−0.968, std=2.407
- N_ψ:    mean=−1.114, std=1.961
- Q_φ:    mean=+134.92, std=83.13

Correlations vs proxy:
- N^off vs N_ψ : pearson=**0.848**, spearman=**0.797**
- Q_φ vs N^off : spearman=**−0.033**  (no signal)
- Q_φ vs N_ψ   : spearman=−0.040       (no signal)

Drop test (200 on-policy states, paired rollouts):

| Run | kernel ε | cf samples | N_ψ vs ΔG | p | Q vs ΔG | p |
|---|---|---|---|---|---|---|
| v1 | 0.05 | 1 | +0.031 | 0.66 | −0.026 | 0.72 |
| v2 | 0.30 | 5 | **+0.164** | **0.02** | +0.129 | 0.07 |

ΔG mean=+2.07, std=54.7.

Findings:
1. N_ψ fits the offline proxy almost perfectly (ρ=0.80).
2. Q-rank baseline has zero correlation with the proxy — Q ranking and necessity ranking are independent.
3. Against ground-truth ΔG, N_ψ is significantly better than chance but only weakly (ρ=0.16, p=0.02).
4. Conclusion: the offline Q-margin proxy captures *something* but not enough — gap between proxy and reality is the open problem.

---

## How we did it (maze2d-large-dense-v2)

### 0. Setup
- RunPod RTX 4090, $0.69/hr (Stage 1 ≈ 30 min, Stage 2 ≈ 6 min)
- Persistent network volume `/workspace` (conda env + code + checkpoints survive pod stop)
- Repo: https://github.com/BobbyZbp/NCA-RL
- Checkpoints backup: https://huggingface.co/Bopeng888/nca-rl-maze2d

### 1. Data adapter (minari → d4rl format)
File: `experiments/checkpoint_probing/nca_iql_maze2d.py`

The original `nca_iql.py` consumes `d4rl.qlearning_dataset(env)`. D4RL pointmaze has been migrated to Gymnasium-Robotics + Minari, so we wrote a thin adapter:

- `load_minari_as_d4rl(name)` iterates minari episodes and emits the dict `{observations, actions, rewards, next_observations, terminals}`.
- `flatten_obs(obs_dict)` concatenates `[observation, desired_goal]` into a 6-dim vector.
- `GymnasiumToGymAdapter` wraps the gymnasium env so eval rollouts return `(obs, r, done, info)` instead of `(obs, r, term, trunc, info)`.

### 2. Two fixes that made Stage 1 stable
First training run had value_loss / q_loss diverging.

Two issues:
- **Pointmaze has no `termination` flags, only `truncation`.** Q bootstrap was unbounded. Fix: treat truncations as terminals (`done = term | trunc`).
- **Reward scale was wrong for IQL hyperparameters.** Fix: CORL-style normalisation — divide rewards by per-trajectory return range, multiply by 1000.

After both fixes, value_loss converges to ≈1, q_loss stays bounded at ≈100.

### 3. Stage 1 (IQL pretrain, 1M steps)
Standard CORL IQL with periodic checkpoints (every 50K steps for resume safety).

Final eval_return ≈ 74 ± 49 — policy learns to navigate the large maze.

### 4. Stage 2 (NCA regression, 200K steps)
Same as antmaze: rejection-sampled counterfactual kernel κ_ε with ε=0.05, L=8 samples per (s,a) for the proxy, two-head ensemble for disagreement.

Train target: `MSE(N_ψ, N^off)` where `N^off(s,a) = Q(s,a) − E_κ[Q(s,ã)]`.

Loss settled at ≈ 4 (target_std² ≈ 6, so explained variance ≈ 35%).

### 5. compute_proxy_maze2d.py (offline correlations)
Sample 1000 random (s, a) from the dataset, compute the three quantities (N^off, N_ψ, Q_φ), save to `probe_results.npz`.

Spearman / Pearson computed in a one-liner with `scipy.stats`.

### 6. drop_test_maze2d.py (ground-truth ΔG)
For each (s, a):
1. Roll out π_θ to collect on-policy states with `(qpos, qvel)` for state restoration.
2. Subsample 200 states uniformly.
3. For each: restore env state, take dataset action a, follow π_θ → G_base.
4. Sample n_cf cf actions ã ~ κ_ε; for each: restore env, take ã, follow π_θ → G_cf.
5. ΔG = G_base − mean(G_cf).
6. Compute Spearman ρ(N_ψ, ΔG) and ρ(Q_φ, ΔG).

State restoration uses `env.unwrapped.set_state(qpos, qvel)`.

### 7. Two drop-test runs
- **v1** (ε=0.05, 1 cf sample): high noise, ρ ≈ 0.
- **v2** (ε=0.30, 5 cf samples): noise reduced, ρ jumps to 0.164 (statistically significant).

The kernel margin and cf sample count are the dominant knobs for drop-test signal. ε too small → cf rollout ≈ baseline rollout → ΔG dominated by env stochasticity.
