# Tier 1 — Probing a pretrained IQL checkpoint

**Status: in progress.** Goal: validate the offline necessity proxy on a real D4RL task by comparing it to the interventional drop signal $\Delta G$ from state-restored paired rollouts.

## Plan

1. **Pretrain IQL** on `antmaze-medium-diverse-v2` (~30 min on a 4090). Fork of `CORL/algorithms/finetune/iql.py` → `nca_iql.py`.
2. **Train NecessityHead** $\mathcal{N}_\psi$ (Stage 2): MLP regression against the Q-margin proxy. Two parallel heads for ensemble disagreement.
3. **Compute proxy** $\mathcal{N}^{\text{off}}$ on 1000 sampled states ($L=8$, $\epsilon=0.05$ in normalised action space).
4. **Drop test:** restore simulator state $s_t$, run paired rollouts (baseline $\pi_\theta$ vs. $\tilde a \sim \kappa_\epsilon$ then $\pi_\theta$). Record $\Delta G(s)$.
5. **Report:** Spearman $\rho_S(\mathcal{N}_\psi, \Delta G)$ + bootstrap CI, decile-mean $\Delta G$, top-vs-bottom drop ratio. Soft quality gate $\rho_S \geq 0.4$.
6. **Heatmap:** 2-D $\mathcal{N}^{\text{off}}$ overlay on `maze2d-large-v1` (paper Fig 3 candidate).

## Files (to be written)

```
nca_iql.py              fork of CORL iql.py with NecessityHead + Stage 2 loop
compute_proxy.py        offline N^off on sampled states
drop_test.py            state-restored paired rollouts + Spearman
plots.py                decile bar plot + maze2d heatmap
configs/
  antmaze.yaml
  maze2d.yaml
```

## Dependencies

- `CORL` (sibling clone at `/home/bobby/icml2026/CORL/`)
- D4RL + MuJoCo (the user has these in their old QT env — may need rebuild)
