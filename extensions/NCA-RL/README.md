# NCA-RL — Necessity-aware Credit Assignment for Offline-to-Online RL

Code repo for the ICML 2026 O2O Workshop short paper **NCA-T**. Submission deadline 2026-05-07 AoE.

The paper proposes a learnable *necessity* score (Q-margin proxy + ensemble MLP head) on top of an offline critic, and validates it via state-restored counterfactual drop tests during online interaction.

## Structure

```
nca-rl/
├── nca/                              core library (NecessityHead, kernels, drop test)
├── experiments/
│   ├── gridworld/                    Tier 0 — closed-form sanity check (DONE)
│   ├── checkpoint_probing/           Tier 1 — NCA-IQL on D4RL antmaze + maze2d
│   └── online_adaptation/            Tier 2 — full O2O with NCA-T (future work)
├── configs/                          per-experiment YAMLs
├── paper/                            ncat.tex + bib + sty
├── scripts/                          shell entry points
└── tests/
```

## Status

| Tier | Status | Headline |
|------|--------|----------|
| 0 — gridworld | ✅ done | suff ratio 0.85, nec ratio 0.35 (2.42× separation) |
| 1 — checkpoint probing | 🔧 in progress | NCA-IQL patch + drop test pending |
| 2 — online adaptation | 📋 future work | Tier 2 of paper §5.3 |

## Setup

```bash
pip install -e .                      # editable install of `nca/`
# experiments/checkpoint_probing depends on CORL — see its README
```
