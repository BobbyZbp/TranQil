# Tier 0 — Gridworld concept validation

**Status: DONE.** Closed-form sanity check of necessity vs. sufficiency on a 5×5 deterministic gridworld with two paths (Path A bottleneck, Path B redundant).

## Run

```bash
python run_gridworld.py
```

~5 seconds, NumPy only. Produces `fig2_gridworld_heatmap.{pdf,png}` and `summary.txt`.

## Headline numbers

| Metric | Path A (bottleneck) | Path B (redundant) | Ratio B/A |
|--------|---------------------|--------------------|-----------|
| Sufficiency $Q^{\pi_\tau}$ | 9.29 | 7.94 | **0.85** (no separation) |
| Necessity $\mathcal{N}^{\pi_\tau}_\kappa$ | 0.98 | 0.35 | **0.35** (strong separation) |

Divergence factor: **2.42×**. Necessity contracts on the redundant region; sufficiency does not.

These numbers feed paper Table 1 and the §6 *Empirical status* paragraph.
