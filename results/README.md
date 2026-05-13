# Results

This directory contains the re-implementation outputs reported in the project's README and final report.

## Headline run

`qt_anchor_runs/qt_anchor_walker2d_medium_replay_v2_seed123/` — QT on `walker2d-medium-replay-v2`, seed 123.

| Quantity | Value |
|---|---|
| Mean normalized score (iter 21, best eval) | **96.21** |
| Best single-episode normalized score | **99.72** |
| Original paper (Hu et al., 2024) | 98.5 ± 1.1 |

## Layout

```
results/
├── qt_anchor_runs/
│   └── qt_anchor_walker2d_medium_replay_v2_seed123/
│       ├── checkpoints/
│       │   ├── best.pt              # Iter 21, mean_normalized_score=96.21 (Git LFS, ~59 MB)
│       │   └── latest.pt            # Most recent checkpoint (gitignored)
│       ├── config_resolved.yaml     # Fully-resolved training config used for this run
│       ├── evaluations.jsonl        # Per-iteration eval (10 episodes each, max_steps=1000)
│       ├── metrics.jsonl            # Per-iteration training losses + target_q_mean
│       ├── rollout_best_score96.mp4 # Rendered 1000-step rollout from best.pt (Git LFS)
│       └── rollout_best_score96.mp4.json
├── bct_anchor_runs/                 # BCT baseline runs (subset, exploratory)
├── hct_anchor_runs/                 # HCT variant runs (exploratory, not reported in README)
└── debug_runs/, readiness_runs/, previews/, run_logs/   # Auxiliary diagnostic outputs
```

## Reading the JSONL files

Each line in `evaluations.jsonl` is one eval iteration with:
- `iteration`, `step` — training progress
- `mean_normalized_score`, `std_normalized_score`, `mean_return` — aggregate metrics
- `normalized_scores`, `episode_returns`, `episode_lengths` — per-episode raw values
- `target_return`, `candidate_target_returns` — RTG values used by the policy

Each line in `metrics.jsonl` is one training iteration with `critic_loss`, `actor_loss`, `bc_loss`, and `target_q_mean`.

To extract the best score:

```bash
python3 -c "
import json
evals = [json.loads(l) for l in open('results/qt_anchor_runs/qt_anchor_walker2d_medium_replay_v2_seed123/evaluations.jsonl')]
best = max(evals, key=lambda x: x['mean_normalized_score'])
print(f'iter={best[\"iteration\"]}, score={best[\"mean_normalized_score\"]:.2f}, return={best[\"mean_return\"]:.1f}')
"
```

## Notes on auxiliary directories

- `bct_anchor_runs/`, `hct_anchor_runs/` — additional baselines explored during development; not part of the headline replication number.
- `debug_runs/`, `readiness_runs/`, `previews/`, `run_logs/` — diagnostic artifacts retained for traceability; safe to ignore when reviewing reproduction.
