# Original QT AntMaze Reproduction Status

Current date: 2026-04-27.

## Goal

Reproduce the official QT AntMaze large-diverse result in the original `charleshsc/QT` codebase, then compare the original implementation on:

- `antmaze-large-diverse-v0`: paper-matched task.
- `antmaze-large-diverse-v2`: diagnostic task used by the TranQil reimplementation.

## Verified Facts

- This repo is the official QT repo: `https://github.com/charleshsc/QT`.
- QT paper reports `53.3 +/- 4.7` on `antmaze-large-diverse-v0`, not v2.
- Upstream `experiment.py` hardcoded AntMaze `dversion = 0`; this repo now adds `--d4rl_version` with default `0`.
- The QT algorithm is intentionally unchanged.

## Local Layout

- Env: `.micromamba/root/envs/original-qt`
- MuJoCo: `.mujoco/mujoco210`
- D4RL cache: `.d4rl/datasets`
- Converted datasets: `D4RL/*.pkl`
- Training outputs: `save/`

## Current Progress

- Added `environment.yml`.
- Added isolated activation/install/verify scripts under `scripts/`.
- Added WSL2 `mujoco-py` builder patch script; verified `mujoco-py` builds `linuxgpuextensionbuilder`.
- Replaced `D4RL/create_dataset.py` with a CLI converter for v0 and v2 AntMaze datasets.
- Added `--d4rl_version` to `experiment.py`; default preserves paper behavior.
- Runtime verified with `torch==2.4.1+cu121`, `transformers==4.5.1`, and `packaging==21.3`.
- Generated `D4RL/antmaze-large-diverse-v0.pkl`: 7,141 trajectories, 1,000,000 samples, 151 MB.
- Generated `D4RL/antmaze-large-diverse-v2.pkl`: 7,189 trajectories, 1,000,000 samples, 151 MB.
- Tiny CPU smoke tests passed for v0 and v2 with `K=5`, `batch_size=2`, `embed_dim=32`, `n_layer=1`, `n_head=1`.
- Full large-diverse v0 diagnostic was restarted with persistent stdout/stderr logging after the first tmux attempt exited before eval.
- Large-diverse v0 output directory: `save/qt_antmaze_v0-antmaze-large-diverse-123-260426-143133`.
- Large-diverse v0 log: `save/qt_antmaze_v0_stdout.log`.
- Large-diverse v0 is no longer running. It completed 32 evaluation rows, then wrote one additional training-loss row before stopping.
- Best large-diverse v0 return mean was `0.3`, normalized score `0.3` in `progress.csv` / `30.0` in the script's printed percent-style best line, at iteration 12. It also hit return `0.3` at iteration 16 and `0.2` at iteration 22.
- Last completed large-diverse v0 evaluation, iteration 32, was return `0.0`, normalized score `0.0`.
- A medium-diverse v0 diagnostic was also run: `save/qt_antmaze_md_v0-antmaze-medium-diverse-123-260427-003438`.
- Medium-diverse v0 is no longer running. It completed 48 evaluation rows, then wrote one additional training-loss row before stopping.
- Best medium-diverse v0 return mean was `0.2`, normalized score `0.2` in `progress.csv` / `20.0` in the script's printed percent-style best line, at iteration 32.
- Last completed medium-diverse v0 evaluation, iteration 48, was return `0.1`, normalized score `0.1`.
- Full large-diverse v2 diagnostic was started on 2026-04-27 at 10:26 America/New_York, then stopped on user request before its first evaluation completed.
- Interrupted v2 stdout/stderr log: `save/qt_antmaze_v2_stdout.log`.
- Interrupted v2 output directory: `save/qt_antmaze_v2-antmaze-large-diverse-123-260427-102636`.
- Fresh medium-diverse v0 run was started on 2026-04-27 at 10:30 America/New_York.
- Active medium-diverse v0 tmux session: `qt_antmaze_md_v0`.
- Active medium-diverse v0 stdout/stderr log: `save/qt_antmaze_md_v0_stdout.log`.
- Active medium-diverse v0 output directory: `save/qt_antmaze_md_v0-antmaze-medium-diverse-123-260427-103040`.

## Next Required Checks

Run in order:

1. Monitor `qt_antmaze_md_v0` until it finishes or fails.
2. Record best medium-diverse v0 return/normalized score from `save/qt_antmaze_md_v0-antmaze-medium-diverse-123-260427-103040/progress.csv`.
3. Compare against the earlier medium-diverse v0 partial best: return `0.2`, normalized score `20.0` in the script's printed convention.
4. After medium-diverse v0 is settled, decide whether to resume the original large-diverse v0/v2 comparison.

Useful monitor command:

```bash
tmux capture-pane -pt qt_antmaze_md_v0:0.0 -S -120
```

## Interpretation Gate

- v0 works and v2 is zero: likely D4RL version mismatch matters.
- v0 zero and v2 zero: debug environment/data/original-run setup before HCT-QT claims.
- v0 works and v2 works: TranQil AntMaze failure is likely implementation/config specific.
- v0 zero but v2 works: unexpected; inspect dataset conversion and evaluation behavior.
