# Datasets

This directory holds the D4RL benchmark datasets used for training and evaluation. The dataset files themselves are **not** committed — they download automatically on the first training run and are cached locally.

## What lives here after setup

```
data/
├── d4rl/
│   ├── walker2d_medium_replay-v2.hdf5   (~180 MB)
│   ├── hopper_medium_replay-v2.hdf5      (~180 MB)
│   └── maze2d-medium-sparse-v1.hdf5      (~50 MB)
└── qt_cache/                             # preprocessed trajectory shards
```

## How to obtain the data

1. Install the environment (see top-level `README.md` → *Reproduction Steps*).
2. Activate it: `source scripts/activate_env.sh`. The activation script points D4RL at `data/d4rl/` via `D4RL_DATASET_DIR`.
3. Run any training or smoke-test script. On the first run for a given task, D4RL will fetch the corresponding HDF5 from the upstream mirror into `data/d4rl/`.

Manual download (optional) — the upstream URLs are listed by the D4RL package: https://github.com/Farama-Foundation/D4RL

## Required disk space

Approximately **410 MB** total for all three tasks plus ~150 MB of preprocessed `qt_cache/` shards after the first epoch.
