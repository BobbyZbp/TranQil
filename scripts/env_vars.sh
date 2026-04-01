#!/usr/bin/env bash

# TranQil runtime variable specification
#
# Purpose:
#   Export the runtime variables required to make the repo-local D4RL +
#   MuJoCo stack work reproducibly on this machine.
#
# Pipeline role:
#   Phase 2: `activation/runtime vars`
#   This file is sourced by `activate_env.sh` and by command wrappers that need
#   a fully configured runtime.
#
# Functionality implemented here:
#   - defines repo-local locations for caches, datasets, and MuJoCo binaries
#   - points `MUJOCO_PY_MUJOCO_PATH` at `.mujoco/mujoco210`
#   - sets headless rendering backends via `MUJOCO_GL=osmesa` and
#     `PYOPENGL_PLATFORM=osmesa`
#   - creates required cache directories on demand
#   - constructs `LD_LIBRARY_PATH` from the active env, MuJoCo's `bin/`, and
#     `/usr/lib/wsl/lib` when present on WSL2
#
# Why this matters:
#   The current environment is not considered runnable if these variables are
#   missing. Using the env's Python binary alone is not enough because
#   `mujoco-py` would otherwise fall back to its default `~/.mujoco/...`
#   assumptions instead of the repo-local setup.
#
# Used by:
#   - `install_d4rl_stack.sh`
#   - `run_smoke_test.sh`
#   - `run_rollout_preview.sh`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TRANQIL_REPO_ROOT="${REPO_ROOT}"
export TRANQIL_CACHE_ROOT="${REPO_ROOT}/.cache"
export TRANQIL_MUJOCO_ROOT="${REPO_ROOT}/.mujoco"
export TRANQIL_MUJOCO_DIR="${TRANQIL_MUJOCO_ROOT}/mujoco210"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${REPO_ROOT}/.micromamba/root}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-${REPO_ROOT}/data/d4rl}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-${TRANQIL_MUJOCO_DIR}}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TRANQIL_CACHE_ROOT}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"

mkdir -p "${D4RL_DATASET_DIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TRANQIL_MUJOCO_DIR}/bin:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${TRANQIL_MUJOCO_DIR}/bin:${LD_LIBRARY_PATH:-}"
fi

if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
fi
