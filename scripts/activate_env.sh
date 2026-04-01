#!/usr/bin/env bash

# TranQil runtime activation helper
#
# Purpose:
#   Prepare the current shell to use the repo-local `tranqil-qt` environment.
#
# Pipeline role:
#   Phase 2: `activation/runtime vars`
#   This script is the standard entrypoint after the base environment exists.
#
# What this script does:
#   - points `PATH` at `.micromamba/root/envs/tranqil-qt/bin`
#   - exports `CONDA_PREFIX` and `MAMBA_ROOT_PREFIX`
#   - sets a repo-local cache root
#   - sources `env_vars.sh` so MuJoCo, D4RL, and library-path variables are set
#   - prints the resolved Python executable for verification
#
# What follows this script:
#   Commands such as `install_d4rl_stack.sh`, `run_smoke_test.sh`, and the
#   preview scripts assume the shell has already been configured this way.
#
# Usage:
#   This file is meant to be sourced, not executed:
#   `source ~/TranQil/scripts/activate_env.sh`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PREFIX="${REPO_ROOT}/.micromamba/root/envs/tranqil-qt"

export MAMBA_ROOT_PREFIX="${REPO_ROOT}/.micromamba/root"
export XDG_CACHE_HOME="${REPO_ROOT}/.micromamba/cache"
export CONDA_PREFIX="${ENV_PREFIX}"
export PATH="${ENV_PREFIX}/bin:${PATH}"

source "${SCRIPT_DIR}/env_vars.sh"

echo "Activated TranQil environment variables for ${ENV_PREFIX}"
echo "Python: $(command -v python)"
