#!/usr/bin/env bash

# TranQil single-env preview entrypoint
#
# Purpose:
#   Provide the standard shell command for rendering one preview rollout.
#
# Pipeline role:
#   Phase 5: `preview rollout validation`
#
# Functionality implemented here:
#   - sources `env_vars.sh`
#   - forwards all CLI arguments to `rollout_preview.py`
#
# What it produces:
#   - one preview media file in `results/previews/`
#   - one adjacent JSON metadata file describing the run
#
# Typical usage:
#   `bash ~/TranQil/scripts/run_rollout_preview.sh --env walker2d-medium-replay-v2 --steps 150`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_vars.sh"

python "${SCRIPT_DIR}/rollout_preview.py" "$@"
