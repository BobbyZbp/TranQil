#!/usr/bin/env bash

# TranQil benchmark preview batch runner
#
# Purpose:
#   Execute the canonical preview rollout for each scoped benchmark task with a
#   shared set of preview parameters.
#
# Pipeline role:
#   Phase 5: `preview rollout validation`
#
# Functionality implemented here:
#   - defines the current benchmark task list:
#       * walker2d-medium-replay-v2
#       * hopper-medium-replay-v2
#       * maze2d-medium-v1
#   - calls `run_rollout_preview.sh` once per task
#   - fixes the current batch-preview settings to:
#       * `--steps 150`
#       * `--fps 20`
#       * `--frame-skip 2`
#
# Output:
#   One preview media file and one JSON summary per scoped task under
#   `results/previews/`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for env_name in \
  walker2d-medium-replay-v2 \
  hopper-medium-replay-v2 \
  maze2d-medium-v1
do
  bash "${SCRIPT_DIR}/run_rollout_preview.sh" \
    --env "${env_name}" \
    --steps 150 \
    --fps 20 \
    --frame-skip 2
done
