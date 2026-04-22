#!/usr/bin/env bash
# TranQil learned-policy rollout smoke test
#
# Purpose:
#   Confirm that a tiny walker checkpoint can be rendered into a learned-policy
#   MuJoCo rollout artifact with a neighboring JSON summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/scripts/qt_workflow_common.sh"
qt_bootstrap

RUN_NAME="qt_rollout_smoke"
CONFIG_PATH="${REPO_ROOT}/configs/qt_walker2d_medium_replay_stable.yaml"

"${QT_PYTHON}" "${REPO_ROOT}/scripts/train_qt.py" \
  --config "${CONFIG_PATH}" \
  --run-name "${RUN_NAME}" \
  --steps 2 \
  --seed 0 \
  --target-return 300.0

CHECKPOINT_PATH="${REPO_ROOT}/results/qt_runs/${RUN_NAME}/checkpoints/best.pt"
ROLLOUT_PATH="$(qt_resolve_rollout_path "${CHECKPOINT_PATH}" "walker2d-medium-replay-v2" 0)"

"${QT_PYTHON}" "${REPO_ROOT}/scripts/render_qt_rollout.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --seed 0 \
  --target-return 300.0 \
  --max-steps 20

SUMMARY_PATH="${ROLLOUT_PATH}.json"
qt_assert_files_exist "${ROLLOUT_PATH}" "${SUMMARY_PATH}"

echo "learned_rollout: ${ROLLOUT_PATH}"
echo "summary: ${SUMMARY_PATH}"
