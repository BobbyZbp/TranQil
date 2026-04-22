#!/usr/bin/env bash
# TranQil walker package smoke test
#
# Purpose:
#   Exercise the one-task packaging flow on a one-seed miniature run so we can
#   verify final-eval, learned rollout, results table generation, and finalized
#   artifact copying without running the full 3-seed completion job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/scripts/qt_workflow_common.sh"
qt_bootstrap

RUN_NAME="qt_walker_package_smoke_seed0"
RUN_DIR="${REPO_ROOT}/results/qt_runs/${RUN_NAME}"
FINALIZED_DIR="${REPO_ROOT}/results/finalized_smoke/walker2d_medium_replay_v2"
CONFIG_PATH="${REPO_ROOT}/configs/qt_walker2d_medium_replay_stable.yaml"
REFERENCE_PATH="${REPO_ROOT}/experiments/reference_results/walker2d_medium_replay_v2_qt_reference.json"

"${QT_PYTHON}" "${REPO_ROOT}/scripts/train_qt.py" \
  --config "${CONFIG_PATH}" \
  --run-name "${RUN_NAME}" \
  --steps 2 \
  --seed 0 \
  --target-return 300.0

qt_eval_checkpoint_pair "${CONFIG_PATH}" "${RUN_DIR}" 0 1 20 300.0

ROLLOUT_PATH="$(qt_resolve_rollout_path "${RUN_DIR}/checkpoints/best.pt" "walker2d-medium-replay-v2" 0)"

"${QT_PYTHON}" "${REPO_ROOT}/scripts/render_qt_rollout.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${RUN_DIR}/checkpoints/best.pt" \
  --seed 0 \
  --target-return 300.0 \
  --max-steps 20

"${QT_PYTHON}" "${REPO_ROOT}/scripts/build_qt_results_table.py" \
  --run-dir "${RUN_DIR}" \
  --reference-results "${REFERENCE_PATH}" \
  --finalized-dir "${FINALIZED_DIR}" \
  --canonical-rollout "${ROLLOUT_PATH}"

qt_assert_files_exist \
  "${FINALIZED_DIR}/manifest.json" \
  "${FINALIZED_DIR}/results_table.csv" \
  "${FINALIZED_DIR}/results_table.md" \
  "${FINALIZED_DIR}/best.pt" \
  "${FINALIZED_DIR}/best_eval.json"

echo "finalized_dir: ${FINALIZED_DIR}"
