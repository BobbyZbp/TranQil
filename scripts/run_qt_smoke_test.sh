#!/usr/bin/env bash
# TranQil QT end-to-end smoke test
#
# Purpose:
#   Run a short training + checkpoint + evaluation cycle for the first QT
#   implementation milestone on walker2d-medium-replay-v2.
#
# Usage:
#   bash scripts/run_qt_smoke_test.sh
#   bash scripts/run_qt_smoke_test.sh --steps 4 --episodes 1 --max-steps 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/scripts/activate_env.sh"

CONFIG_PATH="${REPO_ROOT}/configs/qt_walker2d_medium_replay.yaml"
RUN_NAME="qt_walker2d_medium_replay_smoke"
STEPS=4
EPISODES=1
MAX_STEPS=50

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

python "${REPO_ROOT}/scripts/train_qt.py" \
  --config "${CONFIG_PATH}" \
  --run-name "${RUN_NAME}" \
  --steps "${STEPS}"

CHECKPOINT_PATH="${REPO_ROOT}/results/qt_runs/${RUN_NAME}/checkpoints/latest.pt"
EVAL_OUTPUT_PATH="${REPO_ROOT}/results/qt_runs/${RUN_NAME}/manual_eval.json"

python "${REPO_ROOT}/scripts/eval_qt.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --output-path "${EVAL_OUTPUT_PATH}"
