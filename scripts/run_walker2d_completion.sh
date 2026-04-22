#!/usr/bin/env bash
# TranQil walker2d one-task completion pipeline
#
# Purpose:
#   Run the full walker completion flow: candidate sweep, final 3-seed runs,
#   best-checkpoint selection, learned-policy rollout rendering, and finalized
#   results-package generation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/scripts/qt_workflow_common.sh"
qt_bootstrap

PYTHON="${QT_PYTHON}"
REFERENCE_PATH="${REFERENCE_PATH:-${REPO_ROOT}/experiments/reference_results/walker2d_medium_replay_v2_qt_reference.json}"
FINALIZED_DIR="${FINALIZED_DIR:-${REPO_ROOT}/results/finalized/walker2d_medium_replay_v2}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-10}"
FINAL_EVAL_MAX_STEPS="${FINAL_EVAL_MAX_STEPS:-500}"
FINAL_TARGET_RETURN="${FINAL_TARGET_RETURN:-300.0}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"

CANDIDATE_NAMES=("baseline" "low_eta" "conservative_actor")
CANDIDATE_CONFIGS=(
  "${REPO_ROOT}/configs/qt_walker2d_medium_replay_baseline.yaml"
  "${REPO_ROOT}/configs/qt_walker2d_medium_replay_low_eta.yaml"
  "${REPO_ROOT}/configs/qt_walker2d_medium_replay_stable.yaml"
)

SWEEP_RUN_DIRS=()
for index in "${!CANDIDATE_NAMES[@]}"; do
  candidate_name="${CANDIDATE_NAMES[$index]}"
  candidate_config="${CANDIDATE_CONFIGS[$index]}"
  run_name="qt_walker2d_medium_replay_tune_${candidate_name}_seed0"

  "${PYTHON}" "${REPO_ROOT}/scripts/train_qt.py" \
    --config "${candidate_config}" \
    --run-name "${run_name}" \
    --steps "${TRAIN_STEPS}" \
    --seed 0 \
    --target-return "${FINAL_TARGET_RETURN}"

  run_dir="${REPO_ROOT}/results/qt_runs/${run_name}"
  qt_eval_checkpoint_pair \
    "${candidate_config}" \
    "${run_dir}" \
    0 \
    "${FINAL_EVAL_EPISODES}" \
    "${FINAL_EVAL_MAX_STEPS}" \
    "${FINAL_TARGET_RETURN}"

  SWEEP_RUN_DIRS+=("${run_dir}")
done

SELECTED_CONFIG="$(
  "${PYTHON}" - "${SWEEP_RUN_DIRS[@]}" "${CANDIDATE_CONFIGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

from tranqil.results_table import load_run_artifact_summary, select_best_candidate_run

candidate_count = len(sys.argv[1:]) // 2
run_dirs = sys.argv[1 : 1 + candidate_count]
config_paths = sys.argv[1 + candidate_count :]
runs = [load_run_artifact_summary(path) for path in run_dirs]
selected = select_best_candidate_run(runs)
mapping = {Path(run_dir).name: config for run_dir, config in zip(run_dirs, config_paths)}
print(mapping[Path(selected.run_dir).name])
PY
)"

FINAL_RUN_DIRS=()
for seed in 0 1 2; do
  run_name="qt_walker2d_medium_replay_final_seed${seed}"
  "${PYTHON}" "${REPO_ROOT}/scripts/train_qt.py" \
    --config "${SELECTED_CONFIG}" \
    --run-name "${run_name}" \
    --steps "${TRAIN_STEPS}" \
    --seed "${seed}" \
    --target-return "${FINAL_TARGET_RETURN}"

  run_dir="${REPO_ROOT}/results/qt_runs/${run_name}"
  qt_eval_checkpoint_pair \
    "${SELECTED_CONFIG}" \
    "${run_dir}" \
    "${seed}" \
    "${FINAL_EVAL_EPISODES}" \
    "${FINAL_EVAL_MAX_STEPS}" \
    "${FINAL_TARGET_RETURN}"

  FINAL_RUN_DIRS+=("${run_dir}")
done

FINAL_RUN_ARGS=()
for run_dir in "${FINAL_RUN_DIRS[@]}"; do
  FINAL_RUN_ARGS+=(--run-dir "${run_dir}")
done

"${PYTHON}" "${REPO_ROOT}/scripts/build_qt_results_table.py" \
  "${FINAL_RUN_ARGS[@]}" \
  --reference-results "${REFERENCE_PATH}" \
  --finalized-dir "${FINALIZED_DIR}"

MANIFEST_PATH="${FINALIZED_DIR}/manifest.json"
CANONICAL_CHECKPOINT="$(qt_manifest_field "${MANIFEST_PATH}" "canonical_checkpoint_path")"
CANONICAL_SEED="$(qt_manifest_field "${MANIFEST_PATH}" "canonical_seed")"
CANONICAL_ROLLOUT="$(qt_resolve_rollout_path "${CANONICAL_CHECKPOINT}" "walker2d-medium-replay-v2" "${CANONICAL_SEED}")"
CANONICAL_ROLLOUT_SUFFIX=".${CANONICAL_ROLLOUT##*.}"

"${PYTHON}" "${REPO_ROOT}/scripts/render_qt_rollout.py" \
  --config "${SELECTED_CONFIG}" \
  --checkpoint "${CANONICAL_CHECKPOINT}" \
  --seed "${CANONICAL_SEED}" \
  --target-return "${FINAL_TARGET_RETURN}" \
  --max-steps "${FINAL_EVAL_MAX_STEPS}"

"${PYTHON}" "${REPO_ROOT}/scripts/build_qt_results_table.py" \
  "${FINAL_RUN_ARGS[@]}" \
  --reference-results "${REFERENCE_PATH}" \
  --finalized-dir "${FINALIZED_DIR}" \
  --canonical-rollout "${CANONICAL_ROLLOUT}"

qt_assert_files_exist \
  "${FINALIZED_DIR}/manifest.json" \
  "${FINALIZED_DIR}/results_table.csv" \
  "${FINALIZED_DIR}/results_table.md" \
  "${FINALIZED_DIR}/best.pt" \
  "${FINALIZED_DIR}/best_eval.json" \
  "${FINALIZED_DIR}/canonical_rollout${CANONICAL_ROLLOUT_SUFFIX}"

echo "selected_config: ${SELECTED_CONFIG}"
echo "finalized_dir: ${FINALIZED_DIR}"
echo "canonical_checkpoint: ${CANONICAL_CHECKPOINT}"
echo "canonical_rollout: ${CANONICAL_ROLLOUT}"
