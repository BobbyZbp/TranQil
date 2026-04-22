#!/usr/bin/env bash
# Run the scoped anchor-mode QT replication pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/scripts/qt_workflow_common.sh"
qt_bootstrap

EPISODES="${REPLICATION_EPISODES:-10}"
MAX_STEPS="${REPLICATION_MAX_STEPS:-500}"
STEPS_OVERRIDE="${ANCHOR_STEPS_OVERRIDE:-}"
RUN_BASE_DIR="${RUN_BASE_DIR:-${REPO_ROOT}/results/qt_anchor_runs}"
FINALIZED_BASE_DIR="${FINALIZED_BASE_DIR:-${REPO_ROOT}/results/finalized/scoped_anchor}"
SUMMARY_DIR="${SUMMARY_DIR:-${REPO_ROOT}/results/finalized/scoped_replication}"
TASK_FILTER="${TASK_FILTER:-}"
WALKER_THRESHOLD="${WALKER_THRESHOLD:-93.5}"
HOPPER_THRESHOLD="${HOPPER_THRESHOLD:-97.0}"
MAZE_THRESHOLD="${MAZE_THRESHOLD:-157.0}"

IFS=' ' read -r -a SEEDS <<< "${REPLICATION_SEEDS:-123 124 125}"

TASK_SPECS=(
  "walker2d-medium-replay-v2|configs/qt_anchor_walker2d_medium_replay.yaml|experiments/reference_results/walker2d_medium_replay_v2_qt_reference.json|${WALKER_THRESHOLD}"
  "hopper-medium-replay-v2|configs/qt_anchor_hopper_medium_replay.yaml|experiments/reference_results/hopper_medium_replay_v2_qt_reference.json|${HOPPER_THRESHOLD}"
  "maze2d-medium-v1|configs/qt_anchor_maze2d_medium.yaml|experiments/reference_results/maze2d_medium_v1_qt_reference.json|${MAZE_THRESHOLD}"
)

mkdir -p "${SUMMARY_DIR}"
TASK_ROW_PATHS=()

for task_spec in "${TASK_SPECS[@]}"; do
  IFS='|' read -r ENV_NAME CONFIG_PATH REFERENCE_PATH THRESHOLD <<< "${task_spec}"
  if [[ -n "${TASK_FILTER}" && "${ENV_NAME}" != "${TASK_FILTER}" ]]; then
    continue
  fi
  echo "=== Replication task: ${ENV_NAME} ==="
  echo "config=${CONFIG_PATH} run_base_dir=${RUN_BASE_DIR} finalized_dir_base=${FINALIZED_BASE_DIR}"
  TASK_SLUG="${ENV_NAME//-/_}"
  FINALIZED_DIR="${FINALIZED_BASE_DIR}/${TASK_SLUG}"
  RUN_DIRS=()

  for seed in "${SEEDS[@]}"; do
    RUN_NAME="qt_anchor_${TASK_SLUG}_seed${seed}"
    echo "--- Seed ${seed}: training run_name=${RUN_NAME}"
    TRAIN_CMD=(
      "${QT_PYTHON}" "${REPO_ROOT}/scripts/train_qt.py"
      --config "${REPO_ROOT}/${CONFIG_PATH}"
      --run-name "${RUN_NAME}"
      --base-dir "${RUN_BASE_DIR}"
      --seed "${seed}"
    )
    if [[ -n "${STEPS_OVERRIDE}" ]]; then
      TRAIN_CMD+=(--steps "${STEPS_OVERRIDE}")
    fi
    EXPECTED_LATEST="${RUN_BASE_DIR}/${RUN_NAME}/checkpoints/latest.pt"
    if [[ -f "${EXPECTED_LATEST}" ]]; then
      echo "--- Seed ${seed}: auto-resuming from ${EXPECTED_LATEST}"
      TRAIN_CMD+=(--resume-from "${EXPECTED_LATEST}")
    fi
    "${TRAIN_CMD[@]}"

    RUN_DIR="${RUN_BASE_DIR}/${RUN_NAME}"
    RUN_DIRS+=("${RUN_DIR}")
    echo "--- Seed ${seed}: final evals from ${RUN_DIR}"

    "${QT_PYTHON}" "${REPO_ROOT}/scripts/eval_qt.py" \
      --config "${REPO_ROOT}/${CONFIG_PATH}" \
      --checkpoint "${RUN_DIR}/checkpoints/best.pt" \
      --seed "${seed}" \
      --episodes "${EPISODES}" \
      --max-steps "${MAX_STEPS}" \
      --output-path "${RUN_DIR}/best_eval.json"

    "${QT_PYTHON}" "${REPO_ROOT}/scripts/eval_qt.py" \
      --config "${REPO_ROOT}/${CONFIG_PATH}" \
      --checkpoint "${RUN_DIR}/checkpoints/latest.pt" \
      --seed "${seed}" \
      --episodes "${EPISODES}" \
      --max-steps "${MAX_STEPS}" \
      --output-path "${RUN_DIR}/latest_eval.json"
    echo "--- Seed ${seed}: completed"
  done

  mapfile -t CANONICAL_INFO < <(
    "${QT_PYTHON}" - "${RUN_DIRS[@]}" <<'PY'
import sys
from tranqil.results_table import load_run_artifact_summary, select_canonical_run

runs = [load_run_artifact_summary(path) for path in sys.argv[1:]]
canonical = select_canonical_run(runs)
print(canonical.best_checkpoint_path)
print(canonical.seed)
print(canonical.env_name)
PY
  )

  CANONICAL_CHECKPOINT="${CANONICAL_INFO[0]}"
  CANONICAL_SEED="${CANONICAL_INFO[1]}"
  CANONICAL_ENV="${CANONICAL_INFO[2]}"
  ROLLOUT_PATH="$(qt_resolve_rollout_path "${CANONICAL_CHECKPOINT}" "${CANONICAL_ENV}" "${CANONICAL_SEED}")"
  echo "=== Canonical checkpoint selected: seed=${CANONICAL_SEED} checkpoint=${CANONICAL_CHECKPOINT}"

  "${QT_PYTHON}" "${REPO_ROOT}/scripts/render_qt_rollout.py" \
    --config "${REPO_ROOT}/${CONFIG_PATH}" \
    --checkpoint "${CANONICAL_CHECKPOINT}" \
    --seed "${CANONICAL_SEED}" \
    --max-steps "${MAX_STEPS}"
  echo "=== Canonical rollout written: ${ROLLOUT_PATH}"

  BUILD_ARGS=(
    "${QT_PYTHON}" "${REPO_ROOT}/scripts/build_qt_results_table.py"
    --reference-results "${REPO_ROOT}/${REFERENCE_PATH}"
    --finalized-dir "${FINALIZED_DIR}"
    --canonical-rollout "${ROLLOUT_PATH}"
  )
  for run_dir in "${RUN_DIRS[@]}"; do
    BUILD_ARGS+=(--run-dir "${run_dir}")
  done
  "${BUILD_ARGS[@]}"
  echo "=== Finalized package written: ${FINALIZED_DIR}"

  TASK_ROW_PATH="${SUMMARY_DIR}/${TASK_SLUG}_row.json"
  PASSED="$(
    "${QT_PYTHON}" - "${ENV_NAME}" "${REPO_ROOT}/${REFERENCE_PATH}" "${THRESHOLD}" "${ROLLOUT_PATH}" "${TASK_ROW_PATH}" "${RUN_DIRS[@]}" <<'PY'
import json
import sys
from tranqil.replication_summary import build_task_replication_row

env_name = sys.argv[1]
reference_path = sys.argv[2]
threshold = float(sys.argv[3])
rollout_path = sys.argv[4]
row_path = sys.argv[5]
run_dirs = sys.argv[6:]

row = build_task_replication_row(
    env_name=env_name,
    run_dirs=run_dirs,
    reference_results_path=reference_path,
    threshold=threshold,
    canonical_rollout_path=rollout_path,
)
with open(row_path, "w", encoding="utf-8") as handle:
    json.dump(row, handle, indent=2, sort_keys=True)
print(str(row["passed"]).lower())
PY
  )"
  TASK_ROW_PATHS+=("${TASK_ROW_PATH}")

  if [[ "${PASSED}" != "true" ]]; then
    echo "Threshold check failed for ${ENV_NAME}; stopping before the next task." >&2
    exit 1
  fi
  echo "=== Threshold passed for ${ENV_NAME}; summary row written to ${TASK_ROW_PATH}"
done

"${QT_PYTHON}" - "${SUMMARY_DIR}" "${TASK_ROW_PATHS[@]}" <<'PY'
import json
import sys
from tranqil.replication_summary import write_scoped_replication_summary

output_dir = sys.argv[1]
row_paths = sys.argv[2:]
rows = []
for path in row_paths:
    with open(path, "r", encoding="utf-8") as handle:
        rows.append(json.load(handle))
paths = write_scoped_replication_summary(rows, output_dir=output_dir)
print(json.dumps(paths, indent=2))
PY
