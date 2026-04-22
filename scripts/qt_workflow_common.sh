#!/usr/bin/env bash
# Shared shell helpers for QT smoke tests and completion workflows.

set -euo pipefail

qt_bootstrap() {
  : "${REPO_ROOT:?REPO_ROOT must be set before calling qt_bootstrap}"
  source "${REPO_ROOT}/scripts/activate_env.sh"
  QT_PYTHON="${PYTHON:-python}"
}

qt_eval_checkpoint_pair() {
  local config_path="$1"
  local run_dir="$2"
  local seed="$3"
  local episodes="$4"
  local max_steps="$5"
  local target_return="$6"

  "${QT_PYTHON}" "${REPO_ROOT}/scripts/eval_qt.py" \
    --config "${config_path}" \
    --checkpoint "${run_dir}/checkpoints/best.pt" \
    --seed "${seed}" \
    --episodes "${episodes}" \
    --max-steps "${max_steps}" \
    --target-return "${target_return}" \
    --output-path "${run_dir}/best_eval.json"

  "${QT_PYTHON}" "${REPO_ROOT}/scripts/eval_qt.py" \
    --config "${config_path}" \
    --checkpoint "${run_dir}/checkpoints/latest.pt" \
    --seed "${seed}" \
    --episodes "${episodes}" \
    --max-steps "${max_steps}" \
    --target-return "${target_return}" \
    --output-path "${run_dir}/latest_eval.json"
}

qt_resolve_rollout_path() {
  local checkpoint_path="$1"
  local env_name="$2"
  local seed="$3"

  "${QT_PYTHON}" - "${checkpoint_path}" "${env_name}" "${seed}" <<'PY'
import sys

from tranqil.rendering import mp4_is_supported, resolve_qt_rollout_output_path

print(
    resolve_qt_rollout_output_path(
        checkpoint_path=sys.argv[1],
        env_name=sys.argv[2],
        seed=int(sys.argv[3]),
        output=None,
        output_format="auto",
        mp4_supported=mp4_is_supported(),
    )
)
PY
}

qt_manifest_field() {
  local manifest_path="$1"
  local field_name="$2"

  "${QT_PYTHON}" - "${manifest_path}" "${field_name}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
print(manifest[sys.argv[2]])
PY
}

qt_assert_files_exist() {
  local path
  for path in "$@"; do
    test -f "${path}"
  done
}
