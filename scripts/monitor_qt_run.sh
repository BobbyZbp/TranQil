#!/usr/bin/env bash
# Tail-based health monitor for a running anchor-mode QT training.
# Prints: latest metrics.jsonl line, latest evaluations.jsonl line, and
# flags NaN loss / stale timestamps.
#
# Usage: bash scripts/monitor_qt_run.sh <run_dir>
# Example: bash scripts/monitor_qt_run.sh results/qt_anchor_runs/qt_anchor_walker2d_medium_replay_v2_seed123

set -euo pipefail

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 1
fi

METRICS="${RUN_DIR}/metrics.jsonl"
EVALS="${RUN_DIR}/evaluations.jsonl"

echo "=== Monitor: ${RUN_DIR} ==="
if [[ ! -f "${METRICS}" ]]; then
  echo "metrics.jsonl not yet written — training may still be in iter 1"
  exit 0
fi

MTIME_METRICS=$(stat -c '%Y' "${METRICS}")
NOW=$(date +%s)
AGE=$((NOW - MTIME_METRICS))

echo "metrics.jsonl last updated ${AGE}s ago"

LAST_METRIC=$(tail -n 1 "${METRICS}")
echo "latest train: ${LAST_METRIC}"

if echo "${LAST_METRIC}" | grep -qiE '"(actor_loss|critic_loss|bc_loss|ql_loss)":\s*(nan|-?inf)'; then
  echo "🔴 NaN/Inf detected in latest train metrics"
fi

if [[ -f "${EVALS}" ]]; then
  LAST_EVAL=$(tail -n 1 "${EVALS}")
  echo "latest eval: ${LAST_EVAL}"
  LAST_RETURNS=$(awk -F'"mean_return"' '{print $2}' "${EVALS}" | awk -F':' '{print $2}' | awk -F',' '{print $1}' | tail -n 5 | tr '\n' ' ')
  echo "last 5 mean_return: ${LAST_RETURNS}"
fi

if (( AGE > 1800 )); then
  echo "🟡 metrics.jsonl is stale (>${AGE}s old) — run may be hung"
fi
