#!/usr/bin/env bash

# TranQil data smoke-test entrypoint
#
# Purpose:
#   Run the repository's QT data-pipeline validation in one command.
#
# Pipeline role:
#   Phase 6: `data pipeline validation`
#
# Functionality implemented here:
#   - sources `env_vars.sh`
#   - executes `smoke_test_data.py`
#
# What this validates:
#   - the scoped tasks can be preprocessed through the QT data stack
#   - metadata caches can be written and loaded back correctly
#   - a PyTorch dataloader can emit valid sequence batches

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/env_vars.sh"
python "${SCRIPT_DIR}/smoke_test_data.py" "$@"
