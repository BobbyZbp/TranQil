#!/usr/bin/env bash

# TranQil smoke-test entrypoint
#
# Purpose:
#   Run the repository's scoped environment validation in one command.
#
# Pipeline role:
#   Phase 4: `smoke validation`
#
# Functionality implemented here:
#   - sources `env_vars.sh`
#   - executes `smoke_test_env.py`
#
# What this validates:
#   - target env ids can be constructed through Gym/D4RL
#   - D4RL dataset loading works
#   - the key dataset arrays are present and non-empty
#
# Output:
#   Console-only validation. If datasets are not cached yet, D4RL may download
#   them into `data/d4rl` during the run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/env_vars.sh"
python "${SCRIPT_DIR}/smoke_test_env.py"
