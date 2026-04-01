#!/usr/bin/env bash

# TranQil D4RL runtime installer / repair script
#
# Purpose:
#   Complete the legacy offline-RL software stack on top of the base conda
#   environment defined in `environment.yml`.
#
# Pipeline role:
#   Phase 3: `install/repair stack`
#
# Functionality implemented here:
#   - ensures the repo-local MuJoCo 2.1.0 payload exists under `.mujoco/`
#   - upgrades pip tooling inside the active environment
#   - installs pinned runtime packages used by the current validation path:
#       * `torch`
#       * `gym==0.23.1`
#       * `mujoco-py==2.1.2.14`
#       * `mjrl`
#       * `d4rl`
#   - applies the WSL2-specific `mujoco-py` builder patch when needed
#   - validates that `mujoco-py` imports and compiles successfully
#
# Inputs / assumptions:
#   - the shell is already using the repo-local `tranqil-qt` environment
#   - `env_vars.sh` can be sourced successfully
#   - network access is available when packages or MuJoCo need downloading
#
# Outputs / side effects:
#   - mutates the current Python environment by installing packages
#   - may download MuJoCo and pip wheels
#   - leaves the environment ready for smoke validation
#
# Next step in the pipeline:
#   `run_smoke_test.sh`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env_vars.sh"

if ! command -v python >/dev/null 2>&1; then
  echo "python was not found on PATH. Activate the micromamba environment first." >&2
  exit 1
fi

MUJOCO_TARBALL_URL="https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"
MUJOCO_ARCHIVE="${REPO_ROOT}/.mujoco/mujoco210-linux-x86_64.tar.gz"

mkdir -p "${TRANQIL_MUJOCO_ROOT}"

if [[ ! -d "${TRANQIL_MUJOCO_DIR}" ]]; then
  echo "[install_d4rl_stack] Downloading MuJoCo 2.1.0 into ${TRANQIL_MUJOCO_ROOT}"
  curl -L "${MUJOCO_TARBALL_URL}" -o "${MUJOCO_ARCHIVE}"
  tar -xzf "${MUJOCO_ARCHIVE}" -C "${TRANQIL_MUJOCO_ROOT}"
fi

echo "[install_d4rl_stack] Upgrading pip tooling"
python -m pip install --upgrade pip

echo "[install_d4rl_stack] Installing CPU-first PyTorch for local validation"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.4.1"

echo "[install_d4rl_stack] Installing Gym, mujoco-py, and direct runtime dependencies"
python -m pip install \
  "gym==0.23.1" \
  "mujoco-py==2.1.2.14" \
  "click>=8,<9" \
  "termcolor>=2,<3"

if [[ -d "/usr/lib/wsl/lib" ]]; then
  echo "[install_d4rl_stack] Applying WSL2-specific mujoco-py builder patch"
  python "${SCRIPT_DIR}/patch_mujoco_py_builder.py"
fi

echo "[install_d4rl_stack] Validating mujoco-py build/import"
python - <<'PY'
import mujoco_py

print(f"[install_d4rl_stack] mujoco-py extension: {mujoco_py.cymj.__file__}")
PY

echo "[install_d4rl_stack] Installing mjrl for D4RL's hand/locomotion registration path"
python -m pip install --no-deps \
  "mjrl @ git+https://github.com/aravindr93/mjrl@master"

echo "[install_d4rl_stack] Installing D4RL from the official repository without optional heavy domains"
python -m pip install --no-deps \
  "git+https://github.com/Farama-Foundation/D4RL@master#egg=d4rl"

echo "[install_d4rl_stack] Installation complete"
