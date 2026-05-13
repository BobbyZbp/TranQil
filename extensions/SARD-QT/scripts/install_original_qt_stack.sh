#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v python >/dev/null 2>&1; then
  echo "python was not found on PATH. Source scripts/activate_env.sh first." >&2
  exit 1
fi

MUJOCO_ROOT="${REPO_ROOT}/.mujoco"
MUJOCO_DIR="${MUJOCO_ROOT}/mujoco210"
MUJOCO_ARCHIVE="${MUJOCO_ROOT}/mujoco210-linux-x86_64.tar.gz"
MUJOCO_URL="https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"

mkdir -p "${MUJOCO_ROOT}"
if [[ ! -d "${MUJOCO_DIR}" ]]; then
  echo "[install_original_qt_stack] Downloading MuJoCo 2.1.0"
  curl -L "${MUJOCO_URL}" -o "${MUJOCO_ARCHIVE}"
  tar -xzf "${MUJOCO_ARCHIVE}" -C "${MUJOCO_ROOT}"
fi

echo "[install_original_qt_stack] Upgrading pip tooling"
python -m pip install --upgrade pip

echo "[install_original_qt_stack] Installing PyTorch CUDA 12.1"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.4.1"

echo "[install_original_qt_stack] Installing original QT Python runtime"
python -m pip install \
  "gym==0.23.1" \
  "mujoco-py==2.1.2.14" \
  "packaging<22" \
  "transformers==4.5.1" \
  "tensorboard" \
  "click>=8,<9" \
  "termcolor>=2,<3"

if [[ -d "/usr/lib/wsl/lib" ]]; then
  echo "[install_original_qt_stack] Applying WSL2 mujoco-py builder patch"
  python "${SCRIPT_DIR}/patch_mujoco_py_builder.py"
fi

echo "[install_original_qt_stack] Validating mujoco-py import/build"
python - <<'PY'
import mujoco_py

print(f"mujoco-py extension: {mujoco_py.cymj.__file__}")
PY

echo "[install_original_qt_stack] Installing mjrl"
python -m pip install --no-deps \
  "mjrl @ git+https://github.com/aravindr93/mjrl@master"

echo "[install_original_qt_stack] Installing D4RL"
python -m pip install --no-deps \
  "git+https://github.com/Farama-Foundation/D4RL@master#egg=d4rl"

echo "[install_original_qt_stack] Installation complete"
