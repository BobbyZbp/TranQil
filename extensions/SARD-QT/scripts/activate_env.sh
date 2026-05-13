#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PREFIX="${REPO_ROOT}/.micromamba/root/envs/original-qt"
ENV_PYTHON="${ENV_PREFIX}/bin/python"

die() {
  echo "$*" >&2
  return 1 2>/dev/null || exit 1
}

if [[ ! -x "${ENV_PYTHON}" ]]; then
  die "original-qt env is missing at ${ENV_PREFIX}. Create it with: MAMBA_ROOT_PREFIX=\"${REPO_ROOT}/.micromamba/root\" /home/bobby/TranQil/.micromamba/micromamba create -y -f environment.yml"
fi

export QT_REPO_ROOT="${REPO_ROOT}"
export MAMBA_ROOT_PREFIX="${REPO_ROOT}/.micromamba/root"
export CONDA_PREFIX="${ENV_PREFIX}"
export PATH="${ENV_PREFIX}/bin:${PATH}"

export QT_CACHE_ROOT="${REPO_ROOT}/.cache"
export D4RL_DATASET_DIR="${REPO_ROOT}/.d4rl/datasets"
export MUJOCO_PY_MUJOCO_PATH="${REPO_ROOT}/.mujoco/mujoco210"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export XDG_CACHE_HOME="${QT_CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"

case ":${PYTHONPATH:-}:" in
  *":${REPO_ROOT}:"*) ;;
  *) export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

mkdir -p "${D4RL_DATASET_DIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
if [[ -d "/usr/lib/wsl/lib" ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
fi

echo "Activated original-qt environment at ${ENV_PREFIX}"
echo "Python: $(command -v python)"
