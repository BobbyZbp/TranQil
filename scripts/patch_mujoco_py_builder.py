#!/usr/bin/env python

from __future__ import annotations

"""Patch the installed `mujoco_py` builder for this WSL2 setup.

Purpose:
    Modify the installed `mujoco_py.builder` implementation so WSL2 GPU-library
    detection recognizes `/usr/lib/wsl/lib` early enough to choose the correct
    builder path.

Pipeline role:
    Phase 3: `install/repair stack`
    This script is not a user-facing entrypoint. It is called by
    `install_d4rl_stack.sh` only when the runtime appears to be WSL2.

Functionality implemented here:
    - locate the installed `mujoco_py/builder.py` inside the active env
    - replace the original `get_nvidia_lib_dir()` prefix logic
    - move `/usr/lib/wsl/lib` ahead of the default `nvidia-smi` gate
    - remove the redundant trailing WSL block if present
    - fail loudly if the installed source does not match the expected pattern

Output:
    An in-place patch to the installed package so later `import mujoco_py`
    succeeds under this workspace's WSL2 configuration.
"""

import sys
from pathlib import Path


def main() -> int:
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    builder_path = site_packages / "mujoco_py" / "builder.py"

    if not builder_path.exists():
        raise FileNotFoundError(f"Could not find mujoco_py builder at {builder_path}")

    text = builder_path.read_text()
    patched_prefix = (
        "def get_nvidia_lib_dir():\n"
        "    wsl_path = '/usr/lib/wsl/lib'\n"
        "    if exists(wsl_path):\n"
        "        return wsl_path\n\n"
        '    exists_nvidia_smi = subprocess.call("type nvidia-smi", shell=True,\n'
        "                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0\n"
        "    if not exists_nvidia_smi:\n"
        "        return None\n"
    )
    old_prefix = (
        "def get_nvidia_lib_dir():\n"
        '    exists_nvidia_smi = subprocess.call("type nvidia-smi", shell=True,\n'
        "                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0\n"
        "    if not exists_nvidia_smi:\n"
        "        return None\n"
    )
    trailing_wsl_block = "    wsl_path = '/usr/lib/wsl/lib'\n    if exists(wsl_path):\n        return wsl_path\n\n"

    if patched_prefix in text and trailing_wsl_block not in text:
        print("[patch_mujoco_py_builder] WSL2 patch already applied")
        return 0

    if old_prefix not in text and patched_prefix not in text:
        raise RuntimeError("Could not find the expected WSL2 patch target in mujoco_py/builder.py")

    if old_prefix in text:
        text = text.replace(old_prefix, patched_prefix, 1)

    if trailing_wsl_block in text:
        text = text.replace(trailing_wsl_block, "", 1)

    builder_path.write_text(text)
    print(f"[patch_mujoco_py_builder] Patched {builder_path} to prioritize /usr/lib/wsl/lib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
