#!/usr/bin/env python

from __future__ import annotations

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

    if patched_prefix in text:
        print("[patch_mujoco_py_builder] WSL2 patch already applied")
        return 0

    if old_prefix not in text:
        raise RuntimeError("Could not find expected get_nvidia_lib_dir() block in mujoco_py/builder.py")

    text = text.replace(old_prefix, patched_prefix, 1)
    if trailing_wsl_block in text:
        text = text.replace(trailing_wsl_block, "", text.count(trailing_wsl_block) - 1)

    builder_path.write_text(text)
    print(f"[patch_mujoco_py_builder] Patched {builder_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
