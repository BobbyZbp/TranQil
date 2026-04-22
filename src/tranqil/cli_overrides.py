"""Shared CLI override helpers for QT scripts."""

from __future__ import annotations

import copy
from typing import Any, Mapping


def apply_train_overrides(
    config: Mapping[str, Any],
    *,
    steps: int | None = None,
    run_name: str | None = None,
    base_dir: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    target_return: float | None = None,
) -> dict[str, Any]:
    """Apply supported train-script overrides onto a loaded config."""

    updated = copy.deepcopy(dict(config))
    if steps is not None:
        updated["training"]["steps"] = int(steps)
        if bool(updated.get("anchor", {}).get("enabled", False)):
            updated["anchor"]["max_iters"] = 1
            updated["anchor"]["num_steps_per_iter"] = int(steps)
    if run_name is not None:
        updated["run_name"] = str(run_name)
    if base_dir is not None:
        updated["outputs"]["base_dir"] = str(base_dir)
    if device is not None:
        updated["device"] = str(device)
    if seed is not None:
        updated["seed"] = int(seed)
    if target_return is not None:
        updated["evaluation"]["target_return"] = float(target_return)
    return updated


def apply_eval_overrides(
    config: Mapping[str, Any],
    *,
    device: str | None = None,
    episodes: int | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
    target_return: float | None = None,
) -> dict[str, Any]:
    """Apply supported eval-script overrides onto a loaded config."""

    updated = copy.deepcopy(dict(config))
    if device is not None:
        updated["device"] = str(device)
    if episodes is not None:
        updated["evaluation"]["episodes"] = int(episodes)
    if max_steps is not None:
        updated["evaluation"]["max_steps"] = int(max_steps)
    if seed is not None:
        updated["seed"] = int(seed)
    if target_return is not None:
        updated["evaluation"]["target_return"] = float(target_return)
    return updated
