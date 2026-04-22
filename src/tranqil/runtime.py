"""Runtime utilities for QT training and evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the configured training device."""

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor batch fields to the requested device."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    """Convert arrays and tensors into JSON-safe scalars or lists."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON document to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(dict(payload)), handle, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(dict(payload)), sort_keys=True))
        handle.write("\n")
