"""Shared rendering helpers for random and learned-policy rollouts."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


DEFAULT_PREVIEW_DIR = Path("results/previews")


def sanitize_name(name: str) -> str:
    """Convert an env or artifact name into a filesystem-safe stem."""

    return name.replace("/", "_").replace("-", "_")


def mp4_is_supported() -> bool:
    """Return whether `imageio-ffmpeg` is available for mp4 export."""

    try:
        import imageio_ffmpeg  # noqa: F401
    except Exception:
        return False
    return True


def capture_frame(env) -> np.ndarray:
    """Capture one rgb-array frame from a Gym/D4RL environment."""

    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render("rgb_array")

    if frame is None:
        raise RuntimeError("render(mode='rgb_array') returned None")

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Unexpected frame shape from render: {frame.shape}")
    return frame


def write_preview(output_path: str | Path, frames: list[np.ndarray], fps: int) -> None:
    """Write frames to mp4 or gif depending on the output suffix."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".mp4":
        imageio.mimwrite(output_path, frames, fps=fps)
        return

    if suffix == ".gif":
        duration_ms = max(1, int(1000 / fps))
        imageio.mimsave(output_path, frames, duration=duration_ms, loop=0)
        return

    raise ValueError(f"Unsupported output suffix: {suffix}")


def resolve_random_rollout_output_path(
    *,
    env_name: str,
    output: str | Path | None,
    output_format: str,
    mp4_supported: bool,
) -> Path:
    """Resolve the standard output location for a random-policy preview."""

    if output is not None:
        return Path(output)

    DEFAULT_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".mp4" if output_format in {"auto", "mp4"} and mp4_supported else ".gif"
    return DEFAULT_PREVIEW_DIR / f"{sanitize_name(env_name)}_random{suffix}"


def resolve_qt_rollout_output_path(
    *,
    checkpoint_path: str | Path,
    env_name: str,
    seed: int,
    output: str | Path | None,
    output_format: str,
    mp4_supported: bool,
) -> Path:
    """Resolve the default output path for a learned-policy rollout."""

    if output is not None:
        return Path(output)

    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent.parent
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".mp4" if output_format in {"auto", "mp4"} and mp4_supported else ".gif"
    stem = f"{sanitize_name(env_name)}_{checkpoint_path.stem}_seed{int(seed)}"
    return rollout_dir / f"{stem}{suffix}"


def resolve_summary_path(output_path: str | Path) -> Path:
    """Return the sidecar JSON summary path for a rendered artifact."""

    output_path = Path(output_path)
    return output_path.with_suffix(output_path.suffix + ".json")
