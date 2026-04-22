"""Results aggregation and final package helpers for QT experiments."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .checkpoints import load_checkpoint
from .evaluation import normalize_episode_returns
from .runtime import ensure_directory, write_json


DEFAULT_FINALIZED_DIR = Path("results/finalized/walker2d_medium_replay_v2")

TABLE_COLUMNS = [
    "row_type",
    "label",
    "seed",
    "run_name",
    "best_step",
    "best_mean_return",
    "best_std_return",
    "best_mean_normalized_score",
    "best_std_normalized_score",
    "latest_mean_return",
    "latest_std_return",
    "latest_mean_normalized_score",
    "latest_std_normalized_score",
    "final_eval_episodes",
    "final_eval_target_return",
    "checkpoint_path",
    "rollout_path",
    "source_name",
    "source_note",
    "source_url",
]


@dataclass(frozen=True)
class RunArtifactSummary:
    """Structured view of one finalized seed run."""

    env_name: str
    seed: int
    run_name: str
    run_dir: Path
    best_checkpoint_path: Path
    latest_checkpoint_path: Path
    best_eval_path: Path | None
    latest_eval_path: Path | None
    best_step: int
    latest_step: int
    best_mean_return: float
    best_std_return: float
    best_mean_normalized_score: float | None
    best_std_normalized_score: float | None
    latest_mean_return: float
    latest_std_return: float
    latest_mean_normalized_score: float | None
    latest_std_normalized_score: float | None
    final_eval_episodes: int
    final_eval_target_return: float | None


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _maybe_normalized_fields(env_name: str, payload: Mapping[str, Any]) -> tuple[float | None, float | None]:
    if "mean_normalized_score" in payload:
        return float(payload["mean_normalized_score"]), float(payload["std_normalized_score"])

    episode_returns = payload.get("episode_returns")
    if not episode_returns:
        return None, None

    normalized_scores = normalize_episode_returns(env_name, [float(item) for item in episode_returns])
    if not normalized_scores:
        return None, None

    return float(np.mean(normalized_scores)), float(np.std(normalized_scores))


def load_run_artifact_summary(run_dir: str | Path) -> RunArtifactSummary:
    """Load the summary objects needed for table building and selection."""

    run_dir = Path(run_dir).resolve()
    summary = _load_json(run_dir / "summary.json")

    best_checkpoint_path = Path(summary["best_checkpoint_path"]).resolve()
    latest_checkpoint_path = Path(summary["checkpoint_path"]).resolve()

    best_checkpoint = load_checkpoint(best_checkpoint_path, map_location="cpu")
    latest_checkpoint = load_checkpoint(latest_checkpoint_path, map_location="cpu")
    best_step = int(best_checkpoint["step"])
    latest_step = int(latest_checkpoint["step"])

    best_eval_path = run_dir / "best_eval.json"
    latest_eval_path = run_dir / "latest_eval.json"
    best_eval_payload = _load_json(best_eval_path) if best_eval_path.exists() else dict(summary["best_evaluation"])
    latest_eval_payload = _load_json(latest_eval_path) if latest_eval_path.exists() else dict(summary["latest_evaluation"])

    env_name = str(best_eval_payload["env_name"])
    best_norm_mean, best_norm_std = _maybe_normalized_fields(env_name, best_eval_payload)
    latest_norm_mean, latest_norm_std = _maybe_normalized_fields(env_name, latest_eval_payload)

    return RunArtifactSummary(
        env_name=env_name,
        seed=int(best_eval_payload["seed"]),
        run_name=run_dir.name,
        run_dir=run_dir,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        best_eval_path=best_eval_path if best_eval_path.exists() else None,
        latest_eval_path=latest_eval_path if latest_eval_path.exists() else None,
        best_step=best_step,
        latest_step=latest_step,
        best_mean_return=float(best_eval_payload["mean_return"]),
        best_std_return=float(best_eval_payload["std_return"]),
        best_mean_normalized_score=best_norm_mean,
        best_std_normalized_score=best_norm_std,
        latest_mean_return=float(latest_eval_payload["mean_return"]),
        latest_std_return=float(latest_eval_payload["std_return"]),
        latest_mean_normalized_score=latest_norm_mean,
        latest_std_normalized_score=latest_norm_std,
        final_eval_episodes=int(best_eval_payload["episodes"]),
        final_eval_target_return=(
            float(best_eval_payload["target_return"])
            if best_eval_payload.get("target_return") is not None
            else None
        ),
    )


def select_best_candidate_run(runs: Iterable[RunArtifactSummary]) -> RunArtifactSummary:
    """Pick the walker tuning candidate using the configured tie-break rules."""

    runs = list(runs)
    if not runs:
        raise ValueError("At least one run is required for candidate selection.")

    return min(
        runs,
        key=lambda run: (
            -run.best_mean_return,
            run.best_std_return,
            abs(run.best_mean_return - run.latest_mean_return),
        ),
    )


def select_canonical_run(runs: Iterable[RunArtifactSummary]) -> RunArtifactSummary:
    """Pick the canonical report checkpoint across final seed runs."""

    runs = list(runs)
    if not runs:
        raise ValueError("At least one run is required for canonical selection.")

    return min(
        runs,
        key=lambda run: (
            -run.best_mean_return,
            run.best_std_return,
            run.best_step,
        ),
    )


def _copy_artifact(src: str | Path | None, dst: str | Path) -> str | None:
    if src is None:
        return None

    src_path = Path(src)
    if not src_path.exists():
        return None

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return str(dst_path.resolve())


def _load_reference_rows(reference_results_path: str | Path) -> list[dict[str, Any]]:
    reference_payload = _load_json(reference_results_path)
    rows: list[dict[str, Any]] = []
    for entry in reference_payload.get("entries", []):
        rows.append(
            {
                "row_type": "reference",
                "label": entry.get("label", entry.get("source_name", "reference")),
                "seed": "",
                "run_name": "",
                "best_step": "",
                "best_mean_return": "",
                "best_std_return": "",
                "best_mean_normalized_score": entry.get("mean_normalized_score", ""),
                "best_std_normalized_score": entry.get("std_normalized_score", ""),
                "latest_mean_return": "",
                "latest_std_return": "",
                "latest_mean_normalized_score": "",
                "latest_std_normalized_score": "",
                "final_eval_episodes": entry.get("episodes", ""),
                "final_eval_target_return": entry.get("target_return", ""),
                "checkpoint_path": "",
                "rollout_path": "",
                "source_name": entry.get("source_name", "reference"),
                "source_note": entry.get("source_note", ""),
                "source_url": entry.get("source_url", ""),
            }
        )
    return rows


def _format_table_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_table_value(row.get(key)) for key in TABLE_COLUMNS})


def _write_markdown(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(_format_table_value(row.get(key)) for key in TABLE_COLUMNS) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_results_package(
    *,
    run_dirs: Iterable[str | Path],
    reference_results_path: str | Path,
    finalized_dir: str | Path = DEFAULT_FINALIZED_DIR,
    canonical_rollout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the finalized walker results package and return its manifest payload."""

    run_summaries = sorted(
        [load_run_artifact_summary(path) for path in run_dirs],
        key=lambda run: run.seed,
    )
    canonical_run = select_canonical_run(run_summaries)

    finalized_dir = ensure_directory(finalized_dir).resolve()
    reference_copy_path = finalized_dir / Path(reference_results_path).name
    reference_copy = _copy_artifact(reference_results_path, reference_copy_path)

    canonical_best_checkpoint_path = finalized_dir / "best.pt"
    canonical_best_eval_path = finalized_dir / "best_eval.json"
    copied_best_checkpoint = _copy_artifact(canonical_run.best_checkpoint_path, canonical_best_checkpoint_path)
    copied_best_eval = _copy_artifact(canonical_run.best_eval_path, canonical_best_eval_path)

    copied_rollout_path: str | None = None
    copied_rollout_summary_path: str | None = None
    if canonical_rollout_path is not None:
        rollout_path = Path(canonical_rollout_path)
        copied_rollout_path = _copy_artifact(
            rollout_path,
            finalized_dir / f"canonical_rollout{rollout_path.suffix.lower()}",
        )
        rollout_summary = rollout_path.with_suffix(rollout_path.suffix + ".json")
        if rollout_summary.exists():
            copied_rollout_summary_path = _copy_artifact(
                rollout_summary,
                finalized_dir / f"canonical_rollout{rollout_path.suffix.lower()}.json",
            )

    rows: list[dict[str, Any]] = []
    for run in run_summaries:
        is_canonical = run.run_dir == canonical_run.run_dir
        rows.append(
            {
                "row_type": "seed_run",
                "label": f"seed_{run.seed}",
                "seed": run.seed,
                "run_name": run.run_name,
                "best_step": run.best_step,
                "best_mean_return": run.best_mean_return,
                "best_std_return": run.best_std_return,
                "best_mean_normalized_score": run.best_mean_normalized_score,
                "best_std_normalized_score": run.best_std_normalized_score,
                "latest_mean_return": run.latest_mean_return,
                "latest_std_return": run.latest_std_return,
                "latest_mean_normalized_score": run.latest_mean_normalized_score,
                "latest_std_normalized_score": run.latest_std_normalized_score,
                "final_eval_episodes": run.final_eval_episodes,
                "final_eval_target_return": run.final_eval_target_return,
                "checkpoint_path": copied_best_checkpoint if is_canonical and copied_best_checkpoint else str(run.best_checkpoint_path),
                "rollout_path": copied_rollout_path if is_canonical and copied_rollout_path else "",
                "source_name": "TranQil repo run",
                "source_note": f"best.pt selected from {run.run_name}",
                "source_url": "",
            }
        )
    rows.extend(_load_reference_rows(reference_results_path))

    csv_path = finalized_dir / "results_table.csv"
    markdown_path = finalized_dir / "results_table.md"
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path)

    manifest = {
        "env_name": canonical_run.env_name,
        "seed_run_dirs": [str(run.run_dir) for run in run_summaries],
        "canonical_seed": canonical_run.seed,
        "canonical_run_name": canonical_run.run_name,
        "canonical_checkpoint_path": copied_best_checkpoint or str(canonical_run.best_checkpoint_path),
        "canonical_best_eval_path": copied_best_eval or (str(canonical_run.best_eval_path) if canonical_run.best_eval_path else None),
        "canonical_rollout_path": copied_rollout_path,
        "canonical_rollout_summary_path": copied_rollout_summary_path,
        "reference_results_path": reference_copy,
        "results_table_csv": str(csv_path.resolve()),
        "results_table_md": str(markdown_path.resolve()),
        "selection_criteria": {
            "canonical": [
                "highest best_eval mean return",
                "lower best_eval std return",
                "earlier best checkpoint step",
            ],
        },
        "rows": rows,
    }
    write_json(finalized_dir / "manifest.json", manifest)
    return manifest
