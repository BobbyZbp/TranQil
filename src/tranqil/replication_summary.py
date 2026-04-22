"""Scoped replication summary helpers for QT anchor-mode runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .evaluation import normalize_episode_returns
from .results_table import load_run_artifact_summary, select_canonical_run
from .runtime import ensure_directory, write_json


REPLICATION_TABLE_COLUMNS = [
    "env_name",
    "paper_mean_normalized_score",
    "paper_std_normalized_score",
    "our_mean_normalized_score",
    "our_std_normalized_score",
    "absolute_gap",
    "threshold",
    "passed",
    "canonical_seed",
    "canonical_run_dir",
    "canonical_checkpoint_path",
    "canonical_rollout_path",
    "reference_source_name",
    "reference_source_note",
    "reference_source_url",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_reference_entry(reference_results_path: str | Path) -> dict[str, Any]:
    payload = _load_json(reference_results_path)
    entries = payload.get("entries", [])
    if not entries:
        raise ValueError(f"Reference file has no entries: {reference_results_path}")
    return dict(entries[0])


def _load_eval_scores(run_dir: str | Path) -> list[float]:
    run_dir = Path(run_dir)
    eval_path = run_dir / "best_eval.json"
    if not eval_path.exists():
        summary = _load_json(run_dir / "summary.json")
        payload = dict(summary["best_evaluation"])
    else:
        payload = _load_json(eval_path)

    if "normalized_scores" in payload:
        return [float(item) for item in payload["normalized_scores"]]
    return normalize_episode_returns(
        str(payload["env_name"]),
        [float(item) for item in payload["episode_returns"]],
    )


def build_task_replication_row(
    *,
    env_name: str,
    run_dirs: Iterable[str | Path],
    reference_results_path: str | Path,
    threshold: float,
    canonical_rollout_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one paper-vs-ours summary row for a scoped task."""

    run_dirs = [Path(path).resolve() for path in run_dirs]
    if not run_dirs:
        raise ValueError("At least one run directory is required.")

    reference = _load_reference_entry(reference_results_path)
    normalized_scores: list[float] = []
    for run_dir in run_dirs:
        normalized_scores.extend(_load_eval_scores(run_dir))
    if not normalized_scores:
        raise ValueError(f"No normalized scores found for {env_name}.")

    canonical_run = select_canonical_run(load_run_artifact_summary(path) for path in run_dirs)
    our_mean = float(np.mean(normalized_scores))
    our_std = float(np.std(normalized_scores))
    paper_mean = float(reference["mean_normalized_score"])
    paper_std = float(reference["std_normalized_score"])

    return {
        "env_name": env_name,
        "paper_mean_normalized_score": paper_mean,
        "paper_std_normalized_score": paper_std,
        "our_mean_normalized_score": our_mean,
        "our_std_normalized_score": our_std,
        "absolute_gap": abs(our_mean - paper_mean),
        "threshold": float(threshold),
        "passed": bool(our_mean >= float(threshold)),
        "canonical_seed": int(canonical_run.seed),
        "canonical_run_dir": str(canonical_run.run_dir),
        "canonical_checkpoint_path": str(canonical_run.best_checkpoint_path),
        "canonical_rollout_path": str(Path(canonical_rollout_path).resolve()) if canonical_rollout_path is not None else "",
        "reference_source_name": reference.get("source_name", "reference"),
        "reference_source_note": reference.get("source_note", ""),
        "reference_source_url": reference.get("source_url", ""),
    }


def write_scoped_replication_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write csv/md/json copies of the scoped replication summary table."""

    output_dir = ensure_directory(output_dir)
    rows = [dict(row) for row in rows]

    csv_path = output_dir / "scoped_replication_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPLICATION_TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = output_dir / "scoped_replication_summary.md"
    header = "| " + " | ".join(REPLICATION_TABLE_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in REPLICATION_TABLE_COLUMNS) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in REPLICATION_TABLE_COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = output_dir / "scoped_replication_summary.json"
    write_json(json_path, {"rows": rows})

    return {
        "csv_path": str(csv_path.resolve()),
        "md_path": str(md_path.resolve()),
        "json_path": str(json_path.resolve()),
    }
