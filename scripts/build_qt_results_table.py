#!/usr/bin/env python
"""Aggregate QT run artifacts into a finalized results package."""

from __future__ import annotations

import argparse
import json

from tranqil.results_table import DEFAULT_FINALIZED_DIR, build_results_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a QT results table and finalized package.")
    parser.add_argument(
        "--run-dir",
        action="append",
        dest="run_dirs",
        required=True,
        help="Run directory to include. Pass this flag once per seed run.",
    )
    parser.add_argument(
        "--reference-results",
        required=True,
        help="Path to the checked-in paper/reference results file.",
    )
    parser.add_argument(
        "--finalized-dir",
        default=str(DEFAULT_FINALIZED_DIR),
        help="Directory where the finalized package should be written.",
    )
    parser.add_argument(
        "--canonical-rollout",
        default=None,
        help="Optional learned-policy rollout video to copy into the finalized package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_results_package(
        run_dirs=args.run_dirs,
        reference_results_path=args.reference_results,
        finalized_dir=args.finalized_dir,
        canonical_rollout_path=args.canonical_rollout,
    )
    print(json.dumps(
        {
            "manifest_path": f"{args.finalized_dir}/manifest.json",
            "canonical_checkpoint_path": manifest["canonical_checkpoint_path"],
            "canonical_seed": manifest["canonical_seed"],
            "canonical_rollout_path": manifest["canonical_rollout_path"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
