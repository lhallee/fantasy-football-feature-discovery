"""Command-line entry points for the reproducible project workflow."""

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .constants import CURRENT_SEASON
from .dataset import build_datasets
from .download import download_all
from .export import export_current_players
from .modeling import run_experiments


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fantasy_football",
        description="Build and evaluate compact next-season fantasy projections.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download nflverse inputs.")
    download.add_argument("--force", action="store_true")
    download.add_argument("--workers", type=int, default=4)
    build = subparsers.add_parser("build", help="Build processed Parquet tables.")
    build.add_argument("--snapshot-date", type=date.fromisoformat, default=None)
    train = subparsers.add_parser("train", help="Search, audit, and fit models.")
    train.add_argument(
        "--fast",
        action="store_true",
        help="Write an isolated smoke search below artifacts/smoke.",
    )
    subparsers.add_parser("export", help="Generate the static current-player module.")
    all_steps = subparsers.add_parser("all", help="Run download through export.")
    all_steps.add_argument("--force", action="store_true")
    all_steps.add_argument("--workers", type=int, default=4)
    all_steps.add_argument("--snapshot-date", type=date.fromisoformat, default=None)
    return parser


def main() -> None:
    """Run the selected pipeline stage."""
    arguments = _parser().parse_args()
    root = arguments.root.resolve()
    output: Any
    if arguments.command == "download":
        output = download_all(
            root / "data" / "raw",
            arguments.season,
            force=arguments.force,
            workers=arguments.workers,
        )
    elif arguments.command == "build":
        output = build_datasets(
            root,
            arguments.season,
            snapshot_date=arguments.snapshot_date,
        )
    elif arguments.command == "train":
        output = run_experiments(root, arguments.season, fast=arguments.fast)
    elif arguments.command == "export":
        output = export_current_players(root, arguments.season)
    elif arguments.command == "all":
        download_all(
            root / "data" / "raw",
            arguments.season,
            force=arguments.force,
            workers=arguments.workers,
        )
        build_datasets(
            root,
            arguments.season,
            snapshot_date=arguments.snapshot_date,
        )
        run_experiments(root, arguments.season)
        output = export_current_players(root, arguments.season)
    else:
        raise AssertionError(f"Unhandled command: {arguments.command!r}")
    print(json.dumps(output, default=_serialize, indent=2))
