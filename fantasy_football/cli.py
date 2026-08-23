"""Command-line entry points for the source download and data build."""

import argparse
import json

from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .constants import CURRENT_SEASON
from .dataset import build_datasets
from .download import download_all


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fantasy_football",
        description="Download nflverse inputs and build the processed tables.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="Download nflverse inputs.")
    download.add_argument("--force", action="store_true")
    download.add_argument("--workers", type=int, default=4)
    build = subparsers.add_parser("build", help="Build processed Parquet tables.")
    build.add_argument("--snapshot-date", type=date.fromisoformat, default=None)
    return parser


def main() -> None:
    """Run the selected pipeline stage."""
    arguments = _parser().parse_args()
    root = arguments.root.resolve()
    if arguments.command == "download":
        output: Any = download_all(
            root / "data" / "raw",
            season=arguments.season,
            force=arguments.force,
            workers=arguments.workers,
        )
    else:
        output = build_datasets(root, snapshot_date=arguments.snapshot_date)
    print(json.dumps(output, default=_serialize, indent=2, sort_keys=True))
