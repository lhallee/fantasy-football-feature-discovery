"""Download versioned nflverse files and record a content manifest."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from .constants import CURRENT_SEASON, FIRST_ROSTER_SEASON, FIRST_STAT_SEASON
from .constants import MAX_PROJECT_DATA_BYTES, NFLVERSE_RELEASE_ROOT


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """One public source file and its local destination."""

    dataset: str
    season: int | None
    url: str
    destination: Path


@dataclass(frozen=True, slots=True)
class DownloadRecord:
    """Content metadata for one downloaded file."""

    dataset: str
    season: int | None
    url: str
    path: str
    size_bytes: int
    sha256: str
    retrieved_at_utc: str
    verified_at_utc: str
    reused: bool


def download_specs(
    raw_dir: Path, current_season: int = CURRENT_SEASON
) -> list[DownloadSpec]:
    """Declare all source files needed for a complete run."""
    specs: list[DownloadSpec] = []
    for season in range(FIRST_STAT_SEASON, current_season):
        filename = f"stats_player_week_{season}.parquet"
        specs.append(
            DownloadSpec(
                dataset="weekly_player_stats",
                season=season,
                url=f"{NFLVERSE_RELEASE_ROOT}/stats_player/{filename}",
                destination=raw_dir / "stats" / filename,
            )
        )

    for season in range(FIRST_ROSTER_SEASON, current_season + 1):
        roster_filename = f"roster_weekly_{season}.parquet"
        specs.append(
            DownloadSpec(
                dataset="weekly_rosters",
                season=season,
                url=f"{NFLVERSE_RELEASE_ROOT}/weekly_rosters/{roster_filename}",
                destination=raw_dir / "rosters" / roster_filename,
            )
        )
        depth_filename = f"depth_charts_{season}.parquet"
        specs.append(
            DownloadSpec(
                dataset="depth_charts",
                season=season,
                url=f"{NFLVERSE_RELEASE_ROOT}/depth_charts/{depth_filename}",
                destination=raw_dir / "depth_charts" / depth_filename,
            )
        )

    for season in range(2012, current_season):
        filename = f"snap_counts_{season}.parquet"
        specs.append(
            DownloadSpec(
                dataset="snap_counts",
                season=season,
                url=f"{NFLVERSE_RELEASE_ROOT}/snap_counts/{filename}",
                destination=raw_dir / "snap_counts" / filename,
            )
        )

    specs.append(
        DownloadSpec(
            dataset="players",
            season=None,
            url=f"{NFLVERSE_RELEASE_ROOT}/players/players.parquet",
            destination=raw_dir / "players.parquet",
        )
    )
    specs.extend(
        [
            DownloadSpec(
                dataset="combine",
                season=None,
                url=f"{NFLVERSE_RELEASE_ROOT}/combine/combine.parquet",
                destination=raw_dir / "combine.parquet",
            ),
            DownloadSpec(
                dataset="draft_picks",
                season=None,
                url=f"{NFLVERSE_RELEASE_ROOT}/draft_picks/draft_picks.parquet",
                destination=raw_dir / "draft_picks.parquet",
            ),
        ]
    )
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _retrieval_time(
    destination: Path,
    previous_record: dict[str, Any] | None,
) -> str:
    if previous_record is not None:
        retrieved_at = previous_record.get("retrieved_at_utc")
        if retrieved_at is not None:
            return str(retrieved_at)
        if not previous_record.get("reused", False):
            downloaded_at = previous_record.get("downloaded_at_utc")
            if downloaded_at is not None:
                return str(downloaded_at)

    modified_at = datetime.fromtimestamp(destination.stat().st_mtime, UTC)
    return modified_at.replace(microsecond=0).isoformat()


def _download_one(
    spec: DownloadSpec,
    force: bool,
    raw_dir: Path,
    previous_record: dict[str, Any] | None = None,
) -> DownloadRecord:
    spec.destination.parent.mkdir(parents=True, exist_ok=True)
    reused = spec.destination.is_file() and not force
    if reused and previous_record is not None:
        expected_hash = previous_record.get("sha256")
        reused = expected_hash is None or _sha256(spec.destination) == expected_hash

    verified_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    if not reused:
        temporary_path = spec.destination.with_suffix(spec.destination.suffix + ".part")
        with requests.get(spec.url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as destination:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        destination.write(block)
        temporary_path.replace(spec.destination)
        retrieved_at = verified_at
    else:
        retrieved_at = _retrieval_time(spec.destination, previous_record)

    return DownloadRecord(
        dataset=spec.dataset,
        season=spec.season,
        url=spec.url,
        path=spec.destination.relative_to(raw_dir).as_posix(),
        size_bytes=spec.destination.stat().st_size,
        sha256=_sha256(spec.destination),
        retrieved_at_utc=retrieved_at,
        verified_at_utc=verified_at,
        reused=reused,
    )


def download_all(
    raw_dir: Path,
    current_season: int = CURRENT_SEASON,
    force: bool = False,
    workers: int = 4,
) -> list[DownloadRecord]:
    """Download declared inputs with bounded concurrency and write a manifest."""
    specs = download_specs(raw_dir, current_season)
    manifest_path = raw_dir / "manifest.json"
    previous_records: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_records = {
            str(record["path"]): record for record in previous_manifest.get("files", [])
        }

    records: list[DownloadRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                spec,
                force,
                raw_dir,
                previous_records.get(spec.destination.relative_to(raw_dir).as_posix()),
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda record: (record.dataset, record.season or 0))
    total_bytes = sum(record.size_bytes for record in records)
    if total_bytes > MAX_PROJECT_DATA_BYTES:
        raise RuntimeError(
            f"Raw inputs use {total_bytes:,} bytes, above the "
            f"{MAX_PROJECT_DATA_BYTES:,}-byte project limit."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest = {
        "manifest_version": 2,
        "created_at_utc": created_at,
        "data_cutoff_utc": max(record.retrieved_at_utc for record in records),
        "verified_at_utc": created_at,
        "current_season": current_season,
        "total_bytes": total_bytes,
        "files": [asdict(record) for record in records],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return records
