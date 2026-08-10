"""Tests for content-aware download reuse and provenance timestamps."""

import hashlib

from pathlib import Path

from fantasy_football.download import DownloadSpec, _download_one


def test_reused_download_preserves_retrieval_time(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    destination = raw_dir / "source.parquet"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"frozen public source")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    retrieved_at = "2026-08-10T03:11:22+00:00"
    previous_record = {
        "path": "source.parquet",
        "sha256": digest,
        "retrieved_at_utc": retrieved_at,
        "reused": True,
    }
    spec = DownloadSpec("test", 2026, "https://example.invalid/source", destination)

    record = _download_one(spec, False, raw_dir, previous_record)

    assert record.reused
    assert record.retrieved_at_utc == retrieved_at
    assert record.verified_at_utc >= retrieved_at
    assert record.sha256 == digest
