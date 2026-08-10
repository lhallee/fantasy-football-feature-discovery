"""Tests for experiment path and hash migration records."""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from fantasy_football.phase2_layout import ExperimentLayoutError
from fantasy_football.phase2_layout import load_organization_manifest
from fantasy_football.phase2_layout import verify_historical_file


ROOT = Path(__file__).resolve().parents[1]


def test_repository_organization_records_resolve_and_verify() -> None:
    manifest = load_organization_manifest(ROOT)

    for record in manifest["records"]:
        resolved = verify_historical_file(
            ROOT,
            record["historical_path"],
            record["historical_sha256"],
            manifest,
        )
        assert resolved == (ROOT / record["current_path"]).resolve()


def test_unrecorded_file_requires_its_historical_hash(tmp_path: Path) -> None:
    manifest_path = tmp_path / "experiments" / "organization_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"manifest_version": 1, "records": []}),
        encoding="utf-8",
    )
    source = tmp_path / "source.txt"
    source.write_text("current\n", encoding="utf-8")

    with pytest.raises(ExperimentLayoutError, match="Unrecorded"):
        verify_historical_file(tmp_path, "source.txt", "0" * 64)
