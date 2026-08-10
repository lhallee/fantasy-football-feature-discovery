"""Resolve experiment paths across the documented layout migration."""

from __future__ import annotations

import json
import re

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .provenance import file_sha256


_HASH = re.compile(r"[0-9a-f]{64}\Z")
ORGANIZATION_MANIFEST_PATH = Path("experiments/organization_manifest.json")


class ExperimentLayoutError(RuntimeError):
    """Raised when the experiment organization record is invalid."""


def _repository_root(root: Path) -> Path:
    try:
        repository = root.expanduser().resolve(strict=True)
    except OSError as error:
        raise ExperimentLayoutError(
            f"Repository root does not exist: {root}"
        ) from error
    if not repository.is_dir():
        raise ExperimentLayoutError(f"Repository root is not a directory: {repository}")
    return repository


def load_organization_manifest(root: Path) -> dict[str, Any]:
    """Load and structurally validate the experiment path-migration record."""
    repository = _repository_root(root)
    manifest_path = repository / ORGANIZATION_MANIFEST_PATH
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentLayoutError(
            f"Cannot load experiment organization manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != 1:
        raise ExperimentLayoutError("Organization manifest must use version 1.")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ExperimentLayoutError("Organization manifest must contain records.")

    historical_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ExperimentLayoutError("Organization records must be objects.")
        historical_path = record.get("historical_path")
        current_path = record.get("current_path")
        historical_hash = record.get("historical_sha256")
        current_hash = record.get("current_sha256")
        if (
            not isinstance(historical_path, str)
            or not historical_path
            or not isinstance(current_path, str)
            or not current_path
            or not isinstance(historical_hash, str)
            or _HASH.fullmatch(historical_hash) is None
            or not isinstance(current_hash, str)
            or _HASH.fullmatch(current_hash) is None
        ):
            raise ExperimentLayoutError("Organization record fields are malformed.")
        if historical_path in historical_paths:
            raise ExperimentLayoutError(
                f"Historical path appears twice: {historical_path}"
            )
        historical_paths.add(historical_path)
    return manifest


def _record_by_historical_path(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ExperimentLayoutError("Organization manifest must contain records.")
    return {
        str(record["historical_path"]): record
        for record in records
        if isinstance(record, Mapping)
    }


def resolve_historical_path(
    root: Path,
    historical_path: str,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve a historical repository path to its current physical location."""
    repository = _repository_root(root)
    loaded = load_organization_manifest(repository) if manifest is None else manifest
    record = _record_by_historical_path(loaded).get(historical_path)
    current_path = historical_path if record is None else str(record["current_path"])
    try:
        candidate = (repository / current_path).resolve(strict=True)
        candidate.relative_to(repository)
    except (OSError, ValueError) as error:
        raise ExperimentLayoutError(
            f"Experiment path is missing or unsafe: {current_path}"
        ) from error
    if not candidate.is_file():
        raise ExperimentLayoutError(f"Experiment path is not a file: {current_path}")
    return candidate


def verify_historical_file(
    root: Path,
    historical_path: str,
    expected_historical_sha256: str,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Verify a historical hash through an explicit relocation or maintenance record."""
    if _HASH.fullmatch(expected_historical_sha256) is None:
        raise ExperimentLayoutError("Expected historical SHA-256 is malformed.")
    repository = _repository_root(root)
    loaded = load_organization_manifest(repository) if manifest is None else manifest
    record = _record_by_historical_path(loaded).get(historical_path)
    candidate = resolve_historical_path(repository, historical_path, loaded)
    actual_hash = file_sha256(candidate)
    if record is None:
        if actual_hash != expected_historical_sha256:
            raise ExperimentLayoutError(
                f"Unrecorded experiment file changed: {historical_path}"
            )
        return candidate

    if record["historical_sha256"] != expected_historical_sha256:
        raise ExperimentLayoutError(
            f"Historical hash differs from organization record: {historical_path}"
        )
    if record["current_sha256"] != actual_hash:
        raise ExperimentLayoutError(
            f"Current hash differs from organization record: {historical_path}"
        )
    return candidate
