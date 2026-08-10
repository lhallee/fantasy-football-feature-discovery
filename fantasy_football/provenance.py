"""Project hashes, package versions, and stable storage accounting."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from .constants import MAX_PROJECT_DATA_BYTES


TRACKED_PACKAGES = (
    "joblib",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "threadpoolctl",
    "xgboost",
)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Return versions for direct numerical runtime dependencies."""
    return {
        package: importlib.metadata.version(package) for package in TRACKED_PACKAGES
    }


def project_data_bytes(root: Path, *, include_smoke: bool = False) -> int:
    """Count canonical data and artifacts, optionally including smoke output."""
    paths = [*(root / "data").rglob("*"), *(root / "artifacts").rglob("*")]
    smoke_root = root / "artifacts" / "smoke"
    return sum(
        path.stat().st_size
        for path in paths
        if path.is_file() and (include_smoke or not path.is_relative_to(smoke_root))
    )


def _synchronize_summary_fields(
    root: Path,
    summary_fields: dict[Path, str],
    *,
    include_smoke: bool,
) -> int:
    """Stabilize self-inclusive byte counts for one summary group."""
    for _ in range(10):
        total_bytes = project_data_bytes(root, include_smoke=include_smoke)
        for path, field in summary_fields.items():
            if not path.is_file():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get(field) == total_bytes:
                continue
            summary[field] = total_bytes
            path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        exact_total_bytes = project_data_bytes(root, include_smoke=include_smoke)
        if exact_total_bytes == total_bytes:
            return exact_total_bytes

    raise RuntimeError("Project byte count did not stabilize.")


def synchronize_project_sizes(root: Path, *, update_canonical: bool = True) -> int:
    """Synchronize canonical and isolated smoke byte counts."""
    canonical_fields = {
        root / "data" / "processed" / "build_summary.json": (
            "total_project_data_bytes"
        ),
        root / "artifacts" / "result_summary.json": "data_and_artifact_bytes",
    }
    canonical_bytes = project_data_bytes(root)
    if update_canonical:
        canonical_bytes = _synchronize_summary_fields(
            root,
            canonical_fields,
            include_smoke=False,
        )
    smoke_summary = root / "artifacts" / "smoke" / "result_summary.json"
    complete_bytes = canonical_bytes
    if smoke_summary.is_file():
        complete_bytes = _synchronize_summary_fields(
            root,
            {smoke_summary: "data_and_artifact_bytes"},
            include_smoke=True,
        )
    if complete_bytes > MAX_PROJECT_DATA_BYTES:
        raise RuntimeError(
            f"Project data uses {complete_bytes:,} bytes, above the "
            f"{MAX_PROJECT_DATA_BYTES:,}-byte limit."
        )
    return canonical_bytes
