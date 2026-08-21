"""Isolation and integrity controls for Phase 2 experiment runs."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TREE_DIGEST_CONTRACT = (
    "SHA-256 over sorted records encoded as relative_path NUL decimal_size NUL "
    "raw_file_sha256 LF"
)
PROTECTED_SCOPE_NAMES = (
    "data/raw",
    "data/processed",
    "artifacts",
    "phase1_claim_documents",
    "config",
    "phase1_code_tests_pyproject",
    "players_2026.py",
)
MAX_COMBINED_BYTES = 900 * 1024**2
_RUN_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CURRENT_CLAIM_DOCUMENTS = (
    "README.md",
    "experiments/phase1/EXPERIMENT_LEDGER.md",
    "experiments/phase1/COMPLETION_AUDIT.md",
    "NOTICE.md",
    "docs/data_sources.md",
    "docs/fantasy_football.md",
    "docs/methods_and_results.md",
)
_LEGACY_CLAIM_DOCUMENTS = (
    "README.md",
    "EXPERIMENT_LEDGER.md",
    "COMPLETION_AUDIT.md",
    "NOTICE.md",
    "docs/data_sources.md",
    "docs/fantasy_football.md",
    "docs/methods_and_results.md",
)


class Phase2SafetyError(RuntimeError):
    """Raised when a Phase 2 isolation or integrity invariant fails."""


@dataclass(frozen=True, slots=True)
class TreeDigest:
    """File count, byte count, and digest for one protected tree."""

    files: int
    bytes: int
    sha256: str


@dataclass(slots=True)
class Phase2Run:
    """Integrity state captured around one isolated Phase 2 run."""

    root: Path
    run_dir: Path
    freeze_id: str
    input_hashes: dict[str, str]
    protected_before: dict[str, TreeDigest]
    output_hashes: dict[str, str] = field(default_factory=dict)
    protected_after: dict[str, TreeDigest] = field(default_factory=dict)
    combined_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class _FileRecord:
    relative_path: str
    size_bytes: int
    raw_sha256: bytes


def _repository_root(root: Path) -> Path:
    try:
        resolved = root.expanduser().resolve(strict=True)
    except OSError as error:
        raise Phase2SafetyError(f"Repository root does not exist: {root}") from error
    if not resolved.is_dir():
        raise Phase2SafetyError(f"Repository root is not a directory: {resolved}")
    return resolved


def _regular_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise Phase2SafetyError(f"Required protected directory is missing: {directory}")
    return [path for path in directory.rglob("*") if path.is_file()]


def _phase1_code_files(root: Path) -> list[Path]:
    package_root = root / "fantasy_football"
    test_root = root / "tests"
    files = []
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        is_later_phase = any(
            part.casefold().startswith(("phase2", "phase3"))
            for part in relative.parts
        )
        if path.name != "players_2026.py" and not is_later_phase:
            files.append(path)
    files.extend(
        path
        for path in test_root.rglob("test_*.py")
        if not path.stem.casefold().startswith(("test_phase2", "test_phase3"))
    )
    files.append(root / "pyproject.toml")
    return files


def _claim_document_paths(root: Path) -> list[Path]:
    """Return current paths, with a legacy fallback for frozen test fixtures."""
    phase1_root = root / "experiments" / "phase1"
    relative_paths = (
        _CURRENT_CLAIM_DOCUMENTS if phase1_root.is_dir() else _LEGACY_CLAIM_DOCUMENTS
    )
    return [root / path for path in relative_paths]


def _protected_scope_files(root: Path) -> dict[str, list[Path]]:
    scopes = {
        "data/raw": _regular_files(root / "data" / "raw"),
        "data/processed": _regular_files(root / "data" / "processed"),
        "artifacts": _regular_files(root / "artifacts"),
        "phase1_claim_documents": _claim_document_paths(root),
        "config": _regular_files(root / "config"),
        "phase1_code_tests_pyproject": _phase1_code_files(root),
        "players_2026.py": [root / "fantasy_football" / "players_2026.py"],
    }
    for scope, paths in scopes.items():
        missing = [path for path in paths if not path.is_file()]
        if missing:
            display = ", ".join(str(path) for path in missing)
            raise Phase2SafetyError(
                f"Protected scope {scope!r} is incomplete: {display}"
            )
    return scopes


def _file_record(root: Path, path: Path) -> _FileRecord:
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as error:
        raise Phase2SafetyError(
            f"Protected file is outside the repository: {path}"
        ) from error

    before = path.stat()
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size_bytes += len(block)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or size_bytes != after.st_size
    ):
        raise Phase2SafetyError(f"File changed while it was being hashed: {path}")
    return _FileRecord(relative_path, size_bytes, digest.digest())


def _encoded_record(record: _FileRecord) -> bytes:
    return (
        record.relative_path.encode("utf-8")
        + b"\0"
        + str(record.size_bytes).encode("ascii")
        + b"\0"
        + record.raw_sha256
        + b"\n"
    )


def _digest_records(records: Iterable[_FileRecord]) -> TreeDigest:
    ordered = sorted(
        records,
        key=lambda record: (record.relative_path.casefold(), record.relative_path),
    )
    digest = hashlib.sha256()
    for record in ordered:
        digest.update(_encoded_record(record))
    return TreeDigest(
        files=len(ordered),
        bytes=sum(record.size_bytes for record in ordered),
        sha256=digest.hexdigest(),
    )


def _compute_protected_snapshot(
    root: Path,
) -> tuple[dict[str, TreeDigest], str, dict[str, _FileRecord]]:
    scopes = _protected_scope_files(root)
    records_by_path: dict[str, _FileRecord] = {}
    records_by_scope: dict[str, list[_FileRecord]] = {}
    for scope in PROTECTED_SCOPE_NAMES:
        scope_records: list[_FileRecord] = []
        for path in scopes[scope]:
            record = _file_record(root, path)
            if record.relative_path in records_by_path:
                raise Phase2SafetyError(
                    f"Protected file appears in multiple scopes: {record.relative_path}"
                )
            records_by_path[record.relative_path] = record
            scope_records.append(record)
        records_by_scope[scope] = scope_records
    digests = {
        scope: _digest_records(records_by_scope[scope])
        for scope in PROTECTED_SCOPE_NAMES
    }
    freeze_id = _digest_records(records_by_path.values()).sha256
    return digests, freeze_id, records_by_path


def load_freeze_manifest(root: Path) -> dict[str, Any]:
    """Load and structurally validate the Phase 1 freeze manifest."""
    repository = _repository_root(root)
    current_path = repository / "experiments" / "phase1" / "phase1_freeze_manifest.json"
    legacy_path = repository / "experiments" / "phase2" / "phase1_freeze_manifest.json"
    path = current_path if current_path.is_file() else legacy_path
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase2SafetyError(
            f"Cannot load Phase 1 freeze manifest: {path}"
        ) from error
    if not isinstance(manifest, dict):
        raise Phase2SafetyError("Phase 1 freeze manifest must contain a JSON object.")
    if manifest.get("digest_contract") != TREE_DIGEST_CONTRACT:
        raise Phase2SafetyError(
            "Phase 1 freeze manifest has an unknown digest contract."
        )
    freeze_id = manifest.get("freeze_id")
    if not isinstance(freeze_id, str) or _HASH.fullmatch(freeze_id) is None:
        raise Phase2SafetyError("Phase 1 freeze manifest has an invalid freeze_id.")
    trees = manifest.get("trees")
    if not isinstance(trees, dict) or set(trees) != set(PROTECTED_SCOPE_NAMES):
        raise Phase2SafetyError("Phase 1 freeze manifest must define all seven scopes.")
    critical_files = manifest.get("critical_files")
    if not isinstance(critical_files, dict):
        raise Phase2SafetyError("Phase 1 freeze manifest has invalid critical_files.")
    return manifest


def compute_protected_digests(root: Path) -> dict[str, TreeDigest]:
    """Compute the seven Phase 1 tree digests without accepting changes."""
    repository = _repository_root(root)
    digests, _, _ = _compute_protected_snapshot(repository)
    return digests


def _expected_tree_digest(scope: str, value: Any) -> TreeDigest:
    if not isinstance(value, Mapping):
        raise Phase2SafetyError(f"Freeze manifest tree {scope!r} must be an object.")
    files = value.get("files")
    size_bytes = value.get("bytes")
    sha256 = value.get("sha256")
    if (
        not isinstance(files, int)
        or isinstance(files, bool)
        or files < 0
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
        or not isinstance(sha256, str)
        or _HASH.fullmatch(sha256) is None
    ):
        raise Phase2SafetyError(f"Freeze manifest tree {scope!r} is malformed.")
    return TreeDigest(files, size_bytes, sha256)


def verify_phase1_freeze(
    root: Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, TreeDigest]:
    """Verify every protected tree, critical file, and the aggregate freeze ID."""
    repository = _repository_root(root)
    loaded = load_freeze_manifest(repository) if manifest is None else dict(manifest)
    if loaded.get("digest_contract") != TREE_DIGEST_CONTRACT:
        raise Phase2SafetyError(
            "Phase 1 freeze manifest has an unknown digest contract."
        )
    expected_trees = loaded.get("trees")
    if not isinstance(expected_trees, Mapping) or set(expected_trees) != set(
        PROTECTED_SCOPE_NAMES
    ):
        raise Phase2SafetyError("Phase 1 freeze manifest must define all seven scopes.")

    actual, freeze_id, records = _compute_protected_snapshot(repository)
    for scope in PROTECTED_SCOPE_NAMES:
        expected = _expected_tree_digest(scope, expected_trees[scope])
        if actual[scope] != expected:
            raise Phase2SafetyError(
                f"Protected Phase 1 scope changed: {scope}; "
                f"expected {expected}, found {actual[scope]}."
            )
    expected_files = loaded.get("protected_files")
    expected_bytes = loaded.get("protected_bytes")
    if expected_files != len(records):
        raise Phase2SafetyError(
            f"Protected file count changed: expected {expected_files}, found {len(records)}."
        )
    actual_bytes = sum(record.size_bytes for record in records.values())
    if expected_bytes != actual_bytes:
        raise Phase2SafetyError(
            f"Protected byte count changed: expected {expected_bytes}, found {actual_bytes}."
        )
    if loaded.get("freeze_id") != freeze_id:
        raise Phase2SafetyError(
            f"Aggregate Phase 1 freeze ID changed: expected {loaded.get('freeze_id')}, "
            f"found {freeze_id}."
        )

    critical_files = loaded.get("critical_files")
    if not isinstance(critical_files, Mapping):
        raise Phase2SafetyError("Phase 1 freeze manifest has invalid critical_files.")
    for relative_path, expected_hash in critical_files.items():
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise Phase2SafetyError("Phase 1 critical-file records must be strings.")
        record = records.get(relative_path)
        if record is None:
            raise Phase2SafetyError(
                f"Critical Phase 1 file is outside protected scopes: {relative_path}"
            )
        actual_hash = record.raw_sha256.hex()
        if _HASH.fullmatch(expected_hash) is None or actual_hash != expected_hash:
            raise Phase2SafetyError(
                f"Critical Phase 1 file changed: {relative_path}; "
                f"expected {expected_hash}, found {actual_hash}."
            )
    return actual


def _runs_root(root: Path, *, create: bool) -> Path:
    repository = _repository_root(root)
    phase2_root = repository / "experiments" / "phase2"
    runs_root = phase2_root / "runs"
    if create:
        runs_root.mkdir(parents=True, exist_ok=True)
    if not phase2_root.is_dir() or not runs_root.is_dir():
        raise Phase2SafetyError(f"Phase 2 runs directory is unavailable: {runs_root}")
    if phase2_root.resolve() != phase2_root or runs_root.resolve() != runs_root:
        raise Phase2SafetyError(
            "Phase 2 output directories must not use symbolic links."
        )
    return runs_root


def create_run_directory(root: Path, label: str) -> Path:
    """Create one exclusive, collision-resistant directory below Phase 2 runs."""
    if _RUN_LABEL.fullmatch(label) is None:
        raise ValueError(
            "label must start with an alphanumeric character and contain at most "
            "64 letters, digits, dots, underscores, or hyphens."
        )
    runs_root = _runs_root(root, create=True)
    for _ in range(10):
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_name = f"{timestamp}-{label}-{secrets.token_hex(8)}"
        run_dir = runs_root / run_name
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        if run_dir.resolve().parent != runs_root:
            raise Phase2SafetyError(f"Unsafe Phase 2 run directory: {run_dir}")
        return run_dir
    raise Phase2SafetyError("Could not allocate a unique Phase 2 run directory.")


def _repository_file(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise Phase2SafetyError(f"Input file does not exist: {candidate}") from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise Phase2SafetyError(f"Input must be a repository file: {candidate}")
    return resolved


def hash_inputs(root: Path, paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash declared repository inputs by stable repository-relative path."""
    repository = _repository_root(root)
    records: dict[str, str] = {}
    for value in paths:
        path = _repository_file(repository, value)
        record = _file_record(repository, path)
        if record.relative_path in records:
            raise Phase2SafetyError(f"Input was declared twice: {record.relative_path}")
        records[record.relative_path] = record.raw_sha256.hex()
    return dict(sorted(records.items(), key=lambda item: (item[0].casefold(), item[0])))


def _validated_run_directory(root: Path, run_dir: Path) -> Path:
    runs_root = _runs_root(root, create=False)
    if run_dir.is_symlink():
        raise Phase2SafetyError("Phase 2 run directories must not be symbolic links.")
    try:
        resolved = run_dir.resolve(strict=True)
    except OSError as error:
        raise Phase2SafetyError(
            f"Phase 2 run directory does not exist: {run_dir}"
        ) from error
    if not resolved.is_dir() or resolved.parent != runs_root or resolved.is_symlink():
        raise Phase2SafetyError(
            f"Output directory must be a direct child of {runs_root}: {resolved}"
        )
    return resolved


def hash_outputs(run_dir: Path, *, root: Path | None = None) -> dict[str, str]:
    """Hash every regular output in one isolated Phase 2 run directory."""
    if root is None:
        try:
            inferred_root = run_dir.resolve(strict=True).parents[3]
        except (IndexError, OSError) as error:
            raise Phase2SafetyError(
                f"Cannot infer repository root from {run_dir}"
            ) from error
        repository = _repository_root(inferred_root)
    else:
        repository = _repository_root(root)
    resolved = _validated_run_directory(repository, run_dir)
    entries = list(resolved.rglob("*"))
    symbolic_links = [path for path in entries if path.is_symlink()]
    if symbolic_links:
        display = ", ".join(str(path) for path in symbolic_links)
        raise Phase2SafetyError(
            f"Phase 2 outputs must not contain symbolic links: {display}"
        )
    outputs: dict[str, str] = {}
    for path in entries:
        if not path.is_file():
            continue
        record = _file_record(resolved, path)
        outputs[record.relative_path] = record.raw_sha256.hex()
    return dict(sorted(outputs.items(), key=lambda item: (item[0].casefold(), item[0])))


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def combined_project_bytes(root: Path) -> int:
    """Count Phase 1 data, its catalog, and every stored Phase 2 file."""
    repository = _repository_root(root)
    counted_paths = (
        repository / "data",
        repository / "artifacts",
        repository / "fantasy_football" / "players_2026.py",
        repository / "experiments" / "phase2",
    )
    return sum(_tree_bytes(path) for path in counted_paths)


def enforce_size_limit(
    root: Path,
    limit_bytes: int = MAX_COMBINED_BYTES,
) -> int:
    """Fail when combined Phase 1 and Phase 2 storage exceeds the declared cap."""
    if (
        not isinstance(limit_bytes, int)
        or isinstance(limit_bytes, bool)
        or limit_bytes <= 0
    ):
        raise ValueError("limit_bytes must be a positive integer.")
    total_bytes = combined_project_bytes(root)
    if total_bytes > limit_bytes:
        raise Phase2SafetyError(
            f"Combined Phase 1 and Phase 2 storage uses {total_bytes:,} bytes, "
            f"above the {limit_bytes:,}-byte limit."
        )
    return total_bytes


def _finalize_run(
    run: Phase2Run,
    manifest: Mapping[str, Any],
    input_paths: tuple[str | Path, ...],
    limit_bytes: int,
) -> None:
    errors: list[Exception] = []
    try:
        run.output_hashes = hash_outputs(run.run_dir, root=run.root)
    except Exception as error:  # noqa: BLE001 - aggregate all final safety failures
        errors.append(error)
    try:
        final_inputs = hash_inputs(run.root, input_paths)
        if final_inputs != run.input_hashes:
            errors.append(
                Phase2SafetyError("A declared Phase 2 input changed during the run.")
            )
    except Exception as error:  # noqa: BLE001 - aggregate all final safety failures
        errors.append(error)
    try:
        run.combined_bytes = enforce_size_limit(run.root, limit_bytes)
    except Exception as error:  # noqa: BLE001 - aggregate all final safety failures
        errors.append(error)
    try:
        run.protected_after = verify_phase1_freeze(run.root, manifest)
        if run.protected_after != run.protected_before:
            errors.append(
                Phase2SafetyError("Protected Phase 1 digests changed during the run.")
            )
    except Exception as error:  # noqa: BLE001 - aggregate all final safety failures
        errors.append(error)
    if errors:
        details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        raise Phase2SafetyError(
            f"Phase 2 final safety checks failed: {details}"
        ) from errors[0]


@contextmanager
def isolated_phase2_run(
    root: Path,
    label: str,
    input_paths: Iterable[str | Path] = (),
    *,
    limit_bytes: int = MAX_COMBINED_BYTES,
) -> Iterator[Phase2Run]:
    """Guard one run with exclusive output, hashes, size checks, and freeze checks."""
    repository = _repository_root(root)
    manifest = load_freeze_manifest(repository)
    protected_before = verify_phase1_freeze(repository, manifest)
    enforce_size_limit(repository, limit_bytes)
    declared_inputs = tuple(input_paths)
    input_hashes = hash_inputs(repository, declared_inputs)
    run_dir = create_run_directory(repository, label)
    run = Phase2Run(
        root=repository,
        run_dir=run_dir,
        freeze_id=str(manifest["freeze_id"]),
        input_hashes=input_hashes,
        protected_before=protected_before,
    )
    try:
        yield run
    finally:
        _finalize_run(run, manifest, declared_inputs, limit_bytes)
