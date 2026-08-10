"""Isolation-contract tests for Phase 2 experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fantasy_football.phase2_safety as safety
from fantasy_football.phase2_safety import (
    Phase2SafetyError,
    combined_project_bytes,
    create_run_directory,
    enforce_size_limit,
    hash_outputs,
    isolated_phase2_run,
    load_freeze_manifest,
    verify_phase1_freeze,
)


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _freeze_repository(root: Path) -> None:
    files = {
        "data/raw/source.bin": b"raw-source",
        "data/processed/modeling_table.parquet": b"processed",
        "artifacts/result_summary.json": "{}\n",
        "config/scoring_espn_full_ppr_2026.json": "{}\n",
        "README.md": "readme\n",
        "EXPERIMENT_LEDGER.md": "ledger\n",
        "COMPLETION_AUDIT.md": "audit\n",
        "NOTICE.md": "notice\n",
        "docs/data_sources.md": "sources\n",
        "docs/fantasy_football.md": "fantasy\n",
        "docs/methods_and_results.md": "methods\n",
        "fantasy_football/core.py": "VALUE = 1\n",
        "fantasy_football/players_2026.py": "PLAYERS = ()\n",
        "tests/test_core.py": "def test_core():\n    assert True\n",
        "pyproject.toml": "[project]\nname = 'fixture'\nversion = '0.0.0'\n",
    }
    for relative_path, content in files.items():
        _write(root / relative_path, content)
    (root / "experiments" / "phase2").mkdir(parents=True)

    digests, freeze_id, records = safety._compute_protected_snapshot(root.resolve())
    critical_paths = (
        "data/raw/source.bin",
        "data/processed/modeling_table.parquet",
        "config/scoring_espn_full_ppr_2026.json",
        "fantasy_football/players_2026.py",
    )
    manifest = {
        "created_at_utc": "2026-08-10T12:00:00Z",
        "freeze_id": freeze_id,
        "digest_contract": safety.TREE_DIGEST_CONTRACT,
        "protected_files": len(records),
        "protected_bytes": sum(record.size_bytes for record in records.values()),
        "trees": {
            scope: {
                "files": digest.files,
                "bytes": digest.bytes,
                "sha256": digest.sha256,
            }
            for scope, digest in digests.items()
        },
        "critical_files": {
            relative_path: records[relative_path].raw_sha256.hex()
            for relative_path in critical_paths
        },
    }
    _write(
        root / "experiments" / "phase2" / "phase1_freeze_manifest.json",
        json.dumps(manifest, indent=2) + "\n",
    )


def test_repository_freeze_manifest_verifies_with_phase2_files_excluded() -> None:
    manifest = load_freeze_manifest(ROOT)
    digests = verify_phase1_freeze(ROOT, manifest)

    assert tuple(digests) == safety.PROTECTED_SCOPE_NAMES
    assert digests["phase1_code_tests_pyproject"].files == 19
    assert digests["phase1_code_tests_pyproject"].sha256 == manifest["trees"][
        "phase1_code_tests_pyproject"
    ]["sha256"]


def test_future_phase2_code_and_tests_do_not_change_phase1_digest(tmp_path: Path) -> None:
    _freeze_repository(tmp_path)
    before = verify_phase1_freeze(tmp_path)

    _write(tmp_path / "fantasy_football" / "phase2_candidate.py", "VALUE = 2\n")
    _write(
        tmp_path / "fantasy_football" / "phase2" / "nested.py",
        "VALUE = 3\n",
    )
    _write(
        tmp_path / "tests" / "test_phase2_candidate.py",
        "def test_candidate():\n    assert True\n",
    )

    assert verify_phase1_freeze(tmp_path) == before


def test_isolated_run_hashes_inputs_outputs_and_preserves_phase1(
    tmp_path: Path,
) -> None:
    _freeze_repository(tmp_path)
    plan_path = tmp_path / "experiments" / "phase2" / "plan.json"
    _write(plan_path, '{"candidate": "ridge"}\n')
    protected_artifact = tmp_path / "artifacts" / "result_summary.json"
    artifact_before = protected_artifact.read_bytes()

    with isolated_phase2_run(
        tmp_path,
        "ridge-smoke",
        input_paths=("experiments/phase2/plan.json",),
    ) as run:
        assert run.run_dir.parent == tmp_path / "experiments" / "phase2" / "runs"
        _write(run.run_dir / "metrics.csv", "spearman\n0.75\n")
        _write(run.run_dir / "logs" / "ledger.md", "# Run ledger\n")

    expected_input_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert run.input_hashes == {"experiments/phase2/plan.json": expected_input_hash}
    assert set(run.output_hashes) == {"logs/ledger.md", "metrics.csv"}
    assert hash_outputs(run.run_dir) == run.output_hashes
    assert run.protected_after == run.protected_before
    assert run.combined_bytes == combined_project_bytes(tmp_path)
    assert protected_artifact.read_bytes() == artifact_before


def test_run_directory_creation_is_contained_and_collision_resistant(
    tmp_path: Path,
) -> None:
    _freeze_repository(tmp_path)

    first = create_run_directory(tmp_path, "candidate")
    second = create_run_directory(tmp_path, "candidate")

    assert first != second
    assert first.parent == tmp_path / "experiments" / "phase2" / "runs"
    assert second.parent == first.parent
    with pytest.raises(ValueError, match="label must start"):
        create_run_directory(tmp_path, "../escape")
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(Phase2SafetyError, match="direct child"):
        hash_outputs(outside, root=tmp_path)


def test_isolated_run_detects_protected_and_input_mutation(tmp_path: Path) -> None:
    _freeze_repository(tmp_path)
    plan_path = tmp_path / "experiments" / "phase2" / "plan.json"
    _write(plan_path, "original\n")

    with pytest.raises(Phase2SafetyError, match="final safety checks failed"):
        with isolated_phase2_run(
            tmp_path,
            "mutation-sentinel",
            input_paths=(plan_path,),
        ):
            _write(tmp_path / "artifacts" / "result_summary.json", "changed\n")
            _write(plan_path, "changed\n")


def test_combined_size_guard_counts_phase2_outputs(tmp_path: Path) -> None:
    _freeze_repository(tmp_path)
    initial_bytes = combined_project_bytes(tmp_path)

    assert enforce_size_limit(tmp_path, initial_bytes) == initial_bytes
    _write(tmp_path / "experiments" / "phase2" / "artifacts" / "extra.bin", b"1234")

    assert combined_project_bytes(tmp_path) == initial_bytes + 4
    with pytest.raises(Phase2SafetyError, match="above"):
        enforce_size_limit(tmp_path, initial_bytes)


def test_manifest_tree_corruption_is_rejected(tmp_path: Path) -> None:
    _freeze_repository(tmp_path)
    manifest = load_freeze_manifest(tmp_path)
    manifest["trees"]["artifacts"]["sha256"] = "0" * 64

    with pytest.raises(Phase2SafetyError, match="Protected Phase 1 scope changed"):
        verify_phase1_freeze(tmp_path, manifest)
