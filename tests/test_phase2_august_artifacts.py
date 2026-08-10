"""Release checks for the promoted August 9 Phase 2 production generation."""

from __future__ import annotations

import json
import pandas as pd
import pytest

from pathlib import Path

from fantasy_football.metrics import macro_rank_metric
from fantasy_football.phase2_safety import combined_project_bytes, hash_outputs
from fantasy_football.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "experiments" / "phase2" / "production"
RELEASE_MANIFEST = PRODUCTION / "experiment_manifest_v1.json"


def _manifest() -> dict[str, object]:
    """Load the Phase 2 production release manifest."""
    return json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))


def test_production_release_hash_chain_is_complete() -> None:
    manifest = _manifest()

    assert manifest["production_generation"] == "phase2_august9_v1"
    assert manifest["status"] == "production"
    for record in (manifest["plan"], manifest["prefit_lock"]):
        assert file_sha256(ROOT / record["path"]) == record["sha256"]
    snapshot = manifest["production_snapshot"]
    assert file_sha256(ROOT / snapshot["path"]) == snapshot["sha256"]
    snapshot_manifest = json.loads(
        (ROOT / snapshot["path"]).read_text(encoding="utf-8")
    )
    assert snapshot_manifest["freeze_id"] == snapshot["freeze_id"]
    for stage, record in manifest["runs"].items():
        run_dir = ROOT / record["directory"]
        assert file_sha256(run_dir / "run_manifest.json") == record[
            "run_manifest_sha256"
        ]
        assert file_sha256(run_dir / "output_hashes.json") == record[
            "output_hashes_sha256"
        ]
        declared = json.loads(
            (run_dir / "output_hashes.json").read_text(encoding="utf-8")
        )
        actual = hash_outputs(run_dir, root=ROOT)
        actual.pop("output_hashes.json", None)
        assert actual == declared
        if stage == "discovery":
            assert file_sha256(run_dir / "selection.json") == record[
                "selection_sha256"
            ]
        if stage == "audit":
            assert file_sha256(run_dir / "audit_report.json") == record[
                "audit_report_sha256"
            ]
    for relative_path, expected_hash in manifest["canonical_artifacts"].items():
        assert file_sha256(ROOT / relative_path) == expected_hash
    for relative_path, expected_hash in manifest["documents"].items():
        assert file_sha256(ROOT / relative_path) == expected_hash
    for relative_path, expected_hash in manifest["deliverables"].items():
        assert file_sha256(ROOT / relative_path) == expected_hash
    for relative_path, expected_hash in manifest["release_code"].items():
        assert file_sha256(ROOT / relative_path) == expected_hash


def test_production_metrics_recompute_and_null_controls_pass() -> None:
    summary = json.loads(
        (ROOT / "artifacts" / "result_summary.json").read_text(encoding="utf-8")
    )
    audit = pd.read_parquet(  # (n_audit_players, c_audit)
        ROOT / "artifacts" / "audit_predictions.parquet"
    )
    offense_mask = audit["model_position"].ne("K")  # (n_audit_players,)
    offense = audit.loc[offense_mask].copy()  # (n_offense_audit, c_audit)
    kicker = audit.loc[~offense_mask].copy()  # (n_kicker_audit, c_audit)
    gates = summary["promotion_gates"]

    assert macro_rank_metric(
        offense["actual_points"],
        offense["predicted_points"],
        offense["model_position"],
        offense["target_season"],
    ) == pytest.approx(gates["offense"]["selected"]["spearman"])
    assert macro_rank_metric(
        kicker["actual_points"],
        kicker["predicted_points"],
        kicker["model_position"],
        kicker["target_season"],
    ) == pytest.approx(gates["kicker"]["selected"]["spearman"])
    assert gates["passed"] is True
    assert all(gates["checks"].values())
    assert abs(gates["fixed_prediction_null"]["offense"]["mean_spearman"]) <= 0.015
    assert abs(gates["end_to_end_null"]["offense"]["mean_spearman"]) <= 0.05
    assert abs(gates["end_to_end_null"]["kicker"]["mean_spearman"]) <= 0.08


def test_production_forecast_has_complete_roster_parity_and_safe_lineage() -> None:
    manifest = json.loads(
        (ROOT / "artifacts" / "model_manifest.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_parquet(  # (n_fantasy_players, c_prediction)
        ROOT / "artifacts" / "predictions_2026.parquet"
    )
    roster = pd.read_parquet(  # (n_roster_players, c_roster)
        ROOT / "data" / "processed" / "preseason_players.parquet"
    )
    current_mask = roster["season"].eq(2026)  # (n_roster_players,)
    fantasy_mask = roster["position"].isin(  # (n_roster_players,)
        {"QB", "RB", "FB", "HB", "WR", "TE", "K", "PK"}
    )
    expected_ids = set(roster.loc[current_mask & fantasy_mask, "player_id"])

    assert len(predictions) == 958
    assert predictions["player_id"].is_unique
    assert set(predictions["player_id"]) == expected_ids
    assert predictions["predicted_points"].notna().all()
    assert manifest["retrospective_outcomes_blinded"] is False
    assert manifest["prospective_2026_outcomes_observed"] is False

    discovery_dir = ROOT / "experiments" / "phase2" / "runs" / (
        "20260810T174927548207Z-august9-discovery-"
        "3b07b79602c5-b96dfefdf4d91c71"
    )
    lineage = json.loads(
        (discovery_dir / "feature_lineage.json").read_text(encoding="utf-8")
    )["stable"]
    all_columns = (
        *manifest["feature_columns"]["offense"],
        *manifest["feature_columns"]["kicker"],
    )
    for column in all_columns:
        record = lineage[column]
        for source, offset in zip(
            record["sources"], record["source_offsets"], strict=True
        ):
            if source in {"weekly_stats", "player_seasons"}:
                assert offset >= 1
    lowered = " ".join(all_columns).lower()
    for forbidden in (
        "target_points",
        "fantasy",
        "status",
        "depth",
        "room_",
        "team_context",
    ):
        assert forbidden not in lowered


def test_current_release_size_record_is_exact_and_below_limit() -> None:
    manifest = _manifest()
    storage = manifest["storage"]

    assert storage["combined_project_bytes"] == combined_project_bytes(ROOT)
    assert storage["combined_project_bytes"] < storage["limit_bytes"]
