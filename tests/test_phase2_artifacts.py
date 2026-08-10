"""Validate the frozen Phase 2 experiment outputs."""

import json

from pathlib import Path

import pandas as pd

from fantasy_football.phase2_safety import (
    combined_project_bytes,
    hash_outputs,
    verify_phase1_freeze,
)
from fantasy_football.phase2_layout import verify_historical_file
from fantasy_football.provenance import file_sha256


ROOT = Path(__file__).parents[1]
PHASE2 = ROOT / "experiments" / "phase2"
EXPERIMENT_MANIFEST = json.loads(
    (PHASE2 / "experiment_manifest_v1.json").read_text(encoding="utf-8")
)


def _run_directory(stage: str) -> Path:
    relative = EXPERIMENT_MANIFEST["runs"][stage]["directory"]
    return ROOT / relative


def test_phase1_freeze_still_matches_phase2_manifest() -> None:
    protected = verify_phase1_freeze(ROOT)

    assert len(protected) == 7
    assert EXPERIMENT_MANIFEST["phase1_freeze_id"] == (
        "366b32e659f67cbbcb81a68ee20a66bbe79cdefc004db5c971c23e89b35f17ae"
    )
    assert EXPERIMENT_MANIFEST["combined_phase1_phase2_bytes_after_runs"] == (
        combined_project_bytes(ROOT)
    )


def test_phase2_plan_and_code_hashes_match_recorded_run() -> None:
    forecast_origin = EXPERIMENT_MANIFEST["forecast_origin"]

    assert forecast_origin["intended_definition"] == "post_final_cuts_pre_week_1"
    assert forecast_origin["historical_snapshot_timestamp_verified"] is False
    assert forecast_origin["august_9_catalog_compatible"] is False
    assert EXPERIMENT_MANIFEST["validity"]["status"] == "superseded"
    assert (
        file_sha256(ROOT / EXPERIMENT_MANIFEST["plan"]["path"])
        == (EXPERIMENT_MANIFEST["plan"]["sha256"])
    )
    for relative, expected in EXPERIMENT_MANIFEST["code"].items():
        verify_historical_file(ROOT, relative, expected)


def test_phase2_run_output_hash_manifests_match() -> None:
    for stage in ("discovery", "retrospective"):
        run_dir = _run_directory(stage)
        declared = json.loads(
            (run_dir / "output_hashes.json").read_text(encoding="utf-8")
        )
        actual = hash_outputs(run_dir, root=ROOT)
        actual.pop("output_hashes.json")

        assert actual == declared


def test_discovery_selection_is_locked_into_retrospective_run() -> None:
    discovery_dir = _run_directory("discovery")
    retrospective_dir = _run_directory("retrospective")
    selection = json.loads(
        (discovery_dir / "selection.json").read_text(encoding="utf-8")
    )
    retrospective = json.loads(
        (retrospective_dir / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert selection["selected_candidate"] == "stable_core_room_et_all_points"
    assert retrospective["selected_candidate"] == selection["selected_candidate"]
    assert retrospective["plan_sha256"] == selection["plan_sha256"]
    assert retrospective["validation_seasons"] == [2022, 2023, 2024, 2025]


def test_phase2_result_fails_declared_point_nine_success_rule() -> None:
    discovery = pd.read_csv(_run_directory("discovery") / "candidate_summary.csv")
    retrospective = pd.read_csv(
        _run_directory("retrospective") / "candidate_summary.csv"
    )
    selected = retrospective.loc[
        retrospective["candidate"].eq("stable_core_room_et_all_points")
    ].iloc[0]
    negative_control = discovery.loc[
        discovery["candidate"].eq("negative_control_shuffle_recent_et_64")
    ].iloc[0]
    uncertainty = json.loads(
        (_run_directory("retrospective") / "uncertainty.json").read_text(
            encoding="utf-8"
        )
    )

    assert selected["spearman"] == 0.7661339665600965
    assert selected["spearman"] < 0.9
    assert uncertainty["player_cluster"]["lower_95"] < 0.85
    assert abs(negative_control["spearman"]) < 0.1
