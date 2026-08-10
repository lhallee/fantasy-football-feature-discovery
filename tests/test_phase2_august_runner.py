"""Contract tests for the fixed August 9 production runner."""

import json
import numpy as np
import pandas as pd

from pathlib import Path

from fantasy_football.phase2_august_runner import _audit_gate_report
from fantasy_football.phase2_august_runner import _fixed_prediction_null
from fantasy_football.phase2_august_runner import _permuted_target_table
from fantasy_football.phase2_august_runner import _select_candidate, code_hashes
from fantasy_football.phase2_august_runner import load_august_plan, safe_feature_frame
from fantasy_football.phase2_august_runner import verify_prefit_manifest
from fantasy_football.phase2_features import FeatureLineage, Phase2FeatureBundle
from fantasy_football.phase2_modeling import Phase2Candidate


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "experiments" / "phase2" / "production" / "august9_plan_v1.json"


def _lineage(
    column: str,
    source: str,
    offset: int,
) -> FeatureLineage:
    return FeatureLineage(column, (source,), (offset,), "synthetic contract fixture")


def _feature_bundle() -> Phase2FeatureBundle:
    keys = pd.DataFrame(
        {
            "target_season": [2024, 2024],
            "player_id": ["a", "b"],
            "model_position": ["WR", "K"],
        }
    )  # (2, 3)
    frame = pd.DataFrame(
        {
            "age": [25.0, 26.0],
            "missing_age": [0.0, 0.0],
            "team_changed": [0.0, 0.0],
            "lag1_history_available": [1.0, 1.0],
            "lag1_passing_yards": [100.0, 0.0],
            "lag1_fg_made": [0.0, 30.0],
            "lag1_late6_passing_yards": [40.0, 0.0],
            "room_competitor_count": [3.0, 1.0],
            "team_lag1_passing_yards": [4000.0, 4000.0],
        }
    )  # (2, 9)
    lineage = {
        "age": _lineage("age", "modeling_table", 0),
        "missing_age": _lineage("missing_age", "modeling_table", 0),
        "team_changed": _lineage("team_changed", "modeling_table", 0),
        "lag1_history_available": _lineage(
            "lag1_history_available", "player_seasons", 1
        ),
        "lag1_passing_yards": _lineage("lag1_passing_yards", "weekly_stats", 1),
        "lag1_fg_made": _lineage("lag1_fg_made", "weekly_stats", 1),
        "lag1_late6_passing_yards": _lineage(
            "lag1_late6_passing_yards", "weekly_stats", 1
        ),
        "room_competitor_count": _lineage(
            "room_competitor_count", "modeling_table.target_roster", 0
        ),
        "team_lag1_passing_yards": _lineage(
            "team_lag1_passing_yards", "weekly_stats", 1
        ),
    }
    return Phase2FeatureBundle(keys, frame, lineage, "stable")


def _candidate(identifier: str, recipes: tuple[str, ...]) -> Phase2Candidate:
    return Phase2Candidate(
        identifier=identifier,
        tier="stable",
        recipes=recipes,
        estimator="extra_trees",
        feature_limit=4,
    )


def _prediction_frame(candidate: str, reverse: bool = False) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for season in (2022, 2023):
        for position in ("QB", "RB", "WR", "TE"):
            for index in range(8):
                actual = float(index)
                predicted = float(7 - index if reverse else index)
                records.append(
                    {
                        "player_id": f"{season}-{position}-{index}",
                        "target_season": season,
                        "model_position": position,
                        "target_points": actual,
                        "candidate": candidate,
                        "predicted": predicted,
                        "fold_fit_seconds": 0.0,
                    }
                )
    return pd.DataFrame(records)  # (64, 7)


def test_plan_locks_august_origin_splits_and_candidate_cohorts() -> None:
    plan = load_august_plan(PLAN_PATH)

    assert plan["forecast_origin"] == "2026-08-09"
    assert max(plan["discovery_seasons"]) < min(plan["audit_seasons"])
    assert {candidate["cohort"] for candidate in plan["candidates"]} == {
        "offense",
        "kicker",
    }
    assert plan["candidate_universe"][
        "target_season_roster_membership_used_historically"
    ] is False


def test_safe_feature_pool_excludes_target_context_and_position_mismatches() -> None:
    bundle = _feature_bundle()
    offense = safe_feature_frame(
        bundle,
        _candidate("offense", ("core", "trajectory")),
        "offense",
    )  # (2, 5)
    kicker = safe_feature_frame(
        bundle,
        _candidate("kicker", ("core", "trajectory")),
        "kicker",
    )  # (2, 4)

    assert set(offense) == {
        "age",
        "missing_age",
        "lag1_history_available",
        "lag1_passing_yards",
        "lag1_late6_passing_yards",
    }
    assert set(kicker) == {
        "age",
        "missing_age",
        "lag1_history_available",
        "lag1_fg_made",
    }
    assert not any(column.startswith(("room_", "team_")) for column in offense)


def test_grouped_target_permutation_preserves_each_group_multiset() -> None:
    predictions = _prediction_frame("model")
    shuffled = _permuted_target_table(predictions, 17)  # (64, 7)

    for key, group in predictions.groupby(
        ["target_season", "model_position"], observed=True
    ):
        shuffled_group = shuffled.loc[
            (shuffled["target_season"] == key[0])
            & (shuffled["model_position"] == key[1])
        ]  # (8, 7)
        assert sorted(group["target_points"]) == sorted(shuffled_group["target_points"])
    assert not np.array_equal(
        shuffled["target_points"].to_numpy(),
        predictions["target_points"].to_numpy(),
    )


def test_fixed_prediction_null_scores_permuted_labels_not_real_labels() -> None:
    predictions = _prediction_frame("model")
    diagnostic = _fixed_prediction_null(predictions, repeats=199, seed=41)

    assert abs(diagnostic["mean_spearman"]) < 0.03
    assert diagnostic["lower_95"] < 0.0 < diagnostic["upper_95"]


def test_selection_uses_compactness_only_inside_declared_tolerance() -> None:
    plan = load_august_plan(PLAN_PATH)
    summary = pd.DataFrame(
        {
            "candidate": ["large", "compact", "too_far", "baseline"],
            "cohort": ["offense"] * 4,
            "kind": ["model", "model", "model", "benchmark"],
            "spearman": [0.750, 0.746, 0.740, 0.80],
            "median_selected_feature_count": [128, 32, 16, 1],
            "mae": [30.0, 31.0, 29.0, 25.0],
            "feature_pool_count": [200, 200, 200, 1],
        }
    )  # (4, 7)

    selected = _select_candidate(plan, "offense", summary)

    assert selected["identifier"] == "compact"


def test_promotion_gate_requires_performance_and_both_null_controls() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    selected = {
        "offense": _prediction_frame("offense"),
        "kicker": _prediction_frame("kicker").assign(model_position="K"),
    }
    baseline = {
        "offense": _prediction_frame("offense-baseline", reverse=True),
        "kicker": _prediction_frame("kicker-baseline").assign(model_position="K"),
    }
    centered = {
        cohort: {"mean_spearman": 0.0, "lower_95": -0.1, "upper_95": 0.1}
        for cohort in ("offense", "kicker")
    }

    report = _audit_gate_report(plan, selected, baseline, centered, centered)

    assert report["passed"] is True
    assert all(report["checks"].values())


def test_prefit_code_hash_scope_includes_runner_and_contract_tests() -> None:
    hashes = code_hashes(ROOT)

    assert "fantasy_football/phase2_august_runner.py" in hashes
    assert "tests/test_phase2_august_runner.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_prefit_manifest_binds_plan_code_tests_and_scientific_inputs() -> None:
    plan = load_august_plan(PLAN_PATH)

    manifest = verify_prefit_manifest(ROOT, PLAN_PATH, plan)

    assert manifest["manifest_version"] == 1
    assert manifest["focused_verification"]["failed"] == 0
