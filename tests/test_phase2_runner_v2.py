"""Test the frozen-plan overlay used by the Phase 2 v2 runner."""

import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from types import SimpleNamespace

from fantasy_football import phase2_runner_v2 as v2
from fantasy_football.phase2_cohort import EXCLUDED_STATUS_CODES
from fantasy_football.phase2_runner_v2 import _load_amended_plan
from fantasy_football.phase2_runner_v2 import _position_metric_contract
from fantasy_football.phase2_runner_v2 import _repair_metric_scales
from fantasy_football.phase2_runner_v2 import _run_selected_diagnostic
from fantasy_football.phase2_runner_v2 import _shuffled_target_table
from fantasy_football.phase2_runner_v2 import _validate_selection_integrity
from fantasy_football.phase2_runner_v2 import _verify_prefit_manifest
from fantasy_football.provenance import file_sha256


ROOT = Path(__file__).parents[1]
PLAN = ROOT / "experiments" / "phase2" / "plan_v2.json"


def test_v2_plan_preserves_the_v1_candidate_and_split_menu() -> None:
    effective, amendment, base_path = _load_amended_plan(ROOT, PLAN)

    assert effective["plan_version"] == 2
    assert len(effective["candidates"]) == 27
    assert len(effective["blends"]) == 3
    assert effective["discovery_seasons"] == [2016, 2017, 2018, 2019, 2020, 2021]
    assert effective["retrospective_seasons"] == [2022, 2023, 2024, 2025]
    assert file_sha256(base_path) == amendment["base_plan_sha256"]


def test_v2_prefit_manifest_locks_the_executable_protocol() -> None:
    _, amendment, _ = _load_amended_plan(ROOT, PLAN)

    manifest_path, manifest = _verify_prefit_manifest(ROOT, PLAN, amendment)

    assert manifest_path == ROOT / amendment["prefit_manifest_path"]
    assert manifest["plan_sha256"] == file_sha256(PLAN)
    assert len(manifest["locked_files"]) == 15


def test_v2_plan_changes_only_the_declared_cohort_contract() -> None:
    effective, amendment, _ = _load_amended_plan(ROOT, PLAN)

    contract = effective["cohort_contract"]
    assert amendment["candidate_menu_change"] == "none"
    assert amendment["split_change"] == "none"
    assert amendment["selection_rule_change"] == "none"
    assert contract["apply_before_room_aggregation"] is True
    assert contract["status_application_grain"] == (
        "frozen_upstream_deduplicated_week_1_player_row"
    )
    assert (
        contract["upstream_deduplication_audit"]["deduplicated_to_excluded_status"] == 1
    )
    assert contract["status_is_a_cohort_definition_not_a_model_feature"] is True
    assert contract["historical_snapshot_timestamp_verified"] is False
    assert set(contract["excluded_statuses"]) == set(EXCLUDED_STATUS_CODES)
    assert amendment["metric_contract"]["signed_log_predictions"] == (
        "inverse_transform_before_point_metrics"
    )
    assert "end_to_end_null" in amendment["nonselecting_diagnostics"]
    assert "no_draft" in amendment["nonselecting_diagnostics"]


def test_end_to_end_null_shuffle_is_grouped_deterministic_and_nonmutating() -> None:
    table = pd.DataFrame(  # (16, 4)
        {
            "target_season": [2020] * 8 + [2021] * 8,
            "model_position": ["WR"] * 4 + ["RB"] * 4 + ["WR"] * 4 + ["RB"] * 4,
            "player_id": [f"P{index}" for index in range(16)],
            "target_points": np.arange(16, dtype="float64"),
        }
    )
    original = table.copy(deep=True)  # (16, 4)

    first = _shuffled_target_table(table)  # (16, 4)
    second = _shuffled_target_table(table)  # (16, 4)

    pd.testing.assert_frame_equal(table, original)
    pd.testing.assert_frame_equal(first, second)
    assert not first["target_points"].equals(table["target_points"])
    for keys, source_group in table.groupby(  # four (4, 4) frames
        ["target_season", "model_position"],
        observed=True,
    ):
        shuffled_group = first.loc[  # (4, 4)
            first["target_season"].eq(keys[0]) & first["model_position"].eq(keys[1])
        ]
        assert sorted(source_group["target_points"]) == sorted(
            shuffled_group["target_points"]
        )


def test_end_to_end_null_rejects_missing_targets_before_shuffling() -> None:
    table = pd.DataFrame(  # (2, 4)
        {
            "target_season": [2020, 2020],
            "model_position": ["WR", "WR"],
            "player_id": ["A", "B"],
            "target_points": [10.0, np.nan],
        }
    )

    with pytest.raises(ValueError, match="non-null target"):
        _shuffled_target_table(table)


def test_metric_contract_inverts_logs_and_omits_rank_point_errors() -> None:
    plan = {
        "candidates": [
            {
                "identifier": "log_model",
                "tier": "stable",
                "recipes": ["all"],
                "estimator": "extra_trees",
                "target": "signed_log_points",
            },
            {
                "identifier": "rank_model",
                "tier": "stable",
                "recipes": ["all"],
                "estimator": "extra_trees",
                "target": "rank_percentile",
            },
        ]
    }
    base = pd.DataFrame(  # (8, 7)
        {
            "player_id": [f"P{index}" for index in range(8)],
            "target_season": [2020] * 4 + [2021] * 4,
            "model_position": ["WR"] * 8,
            "target_points": [0.0, 10.0, 30.0, 60.0, 5.0, 20.0, 40.0, 80.0],
            "predicted": [0.0] * 8,
            "candidate": [""] * 8,
            "fold_fit_seconds": [0.1] * 8,
        }
    )
    log_predictions = base.copy()  # (8, 7)
    log_predictions["candidate"] = "log_model"  # (8, 7)
    log_predictions["predicted"] = np.log1p(  # (8, 7)
        log_predictions["target_points"]
    )
    rank_predictions = base.copy()  # (8, 7)
    rank_predictions["candidate"] = "rank_model"  # (8, 7)
    rank_predictions["predicted"] = np.tile(  # (8, 7)
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        2,
    )
    predictions = {
        "log_model": log_predictions,
        "rank_model": rank_predictions,
    }
    fold_frames: list[pd.DataFrame] = []
    summaries = [
        {"candidate": "log_model", "kind": "model"},
        {"candidate": "rank_model", "kind": "model"},
    ]

    validity = _repair_metric_scales(
        plan,
        predictions,
        fold_frames,
        summaries,
    )

    np.testing.assert_allclose(
        predictions["log_model"]["predicted"],
        base["target_points"],
    )
    rank_summary = next(
        record for record in summaries if record["candidate"] == "rank_model"
    )
    assert np.isnan(rank_summary["mae"])
    assert all(
        frame.loc[frame["candidate"].eq("rank_model"), "mae"].isna().all()
        for frame in fold_frames
    )
    assert validity["order_metrics_retained_for_all_candidates"] is True


def test_position_metric_contract_omits_only_scale_dependent_values() -> None:
    metrics = {
        "WR": {
            "spearman": 0.75,
            "kendall_tau_b": 0.6,
            "mae": 0.3,
            "rmse": 0.4,
            "r2": -1.0,
            "rank_mae": 12.0,
            "ndcg_at_roster_cutoff": 0.8,
            "top_k_recall": 0.6,
            "calibration_intercept": 1.0,
            "calibration_slope": 2.0,
        }
    }

    contracted = _position_metric_contract(metrics, point_scale_valid=False)

    assert contracted["WR"]["spearman"] == 0.75
    assert contracted["WR"]["rank_mae"] == 12.0
    assert contracted["WR"]["mae"] is None
    assert contracted["WR"]["calibration_slope"] is None
    assert metrics["WR"]["mae"] == 0.3


def test_retrospective_selection_binds_code_and_prefit_lock() -> None:
    code_hashes = {"fantasy_football/phase2_runner_v2.py": "a" * 64}
    selection = {
        "phase2_code_hashes": dict(code_hashes),
        "prefit_manifest_sha256": "b" * 64,
    }

    _validate_selection_integrity(selection, code_hashes, "b" * 64)

    changed_code = {"fantasy_football/phase2_runner_v2.py": "c" * 64}
    with pytest.raises(ValueError, match="code changed"):
        _validate_selection_integrity(selection, changed_code, "b" * 64)
    with pytest.raises(ValueError, match="pre-fit lock changed"):
        _validate_selection_integrity(selection, code_hashes, "d" * 64)


def test_no_draft_diagnostic_is_not_applicable_when_pool_already_excludes_it() -> None:
    plan = {
        "global_excluded_feature_fragments": [],
        "candidates": [
            {
                "identifier": "already_no_draft",
                "tier": "stable",
                "recipes": ["all"],
                "estimator": "extra_trees",
                "target": "points",
                "excluded_feature_fragments": [
                    "draft_number",
                    "draft_round",
                    "was_drafted",
                ],
            }
        ],
        "blends": [],
    }
    frame = pd.DataFrame(  # (2, 2)
        {"lag1_carries": [1.0, 2.0], "draft_number": [10.0, 20.0]}
    )
    bundles = {"stable": SimpleNamespace(frame=frame)}

    result = _run_selected_diagnostic(
        plan,
        "discovery",
        (2020,),
        bundles,
        {},
        "already_no_draft",
        "no_draft",
    )

    assert result["applicable"] is False
    assert "predictions" not in result
    assert "already excludes" in result["reason"]


def test_blend_diagnostic_clones_components_and_preserves_fixed_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {
        "candidates": [
            {
                "identifier": identifier,
                "tier": "stable",
                "recipes": ["all"],
                "estimator": "extra_trees",
                "target": "points",
            }
            for identifier in ("model_a", "model_b")
        ],
        "blends": [
            {
                "identifier": "fixed_blend",
                "components": ["model_a", "model_b"],
                "weights": [0.25, 0.75],
            }
        ],
    }
    base = pd.DataFrame(  # (8, 7)
        {
            "player_id": [f"P{index}" for index in range(8)],
            "target_season": [2020] * 4 + [2021] * 4,
            "model_position": ["WR"] * 8,
            "target_points": [0.0, 10.0, 20.0, 30.0] * 2,
            "predicted": [0.0] * 8,
            "candidate": [""] * 8,
            "fold_fit_seconds": [0.1] * 8,
        }
    )

    def fake_diagnostic(
        plan: object,
        stage: str,
        validation_seasons: tuple[int, ...],
        bundles: object,
        tables: object,
        selected_identifier: str,
        diagnostic: str,
    ) -> dict[str, object]:
        del plan, stage, validation_seasons, bundles, tables
        predictions = base.copy()  # (8, 7)
        predictions["candidate"] = f"{selected_identifier}__{diagnostic}"
        predictions["predicted"] = (
            np.arange(8, dtype="float64")
            if selected_identifier == "model_a"
            else np.arange(7, -1, -1, dtype="float64")
        )
        selected = pd.DataFrame(  # (1, 3)
            {
                "validation_season": [2020],
                "position_scope": ["pooled"],
                "selected_feature_count": [1],
            }
        )
        return {
            "identifier": f"{selected_identifier}__{diagnostic}",
            "applicable": True,
            "predictions": predictions,
            "selected_features": selected,
            "summary": {
                "feature_pool_count": 2,
                "median_selected_feature_count": 1.0,
            },
        }

    monkeypatch.setattr(v2, "_run_model_diagnostic", fake_diagnostic)
    result = _run_selected_diagnostic(
        plan,
        "discovery",
        (2020, 2021),
        {},
        {},
        "fixed_blend",
        "end_to_end_null",
    )

    predictions = result["predictions"]  # (8, 7)
    assert result["applicable"] is True
    assert predictions["candidate"].eq("fixed_blend__end_to_end_null").all()
    np.testing.assert_allclose(
        predictions["predicted"],
        [0.8125, 0.6875, 0.5625, 0.4375] * 2,
    )
    assert result["summary"]["mae"] is None
