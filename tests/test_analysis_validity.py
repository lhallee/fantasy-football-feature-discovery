"""Independent checks for table grain and headline analytical claims."""

import json
import subprocess
import sys
import numpy as np
import pandas as pd

from pathlib import Path

from fantasy_football.dataset import _new_depth_tier
from fantasy_football.metrics import macro_rank_metric, metric_summary


ROOT = Path(__file__).resolve().parents[1]


def test_cli_imports_and_displays_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "fantasy_football", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "download" in completed.stdout
    assert "build" in completed.stdout


def test_depth_snapshot_date_excludes_later_updates() -> None:
    depth = pd.DataFrame(  # (2, 4)
        {
            "dt": ["2026-08-09T12:00:00Z", "2026-08-10T12:00:00Z"],
            "gsis_id": ["player", "player"],
            "pos_abb": ["RB", "RB"],
            "pos_rank": [1, 2],
        }
    )

    snapshot = _new_depth_tier(depth, 2026, 8, 9)  # (1, 4)

    assert snapshot.iloc[0]["depth_rank"] == 1


def test_processed_tables_have_one_row_per_declared_entity() -> None:
    grains = {
        "player_games": ["player_id", "season", "week"],
        "player_seasons": ["player_id", "season"],
        "preseason_players": ["player_id", "season"],
        "modeling_table": ["player_id", "target_season"],
    }
    for table_name, key_columns in grains.items():
        table = pd.read_parquet(  # (n_rows, n_key_columns)
            ROOT / "data" / "processed" / f"{table_name}.parquet",
            columns=key_columns,
        )
        assert not table[key_columns].isna().any(axis=None)
        assert not table.duplicated(key_columns).any()


def test_headline_offense_metrics_recompute_from_saved_predictions() -> None:
    predictions = pd.read_parquet(  # (n_audit_players, c_audit)
        ROOT / "artifacts" / "audit_predictions.parquet"
    )
    predictions = predictions.loc[  # (n_offense_audit_players, c_audit)
        predictions["model_position"].ne("K")
    ].copy()
    summary = metric_summary(
        predictions["actual_points"],
        predictions["predicted_points"],
        predictions["model_position"],
        predictions["target_season"],
    )
    reported = json.loads((ROOT / "artifacts" / "result_summary.json").read_text())
    expected = reported["promotion_gates"]["offense"]["selected"]

    assert np.isclose(summary["spearman"], expected["spearman"], atol=1e-6)
    assert np.isclose(summary["mae"], expected["mae"], atol=1e-6)
    assert np.isclose(summary["rmse"], expected["rmse"], atol=1e-6)
    assert np.isclose(summary["r2"], expected["r2"], atol=1e-6)


def test_discovery_selection_rejects_smaller_offense_vectors() -> None:
    run_dir = ROOT / "experiments" / "phase2" / "runs" / (
        "20260810T174927548207Z-august9-discovery-"
        "3b07b79602c5-b96dfefdf4d91c71"
    )
    candidates = pd.read_csv(  # (n_candidates, c_metrics)
        run_dir / "candidate_summary.csv"
    )
    selection = json.loads((run_dir / "selection.json").read_text())
    offense_mask = candidates["cohort"].eq("offense")  # (n_candidates,)
    offense = candidates.loc[offense_mask].copy()  # (n_offense_candidates, c_metrics)
    best_rho = float(offense["spearman"].max())
    eligible_mask = offense["spearman"].ge(best_rho - 0.005)  # (n_offense_candidates,)
    eligible = offense.loc[eligible_mask].copy()  # (n_eligible, c_metrics)
    chosen = eligible.sort_values(
        ["median_selected_feature_count", "mae", "candidate"],
        kind="stable",
    ).iloc[0]

    assert chosen["candidate"] == selection["cohorts"]["offense"]["identifier"]
    assert int(chosen["median_selected_feature_count"]) == 228
    smaller_mask = offense["median_selected_feature_count"].lt(228)  # (n_offense_candidates,)
    assert offense.loc[smaller_mask, "spearman"].max() < best_rho - 0.005


def test_selected_columns_have_complete_finite_training_scores() -> None:
    manifest = json.loads((ROOT / "artifacts" / "model_manifest.json").read_text())
    offense = manifest["models"]["offense"]
    selected = offense["feature_columns"]
    scores = offense["feature_selection_scores"]

    assert len(selected) == 228
    assert set(scores) == set(selected)
    assert np.isfinite(list(scores.values())).all()
    lowered = " ".join(selected).lower()
    for forbidden in ("target_points", "fantasy", "status", "depth", "room_"):
        assert forbidden not in lowered


def test_prior_points_baseline_is_weaker_on_primary_metric() -> None:
    predictions = pd.read_parquet(  # (n_audit_players, c_audit)
        ROOT / "artifacts" / "audit_predictions.parquet"
    )
    predictions = predictions.loc[  # (n_offense_audit_players, c_audit)
        predictions["model_position"].ne("K")
    ].copy()
    model_rho = macro_rank_metric(
        predictions["actual_points"],
        predictions["predicted_points"],
        predictions["model_position"],
        predictions["target_season"],
    )
    baseline_rho = macro_rank_metric(
        predictions["actual_points"],
        predictions["previous_points_baseline"],
        predictions["model_position"],
        predictions["target_season"],
    )

    assert model_rho > baseline_rho
