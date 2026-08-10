"""Independent checks for table grain and headline analytical claims."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
        ROOT / "artifacts" / "offense_audit_predictions.parquet"
    )
    summary = metric_summary(
        predictions["actual_points"],
        predictions["predicted_points"],
        predictions["model_position"],
        predictions["target_season"],
    )
    reported = json.loads((ROOT / "artifacts" / "result_summary.json").read_text())
    expected = reported["offense"]["retrospective_2024_2025"]

    assert np.isclose(summary["spearman"], expected["spearman"], atol=1e-6)
    assert np.isclose(summary["mae"], expected["mae"], atol=1e-6)
    assert np.isclose(summary["rmse"], expected["rmse"], atol=1e-6)
    assert np.isclose(summary["r2"], expected["r2"], atol=1e-6)


def test_compact_model_meets_declared_development_thresholds() -> None:
    subsets = pd.read_csv(  # (n_subset_settings, c_subset_metrics)
        ROOT / "artifacts" / "offense_subset_search.csv"
    )
    full = subsets.loc[  # (c_subset_metrics,)
        subsets["feature_atom_count"].eq(subsets["feature_atom_count"].max())
    ].iloc[0]
    compact_prefix = subsets.loc[  # (c_subset_metrics,)
        subsets["selected_by_compact_rule"].eq(True)
    ].iloc[0]

    assert full["spearman"] - compact_prefix["spearman"] <= 0.01
    assert compact_prefix["mae"] <= full["mae"] * 1.02

    manifest = json.loads((ROOT / "artifacts" / "model_manifest.json").read_text())
    selected_atoms = ",".join(manifest["offense_champion"]["selected_atoms"])
    refinement = pd.read_csv(  # (n_refinement_trials, c_refinement)
        ROOT / "artifacts" / "offense_backward_refinement.csv"
    )
    if selected_atoms != compact_prefix["selected_atoms"]:
        selected_mask = refinement["selected_atoms"].eq(  # (n_refinement_trials,)
            selected_atoms
        )
        selected_row = refinement[selected_mask]  # (n_selected_rows, c_refinement)
        assert len(selected_row) == 1
        assert bool(selected_row.iloc[0]["passes_compact_rule"])
    elif not refinement.empty:
        assert not refinement["passes_compact_rule"].any()


def test_selected_atoms_have_complete_finite_importance_evidence() -> None:
    manifest = json.loads((ROOT / "artifacts" / "model_manifest.json").read_text())
    selected = set(manifest["offense_champion"]["selected_atoms"])
    importance = pd.read_csv(  # (n_importance_rows, c_importance)
        ROOT / "artifacts" / "offense_fold_permutation_importance.csv"
    )
    selected_mask = importance["atom"].isin(selected)  # (n_importance_rows,)
    selected_importance = importance[selected_mask]  # (n_selected_rows, c_importance)

    assert set(selected_importance["atom"]) == selected
    assert (
        selected_importance.groupby("atom")["validation_season"].nunique().eq(4).all()
    )
    assert np.isfinite(selected_importance["permutation_importance"]).all()
    assert (
        selected_importance.groupby("atom")["permutation_importance"].mean().gt(0).all()
    )


def test_prior_points_baseline_is_weaker_on_primary_metric() -> None:
    predictions = pd.read_parquet(  # (n_audit_players, c_audit)
        ROOT / "artifacts" / "offense_audit_predictions.parquet"
    )
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
