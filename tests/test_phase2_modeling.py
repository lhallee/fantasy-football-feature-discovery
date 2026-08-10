"""Tests for leakage-safe Phase 2 walk-forward modeling."""

import numpy as np
import pandas as pd

from fantasy_football.phase2_modeling import Phase2Candidate
from fantasy_football.phase2_modeling import blend_walk_forward_predictions
from fantasy_football.phase2_modeling import select_fold_features
from fantasy_football.phase2_modeling import walk_forward_candidate


def _synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for season in range(2014, 2021):
        for position_index, position in enumerate(("QB", "WR")):
            for player_index in range(12):
                signal = float(player_index + position_index)
                records.append(
                    {
                        "player_id": f"{position}-{player_index}",
                        "target_season": season,
                        "model_position": position,
                        "target_points": signal * 10.0 + season % 3,
                        "signal": signal,
                        "noise": float((player_index * 7 + season) % 11),
                        "future_only_pattern": float(season >= 2019),
                    }
                )
    panel = pd.DataFrame(records)  # (168, 7)
    table = panel[  # (168, 4)
        ["player_id", "target_season", "model_position", "target_points"]
    ].copy()
    X = panel[["signal", "noise", "future_only_pattern"]].copy()  # (168, 3)
    return table, X


def test_fold_selector_respects_feature_limit() -> None:
    table, X = _synthetic_panel()
    train_mask = table["target_season"] < 2019  # (168,)
    selected, scores = select_fold_features(
        X.loc[train_mask],
        table.loc[train_mask, "target_points"],
        table.loc[train_mask, "target_season"],
        table.loc[train_mask, "model_position"],
        feature_limit=2,
    )

    assert len(selected) == 2
    assert selected[0] == "signal"
    assert list(scores.index) == selected


def test_walk_forward_candidate_predicts_each_outer_row_once() -> None:
    table, X = _synthetic_panel()
    candidate = Phase2Candidate(
        identifier="ridge_rank",
        tier="stable",
        recipes=("synthetic",),
        estimator="ridge",
        target="rank_percentile",
        feature_limit=2,
        parameters={"alpha": 1.0},
    )

    output = walk_forward_candidate(
        table,
        X,
        candidate,
        validation_seasons=(2019, 2020),
        minimum_training_season=2014,
    )

    assert len(output.predictions) == 48
    assert output.predictions.index.is_unique
    assert set(output.predictions["target_season"]) == {2019, 2020}
    assert output.predictions["predicted"].notna().all()
    assert len(output.fold_metrics) == 2
    assert output.fold_metrics["spearman"].min() > 0.95
    assert output.selected_features["selected_feature_count"].eq(2).all()


def test_rank_blend_uses_only_component_predictions() -> None:
    table, X = _synthetic_panel()
    validation_mask = table["target_season"] == 2020  # (168,)
    base = table.loc[validation_mask].copy()  # (24, 4)
    first = base.assign(candidate="first", predicted=X.loc[validation_mask, "signal"])
    second = base.assign(candidate="second", predicted=-X.loc[validation_mask, "noise"])
    first["fold_fit_seconds"] = 1.0
    second["fold_fit_seconds"] = 1.0

    blended = blend_walk_forward_predictions(
        {"first": first, "second": second},
        "blend",
        ("first", "second"),
        (0.75, 0.25),
    )

    assert len(blended) == len(base)
    assert np.isfinite(blended["predicted"]).all()
    assert blended["candidate"].eq("blend").all()
