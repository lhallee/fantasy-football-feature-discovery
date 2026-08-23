"""Test the missed-time injury label, reserve features, and family mapping."""

import numpy as np
import pandas as pd
import pytest

from fantasy_football.phase2_features import FeatureLineage
from fantasy_football.phase4_labels import (
    PRIMARY_LABEL,
    SECONDARY_LABEL,
    Phase4Bundle,
    aggregate_absence_reports,
    build_missed_time_labels,
    feature_family,
    load_roster_reserve_history,
)


def _reports() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2020, 2020, 2020, 2020, 2020],
            "game_type": ["REG", "REG", "REG", "POST", "REG"],
            "week": [1, 2, 3, 19, 4],
            "gsis_id": ["a", "a", "b", "c", "d"],
            "report_primary_injury": ["Knee", "Knee", None, "Ankle", "Illness"],
            "report_secondary_injury": [None, None, None, None, None],
            "report_status": ["Questionable", "Out", None, "Out", "Out"],
            "practice_primary_injury": [
                "Knee",
                "Knee",
                "Hamstring",
                "Ankle",
                "Illness",
            ],
            "practice_secondary_injury": [None, None, None, None, None],
        }
    )  # (5, 9)


def test_absence_requires_physical_injury_and_out_or_doubtful_status() -> None:
    absences = aggregate_absence_reports(_reports())  # (1, 4)
    assert absences["player_id"].tolist() == ["a"]
    assert absences["absence_report_weeks"].tolist() == [1.0]


def test_reserve_history_excludes_non_injury_codes(tmp_path) -> None:
    rosters = tmp_path / "rosters"
    rosters.mkdir()
    pd.DataFrame(
        {
            "season": [2020] * 6,
            "week": [1, 2, 3, 1, 2, 1],
            "game_type": ["REG", "REG", "REG", "REG", "REG", "POST"],
            "status": ["ACT", "RES", "RES", "RES", "ACT", "RES"],
            "gsis_id": ["a", "a", "a", "b", "b", "a"],
            "status_description_abbr": ["A01", "R01", "R48", "R59", "A01", "R01"],
        }
    ).to_parquet(rosters / "roster_weekly_2020.parquet", index=False)
    history = load_roster_reserve_history(tmp_path, seasons=(2020,))  # (2, 4)
    by_player = history.set_index("player_id")
    assert by_player.loc["a", "reserve_injury_weeks"] == 2.0
    assert by_player.loc["b", "reserve_injury_weeks"] == 0.0
    assert by_player.loc["a", "roster_regular_weeks"] == 3.0


def test_missed_time_label_unions_components_and_blanks_unlabeled_seasons() -> None:
    keys = pd.DataFrame(
        {
            "target_season": [2020, 2020, 2020, 2026],
            "player_id": ["a", "b", "c", "d"],
            "model_position": ["RB", "WR", "TE", "QB"],
        }
    )  # (4, 3)
    secondary = pd.Series([1.0, 1.0, 0.0, 0.0])
    absences = pd.DataFrame(
        {
            "season": [2020],
            "player_id": ["a"],
            "absence_report_weeks": [2.0],
            "absence_reported": [1.0],
        }
    )
    reserve = pd.DataFrame(
        {
            "season": [2020, 2020],
            "player_id": ["b", "c"],
            "reserve_injury_weeks": [4.0, 0.0],
            "roster_regular_weeks": [17.0, 17.0],
        }
    )
    labels = build_missed_time_labels(keys, secondary, absences, reserve)  # (4, 9)
    assert labels[PRIMARY_LABEL].tolist()[:3] == [1.0, 1.0, 0.0]
    assert np.isnan(labels.loc[3, PRIMARY_LABEL])
    assert np.isnan(labels.loc[3, SECONDARY_LABEL])


def test_feature_family_mapping() -> None:
    assert feature_family("lag1_games") == "lag1_participation"
    assert feature_family("lag1_receptions") == "lag1_production"
    assert feature_family("lag1_catch_rate") == "lag1_rates"
    assert feature_family("lag1_late6_receptions") == "lag1_trajectory"
    assert feature_family("lag3_carries") == "lag2_4_history"
    assert feature_family("injury_lag1_reserve_weeks") == "reserve_history_roster"
    assert feature_family("injury_lag2_absence_reported") == "injury_report_history"
    assert feature_family("combine_forty") == "body_combine"
    assert feature_family("position_K") == "position"
    assert feature_family("lag2_history_available") == "history_availability"


def test_bundle_rejects_report_features_with_one_season_offset() -> None:
    keys = pd.DataFrame(
        {"target_season": [2020], "player_id": ["a"], "model_position": ["RB"]}
    )
    frame = pd.DataFrame({"injury_lag1_absence_reported": [0.0]}, dtype="float32")
    labels = pd.DataFrame({PRIMARY_LABEL: [1.0], SECONDARY_LABEL: [1.0]})
    lineage = {
        "injury_lag1_absence_reported": FeatureLineage(
            "injury_lag1_absence_reported",
            ("nflverse_injury_reports",),
            (1,),
            "test",
        )
    }
    with pytest.raises(ValueError, match="not deployable"):
        Phase4Bundle(
            keys,
            keys.copy(),
            labels,
            frame,
            lineage,
            {"injury_lag1_absence_reported": "x"},
        )
