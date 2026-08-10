"""Tests for explicit, leakage-safe Phase 2 feature recipes."""

import pandas as pd
import pytest

from fantasy_football.phase2_cohort import build_phase2_offense_cohort
from fantasy_football.phase2_features import KEY_COLUMNS
from fantasy_football.phase2_features import build_phase2_features


def _stable_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly_records: list[dict[str, object]] = []
    season_carries = {
        2020: {"A": (2.0, 4.0), "B": (1.0, 2.0)},
        2021: {"A": (3.0, 6.0), "B": (2.0, 3.0)},
        2022: {"A": (5.0, 10.0), "B": (3.0, 5.0)},
        2023: {"A": (10.0, 20.0), "B": (4.0, 6.0)},
        2024: {"A": (1_000.0, 2_000.0), "B": (3_000.0, 4_000.0)},
    }
    for season, player_counts in season_carries.items():
        for player_id, carries_by_week in player_counts.items():
            team = "OLD" if player_id == "A" and season == 2023 else "NEW"
            for week, carries in zip((1, 18), carries_by_week, strict=True):
                weekly_records.append(
                    {
                        "season": season,
                        "week": week,
                        "season_type": "REG",
                        "player_id": player_id,
                        "team": team,
                        "carries": carries,
                        "rushing_yards": carries * 5.0,
                        "rushing_tds": float(carries >= 10.0),
                        "rushing_first_downs": carries / 2.0,
                        "rushing_20": float(carries >= 6.0),
                        "receptions": carries / 2.0,
                        "receiving_yards": carries * 4.0,
                        "receiving_first_downs": carries / 4.0,
                        "receiving_20": float(carries >= 6.0),
                        "fantasy_points_ppr": 99_999.0,
                        "rushing_epa": 99_999.0,
                        "target_share": 1.0,
                    }
                )
    weekly_records.append(
        {
            "season": 2023,
            "week": 18,
            "season_type": "REG",
            "player_id": "TEAMMATE",
            "team": "NEW",
            "carries": 7.0,
            "rushing_yards": 21.0,
            "rushing_tds": 0.0,
            "rushing_first_downs": 2.0,
            "rushing_20": 0.0,
            "receptions": 0.0,
            "receiving_yards": 0.0,
            "receiving_first_downs": 0.0,
            "receiving_20": 0.0,
            "fantasy_points_ppr": 99_999.0,
            "rushing_epa": 99_999.0,
            "target_share": 1.0,
        }
    )
    weekly = pd.DataFrame(weekly_records)  # (21, 17)

    player_season_records: list[dict[str, object]] = []
    for season in range(2020, 2025):
        for player_id in ("A", "B"):
            last_team = "OLD" if player_id == "A" and season == 2023 else "NEW"
            player_season_records.append(
                {
                    "season": season,
                    "player_id": player_id,
                    "last_team": last_team,
                    "games": 2.0,
                    "roster_weeks": 18.0,
                    "offense_snaps": 100.0 + season,
                    "st_snaps": 5.0,
                    "active_roster_weeks": 18.0,
                    "target_points": 99_999.0,
                }
            )
    player_seasons = pd.DataFrame(player_season_records)  # (10, 9)

    modeling = pd.DataFrame(  # (2, 18)
        {
            "target_season": [2024, 2024],
            "player_id": ["A", "B"],
            "model_position": ["RB", "RB"],
            "team": ["NEW", "NEW"],
            "age": [25.0, 27.0],
            "years_exp": [3.0, 5.0],
            "height": [71.0, 72.0],
            "weight": [210.0, 220.0],
            "draft_number": [20.0, 120.0],
            "draft_round": [1.0, 4.0],
            "was_drafted": [1.0, 1.0],
            "is_rookie": [0.0, 0.0],
            "team_changed": [1.0, 0.0],
            "combine_forty": [4.45, None],
            "target_points": [300.0, 100.0],
            "fantasy_points_ppr": [300.0, 100.0],
            "status": ["ACT", "ACT"],
            "depth_chart_position": ["RB1", "RB2"],
        }
    )
    return weekly, player_seasons, modeling


def _recent_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly = pd.DataFrame(  # (2, 13)
        {
            "season": [2011, 2012],
            "week": [17, 17],
            "season_type": ["REG", "REG"],
            "player_id": ["A", "A"],
            "team": ["OLD", "OLD"],
            "targets": [6.0, 10.0],
            "receptions": [3.0, 5.0],
            "receiving_yards": [30.0, 60.0],
            "receiving_air_yards": [80.0, 120.0],
            "receiving_yards_after_catch": [12.0, 20.0],
            "fantasy_points_ppr": [50.0, 90.0],
            "receiving_epa": [8.0, 12.0],
            "wopr": [0.7, 0.8],
        }
    )
    player_seasons = pd.DataFrame(  # (2, 9)
        {
            "season": [2011, 2012],
            "player_id": ["A", "A"],
            "last_team": ["OLD", "OLD"],
            "games": [1.0, 1.0],
            "roster_weeks": [17.0, 17.0],
            "offense_snaps": [500.0, 650.0],
            "st_snaps": [10.0, 15.0],
            "active_roster_weeks": [17.0, 17.0],
            "target_points": [50.0, 90.0],
        }
    )
    modeling = pd.DataFrame(  # (2, 7)
        {
            "target_season": [2012, 2013],
            "player_id": ["A", "A"],
            "model_position": ["WR", "WR"],
            "team": ["OLD", "OLD"],
            "age": [24.0, 25.0],
            "target_points": [50.0, 90.0],
            "status": ["ACT", "ACT"],
        }
    )
    return weekly, player_seasons, modeling


def test_stable_recipes_have_expected_values_and_lineage() -> None:
    weekly, player_seasons, modeling = _stable_inputs()
    bundle = build_phase2_features(weekly, player_seasons, modeling)
    frame = bundle.frame  # (2, d)

    assert tuple(bundle.keys.columns) == KEY_COLUMNS
    assert frame.loc[0, "lag1_carries"] == pytest.approx(30.0)
    assert frame.loc[0, "lag4_carries"] == pytest.approx(6.0)
    assert frame.loc[0, "lag1_rush_yards_per_carry"] == pytest.approx(5.0)
    assert frame.loc[0, "lag1_late6_carries"] == pytest.approx(20.0)
    assert frame.loc[0, "room_lag1_other_carries"] == pytest.approx(10.0)
    assert frame.loc[1, "room_lag1_other_carries"] == pytest.approx(30.0)
    assert frame.loc[0, "room_competitor_count"] == pytest.approx(1.0)
    assert frame.loc[0, "team_lag1_carries"] == pytest.approx(17.0)
    assert frame.loc[0, "prior_team_lag1_carries"] == pytest.approx(30.0)
    assert frame.loc[0, "combine_forty"] == pytest.approx(4.45)
    assert frame.loc[1, "missing_combine_forty"] == pytest.approx(1.0)

    assert frame["position_RB"].eq(1.0).all()
    assert frame[["position_QB", "position_WR", "position_TE"]].eq(0.0).all().all()
    assert bundle.lineage["position_RB"].source_offsets == (0,)
    assert bundle.lineage["lag4_carries"].source_offsets == (4,)
    assert bundle.lineage["room_lag1_other_carries"].source_offsets == (0, 1)
    assert bundle.lineage["prior_team_lag1_carries"].source_offsets == (1, 1)

    lowered_columns = " ".join(frame.columns).lower()
    for prohibited in (
        "active_roster",
        "depth",
        "epa",
        "fantasy",
        "status",
        "target_points",
        "target_share",
    ):
        assert prohibited not in lowered_columns


def test_status_cohort_filter_precedes_real_room_feature_aggregation() -> None:
    weekly, player_seasons, modeling = _stable_inputs()
    cut_row = modeling.iloc[[0]].copy()  # (1, 18)
    cut_row.loc[:, "player_id"] = "TEAMMATE"
    cut_row.loc[:, "status"] = "CUT"
    cut_row.loc[:, "target_points"] = 0.0
    with_cut = pd.concat([modeling, cut_row], ignore_index=True)  # (3, 18)

    cohort = build_phase2_offense_cohort(with_cut)
    bundle = build_phase2_features(weekly, player_seasons, cohort.table)
    frame = bundle.frame  # (2, d)

    assert bundle.keys["player_id"].tolist() == ["A", "B"]
    assert frame["room_competitor_count"].tolist() == [1.0, 1.0]
    assert frame["room_lag1_other_carries"].tolist() == [10.0, 30.0]


def test_target_and_future_performance_fields_cannot_change_features() -> None:
    weekly, player_seasons, modeling = _stable_inputs()
    baseline = build_phase2_features(weekly, player_seasons, modeling)

    changed_weekly = weekly.copy()  # (21, 17)
    future_mask = changed_weekly["season"].ge(2024)  # (21,)
    changed_weekly.loc[future_mask, "carries"] = -1_000_000.0
    changed_weekly.loc[:, "fantasy_points_ppr"] = -1_000_000.0
    changed_weekly.loc[:, "rushing_epa"] = -1_000_000.0
    changed_weekly.loc[:, "target_share"] = -1_000_000.0
    changed_player_seasons = player_seasons.copy()  # (10, 9)
    changed_player_seasons.loc[:, "active_roster_weeks"] = -1_000_000.0
    changed_player_seasons.loc[:, "target_points"] = -1_000_000.0
    changed_modeling = modeling.copy()  # (2, 18)
    changed_modeling.loc[:, "target_points"] = -1_000_000.0
    changed_modeling.loc[:, "fantasy_points_ppr"] = -1_000_000.0
    changed_modeling.loc[:, "status"] = "OUT"
    changed_modeling.loc[:, "depth_chart_position"] = "UNKNOWN"

    changed = build_phase2_features(
        changed_weekly,
        changed_player_seasons,
        changed_modeling,
    )

    pd.testing.assert_frame_equal(changed.keys, baseline.keys)
    pd.testing.assert_frame_equal(changed.frame, baseline.frame)
    assert changed.lineage == baseline.lineage


def test_recent_tier_is_era_gated_and_adds_only_lagged_recent_atoms() -> None:
    weekly, player_seasons, modeling = _recent_inputs()
    stable = build_phase2_features(
        weekly,
        player_seasons,
        modeling,
        tier="stable",
        max_lag=1,
    )
    recent = build_phase2_features(
        weekly,
        player_seasons,
        modeling,
        tier="recent",
        max_lag=1,
    )

    assert stable.keys["target_season"].tolist() == [2012, 2013]
    assert recent.keys["target_season"].tolist() == [2013]
    assert "lag1_targets" not in stable.frame
    assert "lag1_offense_snaps" not in stable.frame
    assert recent.frame.loc[0, "lag1_targets"] == pytest.approx(10.0)
    assert recent.frame.loc[0, "lag1_receiving_air_yards"] == pytest.approx(120.0)
    assert recent.frame.loc[0, "lag1_offense_snaps"] == pytest.approx(650.0)
    assert recent.frame.loc[0, "lag1_catch_rate"] == pytest.approx(0.5)

    for record in recent.lineage.values():
        for source, offset in zip(
            record.sources,
            record.source_offsets,
            strict=True,
        ):
            if source in {"weekly_stats", "player_seasons"}:
                assert offset >= 1


def test_builder_rejects_duplicate_player_seasons() -> None:
    weekly, player_seasons, modeling = _stable_inputs()
    duplicated = pd.concat(  # (11, 9)
        [player_seasons, player_seasons.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="unique at season-player grain"):
        build_phase2_features(weekly, duplicated, modeling)
