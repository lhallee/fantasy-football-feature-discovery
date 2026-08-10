"""Tests for explicit ESPN-style scoring reconstruction."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fantasy_football.scoring import ScoringProfile, load_scoring_profile
from fantasy_football.scoring import score_player_games


ROOT = Path(__file__).resolve().parents[1]


def test_scoring_profile_matches_versioned_configuration() -> None:
    configuration = json.loads(
        (ROOT / "config" / "scoring_espn_full_ppr_2026.json").read_text()
    )
    for field, value in asdict(ScoringProfile()).items():
        assert configuration[field] == value
    assert (
        load_scoring_profile(ROOT / "config" / "scoring_espn_full_ppr_2026.json")
        == ScoringProfile()
    )


def test_full_ppr_scoring_uses_raw_events() -> None:
    games = pd.DataFrame(  # (1, 12)
        {
            "position": ["RB"],
            "passing_yards": [250],
            "passing_tds": [2],
            "passing_interceptions": [1],
            "rushing_yards": [30],
            "rushing_tds": [1],
            "receptions": [5],
            "receiving_yards": [70],
            "receiving_tds": [1],
            "rushing_2pt_conversions": [1],
            "fumbles_lost_total": [1],
            "special_teams_tds": [1],
        }
    )

    scored = score_player_games(games, strict=False)  # (1, c_scored)

    assert scored.loc[0, "recomputed_non_ppr_points"] == 44.0
    assert scored.loc[0, "recomputed_half_ppr_points"] == 46.5
    assert scored.loc[0, "recomputed_ppr_points"] == 49.0
    assert scored.loc[0, "target_points"] == 49.0


def test_kicker_distance_buckets_are_separate() -> None:
    games = pd.DataFrame(  # (1, 7)
        {
            "position": ["K"],
            "pat_made": [2],
            "fg_made_30_39": [1],
            "fg_made_40_49": [1],
            "fg_made_50_59": [1],
            "fg_made_60_": [1],
            "fg_missed": [2],
        }
    )

    scored = score_player_games(games, strict=False)  # (1, c_scored)

    assert scored.loc[0, "recomputed_kicker_points"] == 18.0
    assert scored.loc[0, "target_points"] == 18.0


def test_reception_coefficient_controls_full_and_half_ppr() -> None:
    games = pd.DataFrame({"position": ["RB"], "receptions": [4]})  # (1, 2)

    scored = score_player_games(  # (1, c_scored)
        games,
        ScoringProfile(reception=2.0),
        strict=False,
    )

    assert scored.loc[0, "recomputed_half_ppr_points"] == 4.0
    assert scored.loc[0, "recomputed_ppr_points"] == 8.0


def test_kicker_receptions_and_curated_defensive_touchdowns_score() -> None:
    games = pd.DataFrame(  # (2, 3)
        {
            "position": ["K", "WR"],
            "receptions": [1, 0],
            "individual_defensive_tds": [0, 1],
        }
    )

    scored = score_player_games(games, strict=False)  # (2, c_scored)

    assert scored.loc[0, "target_points"] == 1.0
    assert scored.loc[1, "target_points"] == 6.0


def test_strict_scoring_rejects_incomplete_source_schema() -> None:
    games = pd.DataFrame({"position": ["RB"], "receptions": [1]})  # (1, 2)

    try:
        score_player_games(games)
    except ValueError as error:
        assert "missing scoring columns" in str(error)
    else:
        raise AssertionError("Incomplete scoring inputs must fail in strict mode.")
