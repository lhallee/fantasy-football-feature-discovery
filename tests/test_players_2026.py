"""Tests for the generated all-roster query interface."""

import json
import pandas as pd

from pathlib import Path

from fantasy_football.players_2026 import PLAYERS, get_player, query_players


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_contains_full_roster_and_backups() -> None:
    roster = pd.read_parquet(  # (n_preseason_players, c_roster)
        ROOT / "data" / "processed" / "preseason_players.parquet"
    )
    current_mask = roster["season"].eq(2026)  # (n_preseason_players,)
    current = roster[current_mask]  # (n_current_players, c_roster)

    assert {player.player_id for player in PLAYERS} == set(current["player_id"])
    assert len({player.player_id for player in PLAYERS}) == len(PLAYERS)
    assert sum(player.fantasy_eligible for player in PLAYERS) == int(
        current["position"].isin({"QB", "RB", "FB", "HB", "WR", "TE", "K", "PK"}).sum()
    )
    assert any(
        player.fantasy_eligible and (player.team_position_rank or 0) > 1
        for player in PLAYERS
    )


def test_query_filters_and_orders_players() -> None:
    running_backs = query_players(position="RB", limit=20)
    assert len(running_backs) == 20
    assert all(player.model_position == "RB" for player in running_backs)
    assert all(
        left.predicted_points >= right.predicted_points
        for left, right in zip(running_backs, running_backs[1:], strict=False)
    )
    assert get_player(running_backs[0].player_id) == running_backs[0]


def test_kickers_carry_low_confidence_label() -> None:
    kickers = query_players(position="K", limit=None)
    place_kickers = query_players(position="PK", limit=None)
    assert kickers
    assert place_kickers == kickers
    assert all(player.confidence == "low_kicker_rank_evidence" for player in kickers)
    assert all(
        (left.previous_season_points or 0.0) >= (right.previous_season_points or 0.0)
        for left, right in zip(kickers, kickers[1:], strict=False)
    )


def test_catalog_exposes_exact_model_feature_vectors() -> None:
    manifest = json.loads((ROOT / "artifacts" / "model_manifest.json").read_text())
    predictions = pd.read_parquet(  # (n_fantasy_players, c_prediction)
        ROOT / "artifacts" / "predictions_2026.parquet"
    ).set_index("player_id")
    offense_columns = tuple(manifest["feature_columns"]["offense"])
    kicker_columns = tuple(manifest["feature_columns"]["kicker"])
    for player in PLAYERS:
        if not player.fantasy_eligible:
            assert player.selected_atoms == ()
            assert player.feature_values == ()
            continue
        expected = kicker_columns if player.model_position == "K" else offense_columns
        assert tuple(name for name, _ in player.feature_values) == expected
        for name, value in player.feature_values:
            source_value = predictions.loc[
                player.player_id,
                f"feature_value__{name}",
            ]
            if pd.isna(source_value):
                assert value is None
            else:
                assert value == float(source_value)
        cohort = "kicker" if player.model_position == "K" else "offense"
        assert player.selected_atoms == tuple(manifest["models"][cohort]["recipes"])
        assert player.scoring_profile == "espn_full_ppr_2026"


def test_query_aliases_availability_and_baseline_sort() -> None:
    assert query_players(team="ARI", limit=1)
    assert query_players(team="LAR", limit=1)
    retired = query_players(roster_status="RET", fantasy_only=False, limit=None)
    assert retired
    assert all(player.roster_status == "RET" for player in retired)
    kickers = query_players(
        position="K",
        sort_by="previous_season_points",
        limit=10,
    )
    assert all(
        (left.previous_season_points or 0.0) >= (right.previous_season_points or 0.0)
        for left, right in zip(kickers, kickers[1:], strict=False)
    )
    model_kickers = query_players(position="K", sort_by="prediction", limit=10)
    assert all(
        left.predicted_points >= right.predicted_points
        for left, right in zip(model_kickers, model_kickers[1:], strict=False)
    )
