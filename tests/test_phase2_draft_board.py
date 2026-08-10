"""Tests for predicted-score-only draft-board pages."""

from dataclasses import replace

from fantasy_football.phase2_draft_board import DRAFT_POSITIONS, draft_board_pages
from fantasy_football.players_2026 import PLAYERS


def test_real_draft_board_has_position_pages_sorted_by_prediction() -> None:
    pages = draft_board_pages()

    assert tuple(pages) == ("Overall", *DRAFT_POSITIONS)
    assert len(pages["Overall"]) > 900
    for page_name, records in pages.items():
        scores = [record["predicted_points"] for record in records]
        assert scores == sorted(scores, reverse=True), page_name
        assert all(score is not None for score in scores)
        if page_name != "Overall":
            assert {record["position"] for record in records} == {page_name}


def test_kickers_use_model_prediction_instead_of_catalog_query_baseline() -> None:
    pages = draft_board_pages()
    kicker_scores = [record["predicted_points"] for record in pages["K"]]

    assert kicker_scores == sorted(kicker_scores, reverse=True)
    assert len(kicker_scores) == 43


def test_unavailable_players_can_be_included_explicitly() -> None:
    template = next(player for player in PLAYERS if player.fantasy_eligible)
    cut_player = replace(
        template,
        player_id="cut-player",
        name="Cut Player",
        roster_status="CUT",
        predicted_points=999.0,
    )

    available = draft_board_pages((template, cut_player))
    complete = draft_board_pages((template, cut_player), available_only=False)

    assert all(record["player_id"] != "cut-player" for record in available["Overall"])
    assert complete["Overall"][0]["player_id"] == "cut-player"
