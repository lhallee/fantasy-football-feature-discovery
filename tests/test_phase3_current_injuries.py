"""Test the dated current-injury source normalization and player matching."""

import pandas as pd

from fantasy_football.phase3_current_injuries import (
    match_current_injuries,
    parse_mfl_payloads,
)


def _source_rows() -> pd.DataFrame:
    injury_payload = {
        "injuries": {
            "timestamp": "1786709224",
            "injury": [
                {
                    "id": "1",
                    "status": "Questionable",
                    "details": "Hamstring",
                    "exp_return": "Aug 22, 2026",
                },
                {
                    "id": "2",
                    "status": "Retired",
                    "details": "Personal",
                    "exp_return": "Feb 15, 2027",
                },
                {
                    "id": "3",
                    "status": "Questionable",
                    "details": "Illness",
                    "exp_return": "Aug 15, 2026",
                },
            ],
        }
    }
    players = [
        {"id": "1", "name": "Alpha, Alex Jr.", "team": "LVR", "position": "QB"},
        {"id": "2", "name": "Beta, Bea", "team": "BUF", "position": "WR"},
        {"id": "3", "name": "Gamma, Gia", "team": "NYJ", "position": "TE"},
    ]
    return parse_mfl_payloads(injury_payload, players)  # (3, 16)


def test_mfl_parser_excludes_noninjury_unavailability_and_illness() -> None:
    source_rows = _source_rows()  # (3, 16)

    assert source_rows["current_injury_listing"].sum() == 1
    assert source_rows.loc[
        source_rows["current_injury_listing"], "source_player_name"
    ].tolist() == ["Alex Jr. Alpha"]


def test_current_injury_match_screens_every_player_and_normalizes_team() -> None:
    board = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "candidate_name": ["Alex Alpha", "Ben Backup"],
            "team": ["LV", "BUF"],
            "model_position": ["QB", "WR"],
        }
    )  # (2, 4)

    screen = match_current_injuries(
        board,
        _source_rows(),
        as_of_date="2026-08-14",
        retrieved_at_utc="2026-08-14T15:00:00+00:00",
    )

    assert len(screen.player_screen) == 2
    assert screen.current_injuries["player_id"].tolist() == ["a"]
    assert screen.player_screen.loc[
        screen.player_screen["player_id"].eq("b"), "screen_result"
    ].item() == "No current listing found"
