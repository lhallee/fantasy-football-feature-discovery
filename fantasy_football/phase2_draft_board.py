"""Build draft-board pages from the generated 2026 player catalog."""

from __future__ import annotations

import argparse
import json

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .players_2026 import PLAYERS, PlayerProjection


DRAFT_POSITIONS = ("QB", "RB", "WR", "TE", "K")
UNAVAILABLE_STATUSES = frozenset({"CUT", "RET"})


def _draft_record(player: PlayerProjection) -> dict[str, Any]:
    """Return one flat, spreadsheet-ready player record."""
    return {
        "drafted": "",
        "fantasy_team": "",
        "draft_round": None,
        "notes": "",
        "name": player.name,
        "team": player.team,
        "position": player.model_position,
        "predicted_points": player.predicted_points,
        "prediction_low_80": player.prediction_interval_80_low,
        "prediction_high_80": player.prediction_interval_80_high,
        "previous_season_points": player.previous_season_points,
        "position_rank": player.position_rank,
        "overall_point_rank": player.overall_point_rank,
        "team_position_rank": player.team_position_rank,
        "depth_tier": player.depth_tier,
        "depth_rank": player.depth_rank,
        "roster_status": player.roster_status,
        "confidence": player.confidence,
        "out_of_distribution": player.out_of_distribution,
        "age": player.age,
        "years_experience": player.years_experience,
        "draft_year": player.draft_year,
        "nfl_draft_round": player.draft_round,
        "nfl_draft_pick": player.draft_number,
        "college": player.college,
        "player_id": player.player_id,
    }


def _predicted_order(player: PlayerProjection) -> tuple[float, str, str]:
    """Return the deterministic descending-prediction sort key."""
    predicted_points = player.predicted_points
    if predicted_points is None:
        raise ValueError(f"Draft-eligible player {player.player_id} has no prediction.")
    return (-predicted_points, player.name.casefold(), player.player_id)


def draft_board_pages(
    players: Iterable[PlayerProjection] = PLAYERS,
    *,
    available_only: bool = True,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return overall and position pages sorted by predicted points."""
    eligible = tuple(  # (n_eligible_players,)
        player
        for player in players
        if player.fantasy_eligible
        and player.model_position in DRAFT_POSITIONS
        and player.predicted_points is not None
        and (
            not available_only
            or (player.roster_status or "").upper() not in UNAVAILABLE_STATUSES
        )
    )
    ordered = tuple(sorted(eligible, key=_predicted_order))  # (n_eligible_players,)
    pages: dict[str, tuple[dict[str, Any], ...]] = {
        "Overall": tuple(_draft_record(player) for player in ordered)
    }
    for position in DRAFT_POSITIONS:
        position_players = (  # (n_position_players,)
            player for player in ordered if player.model_position == position
        )
        pages[position] = tuple(_draft_record(player) for player in position_players)
    return pages


def write_draft_board_json(
    output_path: Path,
    players: Iterable[PlayerProjection] = PLAYERS,
    *,
    available_only: bool = True,
) -> Path:
    """Write the workbook-neutral draft-board payload."""
    pages = draft_board_pages(players, available_only=available_only)
    payload: Mapping[str, Any] = {
        "season": 2026,
        "sort_contract": "predicted_points_descending",
        "available_only": available_only,
        "pages": pages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> Path:
    """Parse the standalone draft-board payload command."""
    parser = argparse.ArgumentParser(
        description="Build position pages for the 2026 fantasy draft board."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/phase2/artifacts/draft_board_2026.json"),
    )
    return parser.parse_args(argv).output


def main(argv: Sequence[str] | None = None) -> int:
    """Write the draft-board payload and print its path."""
    output_path = write_draft_board_json(_parse_args(argv))
    print(output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
