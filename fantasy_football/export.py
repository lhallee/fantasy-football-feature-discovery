"""Export all current roster players as a typed, directly queryable Python module."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CURRENT_SEASON, FANTASY_POSITIONS, MODEL_POSITION
from .provenance import synchronize_project_sizes


def _value(value: Any, integer: bool = False) -> Any:
    if value is None or pd.isna(value):
        return None
    if integer:
        return int(round(float(value)))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    return value


def _feature_value(value: Any) -> float | None:
    """Preserve one fitted numeric input at full floating-point precision."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)


def _age(birth_date: Any, season: int) -> float | None:
    parsed = pd.to_datetime(birth_date, errors="coerce")
    if pd.isna(parsed):
        return None
    reference = pd.Timestamp(date(season, 9, 1))
    return round(float((reference - parsed).days / 365.2425), 2)


def _module_header(season: int, data_cutoff: str, snapshot_date: str) -> str:
    return f'''"""Generated {season} NFL roster and fantasy projections.

Source cutoff: {data_cutoff}. Preseason snapshot: {snapshot_date}.
Regenerate with ``python -m fantasy_football export``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerProjection:
    """One rostered player and an optional fantasy forecast."""

    player_id: str
    name: str
    team: str
    position: str
    model_position: str | None
    fantasy_eligible: bool
    roster_status: str | None
    jersey_number: int | None
    depth_tier: int | None
    depth_rank: float | None
    years_experience: int | None
    age: float | None
    college: str | None
    birth_date: str | None
    height_inches: int | None
    weight_pounds: int | None
    rookie_year: int | None
    draft_year: int | None
    draft_round: int | None
    draft_number: int | None
    draft_team: str | None
    espn_id: str | None
    sleeper_id: str | None
    pfr_id: str | None
    headshot_url: str | None
    predicted_points: float | None
    previous_season_points: float | None
    prediction_interval_80_low: float | None
    prediction_interval_80_high: float | None
    position_rank: int | None
    overall_point_rank: int | None
    team_position_rank: int | None
    model_name: str | None
    scoring_profile: str | None
    selected_atoms: tuple[str, ...]
    feature_values: tuple[tuple[str, float | None], ...]
    feature_coverage: float | None
    out_of_distribution: bool | None
    confidence: str


PLAYERS: tuple[PlayerProjection, ...] = (
'''


def _module_footer() -> str:
    return ''')

_BY_ID = {player.player_id: player for player in PLAYERS}


def get_player(player_id: str) -> PlayerProjection:
    """Return one player by GSIS ID."""
    return _BY_ID[player_id]


def query_players(
    *,
    position: str | None = None,
    team: str | None = None,
    roster_status: str | None = None,
    name_contains: str | None = None,
    college: str | None = None,
    confidence: str | None = None,
    max_depth_tier: int | None = None,
    max_position_rank: int | None = None,
    min_predicted_points: float | None = None,
    out_of_distribution: bool | None = None,
    fantasy_only: bool = True,
    available_only: bool = True,
    sort_by: str = "auto",
    limit: int | None = 25,
) -> tuple[PlayerProjection, ...]:
    """Filter players and order them by the selected score field."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative or None.")
    if sort_by not in {"auto", "prediction", "previous_season_points"}:
        raise ValueError(
            "sort_by must be 'auto', 'prediction', or 'previous_season_points'."
        )
    selected = PLAYERS
    if fantasy_only:
        selected = tuple(player for player in selected if player.fantasy_eligible)
    if available_only and roster_status is None:
        selected = tuple(
            player
            for player in selected
            if (player.roster_status or "").upper() not in {"CUT", "RET"}
        )
    if position is not None:
        wanted_position = "K" if position.upper() == "PK" else position.upper()
        selected = tuple(
            player
            for player in selected
            if player.position.upper() == wanted_position
            or player.model_position == wanted_position
        )
    if team is not None:
        team_aliases = {
            "ARI": "AZ",
            "JAC": "JAX",
            "LAR": "LA",
            "OAK": "LV",
            "SD": "LAC",
            "STL": "LA",
            "WSH": "WAS",
        }
        wanted_team = team_aliases.get(team.upper(), team.upper())
        selected = tuple(player for player in selected if player.team.upper() == wanted_team)
    if roster_status is not None:
        wanted_status = roster_status.upper()
        selected = tuple(
            player
            for player in selected
            if (player.roster_status or "").upper() == wanted_status
        )
    if name_contains is not None:
        wanted_name = name_contains.casefold()
        selected = tuple(
            player for player in selected if wanted_name in player.name.casefold()
        )
    if college is not None:
        wanted_college = college.casefold()
        selected = tuple(
            player
            for player in selected
            if wanted_college in (player.college or "").casefold()
        )
    if confidence is not None:
        wanted_confidence = confidence.casefold()
        selected = tuple(
            player
            for player in selected
            if player.confidence.casefold() == wanted_confidence
        )
    if max_depth_tier is not None:
        selected = tuple(
            player
            for player in selected
            if player.depth_tier is not None and player.depth_tier <= max_depth_tier
        )
    if max_position_rank is not None:
        selected = tuple(
            player
            for player in selected
            if player.position_rank is not None
            and player.position_rank <= max_position_rank
        )
    if min_predicted_points is not None:
        selected = tuple(
            player
            for player in selected
            if player.predicted_points is not None
            and player.predicted_points >= min_predicted_points
        )
    if out_of_distribution is not None:
        selected = tuple(
            player
            for player in selected
            if player.out_of_distribution is out_of_distribution
        )

    kicker_query = position is not None and position.upper() in {"K", "PK"}
    use_previous_points = sort_by == "previous_season_points" or (
        sort_by == "auto" and kicker_query
    )
    score_field = (
        "previous_season_points" if use_previous_points else "predicted_points"
    )
    ordered = tuple(
        sorted(
            selected,
            key=lambda player: (
                getattr(player, score_field) is None,
                -(getattr(player, score_field) or 0.0),
                player.name.casefold(),
                player.player_id,
            ),
        )
    )
    return ordered if limit is None else ordered[:limit]
'''


def export_current_players(
    root: Path,
    current_season: int = CURRENT_SEASON,
) -> Path:
    """Generate a Python tuple containing every player in the current roster file."""
    preseason = pd.read_parquet(  # (n_preseason_players, c_roster)
        root / "data" / "processed" / "preseason_players.parquet"
    )
    current_mask = preseason["season"].eq(current_season)  # (n_preseason_players,)
    roster = preseason[current_mask].copy()  # (n_current_players, c_roster)
    if roster.empty:
        raise ValueError(f"Processed roster has no rows for season {current_season}.")
    predictions = pd.read_parquet(  # (n_fantasy_players, c_prediction)
        root / "artifacts" / f"predictions_{current_season}.parquet"
    )
    projection_columns = [
        "player_id",
        "model_position",
        "selected_atoms",
        "model_feature_columns",
        "predicted_points",
        "previous_points_baseline",
        "prediction_interval_80_low",
        "prediction_interval_80_high",
        "position_rank",
        "overall_point_rank",
        "team_position_rank",
        "model_name",
        "scoring_profile",
        "feature_coverage",
        "out_of_distribution",
        *[
            column
            for column in predictions.columns
            if column.startswith("feature_value__")
        ],
    ]
    roster = roster.merge(  # (n_current_players, c_export)
        predictions[projection_columns],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    fantasy_mask = roster["position"].isin(FANTASY_POSITIONS)  # (n_current_players,)
    missing_predictions = roster.loc[  # (n_fantasy_players,)
        fantasy_mask,
        "predicted_points",
    ].isna()
    if missing_predictions.any():
        missing_fantasy_mask = fantasy_mask.copy()  # (n_current_players,)
        missing_fantasy_mask.loc[fantasy_mask] = (  # (n_fantasy_players,)
            missing_predictions.to_numpy()
        )
        missing_ids = roster.loc[missing_fantasy_mask, "player_id"].tolist()
        raise RuntimeError(
            f"Missing predictions for {len(missing_ids)} fantasy players: "
            f"{missing_ids[:5]}."
        )
    roster.sort_values(  # (n_current_players, c_export)
        ["team", "position", "full_name", "player_id"],
        inplace=True,
    )

    manifest_path = root / "data" / "raw" / "manifest.json"
    data_cutoff = "unknown"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_cutoff = manifest.get("data_cutoff_utc", manifest["created_at_utc"])

    build_summary_path = root / "data" / "processed" / "build_summary.json"
    snapshot_date = "unknown"
    if build_summary_path.is_file():
        build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
        snapshot_date = str(build_summary.get("snapshot_date", "unknown"))

    lines = [_module_header(current_season, data_cutoff, snapshot_date)]
    roster_records = roster.to_dict("records")  # (n_current_players,)
    for row in roster_records:
        position = str(row["position"])
        fantasy_eligible = position in FANTASY_POSITIONS
        row_feature_names = tuple(
            name
            for name in str(_value(row.get("model_feature_columns")) or "").split(",")
            if name
        )
        feature_values = tuple(
            (
                name,
                _feature_value(row.get(f"feature_value__{name}")),
            )
            for name in row_feature_names
        )
        fields = {
            "player_id": str(row["player_id"]),
            "name": str(row["full_name"]),
            "team": str(row["team"]),
            "position": position,
            "model_position": _value(row.get("model_position"))
            or MODEL_POSITION.get(position),
            "fantasy_eligible": fantasy_eligible,
            "roster_status": _value(row.get("status")),
            "jersey_number": _value(row.get("jersey_number"), integer=True),
            "depth_tier": _value(row.get("depth_tier"), integer=True),
            "depth_rank": _value(row.get("depth_rank")),
            "years_experience": _value(row.get("years_exp"), integer=True),
            "age": _age(row.get("birth_date"), current_season),
            "college": _value(row.get("college")),
            "birth_date": _value(row.get("birth_date")),
            "height_inches": _value(row.get("height"), integer=True),
            "weight_pounds": _value(row.get("weight"), integer=True),
            "rookie_year": _value(row.get("rookie_year"), integer=True),
            "draft_year": _value(row.get("draft_year"), integer=True),
            "draft_round": _value(row.get("draft_round"), integer=True),
            "draft_number": _value(row.get("draft_number"), integer=True),
            "draft_team": _value(row.get("draft_club")),
            "espn_id": _value(row.get("espn_id")),
            "sleeper_id": _value(row.get("sleeper_id")),
            "pfr_id": _value(row.get("pfr_id")),
            "headshot_url": _value(row.get("headshot_url")),
            "predicted_points": _value(row.get("predicted_points")),
            "previous_season_points": _value(row.get("previous_points_baseline")),
            "prediction_interval_80_low": _value(row.get("prediction_interval_80_low")),
            "prediction_interval_80_high": _value(
                row.get("prediction_interval_80_high")
            ),
            "position_rank": _value(row.get("position_rank"), integer=True),
            "overall_point_rank": _value(row.get("overall_point_rank"), integer=True),
            "team_position_rank": _value(row.get("team_position_rank"), integer=True),
            "model_name": _value(row.get("model_name")),
            "scoring_profile": _value(row.get("scoring_profile")),
            "selected_atoms": tuple(
                atom
                for atom in str(_value(row.get("selected_atoms")) or "").split(",")
                if atom
            ),
            "feature_values": feature_values,
            "feature_coverage": _value(row.get("feature_coverage")),
            "out_of_distribution": _value(row.get("out_of_distribution")),
            "confidence": (
                "not_applicable"
                if not fantasy_eligible
                else "not_available"
                if str(row.get("status") or "").upper() in {"CUT", "RET"}
                else "low_kicker_rank_evidence"
                if MODEL_POSITION.get(position) == "K"
                else "limited_rookie_history"
                if _value(row.get("rookie_year"), integer=True) == current_season
                else "out_of_range"
                if bool(_value(row.get("out_of_distribution")) or False)
                else "retrospective"
            ),
        }
        arguments = ", ".join(f"{name}={value!r}" for name, value in fields.items())
        lines.append(f"    PlayerProjection({arguments}),\n")
    lines.append(_module_footer())

    output_path = root / "fantasy_football" / f"players_{current_season}.py"
    output_path.write_text("".join(lines), encoding="utf-8")
    roster.to_parquet(
        root / "artifacts" / f"players_{current_season}.parquet", index=False
    )
    synchronize_project_sizes(root)
    return output_path
