"""Versioned fractional fantasy scoring reconstructed from raw game events."""

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from .constants import KICKER_POSITIONS


@dataclass(frozen=True, slots=True)
class ScoringProfile:
    """Points assigned to each raw scoring event."""

    name: str = "espn_full_ppr_2026"
    passing_yard: float = 0.04
    passing_td: float = 4.0
    passing_interception: float = -2.0
    rushing_yard: float = 0.1
    rushing_td: float = 6.0
    reception: float = 1.0
    receiving_yard: float = 0.1
    receiving_td: float = 6.0
    two_point_conversion: float = 2.0
    fumble_lost: float = -2.0
    return_td: float = 6.0
    fumble_recovery_td: float = 6.0
    defensive_td: float = 6.0
    pat_made: float = 1.0
    fg_made_0_39: float = 3.0
    fg_made_40_49: float = 4.0
    fg_made_50_59: float = 5.0
    fg_made_60_plus: float = 6.0
    fg_missed: float = -1.0


REQUIRED_SCORING_COLUMNS = (
    "position",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
    "fumbles_lost_total",
    "special_teams_tds",
    "fumble_recovery_tds",
    "pat_made",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "fg_missed",
)


def load_scoring_profile(configuration_path: Path) -> ScoringProfile:
    """Load the declared coefficients from a versioned JSON file."""
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    profile_fields = {field.name for field in fields(ScoringProfile)}
    missing = sorted(profile_fields - configuration.keys())
    if missing:
        raise ValueError(f"Scoring configuration is missing fields: {missing}.")
    values = {name: configuration[name] for name in profile_fields}
    return ScoringProfile(**values)


def scoring_profile_digest(configuration_path: Path) -> str:
    """Return the SHA-256 digest that identifies a scoring configuration."""
    return hashlib.sha256(configuration_path.read_bytes()).hexdigest()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric scoring column with missing values set to zero."""
    # frame: (n_games, c_game); return: (n_games,)
    if column not in frame:
        return pd.Series(0.0, index=frame.index, dtype="float64")  # (n_games,)

    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)  # (n_games,)


def score_player_games(
    player_games: pd.DataFrame,
    profile: ScoringProfile | None = None,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Add non-PPR, half-PPR, full-PPR, kicker, and primary point totals."""
    # player_games: (n_games, c_game)
    if strict:
        missing = sorted(set(REQUIRED_SCORING_COLUMNS) - set(player_games.columns))
        if missing:
            raise ValueError(
                f"Player-game input is missing scoring columns: {missing}."
            )

    scoring = profile or ScoringProfile()
    scored_games = player_games.copy()  # (n_games, c_game)

    two_point_conversions = (  # (n_games,)
        _numeric(scored_games, "passing_2pt_conversions")
        + _numeric(scored_games, "rushing_2pt_conversions")
        + _numeric(scored_games, "receiving_2pt_conversions")
    )
    base_points = (  # (n_games,)
        scoring.passing_yard * _numeric(scored_games, "passing_yards")
        + scoring.passing_td * _numeric(scored_games, "passing_tds")
        + scoring.passing_interception * _numeric(scored_games, "passing_interceptions")
        + scoring.rushing_yard * _numeric(scored_games, "rushing_yards")
        + scoring.rushing_td * _numeric(scored_games, "rushing_tds")
        + scoring.receiving_yard * _numeric(scored_games, "receiving_yards")
        + scoring.receiving_td * _numeric(scored_games, "receiving_tds")
        + scoring.two_point_conversion * two_point_conversions
        + scoring.fumble_lost * _numeric(scored_games, "fumbles_lost_total")
        + scoring.return_td * _numeric(scored_games, "special_teams_tds")
        + scoring.fumble_recovery_td * _numeric(scored_games, "fumble_recovery_tds")
        + scoring.defensive_td * _numeric(scored_games, "individual_defensive_tds")
    )
    receptions = _numeric(scored_games, "receptions")  # (n_games,)
    kicker_points = (  # (n_games,)
        scoring.pat_made * _numeric(scored_games, "pat_made")
        + scoring.fg_made_0_39
        * (
            _numeric(scored_games, "fg_made_0_19")
            + _numeric(scored_games, "fg_made_20_29")
            + _numeric(scored_games, "fg_made_30_39")
        )
        + scoring.fg_made_40_49 * _numeric(scored_games, "fg_made_40_49")
        + scoring.fg_made_50_59 * _numeric(scored_games, "fg_made_50_59")
        + scoring.fg_made_60_plus * _numeric(scored_games, "fg_made_60_")
        + scoring.fg_missed * _numeric(scored_games, "fg_missed")
    )

    scored_games["recomputed_non_ppr_points"] = base_points  # (n_games,)
    scored_games["recomputed_half_ppr_points"] = (  # (n_games,)
        base_points + 0.5 * scoring.reception * receptions
    )
    scored_games["recomputed_ppr_points"] = (  # (n_games,)
        base_points + scoring.reception * receptions
    )
    scored_games["recomputed_kicker_points"] = kicker_points  # (n_games,)

    is_kicker = scored_games["position"].isin(KICKER_POSITIONS)  # (n_games,)
    scored_games["target_points"] = scored_games[  # (n_games,)
        "recomputed_ppr_points"
    ]
    scored_games.loc[is_kicker, "target_points"] = (  # (n_kicker_games,)
        scored_games.loc[is_kicker, "recomputed_ppr_points"]
        + scored_games.loc[is_kicker, "recomputed_kicker_points"]
    )

    return scored_games  # (n_games, c_scored)
