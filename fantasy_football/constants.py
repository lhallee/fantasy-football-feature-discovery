"""Stable project constants and declared raw feature atoms."""

from pathlib import Path


CURRENT_SEASON = 2026
FIRST_STAT_SEASON = 1999
FIRST_ROSTER_SEASON = 2002
SNAPSHOT_MONTH = 8
SNAPSHOT_DAY = 9
MAX_PROJECT_DATA_BYTES = 900 * 1024**2

NFLVERSE_RELEASE_ROOT = "https://github.com/nflverse/nflverse-data/releases/download"

FANTASY_POSITIONS = frozenset({"QB", "RB", "FB", "HB", "WR", "TE", "K", "PK"})
OFFENSE_POSITIONS = frozenset({"QB", "RB", "FB", "HB", "WR", "TE"})
KICKER_POSITIONS = frozenset({"K", "PK"})
MODEL_POSITION = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "HB": "RB",
    "WR": "WR",
    "TE": "TE",
    "K": "K",
    "PK": "K",
}

RAW_STAT_COLUMNS = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "passing_2pt_conversions",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles_lost",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_2pt_conversions",
    "special_teams_tds",
    "fumble_recovery_tds",
    "fumbles_lost_total",
    "fg_made",
    "fg_att",
    "fg_missed",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "pat_made",
    "pat_att",
)

GAME_ID_COLUMNS = (
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "position_group",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
)

SOURCE_SCORE_COLUMNS = ("fantasy_points", "fantasy_points_ppr")
GAME_COLUMNS = GAME_ID_COLUMNS + RAW_STAT_COLUMNS + SOURCE_SCORE_COLUMNS

OFFENSE_STAT_ATOMS = (
    "games",
    "active_roster_weeks",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "fumbles_lost_total",
)

UTILIZATION_ATOMS = ("offense_snaps",)

COMBINE_ATOMS = (
    "combine_forty",
    "combine_bench",
    "combine_vertical",
    "combine_broad_jump",
    "combine_cone",
    "combine_shuttle",
)

KICKER_STAT_ATOMS = (
    "games",
    "active_roster_weeks",
    "fg_att",
    "fg_made",
    "fg_missed",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "pat_att",
    "pat_made",
)

METADATA_ATOMS = (
    "age",
    "years_exp",
    "height",
    "weight",
    "draft_number",
    "was_drafted",
    "is_rookie",
    "team_changed",
)

STATUS_PRIORITY = {
    "ACT": 0,
    "RES": 1,
    "DEV": 2,
    "E14": 3,
    "SUS": 4,
    "NWT": 5,
    "CUT": 6,
    "RET": 7,
}


def default_paths(root: Path) -> dict[str, Path]:
    """Return the project paths rooted at ``root``."""
    return {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "artifacts": root / "artifacts",
        "docs": root / "docs",
    }
