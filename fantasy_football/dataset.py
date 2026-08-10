"""Build compact player-game, player-season, and next-season modeling tables."""

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .constants import COMBINE_ATOMS, CURRENT_SEASON, FANTASY_POSITIONS
from .constants import GAME_COLUMNS, MAX_PROJECT_DATA_BYTES
from .constants import MODEL_POSITION, OFFENSE_POSITIONS, RAW_STAT_COLUMNS
from .constants import SNAPSHOT_DAY, SNAPSHOT_MONTH, STATUS_PRIORITY
from .provenance import project_data_bytes, synchronize_project_sizes
from .scoring import REQUIRED_SCORING_COLUMNS, ScoringProfile, load_scoring_profile
from .scoring import score_player_games
from .scoring import scoring_profile_digest


ROSTER_COLUMNS = (
    "season",
    "team",
    "position",
    "depth_chart_position",
    "jersey_number",
    "status",
    "full_name",
    "birth_date",
    "height",
    "weight",
    "college",
    "gsis_id",
    "espn_id",
    "pfr_id",
    "sleeper_id",
    "years_exp",
    "headshot_url",
    "week",
    "game_type",
    "entry_year",
    "rookie_year",
    "draft_club",
    "draft_number",
)

PLAYER_COLUMNS = (
    "gsis_id",
    "display_name",
    "pfr_id",
    "espn_id",
    "birth_date",
    "position_group",
    "position",
    "height",
    "weight",
    "college_name",
    "rookie_season",
    "latest_team",
    "status",
    "draft_year",
    "draft_round",
    "draft_pick",
    "draft_team",
)


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """Key dataset counts and quality checks."""

    current_season: int
    player_game_rows: int
    player_game_players: int
    unidentified_game_rows_dropped: int
    player_season_rows: int
    modeling_rows: int
    current_roster_players: int
    current_fantasy_players: int
    recomputed_ppr_exact_fraction: float
    recomputed_ppr_mean_absolute_delta: float
    depth_tier_coverage: float
    snap_id_match_fraction: float
    snapshot_date: str
    scoring_profile: str
    scoring_config_sha256: str
    target_period: str
    total_project_data_bytes: int
    built_at_utc: str


def _existing_columns(path: Path, requested: tuple[str, ...]) -> list[str]:
    available = set(pq.read_schema(path).names)
    return [column for column in requested if column in available]


def _load_player_games(
    raw_dir: Path,
    scoring_profile: ScoringProfile,
) -> tuple[pd.DataFrame, int]:
    # Return player_games: (n_games, c_scored).
    season_frames: list[pd.DataFrame] = []  # each: (n_season_games, c_scored)
    for path in sorted((raw_dir / "stats").glob("stats_player_week_*.parquet")):
        columns = _existing_columns(path, GAME_COLUMNS)
        missing_required = sorted(set(REQUIRED_SCORING_COLUMNS) - set(columns))
        if missing_required:
            raise ValueError(
                f"{path.name} is missing scoring columns: {missing_required}."
            )
        season_frame = pd.read_parquet(  # (n_season_games, c_available)
            path,
            columns=columns,
        )
        for column in GAME_COLUMNS:
            if column not in season_frame:
                season_frame[column] = np.nan  # (n_season_games,)
        scoring_frame = season_frame.loc[  # (n_season_games, c_game)
            :,
            GAME_COLUMNS,
        ]
        scored_season = score_player_games(  # (n_season_games, c_scored)
            scoring_frame,
            scoring_profile,
        )
        season_frames.append(scored_season)

    if not season_frames:
        raise FileNotFoundError(f"No weekly player statistics found below {raw_dir}.")

    player_games = pd.concat(season_frames, ignore_index=True)  # (n_games, c_scored)
    unidentified_rows = int(player_games["player_id"].isna().sum())
    identified = player_games["player_id"].notna()  # (n_games,)
    player_games = player_games[identified].copy()  # (n_identified_games, c_scored)
    player_games.sort_values(  # (n_identified_games, c_scored)
        ["season", "week", "game_id", "player_id"],
        inplace=True,
    )
    player_games.reset_index(  # (n_identified_games, c_scored)
        drop=True,
        inplace=True,
    )
    return player_games, unidentified_rows


def _aggregate_player_seasons(player_games: pd.DataFrame) -> pd.DataFrame:
    # player_games: (n_games, c_scored)
    regular_mask = player_games["season_type"].eq("REG")  # (n_games,)
    regular_games = player_games[regular_mask].copy()  # (n_regular_games, c_scored)
    numeric_columns = list(RAW_STAT_COLUMNS) + [
        "recomputed_non_ppr_points",
        "recomputed_half_ppr_points",
        "recomputed_ppr_points",
        "recomputed_kicker_points",
        "target_points",
    ]
    grouped = regular_games.groupby(["season", "player_id"], observed=True)
    player_seasons = (  # (n_player_seasons, 2 + n_numeric)
        grouped[numeric_columns].sum(min_count=1).reset_index()
    )
    player_seasons["games"] = (
        grouped["game_id"].nunique().to_numpy()
    )  # (n_player_seasons,)

    ordered_games = regular_games.sort_values(  # (n_regular_games, c_scored)
        ["season", "player_id", "week"]
    )
    last_rows = ordered_games.groupby(  # (n_player_seasons, c_scored)
        ["season", "player_id"],
        observed=True,
    ).tail(1)
    identity = last_rows[  # (n_player_seasons, 5)
        ["season", "player_id", "player_display_name", "position", "team"]
    ].rename(columns={"team": "last_team"})
    player_seasons = player_seasons.merge(  # (n_player_seasons, c_season)
        identity,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    return player_seasons


def _deduplicate_week_one(roster: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # roster: (n_roster_week_rows, c_roster)
    week_one_mask = (  # (n_roster_week_rows,)
        roster["week"].eq(1) & roster["game_type"].eq("REG")
    )
    week_one = roster[week_one_mask].copy()  # (n_week_one_rows, c_roster)
    identified = week_one["gsis_id"].notna()  # (n_week_one_rows,)
    week_one = week_one[identified].copy()  # (n_identified_week_one, c_roster)
    duplicate_rows = int(week_one.duplicated("gsis_id", keep=False).sum())
    week_one["status_priority"] = (  # (n_identified_week_one,)
        week_one["status"].map(STATUS_PRIORITY).fillna(99).astype("int16")
    )
    week_one.sort_values(
        ["gsis_id", "status_priority", "team"],
        inplace=True,
        kind="stable",
    )  # (n_identified_week_one, c_roster + 1)
    week_one.drop_duplicates(  # (n_unique_week_one_players, c_roster + 1)
        "gsis_id",
        keep="first",
        inplace=True,
    )
    week_one.drop(  # (n_unique_week_one_players, c_roster)
        columns="status_priority",
        inplace=True,
    )
    return week_one, duplicate_rows


def _load_roster_tables(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    preseason_frames: list[pd.DataFrame] = []  # each: (n_season_players, c_roster)
    availability_frames: list[pd.DataFrame] = []  # each: (n_player_seasons, 4)
    duplicate_rows = 0
    for path in sorted((raw_dir / "rosters").glob("roster_weekly_*.parquet")):
        columns = _existing_columns(path, ROSTER_COLUMNS)
        roster = pd.read_parquet(
            path, columns=columns
        )  # (n_roster_week_rows, c_available)
        for column in ROSTER_COLUMNS:
            if column not in roster:
                roster[column] = np.nan  # (n_roster_week_rows,)
        roster = roster.loc[:, ROSTER_COLUMNS]  # (n_roster_week_rows, c_roster)

        week_one, season_duplicates = _deduplicate_week_one(roster)
        duplicate_rows += season_duplicates
        preseason_frames.append(week_one)

        regular_mask = roster["game_type"].eq("REG")  # (n_roster_week_rows,)
        regular_roster = roster[
            regular_mask
        ].copy()  # (n_regular_roster_rows, c_roster)
        roster_weeks = (  # (n_player_seasons,)
            regular_roster.groupby(["season", "gsis_id"], observed=True)["week"]
            .nunique()
            .rename("roster_weeks")
        )
        active_weeks = (  # (n_active_player_seasons,)
            regular_roster[regular_roster["status"].eq("ACT")]
            .groupby(["season", "gsis_id"], observed=True)["week"]
            .nunique()
            .rename("active_roster_weeks")
        )
        availability = pd.concat(  # (n_player_seasons, 2)
            [roster_weeks, active_weeks],
            axis=1,
        ).fillna(0)
        availability_frame = availability.reset_index()  # (n_player_seasons, 4)
        availability_frames.append(availability_frame)

    if not preseason_frames:
        raise FileNotFoundError(f"No weekly rosters found below {raw_dir}.")

    preseason = pd.concat(  # (n_preseason_players, c_roster)
        preseason_frames,
        ignore_index=True,
    )
    availability = pd.concat(  # (n_availability_rows, 4)
        availability_frames,
        ignore_index=True,
    )
    availability.rename(  # (n_availability_rows, 4)
        columns={"gsis_id": "player_id"},
        inplace=True,
    )
    return preseason, availability, duplicate_rows


def _new_depth_tier(
    depth: pd.DataFrame,
    season: int,
    snapshot_month: int,
    snapshot_day: int,
) -> pd.DataFrame:
    # depth: (n_depth_rows, c_depth)
    dates = pd.to_datetime(depth["dt"], utc=True, errors="coerce")  # (n_depth_rows,)
    cutoff = pd.Timestamp(
        year=season,
        month=snapshot_month,
        day=snapshot_day,
        hour=23,
        minute=59,
        second=59,
        tz="UTC",
    )
    eligible_mask = dates.le(cutoff)  # (n_depth_rows,)
    eligible = depth[eligible_mask].copy()  # (n_eligible_depth_rows, c_depth)
    if eligible.empty:
        return pd.DataFrame(  # (0, 4)
            columns=["season", "gsis_id", "depth_rank", "depth_tier"]
        )

    latest_timestamp = eligible["dt"].max()
    latest_mask = eligible["dt"].eq(latest_timestamp)  # (n_eligible_depth_rows,)
    snapshot = eligible[latest_mask].copy()  # (n_snapshot_depth_rows, c_depth)
    snapshot["depth_position"] = snapshot[
        "pos_abb"
    ].replace(  # (n_snapshot_depth_rows,)
        {"PK": "K"}
    )
    fantasy_mask = snapshot["depth_position"].isin(  # (n_snapshot_depth_rows,)
        FANTASY_POSITIONS
    )
    snapshot = snapshot[fantasy_mask].copy()  # (n_fantasy_depth_rows, c_depth)
    snapshot["depth_rank"] = pd.to_numeric(  # (n_fantasy_depth_rows,)
        snapshot["pos_rank"], errors="coerce"
    )
    snapshot["depth_tier"] = snapshot["depth_rank"].clip(  # (n_fantasy_depth_rows,)
        upper=3
    )
    is_receiver = snapshot["depth_position"].eq("WR")  # (n_fantasy_depth_rows,)
    snapshot.loc[is_receiver, "depth_tier"] = np.ceil(  # (n_receiver_depth_rows,)
        snapshot.loc[is_receiver, "depth_rank"] / 3.0
    ).clip(upper=3)
    snapshot.sort_values(  # (n_fantasy_depth_rows, c_depth_enriched)
        ["gsis_id", "depth_tier", "depth_rank"],
        inplace=True,
    )
    snapshot.drop_duplicates(  # (n_snapshot_players, c_depth_enriched)
        "gsis_id",
        keep="first",
        inplace=True,
    )
    snapshot["season"] = season  # (n_snapshot_players,)
    return snapshot[  # (n_snapshot_players, 4)
        ["season", "gsis_id", "depth_rank", "depth_tier"]
    ]


def _old_depth_tier(depth: pd.DataFrame, season: int) -> pd.DataFrame:
    # depth: (n_depth_rows, c_depth)
    week_one_mask = depth["week"].eq(1)  # (n_depth_rows,)
    snapshot = depth[week_one_mask].copy()  # (n_week_one_depth_rows, c_depth)
    fantasy_mask = (  # (n_week_one_depth_rows,)
        snapshot["formation"].isin(["Offense", "Special Teams"])
        & snapshot["position"].isin(FANTASY_POSITIONS)
    )
    snapshot = snapshot[fantasy_mask].copy()  # (n_fantasy_depth_rows, c_depth)
    snapshot["depth_rank"] = pd.to_numeric(  # (n_fantasy_depth_rows,)
        snapshot["depth_team"], errors="coerce"
    )
    snapshot["depth_tier"] = snapshot["depth_rank"].clip(  # (n_fantasy_depth_rows,)
        upper=3
    )
    snapshot.sort_values(  # (n_fantasy_depth_rows, c_depth_enriched)
        ["gsis_id", "depth_tier"],
        inplace=True,
    )
    snapshot.drop_duplicates(  # (n_snapshot_players, c_depth_enriched)
        "gsis_id",
        keep="first",
        inplace=True,
    )
    snapshot["season"] = season  # (n_snapshot_players,)
    return snapshot[  # (n_snapshot_players, 4)
        ["season", "gsis_id", "depth_rank", "depth_tier"]
    ]


def _load_depth_tiers(
    raw_dir: Path,
    snapshot_month: int,
    snapshot_day: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []  # each: (n_season_players, 4)
    for path in sorted((raw_dir / "depth_charts").glob("depth_charts_*.parquet")):
        season = int(path.stem.rsplit("_", maxsplit=1)[1])
        depth = pd.read_parquet(path)  # (n_depth_rows, c_depth)
        frame = (  # (n_season_players, 4)
            _new_depth_tier(depth, season, snapshot_month, snapshot_day)
            if "dt" in depth
            else _old_depth_tier(depth, season)
        )
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No depth charts found below {raw_dir}.")
    return pd.concat(frames, ignore_index=True)  # (n_depth_players, 4)


def _load_players(raw_dir: Path) -> pd.DataFrame:
    players = pd.read_parquet(  # (n_players, c_player)
        raw_dir / "players.parquet",
        columns=list(PLAYER_COLUMNS),
    )
    players.rename(
        columns={
            "gsis_id": "player_id",
            "display_name": "master_name",
            "pfr_id": "master_pfr_id",
            "espn_id": "master_espn_id",
            "birth_date": "master_birth_date",
            "position": "master_position",
            "height": "master_height",
            "weight": "master_weight",
            "college_name": "master_college",
            "rookie_season": "master_rookie_season",
            "status": "master_status",
            "draft_pick": "master_draft_number",
            "draft_team": "master_draft_team",
        },
        inplace=True,
    )  # (n_players, c_player)
    return players  # (n_players, c_player)


def _load_snap_counts(
    raw_dir: Path, players: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    # players: (n_players, c_player)
    frames: list[pd.DataFrame] = []  # each: (n_regular_snap_rows, 4)
    for path in sorted((raw_dir / "snap_counts").glob("snap_counts_*.parquet")):
        frame = pd.read_parquet(  # (n_snap_rows, 4)
            path,
            columns=[
                "season",
                "game_type",
                "pfr_player_id",
                "offense_snaps",
            ],
        )
        regular_mask = frame["game_type"].eq("REG")  # (n_snap_rows,)
        regular_frame = frame[regular_mask]  # (n_regular_snap_rows, 4)
        frames.append(regular_frame)

    if not frames:
        return pd.DataFrame(  # (0, 3)
            columns=["season", "player_id", "offense_snaps"]
        ), 0.0

    snaps = pd.concat(frames, ignore_index=True)  # (n_snap_rows, 4)
    pfr_crosswalk = (  # (n_unique_pfr_players, 2)
        players[["player_id", "master_pfr_id"]]
        .dropna()
        .drop_duplicates("master_pfr_id")
    )
    snaps = snaps.merge(  # (n_snap_rows, c_snap_joined)
        pfr_crosswalk,
        left_on="pfr_player_id",
        right_on="master_pfr_id",
        how="left",
        validate="many_to_one",
    )
    match_fraction = float(snaps["player_id"].notna().mean())
    matched = snaps["player_id"].notna()  # (n_snap_rows,)
    snaps = snaps[matched].copy()  # (n_matched_snap_rows, c_snap_joined)
    seasonal_snaps = (  # (n_player_seasons, 3)
        snaps.groupby(["season", "player_id"], observed=True)["offense_snaps"]
        .sum(min_count=1)
        .reset_index()
    )
    return seasonal_snaps, match_fraction


def _load_combine(raw_dir: Path, players: pd.DataFrame) -> pd.DataFrame:
    # players: (n_players, c_player)
    combine = pd.read_parquet(  # (n_combine_rows, 7)
        raw_dir / "combine.parquet",
        columns=[
            "pfr_id",
            "forty",
            "bench",
            "vertical",
            "broad_jump",
            "cone",
            "shuttle",
        ],
    )
    combine.rename(
        columns={
            "pfr_id": "master_pfr_id",
            "forty": "combine_forty",
            "bench": "combine_bench",
            "vertical": "combine_vertical",
            "broad_jump": "combine_broad_jump",
            "cone": "combine_cone",
            "shuttle": "combine_shuttle",
        },
        inplace=True,
    )  # (n_combine_rows, 7)
    combine.sort_values(  # (n_combine_rows, 7)
        "master_pfr_id",
        inplace=True,
    )
    combine.drop_duplicates(  # (n_unique_combine_players, 7)
        "master_pfr_id",
        keep="last",
        inplace=True,
    )
    player_crosswalk = players[  # (n_players, 2)
        ["player_id", "master_pfr_id"]
    ]
    player_combine = player_crosswalk.merge(
        combine,
        on="master_pfr_id",
        how="left",
        validate="many_to_one",
    )[["player_id", *COMBINE_ATOMS]]  # (n_players, 1 + n_combine_atoms)
    return player_combine  # (n_players, 1 + n_combine_atoms)


def _fill_roster_metadata(
    preseason: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:
    # preseason: (n_preseason_players, c_roster); players: (n_players, c_player)
    roster = preseason.rename(
        columns={"gsis_id": "player_id"}
    ).merge(  # (n_preseason_players, c_joined)
        players,
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    fill_pairs = {
        "full_name": "master_name",
        "pfr_id": "master_pfr_id",
        "espn_id": "master_espn_id",
        "birth_date": "master_birth_date",
        "position": "master_position",
        "height": "master_height",
        "weight": "master_weight",
        "college": "master_college",
        "rookie_year": "master_rookie_season",
        "draft_number": "master_draft_number",
        "draft_club": "master_draft_team",
    }
    for roster_column, master_column in fill_pairs.items():
        roster[roster_column] = roster[
            roster_column
        ].combine_first(  # (n_preseason_players,)
            roster[master_column]
        )
    return roster  # (n_preseason_players, c_joined)


def _age_on_september_first(birth_dates: pd.Series, seasons: pd.Series) -> pd.Series:
    # birth_dates, seasons, birth, reference, return: (n_players,)
    birth = pd.to_datetime(birth_dates, errors="coerce")  # (n_players,)
    reference = pd.to_datetime(  # (n_players,)
        seasons.astype("Int64").astype(str) + "-09-01",
        errors="coerce",
    )
    return (reference - birth).dt.days / 365.2425  # (n_players,)


def _build_modeling_table(
    preseason: pd.DataFrame,
    player_seasons: pd.DataFrame,
    combine: pd.DataFrame,
    current_season: int,
) -> pd.DataFrame:
    # preseason: (n_preseason_players, c_roster)
    # player_seasons: (n_player_seasons, c_season); combine: (n_players, c_combine)
    fantasy_mask = preseason["position"].isin(
        FANTASY_POSITIONS
    )  # (n_preseason_players,)
    modeling = preseason[fantasy_mask].copy()  # (n_model_rows, c_roster)
    modeling["model_position"] = modeling["position"].map(  # (n_model_rows,)
        MODEL_POSITION
    )
    modeling.rename(  # (n_model_rows, c_roster)
        columns={"season": "target_season"},
        inplace=True,
    )
    modeling["age"] = _age_on_september_first(  # (n_model_rows,)
        modeling["birth_date"], modeling["target_season"]
    )
    modeling["draft_number"] = pd.to_numeric(  # (n_model_rows,)
        modeling["draft_number"], errors="coerce"
    )
    modeling["was_drafted"] = (
        modeling["draft_number"]
        .notna()
        .astype(  # (n_model_rows,)
            "int8"
        )
    )
    modeling["draft_number"] = modeling["draft_number"].fillna(300.0)  # (n_model_rows,)
    modeling["is_rookie"] = (  # (n_model_rows,)
        modeling["target_season"].eq(
            pd.to_numeric(modeling["rookie_year"], errors="coerce")
        )
        | pd.to_numeric(modeling["years_exp"], errors="coerce").fillna(-1).eq(0)
    ).astype("int8")
    modeling["status_active"] = (
        modeling["status"]
        .eq("ACT")
        .astype(  # (n_model_rows,)
            "int8"
        )
    )
    modeling = modeling.merge(  # (n_model_rows, c_modeling)
        combine, on="player_id", how="left", validate="many_to_one"
    )

    lag_columns = [
        "player_id",
        "season",
        "last_team",
        "games",
        "roster_weeks",
        "active_roster_weeks",
        "offense_snaps",
        *RAW_STAT_COLUMNS,
        "target_points",
    ]
    for lag in (1, 2):
        lagged = player_seasons.reindex(  # (n_player_seasons, c_lag_source)
            columns=lag_columns
        ).copy()
        lagged["target_season"] = lagged["season"] + lag  # (n_player_seasons,)
        lagged.drop(  # (n_player_seasons, c_lag_source)
            columns="season",
            inplace=True,
        )
        lagged.rename(
            columns={
                column: f"lag{lag}_{column}"
                for column in lagged.columns
                if column not in {"player_id", "target_season"}
            },
            inplace=True,
        )  # (n_player_seasons, c_lag_source)
        modeling = modeling.merge(  # (n_model_rows, c_modeling)
            lagged,
            on=["player_id", "target_season"],
            how="left",
            validate="one_to_one",
        )

    target = player_seasons[
        ["player_id", "season", "target_points"]
    ].rename(  # (n_player_seasons, 3)
        columns={"season": "target_season", "target_points": "target_points"}
    )
    modeling = modeling.merge(  # (n_model_rows, c_modeling)
        target,
        on=["player_id", "target_season"],
        how="left",
        validate="one_to_one",
    )
    completed = modeling["target_season"].lt(current_season)  # (n_model_rows,)
    modeling.loc[completed, "target_points"] = modeling.loc[  # (n_completed_rows,)
        completed, "target_points"
    ].fillna(0.0)

    lag_count_columns = [
        column
        for column in modeling.columns
        if column.startswith(("lag1_", "lag2_"))
        and column not in {"lag1_last_team", "lag2_last_team"}
    ]
    modeling[lag_count_columns] = modeling[
        lag_count_columns
    ].fillna(  # (n_model_rows, n_lag_count_columns)
        0.0
    )
    modeling["team_changed"] = (  # (n_model_rows,)
        modeling["lag1_last_team"].notna()
        & modeling["team"].ne(modeling["lag1_last_team"])
    ).astype("int8")
    modeling["previous_points_baseline"] = modeling[
        "lag1_target_points"
    ].fillna(  # (n_model_rows,)
        0.0
    )
    modeling.sort_values(  # (n_model_rows, c_modeling)
        ["target_season", "model_position", "player_id"],
        inplace=True,
    )
    modeling.reset_index(  # (n_model_rows, c_modeling)
        drop=True,
        inplace=True,
    )
    return modeling  # (n_model_rows, c_modeling)


def _normalize_object_columns(frame: pd.DataFrame) -> None:
    """Make mixed historical text columns Arrow-safe without changing missing values."""
    # frame: (n_rows, c_columns); each selected column: (n_rows,)
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].map(  # (n_rows,)
            lambda value: None if value is None or pd.isna(value) else str(value)
        )


def build_datasets(
    root: Path,
    current_season: int = CURRENT_SEASON,
    snapshot_date: date | None = None,
) -> BuildSummary:
    """Build all compact tables and fail if the project data exceeds 900 MiB."""
    snapshot = snapshot_date or date(current_season, SNAPSHOT_MONTH, SNAPSHOT_DAY)
    if snapshot.year != current_season:
        raise ValueError(
            f"Snapshot year {snapshot.year} must match season {current_season}."
        )

    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    scoring_path = root / "config" / "scoring_espn_full_ppr_2026.json"
    scoring_profile = load_scoring_profile(scoring_path)
    player_games, unidentified_game_rows = (
        _load_player_games(  # (n_games, c_scored), scalar
            raw_dir,
            scoring_profile,
        )
    )
    player_seasons = _aggregate_player_seasons(  # (n_player_seasons, c_season)
        player_games
    )
    preseason, availability, duplicate_rows = (
        _load_roster_tables(  # (n_preseason_players, c_roster), (n_availability_rows, 4), scalar
            raw_dir
        )
    )
    players = _load_players(raw_dir)  # (n_players, c_player)
    preseason = _fill_roster_metadata(  # (n_preseason_players, c_joined)
        preseason,
        players,
    )
    depth_tiers = _load_depth_tiers(  # (n_depth_players, 4)
        raw_dir,
        snapshot.month,
        snapshot.day,
    )
    preseason = preseason.merge(  # (n_preseason_players, c_preseason)
        depth_tiers,
        left_on=["season", "player_id"],
        right_on=["season", "gsis_id"],
        how="left",
        validate="one_to_one",
    )
    preseason.drop(  # (n_preseason_players, c_preseason - 1)
        columns="gsis_id",
        inplace=True,
    )
    fantasy_mask = preseason["position"].isin(
        FANTASY_POSITIONS
    )  # (n_preseason_players,)
    preseason.loc[fantasy_mask, "depth_tier"] = preseason.loc[  # (n_fantasy_players,)
        fantasy_mask, "depth_tier"
    ].fillna(3.0)

    seasonal_snaps, snap_match_fraction = (
        _load_snap_counts(  # (n_snap_player_seasons, 3), scalar
            raw_dir,
            players,
        )
    )
    player_seasons = player_seasons.merge(  # (n_player_seasons, c_season)
        availability,
        on=["season", "player_id"],
        how="outer",
        validate="one_to_one",
    )
    player_seasons = player_seasons.merge(  # (n_player_seasons, c_season)
        seasonal_snaps,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    count_columns = ["roster_weeks", "active_roster_weeks", "offense_snaps"]
    player_seasons[count_columns] = player_seasons[
        count_columns
    ].fillna(  # (n_player_seasons, 3)
        0.0
    )
    combine = _load_combine(raw_dir, players)  # (n_players, 1 + n_combine_atoms)
    modeling = _build_modeling_table(  # (n_model_rows, c_modeling)
        preseason,
        player_seasons,
        combine,
        current_season,
    )

    for frame in (player_games, player_seasons, preseason, modeling):
        _normalize_object_columns(frame)

    player_games.to_parquet(processed_dir / "player_games.parquet", index=False)
    player_seasons.to_parquet(processed_dir / "player_seasons.parquet", index=False)
    preseason.to_parquet(processed_dir / "preseason_players.parquet", index=False)
    modeling.to_parquet(processed_dir / "modeling_table.parquet", index=False)

    validation_mask = (  # (n_games,)
        player_games["season_type"].eq("REG")
        & player_games["position"].isin(OFFENSE_POSITIONS)
    )
    score_delta = (  # (n_validation_games_with_scores,)
        player_games.loc[validation_mask, "recomputed_ppr_points"]
        - player_games.loc[validation_mask, "fantasy_points_ppr"]
    ).dropna()
    exact_fraction = float(score_delta.abs().lt(1e-9).mean())
    mean_absolute_delta = float(score_delta.abs().mean())
    depth_coverage = float(preseason.loc[fantasy_mask, "depth_rank"].notna().mean())

    total_bytes = project_data_bytes(root)
    if total_bytes > MAX_PROJECT_DATA_BYTES:
        raise RuntimeError(
            f"Project data uses {total_bytes:,} bytes, above the "
            f"{MAX_PROJECT_DATA_BYTES:,}-byte limit."
        )

    current_mask = preseason["season"].eq(current_season)  # (n_preseason_players,)
    current_roster = preseason[current_mask]  # (n_current_players, c_roster)
    summary = BuildSummary(
        current_season=current_season,
        player_game_rows=len(player_games),
        player_game_players=int(player_games["player_id"].nunique()),
        unidentified_game_rows_dropped=unidentified_game_rows,
        player_season_rows=len(player_seasons),
        modeling_rows=len(modeling),
        current_roster_players=len(current_roster),
        current_fantasy_players=int(
            current_roster["position"].isin(FANTASY_POSITIONS).sum()
        ),
        recomputed_ppr_exact_fraction=exact_fraction,
        recomputed_ppr_mean_absolute_delta=mean_absolute_delta,
        depth_tier_coverage=depth_coverage,
        snap_id_match_fraction=snap_match_fraction,
        snapshot_date=snapshot.isoformat(),
        scoring_profile=scoring_profile.name,
        scoring_config_sha256=scoring_profile_digest(scoring_path),
        target_period="all NFL regular-season games, including Week 18 since 2021",
        total_project_data_bytes=total_bytes,
        built_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )
    report = asdict(summary)
    report["week_one_duplicate_rows_resolved"] = duplicate_rows
    summary_path = processed_dir / "build_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    exact_total_bytes = synchronize_project_sizes(root)
    summary = replace(summary, total_project_data_bytes=exact_total_bytes)
    return summary
