"""Build leakage-safe Phase 2 features from frozen nflverse inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Literal, Mapping


FeatureTier = Literal["stable", "recent"]

KEY_COLUMNS = ("target_season", "player_id", "model_position")
MODEL_POSITIONS = ("QB", "RB", "WR", "TE")
REQUIRED_MODELING_COLUMNS = (*KEY_COLUMNS, "team")
REQUIRED_WEEKLY_COLUMNS = (
    "season",
    "week",
    "season_type",
    "player_id",
    "team",
)
REQUIRED_PLAYER_SEASON_COLUMNS = ("season", "player_id")

STABLE_METADATA_COLUMNS = (
    "age",
    "years_exp",
    "height",
    "weight",
    "draft_number",
    "draft_round",
    "was_drafted",
    "is_rookie",
    "team_changed",
    "combine_forty",
    "combine_bench",
    "combine_vertical",
    "combine_broad_jump",
    "combine_cone",
    "combine_shuttle",
)

STABLE_WEEKLY_COLUMNS = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "sack_yards_lost",
    "sack_fumbles",
    "sack_fumbles_lost",
    "passing_first_downs",
    "passing_10",
    "passing_16",
    "passing_20",
    "passing_40",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_first_downs",
    "rushing_10",
    "rushing_12",
    "rushing_20",
    "rushing_40",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles",
    "receiving_fumbles_lost",
    "receiving_first_downs",
    "receiving_10",
    "receiving_16",
    "receiving_20",
    "receiving_40",
    "special_teams_tds",
    "fumbles_total",
    "fumbles_lost_total",
    "punt_returns",
    "punt_return_yards",
    "kickoff_returns",
    "kickoff_return_yards",
    "fg_made",
    "fg_att",
    "fg_missed",
    "fg_blocked",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "fg_missed_0_19",
    "fg_missed_20_29",
    "fg_missed_30_39",
    "fg_missed_40_49",
    "fg_missed_50_59",
    "fg_missed_60_",
    "fg_made_distance",
    "fg_missed_distance",
    "pat_made",
    "pat_att",
    "pat_missed",
    "pat_blocked",
    "gwfg_made",
    "gwfg_att",
    "gwfg_missed",
    "gwfg_blocked",
    "gwfg_distance",
)

RECENT_WEEKLY_COLUMNS = (
    "targets",
    "passing_air_yards",
    "passing_yards_after_catch",
    "receiving_air_yards",
    "receiving_yards_after_catch",
)

STABLE_PLAYER_SEASON_COLUMNS = ("games", "roster_weeks")
RECENT_PLAYER_SEASON_COLUMNS = ("offense_snaps", "st_snaps")

STABLE_TRAJECTORY_COLUMNS = (
    "attempts",
    "passing_yards",
    "passing_first_downs",
    "passing_20",
    "carries",
    "rushing_yards",
    "rushing_first_downs",
    "rushing_20",
    "receptions",
    "receiving_yards",
    "receiving_first_downs",
    "receiving_20",
    "punt_return_yards",
    "kickoff_return_yards",
    "fg_made",
    "fg_att",
    "pat_att",
)

RECENT_TRAJECTORY_COLUMNS = (
    "targets",
    "passing_air_yards",
    "passing_yards_after_catch",
    "receiving_air_yards",
    "receiving_yards_after_catch",
)

STABLE_TEAM_ENVIRONMENT_COLUMNS = (
    "attempts",
    "passing_yards",
    "passing_tds",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fg_att",
    "pat_att",
)

RECENT_TEAM_ENVIRONMENT_COLUMNS = ("targets",)

STABLE_ROOM_COLUMNS = (
    "attempts",
    "carries",
    "receptions",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "fg_att",
)

RECENT_ROOM_COLUMNS = ("targets",)

STABLE_RATE_RECIPES = (
    ("completion_rate", "completions", "attempts"),
    ("pass_yards_per_attempt", "passing_yards", "attempts"),
    ("pass_td_rate", "passing_tds", "attempts"),
    ("interception_rate", "passing_interceptions", "attempts"),
    ("rush_yards_per_carry", "rushing_yards", "carries"),
    ("rush_first_down_rate", "rushing_first_downs", "carries"),
    ("rush_20_rate", "rushing_20", "carries"),
    ("receiving_yards_per_reception", "receiving_yards", "receptions"),
    ("receiving_first_down_rate", "receiving_first_downs", "receptions"),
    ("receiving_20_rate", "receiving_20", "receptions"),
    ("field_goal_make_rate", "fg_made", "fg_att"),
    ("made_field_goal_mean_distance", "fg_made_distance", "fg_made"),
    ("missed_field_goal_mean_distance", "fg_missed_distance", "fg_missed"),
    ("pat_make_rate", "pat_made", "pat_att"),
)

RECENT_RATE_RECIPES = (
    ("catch_rate", "receptions", "targets"),
    ("receiving_yards_per_target", "receiving_yards", "targets"),
    ("receiving_air_yards_per_target", "receiving_air_yards", "targets"),
    (
        "receiving_yards_after_catch_per_reception",
        "receiving_yards_after_catch",
        "receptions",
    ),
    ("passing_air_yards_per_attempt", "passing_air_yards", "attempts"),
    (
        "passing_yards_after_catch_per_completion",
        "passing_yards_after_catch",
        "completions",
    ),
)

PROHIBITED_FEATURE_FRAGMENTS = (
    "active_roster",
    "air_yards_share",
    "cpoe",
    "depth",
    "epa",
    "fantasy",
    "pacr",
    "racr",
    "status",
    "target_points",
    "target_share",
    "wopr",
)


@dataclass(frozen=True, slots=True)
class FeatureLineage:
    """Describe the source timing and recipe for one numeric feature."""

    column: str
    sources: tuple[str, ...]
    source_offsets: tuple[int, ...]
    recipe: str

    def __post_init__(self) -> None:
        """Validate one lineage record."""
        if not self.sources or len(self.sources) != len(self.source_offsets):
            raise ValueError(
                "Lineage sources and offsets must be nonempty and aligned."
            )
        if any(offset < 0 for offset in self.source_offsets):
            raise ValueError("Source offsets must be zero or positive lookbacks.")


@dataclass(frozen=True, slots=True)
class Phase2FeatureBundle:
    """Keep row keys separate from numeric model inputs."""

    keys: pd.DataFrame
    frame: pd.DataFrame
    lineage: Mapping[str, FeatureLineage]
    tier: FeatureTier

    def __post_init__(self) -> None:
        """Validate alignment, finiteness, and source lineage."""
        if tuple(self.keys.columns) != KEY_COLUMNS:
            raise ValueError(f"Bundle keys must be {KEY_COLUMNS!r}.")
        if len(self.keys) != len(self.frame):
            raise ValueError("Bundle keys and feature rows must have equal length.")
        if not self.keys.index.equals(self.frame.index):
            raise ValueError("Bundle keys and features must use the same index.")
        if self.keys.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(
                "Bundle keys must be unique at target-season player grain."
            )
        if not self.frame.columns.is_unique:
            raise ValueError("Feature columns must be unique.")
        if set(self.lineage) != set(self.frame.columns):
            raise ValueError(
                "Every feature column must have exactly one lineage record."
            )

        feature_values = self.frame.to_numpy(dtype="float64")  # (n, d)
        if not np.isfinite(feature_values).all():
            raise ValueError("Phase 2 features must be finite numeric values.")

        for column in self.frame.columns:
            lowered = column.lower()
            if column in KEY_COLUMNS or column == "team" or lowered.endswith("_id"):
                raise ValueError(f"Identifier leaked into model inputs: {column!r}.")
            if any(fragment in lowered for fragment in PROHIBITED_FEATURE_FRAGMENTS):
                raise ValueError(
                    f"Prohibited feature leaked into model inputs: {column!r}."
                )

            record = self.lineage[column]
            if record.column != column:
                raise ValueError(f"Lineage column mismatch for {column!r}.")
            for source, offset in zip(
                record.sources,
                record.source_offsets,
                strict=True,
            ):
                if source in {"weekly_stats", "player_seasons"} and offset < 1:
                    raise ValueError(
                        f"Outcome source {source!r} is not lagged for {column!r}."
                    )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the numeric model columns in deterministic order."""
        return tuple(self.frame.columns)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing!r}.")


def _numeric(values: pd.Series) -> pd.Series:
    # values: (n,)
    numeric_values = pd.to_numeric(values, errors="coerce")  # (n,)
    return numeric_values.replace([np.inf, -np.inf], np.nan)  # (n,)


def _add_feature(
    features: dict[str, pd.Series],
    lineage: dict[str, FeatureLineage],
    column: str,
    values: pd.Series,
    *,
    sources: tuple[str, ...],
    source_offsets: tuple[int, ...],
    recipe: str,
) -> None:
    # each existing feature and values: (n,)
    if column in features:
        raise ValueError(f"Duplicate Phase 2 feature: {column!r}.")

    numeric_values = _numeric(values).fillna(0.0).astype("float32")  # (n,)
    expected_rows = len(next(iter(features.values()))) if features else len(values)
    if len(numeric_values) != expected_rows:
        raise ValueError(f"Feature {column!r} has the wrong row count.")

    features[column] = numeric_values.reset_index(drop=True)  # (n,)
    lineage[column] = FeatureLineage(
        column=column,
        sources=sources,
        source_offsets=source_offsets,
        recipe=recipe,
    )


def _selected_columns(
    available: pd.Index,
    stable: tuple[str, ...],
    recent: tuple[str, ...],
    tier: FeatureTier,
) -> list[str]:
    declared = stable if tier == "stable" else (*stable, *recent)
    return [column for column in declared if column in available]


def _seasonal_player_counts(
    weekly_stats: pd.DataFrame,
    player_seasons: pd.DataFrame,
    tier: FeatureTier,
) -> tuple[pd.DataFrame, dict[str, str]]:
    # weekly_stats: (n_weekly, c_weekly); player_seasons: (n_seasons, c_seasons)
    weekly_columns = _selected_columns(
        weekly_stats.columns,
        STABLE_WEEKLY_COLUMNS,
        RECENT_WEEKLY_COLUMNS,
        tier,
    )
    supplement_columns = _selected_columns(
        player_seasons.columns,
        STABLE_PLAYER_SEASON_COLUMNS,
        RECENT_PLAYER_SEASON_COLUMNS,
        tier,
    )
    if not weekly_columns and not supplement_columns:
        raise ValueError("No declared Phase 2 raw count columns are available.")

    regular_mask = (
        weekly_stats["season_type"].eq("REG") & weekly_stats["player_id"].notna()
    )  # (n_weekly,)
    regular_weekly = weekly_stats.loc[
        regular_mask,
        ["season", "player_id", *weekly_columns],
    ].copy()  # (n_regular, 2 + c_weekly_selected)
    for column in weekly_columns:
        regular_weekly[column] = _numeric(regular_weekly[column]).fillna(
            0.0
        )  # (n_regular,)

    seasonal_weekly = (
        regular_weekly.groupby(["season", "player_id"], observed=True)[weekly_columns]
        .sum(min_count=1)
        .reset_index()
    )  # (n_player_seasons_weekly, 2 + c_weekly_selected)

    supplements = player_seasons.loc[
        :, ["season", "player_id", *supplement_columns]
    ].copy()  # (n_seasons, 2 + c_supplements)
    if supplements.duplicated(["season", "player_id"]).any():
        raise ValueError("player_seasons must be unique at season-player grain.")
    for column in supplement_columns:
        supplements[column] = _numeric(supplements[column]).fillna(0.0)  # (n_seasons,)

    seasonal_counts = seasonal_weekly.merge(
        supplements,
        on=["season", "player_id"],
        how="outer",
        validate="one_to_one",
    )  # (n_player_seasons_union, 2 + c_raw)
    count_columns = [*weekly_columns, *supplement_columns]
    seasonal_counts[count_columns] = seasonal_counts[count_columns].fillna(
        0.0
    )  # (n_player_seasons_union, c_raw)
    seasonal_counts["history_available"] = 1.0  # (n_player_seasons_union,)

    source_by_column = {column: "weekly_stats" for column in weekly_columns}
    source_by_column.update({column: "player_seasons" for column in supplement_columns})
    return seasonal_counts, source_by_column


def _trajectory_table(
    weekly_stats: pd.DataFrame,
    tier: FeatureTier,
    late_weeks: int,
) -> pd.DataFrame:
    # weekly_stats: (n_weekly, c_weekly)
    trajectory_columns = _selected_columns(
        weekly_stats.columns,
        STABLE_TRAJECTORY_COLUMNS,
        RECENT_TRAJECTORY_COLUMNS,
        tier,
    )
    if not trajectory_columns:
        return pd.DataFrame(columns=["season", "player_id"])

    regular_mask = (
        weekly_stats["season_type"].eq("REG") & weekly_stats["player_id"].notna()
    )  # (n_weekly,)
    weekly_values = weekly_stats.loc[
        regular_mask,
        ["season", "player_id", "week", *trajectory_columns],
    ].copy()  # (n_regular, 3 + c_trajectory)
    weekly_values["week"] = _numeric(weekly_values["week"])  # (n_regular,)
    weekly_values = weekly_values[
        weekly_values["week"].notna()
    ].copy()  # (n_week_rows, 3 + c_trajectory)
    for column in trajectory_columns:
        weekly_values[column] = _numeric(weekly_values[column]).fillna(
            0.0
        )  # (n_week_rows,)

    weekly_values = weekly_values.groupby(
        ["season", "player_id", "week"],
        observed=True,
        as_index=False,
    )[trajectory_columns].sum()  # (n_player_weeks, 3 + c_trajectory)
    season_max_week = (
        weekly_values.groupby("season", observed=True)["week"]
        .max()
        .rename("season_max_week")
        .reset_index()
    )  # (n_source_seasons, 2)
    weekly_values = weekly_values.merge(
        season_max_week,
        on="season",
        how="left",
        validate="many_to_one",
    )  # (n_player_weeks, 4 + c_trajectory)

    late_mask = weekly_values["week"].gt(
        weekly_values["season_max_week"] - late_weeks
    )  # (n_player_weeks,)
    group_columns = ["season", "player_id"]
    season_totals = (
        weekly_values.groupby(group_columns, observed=True)[trajectory_columns]
        .sum()
        .add_prefix("total_")
        .reset_index()
    )  # (n_player_seasons, 2 + c_trajectory)
    late_totals = (
        weekly_values.loc[late_mask]
        .groupby(group_columns, observed=True)[trajectory_columns]
        .sum()
        .add_prefix("late_")
        .reset_index()
    )  # (n_late_player_seasons, 2 + c_trajectory)

    weighted_values = weekly_values.loc[:, group_columns].copy()  # (n_player_weeks, 2)
    weighted_values[trajectory_columns] = weekly_values[trajectory_columns].mul(
        weekly_values["week"], axis=0
    )  # (n_player_weeks, c_trajectory)
    weighted_totals = (
        weighted_values.groupby(group_columns, observed=True)[trajectory_columns]
        .sum()
        .add_prefix("weighted_")
        .reset_index()
    )  # (n_player_seasons, 2 + c_trajectory)

    trajectory = season_totals.merge(
        late_totals,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )  # (n_player_seasons, 2 + 2*c_trajectory)
    trajectory = trajectory.merge(
        weighted_totals,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )  # (n_player_seasons, 2 + 3*c_trajectory)
    trajectory = trajectory.merge(
        season_max_week,
        on="season",
        how="left",
        validate="many_to_one",
    )  # (n_player_seasons, 3 + 3*c_trajectory)
    late_columns = [f"late_{column}" for column in trajectory_columns]
    trajectory[late_columns] = trajectory[late_columns].fillna(
        0.0
    )  # (n_player_seasons, c_trajectory)

    week_count = trajectory["season_max_week"].clip(lower=1.0)  # (n_player_seasons,)
    late_week_count = week_count.clip(upper=float(late_weeks))  # (n_player_seasons,)
    early_week_count = week_count - late_week_count  # (n_player_seasons,)
    sum_week = week_count * (week_count + 1.0) / 2.0  # (n_player_seasons,)
    sum_week_squared = (
        week_count * (week_count + 1.0) * (2.0 * week_count + 1.0) / 6.0
    )  # (n_player_seasons,)
    slope_denominator = week_count * sum_week_squared - sum_week.pow(
        2
    )  # (n_player_seasons,)

    derived = trajectory.loc[:, group_columns].copy()  # (n_player_seasons, 2)
    for column in trajectory_columns:
        total = trajectory[f"total_{column}"]  # (n_player_seasons,)
        late = trajectory[f"late_{column}"]  # (n_player_seasons,)
        weighted = trajectory[f"weighted_{column}"]  # (n_player_seasons,)
        slope = (
            (week_count * weighted - sum_week * total)
            .div(slope_denominator.where(slope_denominator.gt(0.0)))
            .fillna(0.0)
        )  # (n_player_seasons,)
        late_mean = late.div(late_week_count.where(late_week_count.gt(0.0))).fillna(
            0.0
        )  # (n_player_seasons,)
        early_mean = (
            (total - late)
            .div(early_week_count.where(early_week_count.gt(0.0)))
            .fillna(0.0)
        )  # (n_player_seasons,)

        derived[f"late{late_weeks}_{column}"] = (
            late  # (n_player_seasons, c_derived + 1)
        )
        derived[f"weekly_slope_{column}"] = slope  # (n_player_seasons, c_derived + 1)
        derived[f"late{late_weeks}_mean_delta_{column}"] = (
            late_mean - early_mean
        )  # (n_player_seasons, c_derived + 1)

    return derived  # (n_player_seasons, 2 + 3*c_trajectory)


def _team_environment_table(
    weekly_stats: pd.DataFrame,
    tier: FeatureTier,
) -> pd.DataFrame:
    # weekly_stats: (n_weekly, c_weekly)
    environment_columns = _selected_columns(
        weekly_stats.columns,
        STABLE_TEAM_ENVIRONMENT_COLUMNS,
        RECENT_TEAM_ENVIRONMENT_COLUMNS,
        tier,
    )
    if not environment_columns:
        return pd.DataFrame(columns=["season", "team"])

    regular_mask = (
        weekly_stats["season_type"].eq("REG") & weekly_stats["team"].notna()
    )  # (n_weekly,)
    team_values = weekly_stats.loc[
        regular_mask,
        ["season", "team", *environment_columns],
    ].copy()  # (n_regular, 2 + c_environment)
    team_values["team"] = (
        team_values["team"].astype("string").str.upper()
    )  # (n_regular,)
    for column in environment_columns:
        team_values[column] = _numeric(team_values[column]).fillna(0.0)  # (n_regular,)

    team_environment = (
        team_values.groupby(["season", "team"], observed=True)[environment_columns]
        .sum(min_count=1)
        .reset_index()
    )  # (n_team_seasons, 2 + c_environment)
    team_environment["team_environment_available"] = 1.0  # (n_team_seasons,)
    return team_environment


def _add_metadata_features(
    target_rows: pd.DataFrame,
    features: dict[str, pd.Series],
    lineage: dict[str, FeatureLineage],
) -> None:
    # target_rows: (n, c_modeling); each feature: (n,)
    for column in STABLE_METADATA_COLUMNS:
        if column not in target_rows:
            continue

        values = _numeric(target_rows[column])  # (n,)
        missing = values.isna().astype("float32")  # (n,)
        _add_feature(
            features,
            lineage,
            column,
            values,
            sources=("modeling_table",),
            source_offsets=(0,),
            recipe="preseason cutoff value; missing values become zero",
        )
        _add_feature(
            features,
            lineage,
            f"missing_{column}",
            missing,
            sources=("modeling_table",),
            source_offsets=(0,),
            recipe=f"1 when preseason {column} is missing, else 0",
        )


def _add_position_features(
    target_rows: pd.DataFrame,
    features: dict[str, pd.Series],
    lineage: dict[str, FeatureLineage],
) -> None:
    # target_rows: (n, c_modeling); each feature: (n,)
    model_positions = target_rows["model_position"].astype("string").str.upper()  # (n,)
    for position in MODEL_POSITIONS:
        indicator = model_positions.eq(position).astype("float32")  # (n,)
        _add_feature(
            features,
            lineage,
            f"position_{position}",
            indicator,
            sources=("modeling_table.cutoff_metadata",),
            source_offsets=(0,),
            recipe=f"1 when cutoff model_position is {position}, else 0",
        )


def _add_lagged_counts(
    keys: pd.DataFrame,
    seasonal_counts: pd.DataFrame,
    source_by_column: Mapping[str, str],
    max_lag: int,
    features: dict[str, pd.Series],
    lineage: dict[str, FeatureLineage],
) -> None:
    # keys: (n, 3); seasonal_counts: (n_source, 3 + c_counts)
    # each feature: (n,)
    raw_columns = [
        column
        for column in seasonal_counts.columns
        if column not in {"season", "player_id", "history_available"}
    ]
    for lag in range(1, max_lag + 1):
        shifted = seasonal_counts.copy()  # (n_source, 3 + c_counts)
        shifted["target_season"] = shifted["season"] + lag  # (n_source,)
        shifted.rename(columns={"season": "source_season"}, inplace=True)
        merged = keys.loc[:, ["target_season", "player_id"]].merge(
            shifted,
            on=["target_season", "player_id"],
            how="left",
            validate="many_to_one",
            sort=False,
        )  # (n, 3 + c_counts)
        matched = merged["source_season"].notna()  # (n,)
        assert (
            merged.loc[matched, "source_season"]
            == merged.loc[matched, "target_season"] - lag
        ).all()

        _add_feature(
            features,
            lineage,
            f"lag{lag}_history_available",
            merged["history_available"],
            sources=("weekly_stats", "player_seasons"),
            source_offsets=(lag, lag),
            recipe=f"1 when a target_season-{lag} player-season exists, else 0",
        )
        for column in raw_columns:
            _add_feature(
                features,
                lineage,
                f"lag{lag}_{column}",
                merged[column],
                sources=(source_by_column[column],),
                source_offsets=(lag,),
                recipe=(
                    f"regular-season {column} from target_season-{lag}; "
                    "missing history becomes zero"
                ),
            )


def _add_rate_features(
    tier: FeatureTier,
    max_lag: int,
    features: dict[str, pd.Series],
    lineage: dict[str, FeatureLineage],
) -> None:
    # each feature: (n,)
    recipes = (
        STABLE_RATE_RECIPES
        if tier == "stable"
        else (*STABLE_RATE_RECIPES, *RECENT_RATE_RECIPES)
    )
    for lag in range(1, max_lag + 1):
        for rate_name, numerator_name, denominator_name in recipes:
            numerator_column = f"lag{lag}_{numerator_name}"
            denominator_column = f"lag{lag}_{denominator_name}"
            if numerator_column not in features or denominator_column not in features:
                continue

            numerator = features[numerator_column]  # (n,)
            denominator = features[denominator_column]  # (n,)
            rate = numerator.div(denominator.where(denominator.gt(0.0))).fillna(
                0.0
            )  # (n,)
            _add_feature(
                features,
                lineage,
                f"lag{lag}_{rate_name}",
                rate,
                sources=("weekly_stats",),
                source_offsets=(lag,),
                recipe=(
                    f"lag{lag}_{numerator_name} / lag{lag}_{denominator_name}; "
                    "zero when denominator is not positive"
                ),
            )


def _add_trajectory_features(
    keys: pd.DataFrame,
    trajectory: pd.DataFrame,
    late_weeks: int,
    features: dict[str, pd.Series],
    lineage: dict[str, FeatureLineage],
) -> None:
    # keys: (n, 3); trajectory: (n_source, 2 + c_trajectory)
    # each feature: (n,)
    if trajectory.empty:
        return

    shifted = trajectory.copy()  # (n_source, 2 + c_trajectory)
    shifted["target_season"] = shifted["season"] + 1  # (n_source,)
    shifted.rename(columns={"season": "source_season"}, inplace=True)
    merged = keys.loc[:, ["target_season", "player_id"]].merge(
        shifted,
        on=["target_season", "player_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )  # (n, 3 + c_trajectory)
    matched = merged["source_season"].notna()  # (n,)
    assert (
        merged.loc[matched, "source_season"] == merged.loc[matched, "target_season"] - 1
    ).all()

    for column in trajectory.columns:
        if column in {"season", "player_id"}:
            continue
        if column.startswith(f"late{late_weeks}_mean_delta_"):
            recipe = (
                f"target_season-1 mean over its final {late_weeks} league weeks "
                "minus its earlier-week mean; missing weeks count as zero"
            )
        elif column.startswith(f"late{late_weeks}_"):
            recipe = (
                f"sum over the final {late_weeks} regular-season league weeks "
                "of target_season-1; missing weeks count as zero"
            )
        else:
            recipe = (
                "least-squares weekly slope across all regular-season league weeks "
                "of target_season-1; missing weeks count as zero"
            )

        _add_feature(
            features,
            lineage,
            f"lag1_{column}",
            merged[column],
            sources=("weekly_stats",),
            source_offsets=(1,),
            recipe=recipe,
        )


def _add_room_features(
    target_rows: pd.DataFrame,
    tier: FeatureTier,
    features: dict[str, pd.Series],
    lineage: dict[str, FeatureLineage],
) -> None:
    # target_rows: (n, c_modeling); each feature: (n,)
    room_keys = target_rows.loc[
        :, ["target_season", "team", "model_position"]
    ].copy()  # (n, 3)
    room_keys["team"] = room_keys["team"].astype("string").str.upper()  # (n,)
    room_grouper = [
        room_keys["target_season"],
        room_keys["team"],
        room_keys["model_position"],
    ]
    room_size = room_keys.groupby(
        ["target_season", "team", "model_position"],
        dropna=False,
    )["model_position"].transform("size")  # (n,)
    competitor_count = (room_size - 1).clip(lower=0)  # (n,)
    _add_feature(
        features,
        lineage,
        "room_competitor_count",
        competitor_count,
        sources=("modeling_table.target_roster",),
        source_offsets=(0,),
        recipe="other players on the cutoff roster with the same team and position",
    )

    room_columns = (
        STABLE_ROOM_COLUMNS
        if tier == "stable"
        else (*STABLE_ROOM_COLUMNS, *RECENT_ROOM_COLUMNS)
    )
    for column in room_columns:
        lagged_column = f"lag1_{column}"
        if lagged_column not in features:
            continue

        own_value = features[lagged_column]  # (n,)
        room_total = own_value.groupby(room_grouper, dropna=False).transform(
            "sum"
        )  # (n,)
        other_total = room_total - own_value  # (n,)
        own_nonzero = own_value.gt(0.0).astype("float32")  # (n,)
        nonzero_count = own_nonzero.groupby(
            room_grouper,
            dropna=False,
        ).transform("sum")  # (n,)
        nonzero_competitors = nonzero_count - own_nonzero  # (n,)
        other_mean = other_total.div(
            competitor_count.where(competitor_count.gt(0.0))
        ).fillna(0.0)  # (n,)
        own_minus_other_mean = own_value - other_mean  # (n,)

        common = {
            "sources": ("modeling_table.target_roster", "weekly_stats"),
            "source_offsets": (0, 1),
        }
        _add_feature(
            features,
            lineage,
            f"room_lag1_other_{column}",
            other_total,
            recipe=(
                f"sum of lag1_{column} for other cutoff-roster players on the "
                "same team and position"
            ),
            **common,
        )
        _add_feature(
            features,
            lineage,
            f"room_lag1_nonzero_competitors_{column}",
            nonzero_competitors,
            recipe=(
                f"number of other same-team, same-position cutoff-roster players "
                f"with positive lag1_{column}"
            ),
            **common,
        )
        _add_feature(
            features,
            lineage,
            f"room_lag1_margin_{column}",
            own_minus_other_mean,
            recipe=(
                f"own lag1_{column} minus the mean lag1_{column} of other "
                "same-team, same-position cutoff-roster players"
            ),
            **common,
        )


def _add_team_environment_features(
    target_rows: pd.DataFrame,
    team_environment: pd.DataFrame,
    features: dict[str, pd.Series],
    lineage: dict[str, FeatureLineage],
) -> None:
    # target_rows: (n, c_modeling); team_environment: (n_teams, 3 + c_environment)
    # each feature: (n,)
    if team_environment.empty:
        return

    shifted = team_environment.copy()  # (n_teams, 3 + c_environment)
    shifted["target_season"] = shifted["season"] + 1  # (n_teams,)
    shifted.rename(
        columns={"season": "source_season", "team": "normalized_team"},
        inplace=True,
    )
    target_teams = target_rows.loc[:, ["target_season", "team"]].copy()  # (n, 2)
    target_teams["normalized_team"] = (
        target_teams["team"].astype("string").str.upper()
    )  # (n,)
    merged = target_teams.merge(
        shifted,
        on=["target_season", "normalized_team"],
        how="left",
        validate="many_to_one",
        sort=False,
    )  # (n, 4 + c_environment)
    matched = merged["source_season"].notna()  # (n,)
    assert (
        merged.loc[matched, "source_season"] == merged.loc[matched, "target_season"] - 1
    ).all()

    _add_feature(
        features,
        lineage,
        "team_lag1_environment_available",
        merged["team_environment_available"],
        sources=("weekly_stats",),
        source_offsets=(1,),
        recipe="1 when the current team has target_season-1 raw team counts, else 0",
    )
    for column in team_environment.columns:
        if column in {"season", "team", "team_environment_available"}:
            continue
        _add_feature(
            features,
            lineage,
            f"team_lag1_{column}",
            merged[column],
            sources=("weekly_stats",),
            source_offsets=(1,),
            recipe=(
                f"sum of target_season-1 regular-season {column} for the "
                "player's cutoff-roster team"
            ),
        )


def _add_prior_team_environment_features(
    keys: pd.DataFrame,
    player_seasons: pd.DataFrame,
    team_environment: pd.DataFrame,
    features: dict[str, pd.Series],
    lineage: dict[str, FeatureLineage],
) -> None:
    # keys: (n, 3); player_seasons: (n_source, c_seasons)
    # team_environment: (n_teams, 3 + c_environment); each feature: (n,)
    if "last_team" not in player_seasons or team_environment.empty:
        return

    prior_rows = player_seasons.loc[
        :, ["season", "player_id", "last_team"]
    ].copy()  # (n_source, 3)
    if prior_rows.duplicated(["season", "player_id"]).any():
        raise ValueError("player_seasons must be unique at season-player grain.")
    prior_rows["prior_team"] = (
        prior_rows["last_team"].astype("string").str.upper()
    )  # (n_source,)
    prior_rows.drop(columns="last_team", inplace=True)  # (n_source, 3)

    normalized_environment = team_environment.rename(
        columns={"team": "prior_team"}
    ).copy()  # (n_teams, 3 + c_environment)
    prior_environment = prior_rows.merge(
        normalized_environment,
        on=["season", "prior_team"],
        how="left",
        validate="many_to_one",
        sort=False,
    )  # (n_source, 4 + c_environment)
    prior_environment["target_season"] = prior_environment["season"] + 1  # (n_source,)
    prior_environment.rename(
        columns={"season": "source_season"},
        inplace=True,
    )
    merged = keys.loc[:, ["target_season", "player_id"]].merge(
        prior_environment,
        on=["target_season", "player_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )  # (n, 5 + c_environment)
    matched = merged["source_season"].notna()  # (n,)
    assert (
        merged.loc[matched, "source_season"] == merged.loc[matched, "target_season"] - 1
    ).all()

    _add_feature(
        features,
        lineage,
        "prior_team_lag1_environment_available",
        merged["team_environment_available"],
        sources=("player_seasons", "weekly_stats"),
        source_offsets=(1, 1),
        recipe=(
            "1 when the player's final target_season-1 team has raw team counts, else 0"
        ),
    )
    for column in team_environment.columns:
        if column in {"season", "team", "team_environment_available"}:
            continue
        _add_feature(
            features,
            lineage,
            f"prior_team_lag1_{column}",
            merged[column],
            sources=("player_seasons", "weekly_stats"),
            source_offsets=(1, 1),
            recipe=(
                f"sum of target_season-1 regular-season {column} for the "
                "player's final team in that source season"
            ),
        )


def build_phase2_features(
    weekly_stats: pd.DataFrame,
    player_seasons: pd.DataFrame,
    modeling_table: pd.DataFrame,
    *,
    tier: FeatureTier = "stable",
    max_lag: int = 4,
    late_weeks: int = 6,
) -> Phase2FeatureBundle:
    """Build numeric, target-free features with explicit temporal lineage."""
    # weekly_stats: (n_weekly, c_weekly)
    # player_seasons: (n_player_seasons, c_player_seasons)
    # modeling_table: (n_targets, c_modeling)
    if tier not in {"stable", "recent"}:
        raise ValueError("tier must be 'stable' or 'recent'.")
    if not 1 <= max_lag <= 4:
        raise ValueError("max_lag must be between 1 and 4.")
    if late_weeks < 1:
        raise ValueError("late_weeks must be positive.")

    _require_columns(modeling_table, REQUIRED_MODELING_COLUMNS, "modeling_table")
    _require_columns(weekly_stats, REQUIRED_WEEKLY_COLUMNS, "weekly_stats")
    _require_columns(
        player_seasons,
        REQUIRED_PLAYER_SEASON_COLUMNS,
        "player_seasons",
    )
    if modeling_table.duplicated(["target_season", "player_id"]).any():
        raise ValueError("modeling_table must be unique at target-season player grain.")

    target_seasons = _numeric(modeling_table["target_season"])  # (n_targets,)
    if target_seasons.isna().any():
        raise ValueError("modeling_table target seasons must be numeric and non-null.")
    target_mask = pd.Series(True, index=modeling_table.index)  # (n_targets,)
    if tier == "recent":
        target_mask &= target_seasons.ge(2013)  # (n_targets,)
    target_rows = modeling_table.loc[target_mask].copy()  # (n, c_modeling)
    target_rows["target_season"] = _numeric(target_rows["target_season"]).astype(
        "int32"
    )  # (n,)
    target_rows.reset_index(drop=True, inplace=True)

    keys = target_rows.loc[:, list(KEY_COLUMNS)].copy()  # (n, 3)
    feature_values: dict[str, pd.Series] = {}
    lineage: dict[str, FeatureLineage] = {}

    seasonal_counts, source_by_column = _seasonal_player_counts(
        weekly_stats,
        player_seasons,
        tier,
    )  # (n_source, 3 + c_counts), c_counts mappings
    trajectory = _trajectory_table(
        weekly_stats,
        tier,
        late_weeks,
    )  # (n_trajectory, 2 + c_trajectory)
    team_environment = _team_environment_table(
        weekly_stats,
        tier,
    )  # (n_team_seasons, 3 + c_environment)

    _add_metadata_features(target_rows, feature_values, lineage)
    _add_position_features(target_rows, feature_values, lineage)
    _add_lagged_counts(
        keys,
        seasonal_counts,
        source_by_column,
        max_lag,
        feature_values,
        lineage,
    )
    _add_rate_features(tier, max_lag, feature_values, lineage)
    _add_trajectory_features(
        keys,
        trajectory,
        late_weeks,
        feature_values,
        lineage,
    )
    _add_room_features(target_rows, tier, feature_values, lineage)
    _add_team_environment_features(
        target_rows,
        team_environment,
        feature_values,
        lineage,
    )
    _add_prior_team_environment_features(
        keys,
        player_seasons,
        team_environment,
        feature_values,
        lineage,
    )

    assert not any(
        offset < 0 for record in lineage.values() for offset in record.source_offsets
    )
    features = pd.DataFrame(feature_values, index=keys.index)  # (n, d)
    return Phase2FeatureBundle(
        keys=keys,
        frame=features,
        lineage=lineage,
        tier=tier,
    )
