"""Build the missed-time injury label and August-origin injury-history features."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .phase2_features import FeatureLineage
from .phase3_injury import (
    INJURY_FIELDS,
    InjuryFeatureBundle,
    build_injury_feature_bundle,
    load_injury_reports,
    physical_designation_mask,
)


ROSTER_SEASONS = tuple(range(2009, 2026))
RESERVE_LAG_OFFSETS = (1, 2, 3, 4)
REPORT_LAG_OFFSETS = (2, 3, 4, 5)
LABEL_SEASONS = (2013, 2024)
NON_INJURY_RESERVE_CODES = frozenset(
    {"R02", "R03", "R06", "R23", "R27", "R40", "R59", "R62"}
)
ABSENCE_STATUSES = frozenset({"out", "doubtful"})
PRIMARY_LABEL = "missed_time_injury"
SECONDARY_LABEL = "physical_injury_reported"
METADATA_FAMILY_COLUMNS = frozenset(
    {
        "age",
        "years_exp",
        "is_rookie",
        "was_drafted",
        "draft_number",
        "draft_round",
        "team_changed",
    }
)
PARTICIPATION_COLUMNS = frozenset(
    {"games", "roster_weeks", "offense_snaps", "st_snaps"}
)
RATE_FRAGMENTS = ("_rate", "_per_", "mean_distance")


@dataclass(frozen=True, slots=True)
class Phase4Bundle:
    """Aligned keys, metadata, labels, numeric inputs, lineage, and families."""

    keys: pd.DataFrame
    metadata: pd.DataFrame
    labels: pd.DataFrame
    frame: pd.DataFrame
    lineage: Mapping[str, FeatureLineage]
    families: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate alignment, label timing, and lineage coverage."""
        # keys: (n, 3); metadata: (n, c_meta); labels: (n, c_labels); frame: (n, d)
        n = len(self.keys)
        if not (len(self.metadata) == len(self.labels) == len(self.frame) == n):
            raise ValueError("Phase 4 tables must have identical row counts.")
        if self.keys.duplicated(["target_season", "player_id"]).any():
            raise ValueError("Phase 4 keys must be unique at player-season grain.")
        if not self.frame.columns.equals(pd.Index(self.lineage)):
            raise ValueError("Every Phase 4 feature must have ordered lineage.")
        if set(self.families) != set(self.frame.columns):
            raise ValueError("Every Phase 4 feature must have one family.")
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in self.frame.dtypes):
            raise TypeError("Phase 4 model inputs must be numeric.")
        values = self.frame.to_numpy(dtype="float64")  # (n, d)
        if not np.isfinite(values).all():
            raise ValueError("Phase 4 model inputs must be finite.")
        seasons = self.keys["target_season"]  # (n,)
        labeled = seasons.between(*LABEL_SEASONS)  # (n,)
        if self.labels.loc[labeled, PRIMARY_LABEL].isna().any():
            raise ValueError("Labeled seasons must have a complete primary label.")
        if self.labels.loc[~labeled, PRIMARY_LABEL].notna().any():
            raise ValueError("Seasons outside the label window must stay unlabeled.")
        for column, record in self.lineage.items():
            for source, offset in zip(
                record.sources,
                record.source_offsets,
                strict=True,
            ):
                if (
                    source in {"weekly_stats", "player_seasons", "roster_weekly"}
                    and offset < 1
                ):
                    raise ValueError(f"Outcome source is not lagged for {column!r}.")
                if source == "nflverse_injury_reports" and offset < 2:
                    raise ValueError(
                        f"Report feature {column!r} is not deployable for 2026."
                    )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return model columns in fitted order."""
        return tuple(self.frame.columns)

    def labeled_mask(self) -> pd.Series:
        """Return rows whose primary label is observed."""
        return self.keys["target_season"].between(*LABEL_SEASONS)  # (n,)


def load_roster_reserve_history(
    raw_dir: Path,
    seasons: Sequence[int] = ROSTER_SEASONS,
) -> pd.DataFrame:
    """Count regular-season weeks on an injury-compatible reserve status."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = raw_dir / "rosters" / f"roster_weekly_{int(season)}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing roster source: {path}")
        roster = pd.read_parquet(
            path,
            columns=[
                "season",
                "week",
                "game_type",
                "status",
                "gsis_id",
                "status_description_abbr",
            ],
        )  # (n_roster_rows, 6)
        regular = (
            roster["game_type"].astype("string").str.upper().eq("REG")
            & roster["gsis_id"].notna()
        )  # (n_roster_rows,)
        roster = roster.loc[regular].copy()  # (n_regular, 6)
        code = (
            roster["status_description_abbr"].astype("string").str.strip().str.upper()
        )  # (n_regular,)
        non_injury = (
            code.isin(NON_INJURY_RESERVE_CODES).fillna(False).astype("bool")
        )  # (n_regular,)
        reserve_status = (
            roster["status"].astype("string").str.upper().eq("RES").fillna(False)
        ).astype("bool")  # (n_regular,)
        roster["reserve_injury"] = reserve_status & ~non_injury  # (n_regular,)
        roster["week"] = pd.to_numeric(roster["week"], errors="coerce")  # (n_regular,)
        reserve_weeks = (
            roster.loc[roster["reserve_injury"]]
            .groupby("gsis_id", observed=True)["week"]
            .nunique()
            .rename("reserve_injury_weeks")
        )  # (n_reserve_players,)
        roster_weeks = (
            roster.groupby("gsis_id", observed=True)["week"]
            .nunique()
            .rename("roster_regular_weeks")
        )  # (n_players,)
        history = pd.concat(
            [roster_weeks, reserve_weeks], axis=1
        ).reset_index()  # (n_players, 3)
        history.rename(columns={"gsis_id": "player_id"}, inplace=True)
        history["reserve_injury_weeks"] = history["reserve_injury_weeks"].fillna(0.0)
        history["season"] = int(season)
        frames.append(history)
    reserve = pd.concat(frames, ignore_index=True)  # (n_player_seasons, 4)
    reserve["player_id"] = reserve["player_id"].astype("string").str.strip()
    reserve["season"] = reserve["season"].astype("int32")
    reserve["reserve_injury_weeks"] = reserve["reserve_injury_weeks"].astype("float32")
    reserve["roster_regular_weeks"] = reserve["roster_regular_weeks"].astype("float32")
    return reserve.loc[
        :, ["season", "player_id", "reserve_injury_weeks", "roster_regular_weeks"]
    ]  # (n_player_seasons, 4)


def aggregate_absence_reports(reports: pd.DataFrame) -> pd.DataFrame:
    """Count regular-season weeks with a physical Out or Doubtful game status."""
    # reports: (n_reports, c_reports)
    regular = reports.loc[
        reports["game_type"].astype("string").str.upper().eq("REG")
    ].copy()  # (n_regular_reports, c_reports)
    physical = pd.Series(False, index=regular.index)  # (n_regular_reports,)
    for column in INJURY_FIELDS:
        physical |= physical_designation_mask(regular[column]).to_numpy()
    status = (
        regular["report_status"].astype("string").str.strip().str.lower()
    )  # (n_regular_reports,)
    absence = physical & status.isin(ABSENCE_STATUSES).fillna(False).astype(
        "bool"
    )  # (n_regular_reports,)
    rows = regular.loc[absence, ["season", "week", "gsis_id"]]  # (n_absence_rows, 3)
    absences = (
        rows.groupby(["season", "gsis_id"], observed=True)["week"]
        .nunique()
        .rename("absence_report_weeks")
        .reset_index()
        .rename(columns={"gsis_id": "player_id"})
    )  # (n_absent_player_seasons, 3)
    absences["absence_reported"] = 1.0  # (n_absent_player_seasons,)
    absences["season"] = absences["season"].astype("int32")
    absences["absence_report_weeks"] = absences["absence_report_weeks"].astype(
        "float32"
    )
    absences["absence_reported"] = absences["absence_reported"].astype("float32")
    return absences  # (n_absent_player_seasons, 4)


def build_missed_time_labels(
    keys: pd.DataFrame,
    secondary: pd.Series,
    absences: pd.DataFrame,
    reserve: pd.DataFrame,
) -> pd.DataFrame:
    """Join report absences and reserve weeks into the primary label."""
    # keys: (n, 3); secondary: (n,); absences: (n_a, 4); reserve: (n_r, 4)
    labels = keys.loc[:, ["target_season", "player_id"]].copy()  # (n, 2)
    labels = labels.merge(
        absences.rename(columns={"season": "target_season"}),
        on=["target_season", "player_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )  # (n, 4)
    labels = labels.merge(
        reserve.rename(columns={"season": "target_season"}),
        on=["target_season", "player_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )  # (n, 6)
    for column in (
        "absence_report_weeks",
        "absence_reported",
        "reserve_injury_weeks",
        "roster_regular_weeks",
    ):
        labels[column] = (
            pd.to_numeric(labels[column], errors="coerce").astype("float64").fillna(0.0)
        )  # (n,)
    labels["reserve_injury"] = (
        labels["reserve_injury_weeks"].gt(0.0).astype("float64")
    )  # (n,)
    labels[PRIMARY_LABEL] = (
        labels["absence_reported"].gt(0.0) | labels["reserve_injury"].gt(0.0)
    ).astype("float64")  # (n,)
    labels[SECONDARY_LABEL] = (
        pd.to_numeric(secondary, errors="coerce").astype("float64").to_numpy()
    )  # (n,)
    unlabeled = ~labels["target_season"].between(*LABEL_SEASONS)  # (n,)
    for column in (
        "absence_report_weeks",
        "absence_reported",
        "reserve_injury_weeks",
        "reserve_injury",
        PRIMARY_LABEL,
        SECONDARY_LABEL,
    ):
        labels.loc[unlabeled, column] = np.nan
    labels.index = keys.index
    return labels  # (n, 9)


def _lagged_history_features(
    keys: pd.DataFrame,
    history: pd.DataFrame,
    value_columns: Mapping[str, str],
    offsets: Sequence[int],
    *,
    source: str,
) -> tuple[pd.DataFrame, dict[str, FeatureLineage]]:
    """Attach target-minus-offset history with zero for missing evidence."""
    # keys: (n, 3); history: (n_history, 2 + len(value_columns))
    features = pd.DataFrame(index=keys.index)  # (n, 0)
    lineage: dict[str, FeatureLineage] = {}
    lookup = history.rename(columns={"season": "source_season"})  # (n_history, ...)
    for offset in offsets:
        target_keys = keys.loc[:, ["target_season", "player_id"]].copy()  # (n, 2)
        target_keys["source_season"] = target_keys["target_season"] - int(
            offset
        )  # (n,)
        merged = target_keys.merge(
            lookup,
            on=["source_season", "player_id"],
            how="left",
            validate="many_to_one",
            sort=False,
        )  # (n, 3 + len(value_columns))
        for value_column, suffix in value_columns.items():
            column = f"injury_lag{offset}_{suffix}"
            features[column] = (
                pd.to_numeric(merged[value_column], errors="coerce")
                .fillna(0.0)
                .astype("float32")
                .to_numpy()
            )  # (n,)
            lineage[column] = FeatureLineage(
                column=column,
                sources=(source,),
                source_offsets=(int(offset),),
                recipe=(
                    f"{value_column} from target_season-{offset}; missing evidence "
                    "is encoded as 0"
                ),
            )
    return features, lineage


def feature_family(column: str) -> str:
    """Map one model column to a named feature family."""
    if column.startswith("position_"):
        return "position"
    if column.endswith("_per_game"):
        return "lag1_per_game"
    if column.startswith("lag1_share_"):
        return "lag1_usage_share"
    if column.startswith("injury_lag") and column.endswith("_reserve_weeks"):
        return "reserve_history_roster"
    if column.startswith("injury_lag"):
        return "injury_report_history"
    if column.startswith("missing_"):
        return "metadata_missingness"
    if column in METADATA_FAMILY_COLUMNS:
        return "age_experience_draft"
    if column in {"height", "weight"} or column.startswith("combine_"):
        return "body_combine"
    if column.endswith("_history_available"):
        return "history_availability"
    match = re.match(r"^lag(\d)_(.+)$", column)
    if match is None:
        return "other"
    lag = int(match.group(1))
    rest = match.group(2)
    if rest.startswith(("late6_", "weekly_slope_")):
        return "lag1_trajectory"
    if lag > 1:
        return "lag2_4_history"
    if rest in PARTICIPATION_COLUMNS:
        return "lag1_participation"
    if any(fragment in rest for fragment in RATE_FRAGMENTS):
        return "lag1_rates"
    return "lag1_production"


def build_phase4_bundle(root: Path, injury_directory: Path) -> Phase4Bundle:
    """Extend the Phase 3 bundle with reserve, absence, and missed-time labels."""
    base: InjuryFeatureBundle = build_injury_feature_bundle(root, injury_directory)
    reports = load_injury_reports(injury_directory)  # (n_reports, 16)
    absences = aggregate_absence_reports(reports)  # (n_absent_player_seasons, 4)
    reserve = load_roster_reserve_history(
        root / "data" / "raw"
    )  # (n_player_seasons, 4)

    reserve_features, reserve_lineage = _lagged_history_features(
        base.keys,
        reserve.loc[:, ["season", "player_id", "reserve_injury_weeks"]],
        {"reserve_injury_weeks": "reserve_weeks"},
        RESERVE_LAG_OFFSETS,
        source="roster_weekly",
    )  # (n, 4)
    absence_features, absence_lineage = _lagged_history_features(
        base.keys,
        absences.loc[
            :, ["season", "player_id", "absence_report_weeks", "absence_reported"]
        ],
        {
            "absence_report_weeks": "absence_report_weeks",
            "absence_reported": "absence_reported",
        },
        REPORT_LAG_OFFSETS,
        source="nflverse_injury_reports",
    )  # (n, 8)

    frame = pd.concat([base.frame, reserve_features, absence_features], axis=1).astype(
        "float32"
    )  # (n, d)
    lineage: dict[str, FeatureLineage] = dict(base.lineage)
    lineage.update(reserve_lineage)
    lineage.update(absence_lineage)
    families = {column: feature_family(column) for column in frame.columns}
    labels = build_missed_time_labels(
        base.keys,
        base.labels[SECONDARY_LABEL],
        absences,
        reserve,
    )  # (n, 9)
    return Phase4Bundle(
        keys=base.keys,
        metadata=base.metadata,
        labels=labels,
        frame=frame,
        lineage=lineage,
        families=families,
    )
