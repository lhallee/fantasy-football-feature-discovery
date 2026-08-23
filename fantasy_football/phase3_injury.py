"""Build observed physical injury-report labels and August-origin features."""

from __future__ import annotations

import hashlib
import json
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from .phase2_august import build_august_cohort
from .phase2_features import (
    RECENT_WEEKLY_COLUMNS,
    REQUIRED_WEEKLY_COLUMNS,
    STABLE_WEEKLY_COLUMNS,
    FeatureLineage,
    build_phase2_features,
)


INJURY_SOURCE_SEASONS = tuple(range(2009, 2025))
INJURY_HISTORY_OFFSETS = (2, 3, 4, 5)
MODEL_POSITIONS = ("QB", "RB", "WR", "TE", "K")
INJURY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/"
    "injuries_{season}.parquet"
)
INJURY_FIELDS = (
    "report_primary_injury",
    "report_secondary_injury",
    "practice_primary_injury",
    "practice_secondary_injury",
)
NONPHYSICAL_PATTERN = re.compile(
    r"^(?:"
    r"illness|covid.*|not\s+injury\s+related.*|non[- ]injury.*|personal|"
    r"rest(?:ed|ing)?(?:\s+(?:player|veteran|vet))?|"
    r"coach(?:'s)?\s+decision|coaching\s+decision"
    r")$",
    flags=re.IGNORECASE,
)
WEEKLY_FEATURE_COLUMNS = tuple(
    dict.fromkeys(
        (*REQUIRED_WEEKLY_COLUMNS, *STABLE_WEEKLY_COLUMNS, *RECENT_WEEKLY_COLUMNS)
    )
)


@dataclass(frozen=True, slots=True)
class InjuryFeatureBundle:
    """Aligned model keys, audit metadata, labels, features, and lineage."""

    keys: pd.DataFrame
    metadata: pd.DataFrame
    labels: pd.DataFrame
    frame: pd.DataFrame
    lineage: Mapping[str, FeatureLineage]

    def __post_init__(self) -> None:
        """Validate player-season alignment and numeric model inputs."""
        # keys: (n, 3); metadata: (n, c_metadata); labels: (n, c_labels)
        # frame: (n, d); every lineage entry describes one of d columns.
        n = len(self.keys)
        if not (len(self.metadata) == len(self.labels) == len(self.frame) == n):
            raise ValueError("Phase 3 feature tables must have identical row counts.")
        if self.keys.duplicated(["target_season", "player_id"]).any():
            raise ValueError("Phase 3 keys must be unique at player-season grain.")
        if not self.frame.columns.equals(pd.Index(self.lineage)):
            raise ValueError("Every Phase 3 feature must have ordered lineage.")
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in self.frame.dtypes):
            raise TypeError("Phase 3 model inputs must be numeric.")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return model columns in fitted order."""
        return tuple(self.frame.columns)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_injury_reports(
    output_directory: Path,
    *,
    seasons: Sequence[int] = INJURY_SOURCE_SEASONS,
    force: bool = False,
) -> dict[str, object]:
    """Download official nflverse injury reports and record byte-level lineage."""
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    session = requests.Session()
    for season in seasons:
        url = INJURY_URL.format(season=int(season))
        path = output_directory / f"injuries_{int(season)}.parquet"
        reused = path.is_file() and not force
        if not reused:
            temporary = path.with_suffix(".parquet.part")
            response = session.get(url, timeout=60)
            response.raise_for_status()
            temporary.write_bytes(response.content)
            temporary.replace(path)
        records.append(
            {
                "season": int(season),
                "url": url,
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "reused": reused,
            }
        )
    manifest: dict[str, object] = {
        "manifest_version": 1,
        "source": "nflverse official weekly injury reports",
        "files": records,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_injury_reports(input_directory: Path) -> pd.DataFrame:
    """Load and validate all declared injury-report seasons."""
    frames: list[pd.DataFrame] = []
    for season in INJURY_SOURCE_SEASONS:
        path = input_directory / f"injuries_{season}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing injury source: {path}")
        frame = pd.read_parquet(path)  # (n_season_reports, 16)
        missing = {"season", "game_type", "week", "gsis_id", *INJURY_FIELDS}.difference(
            frame.columns
        )
        if missing:
            raise ValueError(f"{path.name} lacks injury columns: {sorted(missing)}")
        frames.append(frame)
    reports = pd.concat(frames, ignore_index=True, sort=False)  # (n_reports, 16)
    reports["season"] = pd.to_numeric(reports["season"], errors="raise").astype(
        "int32"
    )  # (n_reports,)
    reports["week"] = pd.to_numeric(reports["week"], errors="coerce")  # (n_reports,)
    reports["gsis_id"] = reports["gsis_id"].astype("string").str.strip()  # (n_reports,)
    if reports["gsis_id"].isna().any() or reports["gsis_id"].eq("").any():
        raise ValueError("Injury reports contain missing GSIS player identifiers.")
    return reports  # (n_reports, 16)


def physical_designation_mask(values: pd.Series) -> pd.Series:
    """Identify physical designations while excluding named nonphysical reasons."""
    # values: (n,)
    normalized = (
        values.astype("string")
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )  # (n,)
    present = normalized.notna() & normalized.ne("")  # (n,)
    excluded = normalized.fillna("").str.fullmatch(NONPHYSICAL_PATTERN)  # (n,)
    return (present & ~excluded).astype("bool")  # (n,)


def aggregate_injury_history(reports: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regular-season physical injury evidence to player-season rows."""
    # reports: (n_reports, c_reports)
    regular = reports.loc[
        reports["game_type"].astype("string").str.upper().eq("REG")
    ].copy()  # (n_regular_reports, c_reports)
    physical_columns: list[str] = []
    for column in INJURY_FIELDS:
        physical_column = f"physical__{column}"
        regular[physical_column] = physical_designation_mask(
            regular[column]
        )  # (n_regular_reports,)
        physical_columns.append(physical_column)
    regular["physical_designation_count"] = regular.loc[:, physical_columns].sum(
        axis=1
    )  # (n_regular_reports,)
    regular["physical_injury_reported"] = regular["physical_designation_count"].gt(
        0
    )  # (n_regular_reports,)
    physical = regular.loc[
        regular["physical_injury_reported"]
    ].copy()  # (n_physical_rows, c_augmented)

    if physical.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "player_id",
                "physical_injury_reported",
                "physical_injury_report_weeks",
                "physical_designation_count",
            ]
        )
    history = (
        physical.groupby(["season", "gsis_id"], observed=True, sort=True)
        .agg(
            physical_injury_report_weeks=("week", "nunique"),
            physical_designation_count=("physical_designation_count", "sum"),
        )
        .reset_index()
        .rename(columns={"gsis_id": "player_id"})
    )  # (n_positive_player_seasons, 4)
    history["physical_injury_reported"] = 1  # (n_positive_player_seasons,)
    history = history.loc[
        :,
        [
            "season",
            "player_id",
            "physical_injury_reported",
            "physical_injury_report_weeks",
            "physical_designation_count",
        ],
    ]  # (n_positive_player_seasons, 5)
    history["season"] = history["season"].astype(
        "int32"
    )  # (n_positive_player_seasons,)
    history["physical_injury_reported"] = history["physical_injury_reported"].astype(
        "int8"
    )  # (n_positive_player_seasons,)
    return history


def _safe_phase2_columns(
    frame: pd.DataFrame,
    lineage: Mapping[str, FeatureLineage],
) -> list[str]:
    """Select core and trajectory inputs that obey the August-origin contract."""
    # frame: (n, d_all)
    columns: list[str] = []
    for column in frame.columns:
        if column.startswith(("room_", "team_", "prior_team_")):
            continue
        feature_lineage = lineage[column]
        if any(
            source == "modeling_table.target_roster"
            for source in feature_lineage.sources
        ):
            continue
        if any(
            source in {"weekly_stats", "player_seasons"} and offset < 1
            for source, offset in zip(
                feature_lineage.sources,
                feature_lineage.source_offsets,
                strict=True,
            )
        ):
            raise RuntimeError(f"Unlagged outcome source reached Phase 3: {column}")
        columns.append(column)
    if not columns:
        raise ValueError("No leakage-safe Phase 2 columns remain for Phase 3.")
    return columns


def _load_weekly_stats(root: Path) -> pd.DataFrame:
    """Load only columns required by the recent Phase 2 feature recipes."""
    frames: list[pd.DataFrame] = []
    stats_directory = root / "data" / "raw" / "stats"
    for path in sorted(stats_directory.glob("stats_player_week_*.parquet")):
        season = int(path.stem.rsplit("_", maxsplit=1)[-1])
        if not 2009 <= season <= 2025:
            continue
        available = set(pq.read_schema(path).names)
        columns = [column for column in WEEKLY_FEATURE_COLUMNS if column in available]
        frame = pd.read_parquet(
            path, columns=columns
        )  # (n_season_player_weeks, c_weekly)
        for column in WEEKLY_FEATURE_COLUMNS:
            if column not in frame:
                frame[column] = np.nan  # (n_season_player_weeks,)
        frames.append(frame.loc[:, WEEKLY_FEATURE_COLUMNS])
    if not frames:
        raise FileNotFoundError(
            f"No weekly player statistics found below {stats_directory}."
        )
    return pd.concat(
        frames, ignore_index=True, sort=False
    )  # (n_player_weeks, c_weekly)


def _injury_history_features(
    keys: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, FeatureLineage]]:
    """Build target-minus-2 through target-minus-5 injury-history features."""
    # keys: (n, 3); history: (n_positive_player_seasons, 5)
    features = pd.DataFrame(index=keys.index)  # (n, 0)
    lineage: dict[str, FeatureLineage] = {}
    value_columns = (
        "physical_injury_reported",
        "physical_injury_report_weeks",
        "physical_designation_count",
    )
    for offset in INJURY_HISTORY_OFFSETS:
        lookup = history.rename(
            columns={"season": "source_season"}
        )  # (n_positive_player_seasons, 5)
        target_keys = keys.loc[:, ["target_season", "player_id"]].copy()  # (n, 2)
        target_keys["source_season"] = target_keys["target_season"] - offset  # (n,)
        merged = target_keys.merge(
            lookup,
            on=["source_season", "player_id"],
            how="left",
            validate="many_to_one",
            sort=False,
        )  # (n, 2 + c_history)
        for source_column in value_columns:
            column = f"injury_lag{offset}_{source_column.removeprefix('physical_')}"
            features[column] = (
                pd.to_numeric(merged[source_column], errors="coerce")
                .fillna(0.0)
                .astype("float32")
            )  # (n,)
            lineage[column] = FeatureLineage(
                column=column,
                sources=("nflverse_injury_reports",),
                source_offsets=(offset,),
                recipe=(
                    f"{source_column} from target_season-{offset}; missing report "
                    "evidence is encoded as 0"
                ),
            )
    return features, lineage


def build_injury_feature_bundle(
    root: Path,
    injury_directory: Path,
) -> InjuryFeatureBundle:
    """Build the complete labeled and current Phase 3 modeling matrix."""
    reports = load_injury_reports(injury_directory)  # (n_reports, 16)
    history = aggregate_injury_history(reports)  # (n_positive_player_seasons, 5)
    cohort = build_august_cohort(
        root,
        current_season=2026,
        minimum_target_season=2013,
        maximum_target_season=2026,
    )
    candidate_table = cohort.table.loc[
        cohort.table["model_position"].isin(MODEL_POSITIONS)
    ].copy()  # (n_candidates, c_candidates)
    weekly_stats = _load_weekly_stats(root)  # (n_player_weeks, c_weekly)
    player_seasons = pd.read_parquet(
        root / "data" / "processed" / "player_seasons.parquet"
    )  # (n_player_seasons, c_player_seasons)
    phase2 = build_phase2_features(
        weekly_stats,
        player_seasons,
        candidate_table,
        tier="recent",
        max_lag=4,
        late_weeks=6,
    )
    safe_columns = _safe_phase2_columns(phase2.frame, phase2.lineage)
    frame = phase2.frame.loc[:, safe_columns].copy()  # (n_candidates, d_safe)
    lineage = {column: phase2.lineage[column] for column in safe_columns}

    position_k = (
        phase2.keys["model_position"].eq("K").astype("float32")
    )  # (n_candidates,)
    frame["position_K"] = position_k  # (n_candidates, d_safe + 1)
    lineage["position_K"] = FeatureLineage(
        column="position_K",
        sources=("modeling_table.cutoff_metadata",),
        source_offsets=(0,),
        recipe="1 when cutoff model_position is K, else 0",
    )
    injury_features, injury_lineage = _injury_history_features(
        phase2.keys,
        history,
    )  # (n_candidates, 12), 12 entries
    frame = pd.concat([frame, injury_features], axis=1)  # (n_candidates, d_total)
    lineage.update(injury_lineage)

    metadata_columns = [
        "target_season",
        "player_id",
        "model_position",
        "team",
        "candidate_name",
        "roster_status",
        "available_for_draft",
        "cohort_source_kind",
        "is_rookie",
        "target_points",
        "previous_points_baseline",
    ]
    metadata_source = candidate_table.loc[
        :, metadata_columns
    ].copy()  # (n_candidates, 11)
    metadata = phase2.keys.merge(
        metadata_source,
        on=["target_season", "player_id", "model_position"],
        how="left",
        validate="one_to_one",
        sort=False,
    )  # (n_candidates, 11)
    positive_labels = history.rename(
        columns={"season": "target_season"}
    )  # (n_positive_player_seasons, 5)
    labels = phase2.keys.merge(
        positive_labels,
        on=["target_season", "player_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )  # (n_candidates, 6)
    for column in (
        "physical_injury_reported",
        "physical_injury_report_weeks",
        "physical_designation_count",
    ):
        labels[column] = pd.to_numeric(labels[column], errors="coerce").fillna(
            0.0
        )  # (n_candidates,)
    labels["physical_injury_reported"] = labels["physical_injury_reported"].astype(
        "int8"
    )  # (n_candidates,)
    frame = frame.astype("float32")  # (n_candidates, d_total)
    return InjuryFeatureBundle(
        keys=phase2.keys,
        metadata=metadata,
        labels=labels,
        frame=frame,
        lineage=lineage,
    )
