"""August-origin candidate cohorts without target-season roster look-ahead."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Final


MODEL_POSITIONS: Final = ("QB", "RB", "WR", "TE", "K")
POSITION_MAP: Final = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "HB": "RB",
    "WR": "WR",
    "TE": "TE",
    "K": "K",
    "PK": "K",
}
UNAVAILABLE_CURRENT_STATUSES: Final = frozenset(
    {"CUT", "RET", "RFA", "RSR", "TRC", "TRD", "TRT", "UFA"}
)
COMBINE_COLUMNS: Final = (
    "combine_forty",
    "combine_bench",
    "combine_vertical",
    "combine_broad_jump",
    "combine_cone",
    "combine_shuttle",
)
KEY_COLUMNS: Final = ("target_season", "player_id", "model_position")


@dataclass(frozen=True, slots=True)
class AugustCohort:
    """A target-free candidate definition joined to retrospective outcomes."""

    table: pd.DataFrame
    audit: pd.DataFrame
    current_season: int
    roster_lookback: int

    def __post_init__(self) -> None:
        """Validate the candidate grain and temporal contract."""
        required = {
            *KEY_COLUMNS,
            "target_points",
            "previous_points_baseline",
            "cohort_source_kind",
            "cohort_source_season",
        }
        missing = sorted(required.difference(self.table.columns))
        if missing:
            raise ValueError(f"August cohort is missing columns: {missing!r}.")
        if self.table.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError("August cohort keys must be unique.")
        if not self.table["model_position"].isin(MODEL_POSITIONS).all():
            raise ValueError("August cohort contains an unsupported position.")
        current = self.table["target_season"].eq(self.current_season)  # (n_rows,)
        if self.table.loc[current, "target_points"].notna().any():
            raise ValueError("Current-season outcomes must remain unknown.")
        if self.table.loc[~current, "target_points"].isna().any():
            raise ValueError("Historical candidate outcomes must include explicit zeros.")

        prior_source = self.table["cohort_source_kind"].eq("prior_roster")  # (n_rows,)
        if not (
            self.table.loc[prior_source, "cohort_source_season"]
            < self.table.loc[prior_source, "target_season"]
        ).all():
            raise ValueError("Prior-roster cohort evidence must predate the target season.")
        rookie_source = self.table["cohort_source_kind"].eq("draft_or_combine")  # (n_rows,)
        if not (
            self.table.loc[rookie_source, "cohort_source_season"]
            == self.table.loc[rookie_source, "target_season"]
        ).all():
            raise ValueError("Rookie evidence must come from the target-year draft/combine.")
        current_source = self.table["cohort_source_kind"].eq("august_roster")  # (n_rows,)
        if not (
            self.table.loc[current_source, "target_season"].eq(self.current_season)
        ).all():
            raise ValueError("August-roster evidence is allowed only for current inference.")


def _first_nonmissing(values: pd.Series) -> object:
    """Return the first nonmissing value in stable input order."""
    present = values.dropna()  # (n_present,)
    return present.iloc[0] if not present.empty else np.nan


def _position(values: pd.Series) -> pd.Series:
    """Normalize source positions to the five production cohorts."""
    return values.astype("string").str.upper().map(POSITION_MAP)  # (n_rows,)


def _pseudo_id(season: int, pfr_id: object, name: object, school: object) -> str:
    """Build a deterministic ID for draft/combine candidates without GSIS IDs."""
    if pd.notna(pfr_id) and str(pfr_id).strip():
        return f"pfr:{str(pfr_id).strip()}"
    identity = "|".join(str(value or "").casefold() for value in (name, school))
    slug = re.sub(r"[^a-z0-9]+", "-", identity).strip("-")
    return f"prospect:{season}:{slug or 'unknown'}"


def _player_metadata(raw_dir: Path) -> pd.DataFrame:
    """Load stable player identifiers and biographical metadata."""
    columns = [
        "gsis_id",
        "pfr_id",
        "display_name",
        "birth_date",
        "height",
        "weight",
        "rookie_season",
        "draft_year",
        "draft_round",
        "draft_pick",
    ]
    players = pd.read_parquet(raw_dir / "players.parquet", columns=columns)  # (n_players, 10)
    identified = players["gsis_id"].notna()  # (n_players,)
    players = players.loc[identified].copy()  # (n_identified, 10)
    players.rename(
        columns={
            "gsis_id": "player_id",
            "display_name": "master_name",
            "birth_date": "master_birth_date",
            "height": "master_height",
            "weight": "master_weight",
            "rookie_season": "master_rookie_year",
            "draft_year": "master_draft_year",
            "draft_round": "master_draft_round",
            "draft_pick": "master_draft_number",
        },
        inplace=True,
    )
    players.sort_values("player_id", inplace=True, kind="stable")
    players.drop_duplicates("player_id", keep="last", inplace=True)
    return players  # (n_unique_players, 10)


def _pfr_crosswalk(players: pd.DataFrame) -> pd.DataFrame:
    """Return a unique PFR-to-GSIS crosswalk."""
    crosswalk = players.loc[:, ["pfr_id", "player_id"]].dropna().copy()  # (n_mapped, 2)
    crosswalk.sort_values(["pfr_id", "player_id"], inplace=True, kind="stable")
    crosswalk.drop_duplicates("pfr_id", keep="first", inplace=True)
    return crosswalk  # (n_unique_pfr, 2)


def _combine_candidates(raw_dir: Path, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Build target-year prospect candidates from combine participants."""
    combine = pd.read_parquet(raw_dir / "combine.parquet")  # (n_combine, c_combine)
    combine["model_position"] = _position(combine["pos"])  # (n_combine,)
    eligible = combine["model_position"].notna() & combine["season"].notna()  # (n_combine,)
    combine = combine.loc[eligible].copy()  # (n_eligible, c_combine + 1)
    combine["target_season"] = pd.to_numeric(combine["season"], errors="raise").astype(
        "int32"
    )
    combine = combine.merge(
        crosswalk,
        on="pfr_id",
        how="left",
        validate="many_to_one",
    )  # (n_eligible, c_joined)
    missing_id = combine["player_id"].isna()  # (n_eligible,)
    combine.loc[missing_id, "player_id"] = [
        _pseudo_id(season, pfr_id, name, school)
        for season, pfr_id, name, school in zip(
            combine.loc[missing_id, "target_season"],
            combine.loc[missing_id, "pfr_id"],
            combine.loc[missing_id, "player_name"],
            combine.loc[missing_id, "school"],
            strict=True,
        )
    ]
    combine.rename(
        columns={
            "player_name": "candidate_name",
            "ht": "combine_height",
            "wt": "combine_weight",
            "forty": "combine_forty",
            "bench": "combine_bench",
            "vertical": "combine_vertical",
            "broad_jump": "combine_broad_jump",
            "cone": "combine_cone",
            "shuttle": "combine_shuttle",
            "draft_round": "prospect_draft_round",
            "draft_ovr": "prospect_draft_number",
        },
        inplace=True,
    )
    combine["cohort_source_kind"] = "draft_or_combine"
    combine["cohort_source_season"] = combine["target_season"]
    combine["team"] = combine["draft_team"].fillna("UNK")
    selected = [
        "target_season",
        "player_id",
        "model_position",
        "team",
        "candidate_name",
        "pfr_id",
        "combine_height",
        "combine_weight",
        *COMBINE_COLUMNS,
        "prospect_draft_round",
        "prospect_draft_number",
        "cohort_source_kind",
        "cohort_source_season",
    ]
    return combine.loc[:, selected]  # (n_combine_candidates, 19)


def _draft_candidates(raw_dir: Path, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Build target-year rookie candidates from the NFL draft."""
    draft = pd.read_parquet(raw_dir / "draft_picks.parquet")  # (n_draft, c_draft)
    draft["model_position"] = _position(draft["position"])  # (n_draft,)
    eligible = draft["model_position"].notna() & draft["season"].notna()  # (n_draft,)
    draft = draft.loc[eligible].copy()  # (n_eligible, c_draft + 1)
    draft["target_season"] = pd.to_numeric(draft["season"], errors="raise").astype(
        "int32"
    )
    draft.rename(columns={"pfr_player_id": "pfr_id"}, inplace=True)
    draft = draft.merge(
        crosswalk,
        on="pfr_id",
        how="left",
        validate="many_to_one",
    )  # (n_eligible, c_joined)
    missing_id = draft["player_id"].isna()  # (n_eligible,)
    draft.loc[missing_id, "player_id"] = [
        _pseudo_id(season, pfr_id, name, school)
        for season, pfr_id, name, school in zip(
            draft.loc[missing_id, "target_season"],
            draft.loc[missing_id, "pfr_id"],
            draft.loc[missing_id, "pfr_player_name"],
            draft.loc[missing_id, "college"],
            strict=True,
        )
    ]
    draft["candidate_name"] = draft["pfr_player_name"]
    draft["prospect_draft_round"] = pd.to_numeric(draft["round"], errors="coerce")
    draft["prospect_draft_number"] = pd.to_numeric(draft["pick"], errors="coerce")
    draft["team"] = draft["team"].fillna("UNK")
    draft["cohort_source_kind"] = "draft_or_combine"
    draft["cohort_source_season"] = draft["target_season"]
    for column in ("combine_height", "combine_weight", *COMBINE_COLUMNS):
        draft[column] = np.nan
    selected = [
        "target_season",
        "player_id",
        "model_position",
        "team",
        "candidate_name",
        "pfr_id",
        "combine_height",
        "combine_weight",
        *COMBINE_COLUMNS,
        "prospect_draft_round",
        "prospect_draft_number",
        "cohort_source_kind",
        "cohort_source_season",
    ]
    return draft.loc[:, selected]  # (n_draft_candidates, 19)


def _rookie_candidates(raw_dir: Path, players: pd.DataFrame) -> pd.DataFrame:
    """Union combine and draft entrants without using target-season participation."""
    crosswalk = _pfr_crosswalk(players)
    candidates = pd.concat(
        [
            _combine_candidates(raw_dir, crosswalk),
            _draft_candidates(raw_dir, crosswalk),
        ],
        ignore_index=True,
    )  # (n_prospect_records, 19)
    candidates.sort_values(
        ["target_season", "player_id", "prospect_draft_number"],
        na_position="last",
        inplace=True,
        kind="stable",
    )
    value_columns = [
        column
        for column in candidates.columns
        if column not in {"target_season", "player_id"}
    ]
    rookies = (
        candidates.groupby(["target_season", "player_id"], observed=True)[value_columns]
        .agg(_first_nonmissing)
        .reset_index()
    )  # (n_rookie_candidates, 19)
    return rookies


def _prior_roster_candidates(
    raw_dir: Path,
    minimum_target_season: int,
    maximum_target_season: int,
) -> pd.DataFrame:
    """Use only the completed prior season to define historical veterans."""
    frames: list[pd.DataFrame] = []
    for target_season in range(minimum_target_season, maximum_target_season + 1):
        source_season = target_season - 1
        path = raw_dir / "rosters" / f"roster_weekly_{source_season}.parquet"
        roster = pd.read_parquet(
            path,
            columns=["season", "team", "position", "gsis_id", "week", "game_type"],
        )  # (n_roster_rows, 6)
        roster["model_position"] = _position(roster["position"])  # (n_roster_rows,)
        eligible = (
            roster["model_position"].notna()
            & roster["gsis_id"].notna()
            & roster["game_type"].eq("REG")
        )  # (n_roster_rows,)
        roster = roster.loc[eligible].copy()  # (n_eligible, 7)
        roster["week"] = pd.to_numeric(roster["week"], errors="coerce")
        roster.sort_values(
            ["gsis_id", "week", "team"],
            inplace=True,
            kind="stable",
        )
        roster.drop_duplicates("gsis_id", keep="last", inplace=True)
        frame = roster.loc[:, ["gsis_id", "team", "model_position"]].rename(
            columns={"gsis_id": "player_id"}
        )  # (n_veterans, 3)
        frame["target_season"] = target_season
        frame["cohort_source_kind"] = "prior_roster"
        frame["cohort_source_season"] = source_season
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)  # (n_veteran_candidates, 6)


def _current_candidates(processed_dir: Path, current_season: int) -> pd.DataFrame:
    """Use every fantasy player in the actual August roster for inference."""
    roster = pd.read_parquet(processed_dir / "preseason_players.parquet")  # (n_rows, c_roster)
    roster["model_position"] = _position(roster["position"])  # (n_rows,)
    eligible = (
        roster["season"].eq(current_season)
        & roster["model_position"].notna()
        & roster["player_id"].notna()
    )  # (n_rows,)
    roster = roster.loc[eligible].copy()  # (n_current, c_roster + 1)
    roster.sort_values(["player_id", "team"], inplace=True, kind="stable")
    roster.drop_duplicates("player_id", keep="first", inplace=True)
    current = roster.loc[
        :,
        ["player_id", "team", "model_position", "full_name", "status"],
    ].rename(
        columns={"full_name": "candidate_name", "status": "roster_status"}
    )  # (n_current, 5)
    current["target_season"] = current_season
    current["cohort_source_kind"] = "august_roster"
    current["cohort_source_season"] = current_season
    current["available_for_draft"] = ~current["roster_status"].isin(
        UNAVAILABLE_CURRENT_STATUSES
    )
    return current  # (n_current, 9)


def _parse_height(values: pd.Series) -> pd.Series:
    """Convert combine height strings such as ``6-2`` to inches."""
    text = values.astype("string")  # (n_rows,)
    parts = text.str.extract(r"^(\d+)-(\d+)$")  # (n_rows, 2)
    feet = pd.to_numeric(parts[0], errors="coerce")  # (n_rows,)
    inches = pd.to_numeric(parts[1], errors="coerce")  # (n_rows,)
    return feet.mul(12).add(inches)  # (n_rows,)


def _enrich_metadata(
    candidates: pd.DataFrame,
    players: pd.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    """Attach only immutable or target-date-known metadata."""
    enriched = candidates.merge(
        players,
        on="player_id",
        how="left",
        validate="many_to_one",
    )  # (n_candidates, c_joined)
    if {"pfr_id_x", "pfr_id_y"}.issubset(enriched.columns):
        enriched["pfr_id"] = enriched["pfr_id_x"].combine_first(enriched["pfr_id_y"])
        enriched.drop(columns=["pfr_id_x", "pfr_id_y"], inplace=True)

    combine = pd.read_parquet(
        raw_dir / "combine.parquet",
        columns=[
            "season",
            "pfr_id",
            "ht",
            "wt",
            "forty",
            "bench",
            "vertical",
            "broad_jump",
            "cone",
            "shuttle",
        ],
    )  # (n_combine, 10)
    combine = combine[combine["pfr_id"].notna()].copy()  # (n_mapped_combine, 10)
    combine.sort_values(["pfr_id", "season"], inplace=True, kind="stable")
    combine.drop_duplicates("pfr_id", keep="last", inplace=True)
    combine.rename(
        columns={
            "ht": "lookup_combine_height",
            "wt": "lookup_combine_weight",
            "forty": "lookup_combine_forty",
            "bench": "lookup_combine_bench",
            "vertical": "lookup_combine_vertical",
            "broad_jump": "lookup_combine_broad_jump",
            "cone": "lookup_combine_cone",
            "shuttle": "lookup_combine_shuttle",
        },
        inplace=True,
    )
    enriched = enriched.merge(
        combine.drop(columns="season"),
        on="pfr_id",
        how="left",
        validate="many_to_one",
    )  # (n_candidates, c_joined + 8)
    for column in ("combine_height", "combine_weight", *COMBINE_COLUMNS):
        lookup_column = f"lookup_{column}"
        if column not in enriched:
            enriched[column] = enriched[lookup_column]
        else:
            enriched[column] = enriched[column].combine_first(enriched[lookup_column])
        enriched.drop(columns=lookup_column, inplace=True)
    enriched["candidate_name"] = enriched.get("candidate_name").combine_first(
        enriched["master_name"]
    )
    birth = pd.to_datetime(enriched["master_birth_date"], errors="coerce")  # (n_candidates,)
    reference = pd.to_datetime(
        enriched["target_season"].astype("int32").astype(str) + "-09-01",
        errors="coerce",
    )  # (n_candidates,)
    enriched["age"] = (reference - birth).dt.days / 365.2425
    enriched["height"] = pd.to_numeric(enriched["master_height"], errors="coerce")
    if "combine_height" in enriched:
        enriched["height"] = enriched["height"].combine_first(
            _parse_height(enriched["combine_height"])
        )
    enriched["weight"] = pd.to_numeric(enriched["master_weight"], errors="coerce")
    if "combine_weight" in enriched:
        enriched["weight"] = enriched["weight"].combine_first(
            pd.to_numeric(enriched["combine_weight"], errors="coerce")
        )

    rookie_year = pd.to_numeric(enriched["master_rookie_year"], errors="coerce")
    prospect = enriched["cohort_source_kind"].eq("draft_or_combine")  # (n_candidates,)
    rookie_year = rookie_year.where(~rookie_year.isna(), enriched["target_season"].where(prospect))
    enriched["is_rookie"] = enriched["target_season"].eq(rookie_year).astype("int8")
    enriched["years_exp"] = (
        enriched["target_season"].sub(rookie_year).clip(lower=0)
    )

    draft_year = pd.to_numeric(enriched["master_draft_year"], errors="coerce")
    draft_round = pd.to_numeric(enriched["master_draft_round"], errors="coerce")
    draft_number = pd.to_numeric(enriched["master_draft_number"], errors="coerce")
    if "prospect_draft_round" in enriched:
        draft_round = draft_round.combine_first(
            pd.to_numeric(enriched["prospect_draft_round"], errors="coerce")
        )
    if "prospect_draft_number" in enriched:
        draft_number = draft_number.combine_first(
            pd.to_numeric(enriched["prospect_draft_number"], errors="coerce")
        )
    drafted_by_cutoff = draft_year.isna() | draft_year.le(enriched["target_season"])
    enriched["draft_round"] = draft_round.where(drafted_by_cutoff)
    enriched["draft_number"] = draft_number.where(drafted_by_cutoff)
    enriched["was_drafted"] = enriched["draft_number"].notna().astype("int8")
    enriched["team_changed"] = 0
    for column in COMBINE_COLUMNS:
        if column not in enriched:
            enriched[column] = np.nan
    return enriched


def _attach_outcomes(
    candidates: pd.DataFrame,
    player_seasons: pd.DataFrame,
    current_season: int,
) -> pd.DataFrame:
    """Join observed outcomes after cohort construction and preserve current nulls."""
    outcomes = player_seasons.loc[
        :, ["season", "player_id", "target_points"]
    ].rename(columns={"season": "target_season"})  # (n_player_seasons, 3)
    enriched = candidates.merge(
        outcomes,
        on=["target_season", "player_id"],
        how="left",
        validate="many_to_one",
    )  # (n_candidates, c_candidates + 1)
    historical = enriched["target_season"].lt(current_season)  # (n_candidates,)
    enriched.loc[historical, "target_points"] = enriched.loc[
        historical, "target_points"
    ].fillna(0.0)
    enriched.loc[~historical, "target_points"] = np.nan

    previous = player_seasons.loc[
        :, ["season", "player_id", "target_points"]
    ].copy()  # (n_player_seasons, 3)
    previous["target_season"] = previous["season"] + 1
    previous.rename(
        columns={"target_points": "previous_points_baseline"},
        inplace=True,
    )
    enriched = enriched.merge(
        previous.loc[:, ["target_season", "player_id", "previous_points_baseline"]],
        on=["target_season", "player_id"],
        how="left",
        validate="many_to_one",
    )  # (n_candidates, c_candidates + 2)
    enriched["previous_points_baseline"] = enriched[
        "previous_points_baseline"
    ].fillna(0.0)
    return enriched


def build_august_cohort(
    root: Path,
    *,
    current_season: int = 2026,
    minimum_target_season: int = 2006,
    maximum_target_season: int | None = None,
    roster_lookback: int = 1,
) -> AugustCohort:
    """Build the fixed August 9 candidate universe and current inference rows."""
    if roster_lookback != 1:
        raise ValueError("The locked August contract uses exactly one prior roster season.")
    if minimum_target_season < 2003 or minimum_target_season >= current_season:
        raise ValueError("minimum_target_season is outside the supported range.")
    maximum_target = (
        current_season
        if maximum_target_season is None
        else int(maximum_target_season)
    )
    if not minimum_target_season <= maximum_target <= current_season:
        raise ValueError("maximum_target_season is outside the supported range.")

    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    players = _player_metadata(raw_dir)
    player_seasons = pd.read_parquet(processed_dir / "player_seasons.parquet")
    veterans = _prior_roster_candidates(
        raw_dir,
        minimum_target_season,
        min(maximum_target, current_season - 1),
    )
    rookies = _rookie_candidates(raw_dir, players)
    rookies = rookies.loc[
        rookies["target_season"].between(
            minimum_target_season,
            min(maximum_target, current_season - 1),
        )
    ].copy()  # (n_historical_rookies, c_rookie)
    historical = pd.concat([veterans, rookies], ignore_index=True, sort=False)
    historical["source_priority"] = historical["cohort_source_kind"].map(
        {"prior_roster": 0, "draft_or_combine": 1}
    )
    historical.sort_values(
        ["target_season", "player_id", "source_priority"],
        inplace=True,
        kind="stable",
    )
    historical.drop_duplicates(
        ["target_season", "player_id"],
        keep="first",
        inplace=True,
    )
    historical.drop(columns="source_priority", inplace=True)
    candidate_frames = [historical]
    if maximum_target == current_season:
        candidate_frames.append(_current_candidates(processed_dir, current_season))
    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False)
    candidates = _enrich_metadata(candidates, players, raw_dir)
    candidates = _attach_outcomes(candidates, player_seasons, current_season)

    selected_columns = [
        "target_season",
        "player_id",
        "model_position",
        "team",
        "candidate_name",
        "roster_status",
        "available_for_draft",
        "cohort_source_kind",
        "cohort_source_season",
        "age",
        "years_exp",
        "height",
        "weight",
        "draft_number",
        "draft_round",
        "was_drafted",
        "is_rookie",
        "team_changed",
        *COMBINE_COLUMNS,
        "target_points",
        "previous_points_baseline",
    ]
    for column in selected_columns:
        if column not in candidates:
            candidates[column] = np.nan
    table = candidates.loc[:, selected_columns].copy()  # (n_candidates, c_modeling)
    table["target_season"] = pd.to_numeric(
        table["target_season"], errors="raise"
    ).astype("int32")
    table["available_for_draft"] = (
        table["available_for_draft"].astype("boolean").fillna(True).astype("bool")
    )
    table.sort_values(
        ["target_season", "model_position", "player_id"],
        inplace=True,
        kind="stable",
    )
    table.reset_index(drop=True, inplace=True)

    audit = (
        table.assign(
            zero_target=table["target_points"].eq(0.0),
            pseudo_id=table["player_id"].str.startswith(("pfr:", "prospect:")),
        )
        .groupby(
            ["target_season", "cohort_source_kind"],
            observed=True,
            dropna=False,
        )
        .agg(
            rows=("player_id", "size"),
            zero_targets=("zero_target", "sum"),
            pseudo_ids=("pseudo_id", "sum"),
        )
        .reset_index()
    )  # (n_season_source_groups, 5)
    return AugustCohort(table, audit, current_season, roster_lookback)
