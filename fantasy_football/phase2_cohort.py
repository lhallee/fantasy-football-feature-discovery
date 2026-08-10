"""Build the strict post-cut offense cohort for Phase 2 experiments."""

from __future__ import annotations

import pandas as pd

from dataclasses import dataclass


KEY_COLUMNS = ("target_season", "player_id", "model_position")
REQUIRED_COLUMNS = (*KEY_COLUMNS, "team", "status")
REQUIRED_STRING_KEYS = ("player_id", "model_position", "team")
OFFENSE_POSITIONS = ("QB", "RB", "WR", "TE")
TEAM_CONTROLLED_STATUS_CODES = (
    "ACT",
    "DEV",
    "RES",
    "INA",
    "EXE",
    "E14",
    "PUP",
    "SUS",
    "RSN",
)
EXCLUDED_STATUS_CODES = (
    "CUT",
    "RET",
    "UFA",
    "RFA",
    "NWT",
    "RSR",
    "E01",
    "TRC",
    "TRD",
    "TRT",
    "MISSING",
)
AUDITED_STATUS_CODES = (*TEAM_CONTROLLED_STATUS_CODES, *EXCLUDED_STATUS_CODES)


@dataclass(frozen=True, slots=True)
class Phase2Cohort:
    """Hold the filtered offense rows and their status audit."""

    table: pd.DataFrame
    status_counts: pd.DataFrame
    rows_before: int
    rows_after: int

    def __post_init__(self) -> None:
        """Check the returned cohort and audit invariants."""
        if tuple(self.status_counts.columns) != (
            "status",
            "rows_before",
            "rows_after",
        ):
            raise ValueError("Status audit columns do not match the cohort contract.")
        if self.rows_before < self.rows_after or self.rows_after != len(self.table):
            raise ValueError("Cohort row counts are inconsistent.")
        if int(self.status_counts["rows_before"].sum()) != self.rows_before:
            raise ValueError("Status audit does not account for every input row.")
        if int(self.status_counts["rows_after"].sum()) != self.rows_after:
            raise ValueError("Status audit does not account for every retained row.")
        if self.table["status"].isna().any():
            raise ValueError("Retained cohort statuses must be non-null.")
        for column in REQUIRED_STRING_KEYS:
            values = self.table[column].astype("string").str.strip()  # (n_after,)
            if values.isna().any() or values.eq("").any():
                raise ValueError(f"Retained cohort {column} values must be present.")
        if self.table["target_season"].isna().any():
            raise ValueError("Retained cohort target seasons must be present.")
        if not self.table["status"].isin(TEAM_CONTROLLED_STATUS_CODES).all():
            raise ValueError("Retained cohort contains a non-allowlisted status.")
        if not self.table["model_position"].isin(OFFENSE_POSITIONS).all():
            raise ValueError("Retained cohort contains a non-offense position.")
        if self.table.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError("Retained cohort keys must be unique.")


def _require_columns(modeling_table: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(modeling_table.columns))
    if missing_columns:
        raise ValueError(
            f"modeling_table is missing required columns: {missing_columns!r}."
        )


def build_phase2_offense_cohort(modeling_table: pd.DataFrame) -> Phase2Cohort:
    """Filter offense rows with a fixed, audited post-cut status vocabulary."""
    # modeling_table: (n, c)
    _require_columns(modeling_table)
    rows_before = len(modeling_table)
    cohort_source = modeling_table.copy()  # (n, c)
    for column in REQUIRED_STRING_KEYS:
        normalized = (  # (n,)
            cohort_source[column].astype("string").str.strip().str.upper()
        )
        invalid = normalized.isna() | normalized.eq("")  # (n,)
        if invalid.any():
            raise ValueError(
                f"{column} contains {int(invalid.sum())} missing value(s)."
            )
        cohort_source[column] = normalized  # (n, c)
    target_season = pd.to_numeric(  # (n,)
        cohort_source["target_season"],
        errors="coerce",
    )
    invalid_season = target_season.isna() | target_season.mod(1).ne(0)  # (n,)
    if invalid_season.any():
        raise ValueError(
            f"target_season contains {int(invalid_season.sum())} invalid value(s)."
        )
    cohort_source["target_season"] = target_season.astype("int64")  # (n, c)
    duplicate_keys = cohort_source.duplicated(list(KEY_COLUMNS))  # (n,)
    if duplicate_keys.any():
        raise ValueError(
            "modeling_table must be unique by target_season, player_id, and "
            "model_position after key normalization."
        )

    normalized_status = (
        cohort_source["status"].astype("string").str.strip().str.upper()
    )  # (n,)
    missing_status = normalized_status.isna() | normalized_status.eq("")  # (n,)
    normalized_status = normalized_status.mask(  # (n,)
        missing_status,
        "MISSING",
    )

    known_status = normalized_status.isin(AUDITED_STATUS_CODES)  # (n,)
    if not known_status.all():
        unexpected_values = tuple(
            sorted(normalized_status.loc[~known_status].unique().tolist())
        )
        raise ValueError(f"Unexpected status value(s): {unexpected_values!r}.")

    cohort_source["status"] = normalized_status  # (n, c)
    offense_row = cohort_source["model_position"].isin(OFFENSE_POSITIONS)  # (n,)
    team_controlled = normalized_status.isin(TEAM_CONTROLLED_STATUS_CODES)  # (n,)
    retained_row = offense_row & team_controlled  # (n,)
    filtered_table = cohort_source.loc[retained_row].copy()  # (n_after, c)
    filtered_table.reset_index(drop=True, inplace=True)  # (n_after, c)
    rows_after = len(filtered_table)

    before_by_status = normalized_status.value_counts(sort=False)  # (s_present,)
    after_by_status = filtered_table["status"].value_counts(sort=False)  # (s_kept,)
    status_counts = pd.DataFrame(  # (20, 3)
        {
            "status": AUDITED_STATUS_CODES,
            "rows_before": [
                int(before_by_status.get(status, 0)) for status in AUDITED_STATUS_CODES
            ],
            "rows_after": [
                int(after_by_status.get(status, 0)) for status in AUDITED_STATUS_CODES
            ],
        }
    )

    assert not filtered_table["status"].isin(EXCLUDED_STATUS_CODES).any()
    return Phase2Cohort(
        table=filtered_table,
        status_counts=status_counts,
        rows_before=rows_before,
        rows_after=rows_after,
    )
