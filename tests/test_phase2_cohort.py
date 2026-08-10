"""Tests for the strict Phase 2 offense cohort boundary."""

import pandas as pd
import pytest

from fantasy_football.phase2_cohort import EXCLUDED_STATUS_CODES
from fantasy_football.phase2_cohort import TEAM_CONTROLLED_STATUS_CODES
from fantasy_football.phase2_cohort import build_phase2_offense_cohort


def _mixed_status_table() -> pd.DataFrame:
    return pd.DataFrame(  # (11, 6)
        {
            "target_season": [2026] * 11,
            "player_id": [f"P{index}" for index in range(11)],
            "team": ["AAA"] * 11,
            "model_position": [
                "RB",
                "WR",
                "TE",
                "RB",
                "QB",
                "WR",
                "TE",
                "RB",
                "K",
                "QB",
                "TE",
            ],
            "status": [
                " act ",
                "dev",
                "RES",
                "CUT",
                "RET",
                "UFA",
                "RFA",
                "TRD",
                "ACT",
                None,
                "   ",
            ],
            "signal": list(range(11)),
        }
    )


def test_cohort_includes_allowlist_and_excludes_departed_or_nonoffense_rows() -> None:
    modeling_table = _mixed_status_table()  # (11, 6)
    cohort = build_phase2_offense_cohort(modeling_table)
    filtered_table = cohort.table  # (3, 6)

    assert cohort.rows_before == 11
    assert cohort.rows_after == 3
    assert filtered_table["player_id"].tolist() == ["P0", "P1", "P2"]
    assert filtered_table["status"].tolist() == ["ACT", "DEV", "RES"]
    assert not filtered_table["status"].isin(EXCLUDED_STATUS_CODES).any()

    audit_by_status = cohort.status_counts.set_index("status")  # (20, 2)
    assert audit_by_status.loc["ACT", "rows_before"] == 2
    assert audit_by_status.loc["ACT", "rows_after"] == 1
    assert audit_by_status.loc["CUT", "rows_before"] == 1
    assert audit_by_status.loc["CUT", "rows_after"] == 0
    assert audit_by_status.loc["MISSING", "rows_before"] == 2
    assert audit_by_status.loc["MISSING", "rows_after"] == 0

    room_sizes = filtered_table.groupby(  # (3,)
        ["target_season", "team", "model_position"],
        observed=True,
    ).size()
    assert room_sizes.loc[(2026, "AAA", "RB")] == 1


def test_every_fixed_team_controlled_status_is_retained() -> None:
    n_statuses = len(TEAM_CONTROLLED_STATUS_CODES)
    modeling_table = pd.DataFrame(  # (9, 5)
        {
            "target_season": [2026] * n_statuses,
            "player_id": [f"P{index}" for index in range(n_statuses)],
            "team": ["AAA"] * n_statuses,
            "model_position": ["WR"] * n_statuses,
            "status": list(TEAM_CONTROLLED_STATUS_CODES),
        }
    )

    cohort = build_phase2_offense_cohort(modeling_table)

    assert cohort.rows_after == n_statuses
    assert tuple(cohort.table["status"]) == TEAM_CONTROLLED_STATUS_CODES
    assert cohort.status_counts["rows_after"].sum() == n_statuses


def test_waived_released_and_unmapped_codes_are_excluded() -> None:
    excluded = ("NWT", "RSR", "E01", "TRC", "TRD", "TRT")
    n_statuses = len(excluded)
    modeling_table = pd.DataFrame(  # (6, 5)
        {
            "target_season": [2026] * n_statuses,
            "player_id": [f"P{index}" for index in range(n_statuses)],
            "team": ["AAA"] * n_statuses,
            "model_position": ["WR"] * n_statuses,
            "status": list(excluded),
        }
    )

    cohort = build_phase2_offense_cohort(modeling_table)

    assert cohort.rows_after == 0
    assert (
        cohort.status_counts.loc[
            cohort.status_counts["status"].isin(excluded), "rows_before"
        ].sum()
        == n_statuses
    )


def test_cohort_rejects_unexpected_nonmissing_status() -> None:
    modeling_table = pd.DataFrame(  # (1, 5)
        {
            "target_season": [2026],
            "player_id": ["P0"],
            "team": ["AAA"],
            "model_position": ["RB"],
            "status": [" out "],
        }
    )

    with pytest.raises(ValueError, match="Unexpected status"):
        build_phase2_offense_cohort(modeling_table)


def test_cohort_does_not_mutate_the_caller() -> None:
    modeling_table = _mixed_status_table()  # (11, 6)
    original = modeling_table.copy(deep=True)  # (11, 6)

    cohort = build_phase2_offense_cohort(modeling_table)

    pd.testing.assert_frame_equal(modeling_table, original)
    assert cohort.table is not modeling_table
    assert modeling_table.loc[0, "status"] == " act "
    assert cohort.table.loc[0, "status"] == "ACT"


def test_cohort_rejects_duplicate_keys() -> None:
    modeling_table = _mixed_status_table()  # (11, 6)
    duplicated = pd.concat(  # (12, 6)
        [modeling_table, modeling_table.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="must be unique"):
        build_phase2_offense_cohort(duplicated)


@pytest.mark.parametrize(
    "column", ["target_season", "player_id", "team", "model_position"]
)
def test_cohort_rejects_missing_room_or_identity_keys(column: str) -> None:
    modeling_table = _mixed_status_table()  # (11, 6)
    modeling_table.loc[0, column] = None

    with pytest.raises(ValueError, match=column):
        build_phase2_offense_cohort(modeling_table)


def test_cohort_requires_contract_columns() -> None:
    modeling_table = _mixed_status_table().drop(columns="team")  # (11, 5)

    with pytest.raises(ValueError, match="missing required columns"):
        build_phase2_offense_cohort(modeling_table)
