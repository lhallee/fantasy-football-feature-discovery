"""Test Phase 3 injury-report labels and temporal lineage."""

import pandas as pd

from fantasy_football.phase3_injury import (
    INJURY_FIELDS,
    aggregate_injury_history,
    physical_designation_mask,
)


def test_physical_designations_exclude_only_named_nonphysical_reasons() -> None:
    values = pd.Series(
        [
            None,
            "Illness",
            "Not injury related - personal matter",
            "Resting Veteran",
            "COVID Ramp Up",
            "Knee",
            "Illness, Shoulder",
            "Calf, Resting Veteran",
        ],
        dtype="string",
    )  # (8,)

    observed = physical_designation_mask(values)  # (8,)

    assert observed.tolist() == [False, False, False, False, False, True, True, True]


def test_aggregate_history_uses_regular_season_physical_rows_only() -> None:
    reports = pd.DataFrame(
        {
            "season": [2020, 2020, 2020, 2020],
            "game_type": ["REG", "REG", "REG", "WC"],
            "week": [1, 2, 3, 19],
            "gsis_id": ["p1", "p1", "p2", "p3"],
            "report_primary_injury": ["Knee", "Rest", "Illness", "Ankle"],
            "report_secondary_injury": [None, None, None, None],
            "practice_primary_injury": [None, "Shoulder", None, None],
            "practice_secondary_injury": [None, None, None, None],
        }
    )  # (4, 8)

    history = aggregate_injury_history(reports)  # (1, 5)

    assert tuple(INJURY_FIELDS) == tuple(
        column
        for column in reports.columns
        if column.endswith(("primary_injury", "secondary_injury"))
    )
    assert history["player_id"].tolist() == ["p1"]
    assert history["physical_injury_reported"].tolist() == [1]
    assert history["physical_injury_report_weeks"].tolist() == [2]
    assert history["physical_designation_count"].tolist() == [2]
