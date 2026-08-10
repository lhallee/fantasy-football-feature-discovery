"""Tests for the fixed August-origin Phase 2 candidate universe."""

from pathlib import Path

import pandas as pd
import pytest

from fantasy_football.phase2_august import (
    KEY_COLUMNS,
    MODEL_POSITIONS,
    POSITION_MAP,
    build_august_cohort,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def august_table() -> pd.DataFrame:
    """Build the real protected-data cohort once for contract tests."""
    return build_august_cohort(ROOT).table


def test_historical_membership_uses_no_target_season_roster(
    august_table: pd.DataFrame,
) -> None:
    """Historical rows must come from prior rosters or target-year prospects."""
    historical = august_table[august_table["target_season"].lt(2026)]  # (n_history, c)
    assert set(historical["cohort_source_kind"]) == {
        "prior_roster",
        "draft_or_combine",
    }
    prior = historical["cohort_source_kind"].eq("prior_roster")  # (n_history,)
    assert (
        historical.loc[prior, "cohort_source_season"]
        == historical.loc[prior, "target_season"] - 1
    ).all()
    prospects = historical["cohort_source_kind"].eq("draft_or_combine")  # (n_history,)
    assert (
        historical.loc[prospects, "cohort_source_season"]
        == historical.loc[prospects, "target_season"]
    ).all()


def test_nonparticipants_are_explicit_zero_targets(august_table: pd.DataFrame) -> None:
    """Outcome joins happen after selection and retain failed prospects as zeros."""
    historical = august_table[august_table["target_season"].lt(2026)]  # (n_history, c)
    assert historical["target_points"].notna().all()
    prospects = historical["cohort_source_kind"].eq("draft_or_combine")  # (n_history,)
    pseudo = historical["player_id"].str.startswith(("pfr:", "prospect:"))  # (n_history,)
    assert pseudo.any()
    assert historical.loc[prospects & pseudo, "target_points"].eq(0.0).all()


def test_current_rows_are_complete_august_roster_with_availability_flag(
    august_table: pd.DataFrame,
) -> None:
    """Current inference may use the frozen August roster but no 2026 outcomes."""
    current = august_table[august_table["target_season"].eq(2026)]  # (n_current, c)
    source = pd.read_parquet(ROOT / "data" / "processed" / "preseason_players.parquet")
    model_position = source["position"].astype("string").str.upper().map(POSITION_MAP)
    unavailable = source["status"].astype("string").str.upper().isin(
        {"CUT", "RET", "RFA", "RSR", "TRC", "TRD", "TRT", "UFA"}
    )
    expected = source[
        source["season"].eq(2026)
        & model_position.notna()
        & source["player_id"].notna()
    ]["player_id"].drop_duplicates()
    assert set(current["player_id"]) == set(expected)
    expected_available = set(
        source.loc[
            source["season"].eq(2026)
            & model_position.notna()
            & source["player_id"].notna()
            & ~unavailable,
            "player_id",
        ]
    )
    actual_available = set(current.loc[current["available_for_draft"], "player_id"])
    assert actual_available == expected_available
    assert current["target_points"].isna().all()
    assert current["cohort_source_kind"].eq("august_roster").all()


def test_august_keys_and_positions_are_complete(august_table: pd.DataFrame) -> None:
    """Every row must have one unique supported player-season-position key."""
    assert not august_table.duplicated(list(KEY_COLUMNS)).any()
    assert august_table.loc[:, list(KEY_COLUMNS)].notna().all().all()
    assert set(august_table["model_position"]).issubset(MODEL_POSITIONS)


def test_discovery_build_never_materializes_later_target_seasons() -> None:
    """A bounded build must exclude audit and current rows before outcome joins."""
    discovery = build_august_cohort(ROOT, maximum_target_season=2021).table

    assert int(discovery["target_season"].max()) == 2021
    assert discovery["target_points"].notna().all()
    assert not discovery["cohort_source_kind"].eq("august_roster").any()
