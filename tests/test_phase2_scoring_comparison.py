"""Tests for common reception-format scoring comparisons."""

import numpy as np
import pandas as pd
import pytest

from fantasy_football.phase2_scoring_comparison import aggregate_player_seasons
from fantasy_football.phase2_scoring_comparison import correlation_records
from fantasy_football.phase2_scoring_comparison import scoring_correlations


def _player_games() -> pd.DataFrame:
    """Return a compact offense-and-kicker scoring fixture."""
    return pd.DataFrame(  # (5, 8)
        {
            "player_id": ["qb", "wr", "wr", "k", "db"],
            "position": ["QB", "WR", "WR", "K", "DB"],
            "season": [2024, 2024, 2024, 2024, 2024],
            "season_type": ["REG"] * 5,
            "recomputed_non_ppr_points": [20.0, 10.0, 5.0, 0.0, 8.0],
            "recomputed_half_ppr_points": [20.0, 12.0, 6.0, 0.0, 8.0],
            "recomputed_ppr_points": [20.0, 14.0, 7.0, 0.0, 8.0],
            "recomputed_kicker_points": [0.0, 0.0, 0.0, 9.0, 0.0],
        }
    )


def test_player_season_aggregation_adds_kicker_points_and_merges_games() -> None:
    season_scores = aggregate_player_seasons(_player_games())

    assert len(season_scores) == 3
    wide_receiver = season_scores.loc[season_scores["player_id"].eq("wr")].iloc[0]
    kicker = season_scores.loc[season_scores["player_id"].eq("k")].iloc[0]
    assert wide_receiver["non_ppr_points"] == 15.0
    assert wide_receiver["half_ppr_points"] == 18.0
    assert wide_receiver["full_ppr_points"] == 21.0
    assert (
        kicker[["non_ppr_points", "half_ppr_points", "full_ppr_points"]].eq(9.0).all()
    )


def test_correlation_matrices_are_symmetric_and_auditable() -> None:
    season_scores = aggregate_player_seasons(_player_games())
    correlations = scoring_correlations(season_scores)
    records = correlation_records(correlations, len(season_scores))

    assert set(correlations) == {"Pearson", "Spearman"}
    for matrix in correlations.values():
        # matrix: (3, 3)
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 1.0)
    assert records.shape == (18, 5)
    assert records["correlation"].between(-1.0, 1.0).all()
    assert records["player_seasons"].eq(3).all()


def test_missing_scoring_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        aggregate_player_seasons(_player_games().drop(columns="season_type"))
