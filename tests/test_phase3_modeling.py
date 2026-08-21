"""Test Phase 3 candidate selection, controls, and probability metrics."""

import numpy as np
import pandas as pd

from fantasy_football.phase3_modeling import (
    candidate_grid,
    feature_association_table,
    injury_probability_from_log_odds,
    permute_labels_within_season_position,
    probability_metrics,
    select_candidate,
)


def test_candidate_grid_matches_the_locked_plan() -> None:
    candidates = candidate_grid()

    assert len(candidates) == 21
    assert len({candidate.identifier for candidate in candidates}) == 21
    assert sum(candidate.family == "elastic_net" for candidate in candidates) == 8
    assert sum(candidate.family == "extra_trees" for candidate in candidates) == 8
    assert sum(candidate.family == "hist_gradient" for candidate in candidates) == 4


def test_near_best_selection_uses_log_loss_then_feature_count() -> None:
    summary = pd.DataFrame(
        {
            "candidate": ["highest_auc", "better_probability", "too_far"],
            "mean_roc_auc": [0.700, 0.697, 0.694],
            "mean_log_loss": [0.64, 0.61, 0.55],
            "mean_feature_count": [16.0, 32.0, 8.0],
        }
    )  # (3, 4)

    assert select_candidate(summary) == "better_probability"


def test_grouped_permutation_preserves_each_group_prevalence() -> None:
    metadata = pd.DataFrame(
        {
            "target_season": [2020, 2020, 2020, 2021, 2021, 2021],
            "model_position": ["RB", "RB", "WR", "RB", "RB", "WR"],
        }
    )  # (6, 2)
    labels = pd.Series([0, 1, 1, 1, 0, 0], dtype="int8")  # (6,)

    permuted = permute_labels_within_season_position(labels, metadata, seed=7)  # (6,)
    original_counts = metadata.assign(label=labels).groupby(
        ["target_season", "model_position"]
    )["label"].sum()  # (4,)
    permuted_counts = metadata.assign(label=permuted).groupby(
        ["target_season", "model_position"]
    )["label"].sum()  # (4,)

    pd.testing.assert_series_equal(original_counts, permuted_counts)


def test_associations_and_probability_metrics_have_expected_direction() -> None:
    frame = pd.DataFrame(
        {
            "risk": [0.0, 0.1, 0.8, 0.9],
            "protective": [0.9, 0.8, 0.1, 0.0],
        }
    )  # (4, 2)
    labels = pd.Series([0, 0, 1, 1], dtype="int8")  # (4,)

    associations = feature_association_table(frame, labels)  # (2, 13)
    metrics = probability_metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]))

    assert set(associations["direction"]) == {"higher_risk", "lower_risk"}
    assert associations["discriminative_auc"].eq(1.0).all()
    assert metrics["roc_auc"] == 1.0
    np.testing.assert_allclose(
        injury_probability_from_log_odds(np.array([0.0])),
        np.array([0.5]),
    )
