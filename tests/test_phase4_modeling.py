"""Test Phase 4 selection rules, combination scoring, and interval tables."""

import numpy as np
import pandas as pd
import pytest

from fantasy_football.phase4_modeling import (
    PHASE2_REFERENCE,
    feature_association_table,
    PointsCandidate,
    apply_residual_intervals,
    assemble_candidate_predictions,
    combined_score,
    evaluate_injury_weights,
    permutation_importance,
    permute_within_groups,
    probability_metrics,
    residual_interval_table,
    select_injury_learner,
    select_injury_weight,
    select_points_candidate,
)


def test_points_selection_prefers_reference_inside_tolerance() -> None:
    summary = pd.DataFrame(
        {
            "candidate": [PHASE2_REFERENCE, "ext_et_all", "blend"],
            "components": [1, 1, 2],
            "spearman": [0.700, 0.704, 0.7049],
            "mae": [25.0, 24.0, 23.0],
        }
    )
    assert select_points_candidate(summary) == PHASE2_REFERENCE
    summary.loc[0, "spearman"] = 0.690
    assert select_points_candidate(summary) == "ext_et_all"


def test_injury_selection_uses_log_loss_then_simplicity() -> None:
    summary = pd.DataFrame(
        {
            "learner": ["et", "hgb", "elastic"],
            "family": ["extra_trees", "hist_gradient", "elastic_net"],
            "roc_auc": [0.750, 0.748, 0.740],
            "log_loss": [0.48, 0.47, 0.50],
        }
    )
    assert select_injury_learner(summary) == "hgb"


def test_blend_averages_components_on_shared_keys() -> None:
    keys = {
        "target_season": [2020, 2020],
        "player_id": ["a", "b"],
        "model_position": ["RB", "RB"],
    }
    first = pd.DataFrame(
        {**keys, "actual_points": [10.0, 20.0], "predicted": [8.0, 30.0]}
    )
    second = pd.DataFrame(
        {**keys, "actual_points": [10.0, 20.0], "predicted": [12.0, 10.0]}
    )
    blended = assemble_candidate_predictions(
        {"one": first, "two": second}, PointsCandidate("blend", ("one", "two"))
    )
    assert blended["predicted"].tolist() == [10.0, 20.0]


def test_combined_score_and_weight_rule() -> None:
    score = combined_score(pd.Series([1.0, 0.5]), pd.Series([0.5, 1.0]), 0.2)
    assert score.tolist() == pytest.approx([90.0, 60.0])
    table = pd.DataFrame(
        {"injury_weight": [0.0, 0.1, 0.2, 0.3], "spearman": [0.70, 0.698, 0.696, 0.68]}
    )
    assert select_injury_weight(table) == 0.2


def test_evaluate_injury_weights_returns_one_row_per_weight() -> None:
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {
            "target_season": [2020] * n,
            "player_id": [f"p{i}" for i in range(n)],
            "model_position": ["RB"] * n,
            "actual_points": rng.gamma(2.0, 40.0, size=n),
            "predicted_points": rng.gamma(2.0, 40.0, size=n),
            "injury_probability": rng.uniform(0.0, 1.0, size=n),
        }
    )
    table = evaluate_injury_weights(frame, (0.0, 0.5))
    assert table["injury_weight"].tolist() == [0.0, 0.5]
    assert table["bust_rate_top_k"].between(0.0, 1.0).all()


def test_residual_intervals_cover_predictions() -> None:
    rng = np.random.default_rng(1)
    n = 90
    audit = pd.DataFrame(
        {
            "model_position": ["WR"] * n,
            "predicted": rng.uniform(0.0, 200.0, size=n),
        }
    )
    audit["actual_points"] = audit["predicted"] + rng.normal(0.0, 20.0, size=n)
    table = residual_interval_table(audit)
    assert len(table) == 3
    current = pd.DataFrame(
        {"model_position": ["WR", "WR"], "predicted_points": [5.0, 190.0]}
    )
    bounded = apply_residual_intervals(current, table)
    assert bounded["points_p10"].notna().all()
    assert (bounded["points_p10"] <= bounded["points_p90"]).all()


def test_permutation_keeps_group_totals() -> None:
    metadata = pd.DataFrame(
        {"target_season": [2020] * 4 + [2021] * 4, "model_position": ["RB"] * 8}
    )
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    permuted = permute_within_groups(values, metadata, seed=3)
    assert sorted(permuted.iloc[:4]) == [1.0, 2.0, 3.0, 4.0]
    assert sorted(permuted.iloc[4:]) == [10.0, 20.0, 30.0, 40.0]


def test_permutation_importance_flags_used_column() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {"signal": rng.normal(size=200), "noise": rng.normal(size=200)}
    )
    target = frame["signal"].to_numpy()
    table = permutation_importance(
        lambda data: data["signal"].to_numpy(),
        frame,
        lambda predicted: float(np.corrcoef(predicted, target)[0, 1]),
        ["signal", "noise"],
        repeats=2,
    )
    by_feature = table.set_index("feature")["importance"]
    assert by_feature["signal"] > 0.5
    assert abs(by_feature["noise"]) < 1e-9


def test_associations_and_probability_metrics_have_expected_direction() -> None:
    rng = np.random.default_rng(4)
    n = 400
    labels = pd.Series(rng.integers(0, 2, size=n))
    frame = pd.DataFrame(
        {
            "signal": labels * 2.0 + rng.normal(size=n),
            "noise": rng.normal(size=n),
            "constant": np.zeros(n),
        }
    )
    table = feature_association_table(frame, labels).set_index("feature")
    assert table.loc["signal", "direction"] == "higher_risk"
    assert (
        table.loc["signal", "discriminative_auc"]
        > table.loc["noise", "discriminative_auc"]
    )
    assert table.loc["constant", "univariate_auc"] == 0.5
    probability = 1.0 / (1.0 + np.exp(-(frame["signal"] - 1.0)))
    metrics = probability_metrics(labels, probability.to_numpy())
    assert 0.8 < metrics["roc_auc"] <= 1.0
    assert metrics["prevalence"] == pytest.approx(float(labels.mean()))
