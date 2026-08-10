"""Position-aware regression and ranking metrics."""

import numpy as np
import pandas as pd

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    r2_score,
)


TOP_K = {"QB": 12, "RB": 24, "WR": 36, "TE": 12, "K": 12}


def _safe_spearman(observed: pd.Series, predicted: pd.Series) -> float:
    # observed, predicted: (n_group,)
    if observed.nunique() < 2 or predicted.nunique() < 2:
        return 0.0
    coefficient = spearmanr(observed, predicted).statistic
    return float(0.0 if np.isnan(coefficient) else coefficient)


def _safe_kendall(observed: pd.Series, predicted: pd.Series) -> float:
    # observed, predicted: (n_group,)
    if observed.nunique() < 2 or predicted.nunique() < 2:
        return 0.0
    coefficient = kendalltau(observed, predicted, variant="b").statistic
    return float(0.0 if np.isnan(coefficient) else coefficient)


def macro_rank_metric(
    observed: pd.Series,
    predicted: pd.Series,
    positions: pd.Series,
    seasons: pd.Series | None = None,
    metric: str = "spearman",
) -> float:
    """Average a rank metric over position-season groups."""
    # observed, predicted, positions, seasons: (n_rows,)
    metric_frame = pd.DataFrame(  # (n_rows, 4)
        {
            "observed": observed,
            "predicted": predicted,
            "position": positions,
            "season": 0 if seasons is None else seasons,
        }
    )
    values: list[float] = []  # (n_groups,)
    for _, group in metric_frame.groupby(["season", "position"], observed=True):
        # group: (n_group, 4)
        if metric == "spearman":
            values.append(_safe_spearman(group["observed"], group["predicted"]))
        elif metric == "kendall":
            values.append(_safe_kendall(group["observed"], group["predicted"]))
        else:
            raise ValueError(f"Unknown rank metric: {metric!r}")
    return float(np.mean(values)) if values else 0.0


def _mean_ndcg(metric_frame: pd.DataFrame) -> float:
    # metric_frame: (n_rows, 4)
    values: list[float] = []  # (n_scored_groups,)
    for (_, position), group in metric_frame.groupby(
        ["season", "position"], observed=True
    ):
        # group: (n_group, 4)
        k = min(TOP_K.get(str(position), 12), len(group))
        nonnegative_observed = group["observed"].clip(lower=0.0)  # (n_group,)
        if k == 0 or nonnegative_observed.max() <= 0:
            continue
        observed = nonnegative_observed.to_numpy(dtype="float64")[None, :]  # (1, n)
        predicted = group["predicted"].to_numpy(dtype="float64")[None, :]  # (1, n)
        values.append(float(ndcg_score(observed, predicted, k=k)))
    return float(np.mean(values)) if values else 0.0


def _top_k_recall(metric_frame: pd.DataFrame) -> float:
    # metric_frame: (n_rows, 4)
    values: list[float] = []  # (n_scored_groups,)
    for (_, position), group in metric_frame.groupby(
        ["season", "position"], observed=True
    ):
        # group: (n_group, 4)
        k = min(TOP_K.get(str(position), 12), len(group))
        if k == 0:
            continue
        actual = set(group.nlargest(k, "observed").index)
        forecast = set(group.nlargest(k, "predicted").index)
        values.append(len(actual & forecast) / k)
    return float(np.mean(values)) if values else 0.0


def metric_summary(
    observed: pd.Series,
    predicted: pd.Series,
    positions: pd.Series,
    seasons: pd.Series,
) -> dict[str, float]:
    """Calculate declared point and within-position ranking metrics."""
    # observed, predicted, positions, seasons: (n_rows,)
    metric_frame = pd.DataFrame(  # (n_rows, 4)
        {
            "observed": observed.astype("float64"),
            "predicted": predicted.astype("float64"),
            "position": positions,
            "season": seasons,
        }
    )
    observed_values = metric_frame["observed"].to_numpy()  # (n_rows,)
    predicted_values = metric_frame["predicted"].to_numpy()  # (n_rows,)

    actual_rank = metric_frame.groupby(  # (n_rows,)
        ["season", "position"], observed=True
    )["observed"].rank(method="average", ascending=False)
    predicted_rank = metric_frame.groupby(  # (n_rows,)
        ["season", "position"], observed=True
    )["predicted"].rank(method="average", ascending=False)

    if np.std(predicted_values) > 0:
        calibration = np.polyfit(  # (2,)
            predicted_values,
            observed_values,
            deg=1,
        )
        calibration_slope, calibration_intercept = calibration
    else:
        calibration_slope = 0.0
        calibration_intercept = float(np.mean(observed_values))

    return {
        "spearman": macro_rank_metric(
            metric_frame["observed"],
            metric_frame["predicted"],
            metric_frame["position"],
            metric_frame["season"],
            "spearman",
        ),
        "kendall_tau_b": macro_rank_metric(
            metric_frame["observed"],
            metric_frame["predicted"],
            metric_frame["position"],
            metric_frame["season"],
            "kendall",
        ),
        "mae": float(mean_absolute_error(observed_values, predicted_values)),
        "rmse": float(mean_squared_error(observed_values, predicted_values) ** 0.5),
        "r2": float(r2_score(observed_values, predicted_values)),
        "rank_mae": float((actual_rank - predicted_rank).abs().mean()),
        "ndcg_at_roster_cutoff": _mean_ndcg(metric_frame),
        "top_k_recall": _top_k_recall(metric_frame),
        "calibration_intercept": float(calibration_intercept),
        "calibration_slope": float(calibration_slope),
    }


def clustered_spearman_interval(
    prediction_frame: pd.DataFrame,
    prediction_column: str,
    baseline_column: str | None = None,
    samples: int = 500,
    seed: int = 20260809,
) -> dict[str, float]:
    """Bootstrap players as clusters and return metric or paired-difference limits."""
    # prediction_frame: (n_rows, c_predictions)
    point_estimate = macro_rank_metric(
        prediction_frame["actual_points"],
        prediction_frame[prediction_column],
        prediction_frame["model_position"],
        prediction_frame["target_season"],
    )
    if baseline_column is not None:
        point_estimate -= macro_rank_metric(
            prediction_frame["actual_points"],
            prediction_frame[baseline_column],
            prediction_frame["model_position"],
            prediction_frame["target_season"],
        )
    player_codes, player_ids = pd.factorize(  # (n_rows,), (n_players,)
        prediction_frame["player_id"],
        sort=False,
    )
    row_numbers = np.arange(len(prediction_frame))  # (n_rows,)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype="float64")  # (n_bootstrap_samples,)
    for sample_index in range(samples):
        sampled_codes = rng.integers(  # (n_players,)
            0,
            len(player_ids),
            size=len(player_ids),
        )
        cluster_counts = np.bincount(  # (n_players,)
            sampled_codes,
            minlength=len(player_ids),
        )
        sampled_rows = np.repeat(  # (n_bootstrap_rows,)
            row_numbers,
            cluster_counts[player_codes],
        )
        sampled = prediction_frame.iloc[
            sampled_rows
        ]  # (n_bootstrap_rows, c_predictions)
        prediction_score = macro_rank_metric(
            sampled["actual_points"],
            sampled[prediction_column],
            sampled["model_position"],
            sampled["target_season"],
        )
        if baseline_column is None:
            estimates[sample_index] = prediction_score
        else:
            baseline_score = macro_rank_metric(
                sampled["actual_points"],
                sampled[baseline_column],
                sampled["model_position"],
                sampled["target_season"],
            )
            estimates[sample_index] = prediction_score - baseline_score

    return {
        "estimate": point_estimate,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": samples,
    }
