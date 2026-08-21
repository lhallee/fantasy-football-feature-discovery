"""Chronological binary classification for physical injury-report risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Mapping, Sequence

from scipy.special import expit, logit
from scipy.stats import mannwhitneyu, pointbiserialr, spearmanr
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


DEVELOPMENT_SEASONS = (2017, 2018, 2019, 2020, 2021)
AUDIT_SEASONS = (2022, 2023, 2024)
RANDOM_SEED = 20260814
MAX_THREADS = 4


@dataclass(frozen=True, slots=True)
class InjuryCandidate:
    """One fixed classifier and feature-count specification."""

    identifier: str
    family: str
    feature_limit: int | None
    parameters: Mapping[str, float | int]


@dataclass(slots=True)
class FittedInjuryModel:
    """Selected feature order, base classifier, and held-out Platt calibrator."""

    candidate: InjuryCandidate
    feature_columns: tuple[str, ...]
    estimator: Pipeline
    calibrator: LogisticRegression

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return calibrated positive-class probabilities."""
        # frame: (n, d_available)
        X = frame.loc[:, self.feature_columns]  # (n, d_selected)
        with threadpool_limits(limits=MAX_THREADS):
            raw_probability = self.estimator.predict_proba(X)[:, 1]  # (n,)
            calibration_X = _calibration_matrix(raw_probability)  # (n, 1)
            probability = self.calibrator.predict_proba(calibration_X)[:, 1]  # (n,)
        return probability  # (n,)


def candidate_grid() -> tuple[InjuryCandidate, ...]:
    """Return the exact predeclared Phase 3 model grid."""
    candidates = [
        InjuryCandidate("age_position_logistic", "baseline", None, {"C": 1.0})
    ]
    for feature_limit in (16, 32, 64, None):
        label = "all" if feature_limit is None else str(feature_limit)
        for c in (0.05, 0.2):
            candidates.append(
                InjuryCandidate(
                    f"elastic_{label}_c{c:g}",
                    "elastic_net",
                    feature_limit,
                    {"C": c, "l1_ratio": 0.9},
                )
            )
        for minimum_leaf in (20, 50):
            candidates.append(
                InjuryCandidate(
                    f"extra_trees_{label}_leaf{minimum_leaf}",
                    "extra_trees",
                    feature_limit,
                    {
                        "n_estimators": 300,
                        "min_samples_leaf": minimum_leaf,
                        "max_features": 0.7,
                    },
                )
            )
    for feature_limit in (32, 64):
        for maximum_leaves in (7, 15):
            candidates.append(
                InjuryCandidate(
                    f"hist_gradient_{feature_limit}_leaves{maximum_leaves}",
                    "hist_gradient",
                    feature_limit,
                    {
                        "max_leaf_nodes": maximum_leaves,
                        "learning_rate": 0.05,
                        "l2_regularization": 3.0,
                    },
                )
            )
    return tuple(candidates)


def _numeric_vector(values: pd.Series) -> np.ndarray:
    """Return a median-filled finite vector for one univariate association."""
    # values: (n,)
    vector = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")  # (n,)
    finite = np.isfinite(vector)  # (n,)
    median = float(np.nanmedian(vector[finite])) if finite.any() else 0.0
    vector = np.where(finite, vector, median)  # (n,)
    return vector  # (n,)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjust a vector of p-values with the Benjamini-Hochberg procedure."""
    # p_values: (d,)
    d = p_values.size
    order = np.argsort(p_values)  # (d,)
    ranked = p_values[order]  # (d,)
    adjusted_ranked = ranked * d / np.arange(1, d + 1)  # (d,)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]  # (d,)
    adjusted = np.empty(d, dtype="float64")  # (d,)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)  # (d,)
    return adjusted  # (d,)


def feature_association_table(
    frame: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """Measure discovery-only univariate association for every model input."""
    # frame: (n, d); labels: (n,)
    y = pd.to_numeric(labels, errors="raise").to_numpy(dtype="int8")  # (n,)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Feature associations require both binary classes.")
    records: list[dict[str, float | int | str]] = []
    for column in frame.columns:
        x = _numeric_vector(frame[column])  # (n,)
        if np.unique(x).size < 2:
            auc = 0.5
            rho = 0.0
            point_biserial = 0.0
            p_value = 1.0
        else:
            auc = float(roc_auc_score(y, x))
            rho = float(spearmanr(x, y).statistic)
            point_biserial = float(pointbiserialr(y, x).statistic)
            p_value = float(
                mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided").pvalue
            )
        records.append(
            {
                "feature": column,
                "rows": int(y.size),
                "positive_rows": int(y.sum()),
                "positive_mean": float(x[y == 1].mean()),
                "negative_mean": float(x[y == 0].mean()),
                "spearman_rho": rho,
                "point_biserial_r": point_biserial,
                "univariate_auc": auc,
                "discriminative_auc": max(auc, 1.0 - auc),
                "direction": "higher_risk" if auc >= 0.5 else "lower_risk",
                "p_value": p_value,
            }
        )
    associations = pd.DataFrame.from_records(records)  # (d, 11)
    associations["q_value"] = _benjamini_hochberg(
        associations["p_value"].to_numpy(dtype="float64")
    )  # (d,)
    associations.sort_values(
        ["discriminative_auc", "feature"],
        ascending=[False, True],
        inplace=True,
        kind="stable",
    )
    associations.reset_index(drop=True, inplace=True)
    associations["association_rank"] = np.arange(1, len(associations) + 1)  # (d,)
    return associations  # (d, 13)


def rank_features(frame: pd.DataFrame, labels: pd.Series) -> tuple[str, ...]:
    """Rank features on the base-training rows only."""
    associations = feature_association_table(frame, labels)  # (d, 13)
    return tuple(associations["feature"].astype(str))


def _baseline_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    candidates = (
        "age",
        "missing_age",
        "position_QB",
        "position_RB",
        "position_WR",
        "position_TE",
        "position_K",
    )
    columns = tuple(column for column in candidates if column in frame.columns)
    if len(columns) < 6:
        raise ValueError("Age-position baseline columns are incomplete.")
    return columns


def select_feature_columns(
    frame: pd.DataFrame,
    labels: pd.Series,
    candidate: InjuryCandidate,
) -> tuple[str, ...]:
    """Apply the candidate's fold-local feature-count rule."""
    # frame: (n_base, d); labels: (n_base,)
    if candidate.family == "baseline":
        return _baseline_columns(frame)
    ranking = rank_features(frame, labels)
    if candidate.feature_limit is None:
        return ranking
    return ranking[: candidate.feature_limit]


def _classifier(candidate: InjuryCandidate, columns: Sequence[str]) -> Pipeline:
    """Construct one deterministic, laptop-sized binary classifier."""
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
                list(columns),
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if candidate.family in {"baseline", "elastic_net"}:
        scaling = StandardScaler()
        if candidate.family == "baseline":
            classifier: BaseEstimator = LogisticRegression(
                C=float(candidate.parameters["C"]),
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            )
        else:
            classifier = LogisticRegression(
                C=float(candidate.parameters["C"]),
                class_weight="balanced",
                l1_ratio=float(candidate.parameters["l1_ratio"]),
                max_iter=3000,
                penalty="elasticnet",
                random_state=RANDOM_SEED,
                solver="saga",
            )
        return Pipeline(
            [("preprocessing", preprocessing), ("scaling", scaling), ("model", classifier)]
        )
    if candidate.family == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=int(candidate.parameters["n_estimators"]),
            min_samples_leaf=int(candidate.parameters["min_samples_leaf"]),
            max_features=float(candidate.parameters["max_features"]),
            class_weight="balanced_subsample",
            n_jobs=MAX_THREADS,
            random_state=RANDOM_SEED,
        )
    elif candidate.family == "hist_gradient":
        classifier = HistGradientBoostingClassifier(
            max_leaf_nodes=int(candidate.parameters["max_leaf_nodes"]),
            learning_rate=float(candidate.parameters["learning_rate"]),
            l2_regularization=float(candidate.parameters["l2_regularization"]),
            max_iter=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown injury candidate family: {candidate.family!r}")
    return Pipeline([("preprocessing", preprocessing), ("model", classifier)])


def _calibration_matrix(probability: np.ndarray) -> np.ndarray:
    """Transform probabilities to one finite log-odds predictor."""
    # probability: (n,)
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)  # (n,)
    return logit(clipped).reshape(-1, 1)  # (n, 1)


def fit_calibrated_model(
    base_frame: pd.DataFrame,
    base_labels: pd.Series,
    calibration_frame: pd.DataFrame,
    calibration_labels: pd.Series,
    candidate: InjuryCandidate,
) -> FittedInjuryModel:
    """Fit feature selection, a base model, and held-out sigmoid calibration."""
    # base_frame: (n_base, d); base_labels: (n_base,)
    # calibration_frame: (n_calibration, d); calibration_labels: (n_calibration,)
    columns = select_feature_columns(base_frame, base_labels, candidate)
    estimator = _classifier(candidate, columns)
    base_X = base_frame.loc[:, columns]  # (n_base, d_selected)
    calibration_X = calibration_frame.loc[:, columns]  # (n_calibration, d_selected)
    base_y = pd.to_numeric(base_labels, errors="raise").to_numpy(dtype="int8")  # (n_base,)
    calibration_y = pd.to_numeric(
        calibration_labels, errors="raise"
    ).to_numpy(dtype="int8")  # (n_calibration,)
    if set(np.unique(base_y)) != {0, 1} or set(np.unique(calibration_y)) != {0, 1}:
        raise ValueError("Base and calibration partitions must contain both classes.")
    with threadpool_limits(limits=MAX_THREADS):
        estimator.fit(base_X, base_y)
        raw_probability = estimator.predict_proba(calibration_X)[:, 1]  # (n_calibration,)
        platt_X = _calibration_matrix(raw_probability)  # (n_calibration, 1)
        calibrator = LogisticRegression(
            max_iter=1000,
            penalty=None,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        calibrator.fit(platt_X, calibration_y)
    return FittedInjuryModel(candidate, columns, estimator, calibrator)


def probability_metrics(labels: pd.Series, probability: np.ndarray) -> dict[str, float]:
    """Compute discrimination, probability error, and calibration metrics."""
    # labels: (n,); probability: (n,)
    y = pd.to_numeric(labels, errors="raise").to_numpy(dtype="int8")  # (n,)
    p = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1.0 - 1e-6)  # (n,)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Probability metrics require both binary classes.")
    calibration_X = _calibration_matrix(p)  # (n, 1)
    calibration = LogisticRegression(
        max_iter=1000,
        penalty=None,
        solver="lbfgs",
    )
    calibration.fit(calibration_X, y)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "prevalence": float(y.mean()),
    }


def evaluate_candidate(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: pd.Series,
    candidate: InjuryCandidate,
    validation_seasons: Sequence[int],
    *,
    fit_labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one fixed procedure on expanding chronological folds."""
    # frame: (n, d); metadata: (n, c_metadata); labels: (n,)
    fitted_labels = labels if fit_labels is None else fit_labels  # (n,)
    fold_records: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_records: list[dict[str, int | str]] = []
    seasons = pd.to_numeric(metadata["target_season"], errors="raise")  # (n,)
    for validation_season in validation_seasons:
        base_mask = seasons.le(validation_season - 2) & seasons.ge(2013)  # (n,)
        calibration_mask = seasons.eq(validation_season - 1)  # (n,)
        validation_mask = seasons.eq(validation_season)  # (n,)
        model = fit_calibrated_model(
            frame.loc[base_mask],
            fitted_labels.loc[base_mask],
            frame.loc[calibration_mask],
            fitted_labels.loc[calibration_mask],
            candidate,
        )
        probability = model.predict_proba(frame.loc[validation_mask])  # (n_validation,)
        metrics = probability_metrics(labels.loc[validation_mask], probability)
        fold_records.append(
            {
                "candidate": candidate.identifier,
                "validation_season": int(validation_season),
                "base_rows": int(base_mask.sum()),
                "calibration_rows": int(calibration_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "feature_count": len(model.feature_columns),
                **metrics,
            }
        )
        predictions = metadata.loc[
            validation_mask,
            ["target_season", "player_id", "model_position", "team", "candidate_name"],
        ].copy()  # (n_validation, 5)
        predictions["physical_injury_reported"] = labels.loc[
            validation_mask
        ].to_numpy(dtype="int8")  # (n_validation,)
        predictions["injury_probability"] = probability  # (n_validation,)
        predictions["candidate"] = candidate.identifier  # (n_validation,)
        prediction_frames.append(predictions)
        feature_records.extend(
            {
                "candidate": candidate.identifier,
                "validation_season": int(validation_season),
                "feature_rank": rank,
                "feature": column,
            }
            for rank, column in enumerate(model.feature_columns, start=1)
        )
    fold_metrics = pd.DataFrame.from_records(fold_records)  # (f, 13)
    predictions = pd.concat(prediction_frames, ignore_index=True)  # (n_validation_total, 8)
    selected_features = pd.DataFrame.from_records(feature_records)  # (sum_f d_selected, 4)
    return fold_metrics, predictions, selected_features


def summarize_candidates(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate equal-weight chronological fold metrics by candidate."""
    metric_columns = [
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
        "calibration_intercept",
        "calibration_slope",
        "prevalence",
        "feature_count",
    ]
    summary = (
        fold_metrics.groupby("candidate", observed=True, sort=True)[metric_columns]
        .mean()
        .add_prefix("mean_")
        .reset_index()
    )  # (n_candidates, 1 + m_metrics)
    fold_counts = (
        fold_metrics.groupby("candidate", observed=True)
        .size()
        .rename("folds")
        .reset_index()
    )  # (n_candidates, 2)
    return summary.merge(
        fold_counts,
        on="candidate",
        how="left",
        validate="one_to_one",
    )  # (n_candidates, 2 + m_metrics)


def select_candidate(summary: pd.DataFrame) -> str:
    """Apply the locked near-best discrimination and probability-error rule."""
    # summary: (n_candidates, c_summary)
    best_auc = float(summary["mean_roc_auc"].max())
    eligible = summary.loc[
        summary["mean_roc_auc"].ge(best_auc - 0.005)
    ].copy()  # (n_eligible, c_summary)
    eligible.sort_values(
        ["mean_log_loss", "mean_feature_count", "candidate"],
        ascending=[True, True, True],
        inplace=True,
        kind="stable",
    )
    return str(eligible.iloc[0]["candidate"])


def permute_labels_within_season_position(
    labels: pd.Series,
    metadata: pd.DataFrame,
    seed: int,
) -> pd.Series:
    """Shuffle labels within season and position without changing group prevalence."""
    # labels: (n,); metadata: (n, c_metadata)
    rng = np.random.default_rng(seed)
    permuted = labels.copy()  # (n,)
    groups = metadata.groupby(
        ["target_season", "model_position"],
        observed=True,
        sort=True,
    ).indices
    for indices in groups.values():
        index_array = np.asarray(indices, dtype="int64")  # (n_group,)
        values = labels.iloc[index_array].to_numpy(copy=True)  # (n_group,)
        permuted.iloc[index_array] = rng.permutation(values)  # (n_group,)
    return permuted


def clustered_auc_interval(
    predictions: pd.DataFrame,
    *,
    repetitions: int = 2000,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int]:
    """Bootstrap pooled ROC AUC by resampling player identifiers."""
    # predictions: (n, c_predictions)
    observed = float(
        roc_auc_score(
            predictions["physical_injury_reported"],
            predictions["injury_probability"],
        )
    )
    player_ids = predictions["player_id"].drop_duplicates().to_numpy()  # (p,)
    row_groups = {
        player_id: group.index.to_numpy(dtype="int64")
        for player_id, group in predictions.groupby("player_id", sort=False)
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled_players = rng.choice(player_ids, size=player_ids.size, replace=True)  # (p,)
        sampled_indices = np.concatenate(
            [row_groups[player_id] for player_id in sampled_players]
        )  # (n_bootstrap,)
        sample = predictions.iloc[sampled_indices]  # (n_bootstrap, c_predictions)
        if sample["physical_injury_reported"].nunique() < 2:
            continue
        estimates.append(
            float(
                roc_auc_score(
                    sample["physical_injury_reported"],
                    sample["injury_probability"],
                )
            )
        )
    distribution = np.asarray(estimates, dtype="float64")  # (b_valid,)
    if distribution.size < repetitions * 0.95:
        raise RuntimeError("Too many clustered bootstrap samples lacked both classes.")
    lower, upper = np.quantile(distribution, [0.025, 0.975])  # (2,)
    return {
        "estimate": observed,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "repetitions_requested": int(repetitions),
        "repetitions_valid": int(distribution.size),
        "cluster_count": int(player_ids.size),
    }


def injury_probability_from_log_odds(log_odds: np.ndarray) -> np.ndarray:
    """Expose the stable inverse-logit transform for focused tests."""
    # log_odds: (n,)
    return expit(np.asarray(log_odds, dtype="float64"))  # (n,)
