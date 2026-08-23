"""Chronological points regression, injury classification, and combination tools."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from scipy.special import logit
from scipy.stats import mannwhitneyu, pointbiserialr, spearmanr
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
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

from .metrics import TOP_K, macro_rank_metric, metric_summary
from .models import CompatibleXGBRegressor


OFFENSE_POSITIONS = ("QB", "RB", "WR", "TE")
MINIMUM_TRAINING_SEASON = 2013
POINTS_DISCOVERY_SEASONS = (2016, 2017, 2018, 2019, 2020, 2021)
POINTS_AUDIT_SEASONS = (2022, 2023, 2024, 2025)
INJURY_DEVELOPMENT_SEASONS = (2017, 2018, 2019, 2020, 2021)
INJURY_AUDIT_SEASONS = (2022, 2023, 2024)
INJURY_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
SELECTION_TOLERANCE = 0.005
RANDOM_SEED = 20260822
MAX_THREADS = 8
PHASE2_REFERENCE = "phase2_production"
KEY_COLUMNS = ("target_season", "player_id", "model_position")


@dataclass(frozen=True, slots=True)
class PointsLearner:
    """One fixed regression learner."""

    identifier: str
    estimator: str
    parameters: Mapping[str, float | int]


@dataclass(frozen=True, slots=True)
class PointsCandidate:
    """A candidate forecast: one learner or an equal-weight blend."""

    identifier: str
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InjuryLearner:
    """One fixed classifier specification."""

    identifier: str
    family: str
    parameters: Mapping[str, float | int]


@dataclass(slots=True)
class FittedInjuryModel:
    """Columns, base classifier, and held-out Platt calibrator."""

    learner: InjuryLearner
    feature_columns: tuple[str, ...]
    estimator: Pipeline
    calibrator: LogisticRegression

    def raw_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return uncalibrated positive-class probabilities."""
        # frame: (n, d_available)
        X = frame.loc[:, list(self.feature_columns)]  # (n, d_selected)
        with threadpool_limits(limits=MAX_THREADS):
            return self.estimator.predict_proba(X)[:, 1]  # (n,)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return calibrated positive-class probabilities."""
        raw = self.raw_proba(frame)  # (n,)
        return self.calibrator.predict_proba(_calibration_matrix(raw))[:, 1]  # (n,)


def points_learner_grid() -> tuple[PointsLearner, ...]:
    """Return the predeclared extended-feature learners."""
    return (
        PointsLearner(
            "ext_et_all",
            "extra_trees",
            {
                "n_estimators": 180,
                "max_depth": 14,
                "min_samples_leaf": 5,
                "max_features": 0.7,
            },
        ),
        PointsLearner(
            "ext_hgb_all",
            "hist_gradient_boosting",
            {
                "max_iter": 220,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
                "l2_regularization": 3.0,
                "min_samples_leaf": 20,
            },
        ),
        PointsLearner(
            "ext_xgb_all",
            "xgboost",
            {
                "n_estimators": 280,
                "learning_rate": 0.03,
                "max_depth": 3,
                "min_child_weight": 5,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "reg_alpha": 0.1,
            },
        ),
    )


def points_candidate_grid(
    reference: str = PHASE2_REFERENCE,
) -> tuple[PointsCandidate, ...]:
    """Return the predeclared candidate forecasts including blends with the reference."""
    short = reference.removesuffix("_production")
    return (
        PointsCandidate(reference, (reference,)),
        PointsCandidate("ext_et_all", ("ext_et_all",)),
        PointsCandidate("ext_hgb_all", ("ext_hgb_all",)),
        PointsCandidate("ext_xgb_all", ("ext_xgb_all",)),
        PointsCandidate("blend_ext_et_hgb", ("ext_et_all", "ext_hgb_all")),
        PointsCandidate("blend_ext_all3", ("ext_et_all", "ext_hgb_all", "ext_xgb_all")),
        PointsCandidate(f"blend_{short}_ext_et", (reference, "ext_et_all")),
        PointsCandidate(
            f"blend_{short}_ext_all3",
            (reference, "ext_et_all", "ext_hgb_all", "ext_xgb_all"),
        ),
    )


def injury_learner_grid() -> tuple[InjuryLearner, ...]:
    """Return the predeclared injury classifiers."""
    return (
        InjuryLearner("age_position_logistic", "age_position_baseline", {"C": 1.0}),
        InjuryLearner("prior_games_logistic", "exposure_baseline", {"C": 1.0}),
        InjuryLearner("elastic_all_c0.05", "elastic_net", {"C": 0.05, "l1_ratio": 0.9}),
        InjuryLearner(
            "extra_trees_all_leaf20",
            "extra_trees",
            {"n_estimators": 300, "min_samples_leaf": 20, "max_features": 0.7},
        ),
        InjuryLearner(
            "extra_trees_all_leaf50",
            "extra_trees",
            {"n_estimators": 300, "min_samples_leaf": 50, "max_features": 0.7},
        ),
        InjuryLearner(
            "hist_gradient_all_leaves15",
            "hist_gradient",
            {
                "max_leaf_nodes": 15,
                "learning_rate": 0.05,
                "l2_regularization": 3.0,
                "max_iter": 200,
            },
        ),
    )


FAMILY_SIMPLICITY = {
    "age_position_baseline": 0,
    "exposure_baseline": 1,
    "elastic_net": 2,
    "hist_gradient": 3,
    "extra_trees": 4,
}


def _regressor(learner: PointsLearner) -> Pipeline:
    """Build one deterministic regression pipeline."""
    parameters = dict(learner.parameters)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if learner.estimator == "extra_trees":
        model: BaseEstimator = ExtraTreesRegressor(
            n_estimators=int(parameters.pop("n_estimators")),
            n_jobs=MAX_THREADS,
            random_state=RANDOM_SEED,
            **parameters,
        )
    elif learner.estimator == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=int(parameters.pop("max_iter")),
            random_state=RANDOM_SEED,
            **parameters,
        )
    elif learner.estimator == "xgboost":
        model = CompatibleXGBRegressor(
            {
                "n_estimators": int(parameters.pop("n_estimators")),
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": MAX_THREADS,
                "random_state": RANDOM_SEED,
                **parameters,
            }
        )
    else:
        raise ValueError(f"Unknown points estimator: {learner.estimator!r}")
    return Pipeline([("imputer", imputer), ("model", model)])


def fit_points_learner(
    frame: pd.DataFrame,
    targets: pd.Series,
    learner: PointsLearner,
    columns: Sequence[str],
) -> Pipeline:
    """Fit one learner on the given rows and columns."""
    # frame: (n_train, d); targets: (n_train,)
    pipeline = _regressor(learner)
    with threadpool_limits(limits=MAX_THREADS):
        pipeline.fit(frame.loc[:, list(columns)], targets.to_numpy(dtype="float64"))
    return pipeline


def predict_points(
    pipeline: Pipeline, frame: pd.DataFrame, columns: Sequence[str]
) -> np.ndarray:
    """Predict nonnegative points."""
    # frame: (n, d)
    with threadpool_limits(limits=MAX_THREADS):
        predicted = pipeline.predict(frame.loc[:, list(columns)])  # (n,)
    return np.clip(np.asarray(predicted, dtype="float64"), 0.0, None)  # (n,)


def walk_forward_points(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    targets: pd.Series,
    learner: PointsLearner,
    validation_seasons: Sequence[int],
    *,
    columns: Sequence[str] | None = None,
    fit_targets: pd.Series | None = None,
    minimum_training_season: int = MINIMUM_TRAINING_SEASON,
) -> tuple[pd.DataFrame, dict[int, Pipeline]]:
    """Train through season y-1 and predict season y for each validation season."""
    # frame: (n, d); metadata: (n, c_meta); targets: (n,)
    used_columns = tuple(frame.columns) if columns is None else tuple(columns)
    training_targets = targets if fit_targets is None else fit_targets  # (n,)
    seasons = pd.to_numeric(metadata["target_season"], errors="raise")  # (n,)
    frames: list[pd.DataFrame] = []
    models: dict[int, Pipeline] = {}
    for season in validation_seasons:
        train_mask = seasons.between(
            int(minimum_training_season), int(season) - 1
        )  # (n,)
        validation_mask = seasons.eq(int(season))  # (n,)
        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            raise ValueError(f"Season {season} lacks training or validation rows.")
        pipeline = fit_points_learner(
            frame.loc[train_mask],
            training_targets.loc[train_mask],
            learner,
            used_columns,
        )
        predicted = predict_points(
            pipeline, frame.loc[validation_mask], used_columns
        )  # (n_validation,)
        rows = metadata.loc[
            validation_mask, list(KEY_COLUMNS)
        ].copy()  # (n_validation, 3)
        rows["actual_points"] = targets.loc[validation_mask].to_numpy(dtype="float64")
        rows["predicted"] = predicted
        rows["learner"] = learner.identifier
        frames.append(rows)
        models[int(season)] = pipeline
    return pd.concat(frames, ignore_index=True), models  # (n_validation_total, 6)


def assemble_candidate_predictions(
    component_predictions: Mapping[str, pd.DataFrame],
    candidate: PointsCandidate,
) -> pd.DataFrame:
    """Average component predictions on shared keys."""
    merged: pd.DataFrame | None = None
    for component in candidate.components:
        part = (
            component_predictions[component]
            .loc[:, [*KEY_COLUMNS, "actual_points", "predicted"]]
            .rename(columns={"predicted": f"predicted__{component}"})
        )  # (n_component, 5)
        if merged is None:
            merged = part
        else:
            merged = merged.merge(
                part.drop(columns="actual_points"),
                on=list(KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
            )
    if merged is None:
        raise ValueError("A candidate needs at least one component.")
    predicted_columns = [
        f"predicted__{component}" for component in candidate.components
    ]
    merged["predicted"] = merged[predicted_columns].mean(axis=1).clip(lower=0.0)  # (n,)
    merged["candidate"] = candidate.identifier
    return merged.drop(columns=predicted_columns)  # (n, 6)


def points_summary(predictions: pd.DataFrame) -> dict[str, float]:
    """Summarize point and rank metrics on one prediction table."""
    # predictions: (n, c)
    return metric_summary(
        predictions["actual_points"],
        predictions["predicted"],
        predictions["model_position"],
        predictions["target_season"],
    )


def points_fold_table(predictions: pd.DataFrame, identifier: str) -> pd.DataFrame:
    """Return per-season and per-position Spearman values for one candidate."""
    records: list[dict[str, float | int | str]] = []
    for (season, position), group in predictions.groupby(
        ["target_season", "model_position"], observed=True
    ):
        summary = metric_summary(
            group["actual_points"],
            group["predicted"],
            group["model_position"],
            group["target_season"],
        )
        records.append(
            {
                "candidate": identifier,
                "target_season": int(season),
                "model_position": str(position),
                "rows": int(len(group)),
                "spearman": summary["spearman"],
                "mae": summary["mae"],
                "ndcg_at_roster_cutoff": summary["ndcg_at_roster_cutoff"],
                "top_k_recall": summary["top_k_recall"],
            }
        )
    return pd.DataFrame.from_records(records)  # (n_groups, 8)


def select_points_candidate(
    summary: pd.DataFrame, reference: str = PHASE2_REFERENCE
) -> str:
    """Apply the predeclared tolerance rule with the production reference preferred."""
    # summary: (n_candidates, c); needs candidate, spearman, mae, components
    best = float(summary["spearman"].max())
    eligible = summary.loc[summary["spearman"].ge(best - SELECTION_TOLERANCE)].copy()
    eligible["reference_first"] = (~eligible["candidate"].eq(reference)).astype("int8")
    eligible.sort_values(
        ["reference_first", "components", "mae", "candidate"],
        ascending=[True, True, True, True],
        inplace=True,
        kind="stable",
    )
    return str(eligible.iloc[0]["candidate"])


def _calibration_matrix(probability: np.ndarray) -> np.ndarray:
    """Transform probabilities to one finite log-odds predictor."""
    # probability: (n,)
    clipped = np.clip(
        np.asarray(probability, dtype="float64"), 1e-6, 1.0 - 1e-6
    )  # (n,)
    return logit(clipped).reshape(-1, 1)  # (n, 1)


def injury_learner_columns(
    learner: InjuryLearner,
    available: Sequence[str],
) -> tuple[str, ...]:
    """Return the columns a learner may see."""
    positions = tuple(column for column in available if column.startswith("position_"))
    if learner.family == "age_position_baseline":
        columns = ("age", "missing_age", *positions)
    elif learner.family == "exposure_baseline":
        columns = ("lag1_games", *positions)
    else:
        columns = tuple(available)
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"Injury learner columns are unavailable: {missing!r}")
    return columns


def _classifier(learner: InjuryLearner, columns: Sequence[str]) -> Pipeline:
    """Build one deterministic classification pipeline."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    parameters = dict(learner.parameters)
    if learner.family in {"age_position_baseline", "exposure_baseline"}:
        model: BaseEstimator = LogisticRegression(
            C=float(parameters["C"]),
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler()), ("model", model)]
        )
    if learner.family == "elastic_net":
        model = LogisticRegression(
            C=float(parameters["C"]),
            class_weight="balanced",
            l1_ratio=float(parameters["l1_ratio"]),
            max_iter=3000,
            random_state=RANDOM_SEED,
            solver="saga",
        )
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler()), ("model", model)]
        )
    if learner.family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=int(parameters["n_estimators"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            class_weight="balanced_subsample",
            n_jobs=MAX_THREADS,
            random_state=RANDOM_SEED,
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    if learner.family == "hist_gradient":
        model = HistGradientBoostingClassifier(
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            learning_rate=float(parameters["learning_rate"]),
            l2_regularization=float(parameters["l2_regularization"]),
            max_iter=int(parameters["max_iter"]),
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    raise ValueError(f"Unknown injury family: {learner.family!r}")


def fit_calibrated_injury_model(
    base_frame: pd.DataFrame,
    base_labels: pd.Series,
    calibration_frame: pd.DataFrame,
    calibration_labels: pd.Series,
    learner: InjuryLearner,
    columns: Sequence[str],
) -> FittedInjuryModel:
    """Fit a base classifier and a held-out sigmoid calibrator."""
    # base_frame: (n_base, d); calibration_frame: (n_cal, d)
    used = injury_learner_columns(learner, tuple(columns))
    estimator = _classifier(learner, used)
    base_y = base_labels.to_numpy(dtype="int8")  # (n_base,)
    calibration_y = calibration_labels.to_numpy(dtype="int8")  # (n_cal,)
    if set(np.unique(base_y)) != {0, 1} or set(np.unique(calibration_y)) != {0, 1}:
        raise ValueError("Base and calibration partitions need both classes.")
    with threadpool_limits(limits=MAX_THREADS):
        estimator.fit(base_frame.loc[:, list(used)], base_y)
        raw = estimator.predict_proba(calibration_frame.loc[:, list(used)])[
            :, 1
        ]  # (n_cal,)
        calibrator = LogisticRegression(
            C=np.inf, max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs"
        )
        calibrator.fit(_calibration_matrix(raw), calibration_y)
    return FittedInjuryModel(learner, used, estimator, calibrator)


def walk_forward_injury(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: pd.Series,
    learner: InjuryLearner,
    validation_seasons: Sequence[int],
    *,
    columns: Sequence[str] | None = None,
    fit_labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[int, FittedInjuryModel]]:
    """Base through y-2, calibrate on y-1, and predict y for each season."""
    # frame: (n, d); metadata: (n, c_meta); labels: (n,)
    used_columns = tuple(frame.columns) if columns is None else tuple(columns)
    training_labels = labels if fit_labels is None else fit_labels  # (n,)
    seasons = pd.to_numeric(metadata["target_season"], errors="raise")  # (n,)
    frames: list[pd.DataFrame] = []
    models: dict[int, FittedInjuryModel] = {}
    for season in validation_seasons:
        base_mask = seasons.between(MINIMUM_TRAINING_SEASON, int(season) - 2)  # (n,)
        calibration_mask = seasons.eq(int(season) - 1)  # (n,)
        validation_mask = seasons.eq(int(season))  # (n,)
        model = fit_calibrated_injury_model(
            frame.loc[base_mask],
            training_labels.loc[base_mask],
            frame.loc[calibration_mask],
            training_labels.loc[calibration_mask],
            learner,
            used_columns,
        )
        probability = model.predict_proba(frame.loc[validation_mask])  # (n_validation,)
        rows = metadata.loc[
            validation_mask, list(KEY_COLUMNS)
        ].copy()  # (n_validation, 3)
        rows["label"] = labels.loc[validation_mask].to_numpy(dtype="int8")
        rows["probability"] = probability
        rows["learner"] = learner.identifier
        frames.append(rows)
        models[int(season)] = model
    return pd.concat(frames, ignore_index=True), models  # (n_validation_total, 6)


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


def probability_metrics(labels: pd.Series, probability: np.ndarray) -> dict[str, float]:
    """Compute discrimination, probability error, and calibration metrics."""
    # labels: (n,); probability: (n,)
    y = pd.to_numeric(labels, errors="raise").to_numpy(dtype="int8")  # (n,)
    p = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1.0 - 1e-6)  # (n,)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Probability metrics require both binary classes.")
    calibration_X = _calibration_matrix(p)  # (n, 1)
    calibration = LogisticRegression(C=np.inf, max_iter=1000, solver="lbfgs")
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


def select_injury_learner(summary: pd.DataFrame) -> str:
    """Apply the predeclared AUC tolerance, log-loss, and simplicity rule."""
    # summary: (n_learners, c); needs learner, family, roc_auc, log_loss
    best = float(summary["roc_auc"].max())
    eligible = summary.loc[summary["roc_auc"].ge(best - SELECTION_TOLERANCE)].copy()
    eligible["simplicity"] = eligible["family"].map(FAMILY_SIMPLICITY).astype("int8")
    eligible.sort_values(
        ["log_loss", "simplicity", "learner"],
        ascending=[True, True, True],
        inplace=True,
        kind="stable",
    )
    return str(eligible.iloc[0]["learner"])


def permute_within_groups(
    values: pd.Series,
    metadata: pd.DataFrame,
    seed: int,
) -> pd.Series:
    """Shuffle values independently within season and position."""
    # values: (n,); metadata: (n, c_meta)
    rng = np.random.default_rng(seed)
    permuted = values.copy()  # (n,)
    groups = metadata.groupby(
        ["target_season", "model_position"], observed=True, sort=True
    ).indices
    for indices in groups.values():
        index_array = np.asarray(indices, dtype="int64")  # (n_group,)
        permuted.iloc[index_array] = rng.permutation(
            values.iloc[index_array].to_numpy(copy=True)
        )
    return permuted


def permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    frame: pd.DataFrame,
    score: Callable[[np.ndarray], float],
    columns: Sequence[str],
    *,
    repeats: int = 1,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Score drop from shuffling each column on held-out rows."""
    # frame: (n_validation, d)
    rng = np.random.default_rng(seed)
    baseline = score(predict(frame))
    records: list[dict[str, float | str]] = []
    working = frame.copy()  # (n_validation, d)
    for column in columns:
        original = working[column].to_numpy(copy=True)  # (n_validation,)
        drops: list[float] = []
        for _ in range(repeats):
            working[column] = rng.permutation(original)
            drops.append(baseline - score(predict(working)))
        working[column] = original
        records.append({"feature": column, "importance": float(np.mean(drops))})
    return pd.DataFrame.from_records(records)  # (d, 2)


def spearman_scorer(predictions: pd.DataFrame) -> Callable[[np.ndarray], float]:
    """Return a scorer for mean within-position-season Spearman."""
    positions = predictions["model_position"].reset_index(drop=True)  # (n,)
    seasons = predictions["target_season"].reset_index(drop=True)  # (n,)
    actual = predictions["actual_points"].reset_index(drop=True)  # (n,)

    def score(predicted: np.ndarray) -> float:
        return macro_rank_metric(actual, pd.Series(predicted), positions, seasons)

    return score


def auc_scorer(labels: np.ndarray) -> Callable[[np.ndarray], float]:
    """Return a ROC AUC scorer bound to fixed labels."""
    # labels: (n,)
    y = np.asarray(labels, dtype="int8")

    def score(probability: np.ndarray) -> float:
        return float(roc_auc_score(y, probability))

    return score


def combined_score(
    points_percentile: pd.Series,
    health_probability: pd.Series,
    injury_weight: float,
) -> pd.Series:
    """Blend within-position fantasy percentile with health probability."""
    # points_percentile, health_probability: (n,)
    if not 0.0 <= injury_weight <= 1.0:
        raise ValueError("injury_weight must lie from 0 through 1.")
    return 100.0 * (
        (1.0 - injury_weight) * points_percentile + injury_weight * health_probability
    )  # (n,)


def _bust_rate(frame: pd.DataFrame, score_column: str) -> float:
    """Share of top-k picks by score that finish outside the top 2k by points."""
    # frame: (n, c) with model_position, target_season, actual_points, score_column
    values: list[float] = []
    for (_, position), group in frame.groupby(
        ["target_season", "model_position"], observed=True
    ):
        k = min(TOP_K.get(str(position), 12), len(group))
        if k == 0:
            continue
        picks = group.nlargest(k, score_column)  # (k, c)
        actual_rank = group["actual_points"].rank(
            method="min", ascending=False
        )  # (n_group,)
        busts = actual_rank.loc[picks.index].gt(2 * k)  # (k,)
        values.append(float(busts.mean()))
    return float(np.mean(values)) if values else 0.0


def evaluate_injury_weights(
    frame: pd.DataFrame,
    weights: Sequence[float] = INJURY_WEIGHTS,
) -> pd.DataFrame:
    """Evaluate combined scores against realized points for each weight."""
    # frame: (n, c) with actual_points, predicted_points, injury_probability, keys
    working = frame.copy()  # (n, c)
    working["points_percentile"] = working.groupby(
        ["target_season", "model_position"], observed=True
    )["predicted_points"].rank(method="average", pct=True)  # (n,)
    working["health_probability"] = 1.0 - working["injury_probability"].clip(
        0.0, 1.0
    )  # (n,)
    records: list[dict[str, float]] = []
    for weight in weights:
        working["score"] = combined_score(
            working["points_percentile"], working["health_probability"], float(weight)
        )  # (n,)
        summary = metric_summary(
            working["actual_points"],
            working["score"],
            working["model_position"],
            working["target_season"],
        )
        records.append(
            {
                "injury_weight": float(weight),
                "spearman": summary["spearman"],
                "ndcg_at_roster_cutoff": summary["ndcg_at_roster_cutoff"],
                "top_k_recall": summary["top_k_recall"],
                "bust_rate_top_k": _bust_rate(working, "score"),
                "rows": float(len(working)),
            }
        )
    return pd.DataFrame.from_records(records)  # (n_weights, 6)


def select_injury_weight(table: pd.DataFrame) -> float:
    """Largest weight whose Spearman is within tolerance of the best weight."""
    # table: (n_weights, c)
    best = float(table["spearman"].max())
    eligible = table.loc[table["spearman"].ge(best - SELECTION_TOLERANCE)]
    return float(eligible["injury_weight"].max())


def residual_interval_table(
    audit_predictions: pd.DataFrame,
    *,
    bins: int = 3,
) -> pd.DataFrame:
    """Residual percentiles within position and predicted-point tercile."""
    # audit_predictions: (n, c) with model_position, predicted, actual_points
    records: list[dict[str, float | int | str]] = []
    for position, group in audit_predictions.groupby("model_position", observed=True):
        edges = np.quantile(
            group["predicted"].to_numpy(dtype="float64"),
            np.linspace(0.0, 1.0, bins + 1),
        )  # (bins + 1,)
        edges[0] = -np.inf
        edges[-1] = np.inf
        residuals = group["actual_points"] - group["predicted"]  # (n_position,)
        assignment = np.digitize(
            group["predicted"], edges[1:-1], right=True
        )  # (n_position,)
        for index in range(bins):
            selected = residuals.loc[assignment == index]  # (n_bin,)
            if len(selected) < 20:
                selected = residuals
            records.append(
                {
                    "model_position": str(position),
                    "prediction_bin": index,
                    "lower_edge": float(edges[index]),
                    "upper_edge": float(edges[index + 1]),
                    "rows": int(len(selected)),
                    "residual_p10": float(np.quantile(selected, 0.10)),
                    "residual_p50": float(np.quantile(selected, 0.50)),
                    "residual_p90": float(np.quantile(selected, 0.90)),
                }
            )
    return pd.DataFrame.from_records(records)  # (positions * bins, 8)


def apply_residual_intervals(
    predictions: pd.DataFrame,
    table: pd.DataFrame,
    *,
    prediction_column: str = "predicted_points",
) -> pd.DataFrame:
    """Attach empirical 10th and 90th percentile point bounds."""
    # predictions: (n, c); table: (positions * bins, 8)
    result = predictions.copy()  # (n, c)
    result["points_p10"] = np.nan
    result["points_p90"] = np.nan
    for row in table.itertuples(index=False):
        mask = (
            result["model_position"].eq(row.model_position)
            & result[prediction_column].gt(row.lower_edge)
            & result[prediction_column].le(row.upper_edge)
        )  # (n,)
        result.loc[mask, "points_p10"] = (
            result.loc[mask, prediction_column] + row.residual_p10
        ).clip(lower=0.0)
        result.loc[mask, "points_p90"] = (
            result.loc[mask, prediction_column] + row.residual_p90
        )
    return result  # (n, c + 2)


def clustered_metric_interval(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    samples: int = 500,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int]:
    """Player-cluster bootstrap of any statistic computed on a prediction table."""
    # frame: (n, c) with player_id
    estimate = float(statistic(frame))
    codes, ids = pd.factorize(frame["player_id"], sort=False)  # (n,), (p,)
    rows = np.arange(len(frame))  # (n,)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        sampled = rng.integers(0, len(ids), size=len(ids))  # (p,)
        counts = np.bincount(sampled, minlength=len(ids))  # (p,)
        resampled = frame.iloc[np.repeat(rows, counts[codes])]  # (n_boot, c)
        try:
            values.append(float(statistic(resampled)))
        except ValueError:
            continue
    distribution = np.asarray(values, dtype="float64")  # (b_valid,)
    return {
        "estimate": estimate,
        "lower_95": float(np.quantile(distribution, 0.025)),
        "upper_95": float(np.quantile(distribution, 0.975)),
        "bootstrap_samples": int(distribution.size),
    }
