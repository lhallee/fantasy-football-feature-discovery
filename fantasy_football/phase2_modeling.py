"""Leakage-safe walk-forward modeling for isolated Phase 2 experiments."""

from __future__ import annotations

import time

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import metric_summary
from .models import CompatibleXGBRegressor, MAX_CPU_THREADS, RANDOM_SEED


@dataclass(frozen=True, slots=True)
class Phase2Candidate:
    """One declared Phase 2 model configuration."""

    identifier: str
    tier: str
    recipes: tuple[str, ...]
    estimator: str
    target: str = "points"
    feature_limit: int | None = None
    excluded_feature_fragments: tuple[str, ...] = ()
    position_specific: bool = False
    tie_predicted_zeros: bool = False
    shuffle_target: bool = False
    benchmark_only: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> Phase2Candidate:
        """Construct a candidate from a validated JSON-like mapping."""
        return cls(
            identifier=str(values["identifier"]),
            tier=str(values["tier"]),
            recipes=tuple(str(value) for value in values["recipes"]),
            estimator=str(values["estimator"]),
            target=str(values.get("target", "points")),
            feature_limit=(
                None
                if values.get("feature_limit") is None
                else int(values["feature_limit"])
            ),
            excluded_feature_fragments=tuple(
                str(value) for value in values.get("excluded_feature_fragments", ())
            ),
            position_specific=bool(values.get("position_specific", False)),
            tie_predicted_zeros=bool(values.get("tie_predicted_zeros", False)),
            shuffle_target=bool(values.get("shuffle_target", False)),
            benchmark_only=bool(values.get("benchmark_only", False)),
            parameters=dict(values.get("parameters", {})),
        )


@dataclass(frozen=True, slots=True)
class WalkForwardOutput:
    """Predictions, fold metrics, and fold-local feature decisions."""

    predictions: pd.DataFrame  # (n_validation_rows, 8)
    fold_metrics: pd.DataFrame  # (n_validation_folds, m_metrics)
    selected_features: pd.DataFrame  # (n_selected_feature_records, 7)


def _rank_target(
    observed: pd.Series,
    seasons: pd.Series,
    positions: pd.Series,
) -> pd.Series:
    """Return tied percent ranks formed only within supplied rows."""
    # observed, seasons, positions: (n_rows,)
    target_frame = pd.DataFrame(  # (n_rows, 3)
        {"observed": observed, "season": seasons, "position": positions},
        index=observed.index,
    )
    ranked = target_frame.groupby(  # (n_rows,)
        ["season", "position"], observed=True
    )["observed"].rank(method="average", pct=True)
    return ranked.astype("float64")  # (n_rows,)


def _transform_target(
    observed: pd.Series,
    seasons: pd.Series,
    positions: pd.Series,
    target: str,
) -> pd.Series:
    """Apply a declared target transform inside a training fold."""
    # observed, seasons, positions: (n_train,)
    if target == "points":
        return observed.astype("float64")  # (n_train,)
    if target == "signed_log_points":
        values = observed.to_numpy(dtype="float64")  # (n_train,)
        transformed = np.sign(values) * np.log1p(np.abs(values))  # (n_train,)
        return pd.Series(transformed, index=observed.index)  # (n_train,)
    if target == "rank_percentile":
        return _rank_target(observed, seasons, positions)  # (n_train,)
    if target == "hurdle_rank":
        return _rank_target(observed, seasons, positions)  # (n_train,)
    raise ValueError(f"Unknown Phase 2 target: {target!r}")


def _fold_feature_scores(
    X_train: pd.DataFrame,
    observed: pd.Series,
    seasons: pd.Series,
    positions: pd.Series,
) -> pd.Series:
    """Score features by fold-local within-group rank association."""
    # X_train: (n_train, d_features); observed, seasons, positions: (n_train,)
    group_keys = pd.DataFrame(  # (n_train, 2)
        {"season": seasons, "position": positions},
        index=X_train.index,
    )
    X_ranked = X_train.groupby(  # (n_train, d_features)
        [group_keys["season"], group_keys["position"]], observed=True
    ).rank(method="average", pct=True)
    y_ranked = _rank_target(observed, seasons, positions)  # (n_train,)
    varying = X_ranked.nunique(dropna=True).gt(1)  # (d_features,)
    scores = pd.Series(0.0, index=X_ranked.columns)  # (d_features,)
    scores.loc[varying] = X_ranked.loc[:, varying].corrwith(y_ranked).abs()
    return scores.fillna(0.0).sort_index()  # (d_features,)


def select_fold_features(
    X_train: pd.DataFrame,
    observed: pd.Series,
    seasons: pd.Series,
    positions: pd.Series,
    feature_limit: int | None,
) -> tuple[list[str], pd.Series]:
    """Select no more than ``feature_limit`` columns using training rows only."""
    # X_train: (n_train, d_features); observed, seasons, positions: (n_train,)
    scores = _fold_feature_scores(  # (d_features,)
        X_train,
        observed,
        seasons,
        positions,
    )
    ordered = sorted(  # (d_features,)
        scores.index,
        key=lambda column: (-float(scores[column]), column),
    )
    if feature_limit is not None:
        ordered = ordered[: min(feature_limit, len(ordered))]  # (d_selected,)
    return ordered, scores.loc[ordered]  # (d_selected,), (d_selected,)


def _regression_pipeline(candidate: Phase2Candidate) -> Pipeline:
    """Build one fold-local regression pipeline."""
    parameters = dict(candidate.parameters)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)

    if candidate.estimator == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=int(parameters.pop("n_estimators", 240)),
            n_jobs=MAX_CPU_THREADS,
            random_state=RANDOM_SEED,
            **parameters,
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    if candidate.estimator == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=int(parameters.pop("max_iter", 250)),
            random_state=RANDOM_SEED,
            **parameters,
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    if candidate.estimator == "xgboost":
        model = CompatibleXGBRegressor(
            {
                "n_estimators": int(parameters.pop("n_estimators", 320)),
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": MAX_CPU_THREADS,
                "random_state": RANDOM_SEED,
                **parameters,
            }
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    if candidate.estimator == "ridge":
        model = Ridge(**parameters)
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler()), ("model", model)]
        )
    if candidate.estimator == "hurdle_extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=int(parameters.pop("n_estimators", 240)),
            n_jobs=MAX_CPU_THREADS,
            random_state=RANDOM_SEED,
            **parameters,
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    raise ValueError(f"Unknown Phase 2 estimator: {candidate.estimator!r}")


def _hurdle_predictions(
    candidate: Phase2Candidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    transformed_target: pd.Series,
) -> np.ndarray:
    """Fit a fold-local nonzero gate and conditional rank regressor."""
    # X_train: (n_train, d_selected); X_valid: (n_valid, d_selected)
    parameters = dict(candidate.parameters)
    n_estimators = int(parameters.pop("n_estimators", 240))
    conditional_weight = float(parameters.pop("conditional_weight", 0.25))
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    n_jobs=MAX_CPU_THREADS,
                    random_state=RANDOM_SEED,
                    **parameters,
                ),
            ),
        ]
    )
    regressor_candidate = Phase2Candidate(
        identifier=candidate.identifier,
        tier=candidate.tier,
        recipes=candidate.recipes,
        estimator="extra_trees",
        target="rank_percentile",
        parameters={"n_estimators": n_estimators, **parameters},
    )
    regressor = _regression_pipeline(regressor_candidate)

    positive = (y_train > 0).astype("int8")  # (n_train,)
    classifier.fit(X_train, positive)
    regressor.fit(X_train, transformed_target)
    positive_probability = classifier.predict_proba(X_valid)[:, 1]  # (n_valid,)
    conditional_score = regressor.predict(X_valid)  # (n_valid,)
    return positive_probability + conditional_weight * conditional_score  # (n_valid,)


def _tie_bottom_predictions(
    predictions: np.ndarray,
    y_train: pd.Series,
    seasons: pd.Series,
) -> np.ndarray:
    """Tie the expected zero-count bottom group using prior prevalence only."""
    # predictions: (n_valid,); y_train, seasons: (n_train,)
    recent_years = sorted(seasons.unique())[-3:]
    recent_mask = seasons.isin(recent_years)  # (n_train,)
    zero_fraction = float((y_train.loc[recent_mask] <= 0).mean())
    n_tied = min(len(predictions), max(0, round(zero_fraction * len(predictions))))
    adjusted = predictions.copy()  # (n_valid,)
    if n_tied > 1:
        bottom = np.argsort(adjusted, kind="stable")[:n_tied]  # (n_tied,)
        adjusted[bottom] = float(np.min(adjusted))
    return adjusted  # (n_valid,)


def _fit_scope(
    candidate: Phase2Candidate,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    train_table: pd.DataFrame,
    valid_table: pd.DataFrame,
    validation_season: int,
    position_scope: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit one pooled or position-specific scope and return predictions."""
    # X_train: (n_train, d_features); X_valid: (n_valid, d_features)
    selected, scores = select_fold_features(
        X_train,
        train_table["target_points"],
        train_table["target_season"],
        train_table["model_position"],
        candidate.feature_limit,
    )
    X_train_selected = X_train.loc[:, selected]  # (n_train, d_selected)
    X_valid_selected = X_valid.loc[:, selected]  # (n_valid, d_selected)
    transformed_target = _transform_target(  # (n_train,)
        train_table["target_points"],
        train_table["target_season"],
        train_table["model_position"],
        candidate.target,
    )
    if candidate.shuffle_target:
        generator = np.random.default_rng(RANDOM_SEED + validation_season)
        shuffled = generator.permutation(  # (n_train,)
            transformed_target.to_numpy(dtype="float64")
        )
        transformed_target = pd.Series(shuffled, index=transformed_target.index)

    if candidate.estimator == "hurdle_extra_trees":
        predictions = _hurdle_predictions(  # (n_valid,)
            candidate,
            X_train_selected,
            train_table["target_points"],
            X_valid_selected,
            transformed_target,
        )
    else:
        estimator = _regression_pipeline(candidate)
        estimator.fit(X_train_selected, transformed_target)
        predictions = estimator.predict(X_valid_selected)  # (n_valid,)

    if candidate.tie_predicted_zeros:
        predictions = _tie_bottom_predictions(  # (n_valid,)
            predictions,
            train_table["target_points"],
            train_table["target_season"],
        )

    selection_records = [
        {
            "candidate": candidate.identifier,
            "validation_season": validation_season,
            "position_scope": position_scope,
            "feature": column,
            "selection_rank": rank,
            "selection_score": float(scores[column]),
            "selected_feature_count": len(selected),
        }
        for rank, column in enumerate(selected, start=1)
    ]
    return np.asarray(predictions, dtype="float64"), selection_records  # (n_valid,)


def walk_forward_candidate(
    table: pd.DataFrame,
    X: pd.DataFrame,
    candidate: Phase2Candidate,
    validation_seasons: tuple[int, ...],
    minimum_training_season: int,
) -> WalkForwardOutput:
    """Evaluate one candidate using strictly chronological outer folds."""
    # table: (n_player_seasons, c_keys); X: (n_player_seasons, d_features)
    if not table.index.equals(X.index):
        raise ValueError("Phase 2 table and feature indices do not align.")

    prediction_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []

    for validation_season in validation_seasons:
        train_mask = (  # (n_player_seasons,)
            (table["target_season"] >= minimum_training_season)
            & (table["target_season"] < validation_season)
            & table["target_points"].notna()
        )
        valid_mask = (  # (n_player_seasons,)
            (table["target_season"] == validation_season)
            & table["target_points"].notna()
        )
        train_table = table.loc[train_mask]  # (n_train, c_keys)
        valid_table = table.loc[valid_mask]  # (n_valid, c_keys)
        X_train = X.loc[train_mask]  # (n_train, d_features)
        X_valid = X.loc[valid_mask]  # (n_valid, d_features)
        if train_table.empty or valid_table.empty:
            raise ValueError(f"Empty Phase 2 fold for season {validation_season}.")
        if int(train_table["target_season"].max()) >= validation_season:
            raise AssertionError("Phase 2 training rows are not strictly historical.")

        started = time.perf_counter()
        fold_predictions = pd.Series(  # (n_valid,)
            index=valid_table.index,
            dtype="float64",
        )
        scopes = (
            tuple(sorted(valid_table["model_position"].unique()))
            if candidate.position_specific
            else ("ALL",)
        )
        for scope in scopes:
            if scope == "ALL":
                scope_train_mask = pd.Series(
                    True, index=train_table.index
                )  # (n_train,)
                scope_valid_mask = pd.Series(
                    True, index=valid_table.index
                )  # (n_valid,)
            else:
                scope_train_mask = train_table["model_position"].eq(scope)  # (n_train,)
                scope_valid_mask = valid_table["model_position"].eq(scope)  # (n_valid,)

            scope_predictions, scope_selections = _fit_scope(
                candidate,
                X_train.loc[scope_train_mask],
                X_valid.loc[scope_valid_mask],
                train_table.loc[scope_train_mask],
                valid_table.loc[scope_valid_mask],
                validation_season,
                str(scope),
            )
            fold_predictions.loc[scope_valid_mask] = scope_predictions
            selection_records.extend(scope_selections)

        if fold_predictions.isna().any():
            raise AssertionError("A Phase 2 validation row has no prediction.")

        fold_frame = valid_table[  # (n_valid, 5)
            ["player_id", "target_season", "model_position", "target_points"]
        ].copy()
        fold_frame["candidate"] = candidate.identifier
        fold_frame["predicted"] = fold_predictions  # (n_valid,)
        fold_frame["fold_fit_seconds"] = time.perf_counter() - started
        prediction_frames.append(fold_frame)

        fold_summary = metric_summary(
            fold_frame["target_points"],
            fold_frame["predicted"],
            fold_frame["model_position"],
            fold_frame["target_season"],
        )
        metric_records.append(
            {
                "candidate": candidate.identifier,
                "validation_season": validation_season,
                "rows": len(fold_frame),
                "fit_seconds": float(fold_frame["fold_fit_seconds"].iloc[0]),
                **fold_summary,
            }
        )

    predictions = pd.concat(prediction_frames).sort_index()  # (n_validation_rows, 7)
    fold_metrics = pd.DataFrame(metric_records)  # (n_validation_folds, m_metrics)
    selected_features = pd.DataFrame(selection_records)  # (n_selected_records, 7)
    return WalkForwardOutput(predictions, fold_metrics, selected_features)


def blend_walk_forward_predictions(
    prediction_frames: dict[str, pd.DataFrame],
    identifier: str,
    components: tuple[str, ...],
    weights: tuple[float, ...],
) -> pd.DataFrame:
    """Blend prediction ranks without observing validation outcomes."""
    if len(components) != len(weights) or not components:
        raise ValueError("Blend components and weights must have equal nonzero length.")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Blend weights must sum to one.")

    base = prediction_frames[components[0]][  # (n_validation_rows, 5)
        ["player_id", "target_season", "model_position", "target_points"]
    ].copy()
    blended = pd.Series(0.0, index=base.index)  # (n_validation_rows,)
    for component, weight in zip(components, weights, strict=True):
        frame = prediction_frames[component]  # (n_validation_rows, 7)
        if not frame.index.equals(base.index):
            raise ValueError("Blend component row indices do not align.")
        ranks = frame.groupby(  # (n_validation_rows,)
            ["target_season", "model_position"], observed=True
        )["predicted"].rank(method="average", pct=True)
        blended = blended + weight * ranks  # (n_validation_rows,)

    base["candidate"] = identifier
    base["predicted"] = blended
    base["fold_fit_seconds"] = 0.0
    return base  # (n_validation_rows, 7)


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Summarize combined walk-forward predictions."""
    # predictions: (n_validation_rows, 7)
    return metric_summary(
        predictions["target_points"],
        predictions["predicted"],
        predictions["model_position"],
        predictions["target_season"],
    )
