"""Bounded, laptop-friendly model families and hyperparameter grids."""

import os
from dataclasses import dataclass
from typing import Any

import xgboost as xgb

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


RANDOM_SEED = 20260809
MAX_CPU_THREADS = 4
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(MAX_CPU_THREADS))


class CompatibleXGBRegressor(RegressorMixin, BaseEstimator):
    """Adapt XGBoost 2.1 to the estimator tags required by scikit-learn 1.6."""

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters = parameters

    def fit(self, X: Any, y: Any) -> "CompatibleXGBRegressor":
        """Fit the contained XGBoost regressor."""
        # X: (n_samples, n_features); y: (n_samples,)
        self.estimator_ = xgb.XGBRegressor(**self.parameters)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        """Predict with the fitted contained regressor."""
        # X: (n_samples, n_features)
        check_is_fitted(self, "estimator_")
        predictions = self.estimator_.predict(X)  # (n_samples,)
        return predictions


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A deterministic model-family setting."""

    identifier: str
    family: str
    parameters: dict[str, Any]


def candidate_specs(fast: bool = False) -> list[ModelSpec]:
    """Return a small declared grid across linear and nonlinear families."""
    settings: list[tuple[str, dict[str, Any]]] = []
    settings.extend(("ridge", {"alpha": alpha}) for alpha in (1.0, 10.0, 100.0))
    settings.extend(
        ("elastic_net", {"alpha": alpha, "l1_ratio": l1_ratio})
        for alpha in (0.01, 0.05, 0.2)
        for l1_ratio in (0.8, 0.95, 1.0)
    )
    settings.extend(
        [
            (
                "random_forest",
                {"max_depth": None, "min_samples_leaf": 2, "max_features": 0.7},
            ),
            (
                "random_forest",
                {"max_depth": 12, "min_samples_leaf": 5, "max_features": 1.0},
            ),
            (
                "random_forest",
                {"max_depth": 8, "min_samples_leaf": 10, "max_features": 0.7},
            ),
            (
                "extra_trees",
                {"max_depth": None, "min_samples_leaf": 2, "max_features": 0.7},
            ),
            (
                "extra_trees",
                {"max_depth": 12, "min_samples_leaf": 5, "max_features": 1.0},
            ),
            (
                "extra_trees",
                {"max_depth": 8, "min_samples_leaf": 10, "max_features": 0.7},
            ),
            (
                "hist_gradient_boosting",
                {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
            ),
            (
                "hist_gradient_boosting",
                {"learning_rate": 0.08, "max_leaf_nodes": 31, "l2_regularization": 3.0},
            ),
            (
                "hist_gradient_boosting",
                {"learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 5.0},
            ),
            (
                "xgboost",
                {
                    "learning_rate": 0.03,
                    "max_depth": 2,
                    "min_child_weight": 5,
                    "reg_alpha": 0.0,
                },
            ),
            (
                "xgboost",
                {
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "min_child_weight": 5,
                    "reg_alpha": 0.1,
                },
            ),
            (
                "xgboost",
                {
                    "learning_rate": 0.03,
                    "max_depth": 4,
                    "min_child_weight": 10,
                    "reg_alpha": 0.5,
                },
            ),
            (
                "xgboost",
                {
                    "learning_rate": 0.08,
                    "max_depth": 2,
                    "min_child_weight": 10,
                    "reg_alpha": 1.0,
                },
            ),
            ("svr", {"C": 1.0, "epsilon": 0.1, "gamma": "scale"}),
            ("svr", {"C": 10.0, "epsilon": 0.2, "gamma": "scale"}),
            ("svr", {"C": 30.0, "epsilon": 0.5, "gamma": 0.01}),
            ("mlp", {"hidden_layer_sizes": (32,), "alpha": 0.01}),
            ("mlp", {"hidden_layer_sizes": (64, 32), "alpha": 0.1}),
        ]
    )
    if fast:
        settings = [
            setting
            for index, setting in enumerate(settings)
            if index in {0, 5, 8, 14, 17, 18, 22}
        ]

    family_counts: dict[str, int] = {}
    specs: list[ModelSpec] = []
    for family, parameters in settings:
        family_counts[family] = family_counts.get(family, 0) + 1
        identifier = f"{family}_{family_counts[family]:02d}"
        specs.append(ModelSpec(identifier, family, parameters))
    return specs


def make_estimator(spec: ModelSpec) -> Pipeline:
    """Construct a regression pipeline with fold-local imputation and scaling."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if spec.family == "ridge":
        model: RegressorMixin = Ridge(**spec.parameters)
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler()), ("model", model)]
        )
    if spec.family == "elastic_net":
        model = ElasticNet(
            **spec.parameters,
            max_iter=50_000,
            tol=1e-3,
            selection="random",
            random_state=RANDOM_SEED,
        )
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler()), ("model", model)]
        )
    if spec.family == "random_forest":
        model = RandomForestRegressor(
            **spec.parameters,
            n_estimators=240,
            n_jobs=MAX_CPU_THREADS,
            random_state=RANDOM_SEED,
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    if spec.family == "extra_trees":
        model = ExtraTreesRegressor(
            **spec.parameters,
            n_estimators=240,
            n_jobs=MAX_CPU_THREADS,
            random_state=RANDOM_SEED,
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    if spec.family == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            **spec.parameters,
            max_iter=250,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    if spec.family == "xgboost":
        model = CompatibleXGBRegressor(
            {
                **spec.parameters,
                "n_estimators": 320,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "n_jobs": MAX_CPU_THREADS,
                "random_state": RANDOM_SEED,
            }
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    if spec.family == "svr":
        model = SVR(**spec.parameters, kernel="rbf", cache_size=512)
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler()), ("model", model)]
        )
    if spec.family == "mlp":
        model = MLPRegressor(
            **spec.parameters,
            activation="relu",
            early_stopping=True,
            max_iter=800,
            n_iter_no_change=20,
            random_state=RANDOM_SEED,
        )
        return Pipeline(
            [("imputer", imputer), ("scaler", StandardScaler()), ("model", model)]
        )
    raise ValueError(f"Unknown model family: {spec.family!r}")
