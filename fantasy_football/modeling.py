"""Temporal model search, grouped feature minimization, audits, and forecasts."""

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, ttest_1samp
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .constants import CURRENT_SEASON
from .features import FeatureSet, build_features
from .metrics import clustered_spearman_interval, macro_rank_metric, metric_summary
from .models import MAX_CPU_THREADS, ModelSpec, candidate_specs, make_estimator
from .provenance import file_sha256, package_versions, synchronize_project_sizes


DEVELOPMENT_SEASONS = (2020, 2021, 2022, 2023)
AUDIT_SEASONS = (2024, 2025)
SEARCH_WINDOW = 8
RANDOM_SEED = 20260809


@dataclass(frozen=True, slots=True)
class Champion:
    """Frozen model and feature-selection decision for one player cohort."""

    cohort: str
    spec: ModelSpec
    tier: str
    training_window: int | None
    selected_atoms: tuple[str, ...]
    development_spearman: float
    development_mae: float


@dataclass(frozen=True, slots=True)
class ExperimentOutput:
    """Paths and headline decisions from one complete experiment run."""

    offense_champion: Champion
    kicker_champion: Champion
    current_prediction_rows: int
    artifact_dir: str


@dataclass(frozen=True, slots=True)
class CohortOutput:
    """Persisted evaluation and current forecasts for one cohort."""

    champion: Champion
    ranking_elastic_spec: ModelSpec
    current: pd.DataFrame  # (n_current_players, c_current)
    audit_predictions: pd.DataFrame  # (n_audit_players, c_audit)
    audit_metrics: pd.DataFrame  # (n_metric_rows, c_metrics)
    feature_columns: tuple[str, ...]
    intervals: dict[str, dict[str, float]]


def _cohort_table(modeling_table: pd.DataFrame, cohort: str) -> pd.DataFrame:
    # modeling_table: (n_player_seasons, c_modeling)
    if cohort == "offense":
        cohort_mask = modeling_table["model_position"].ne("K")  # (n_player_seasons,)
    elif cohort == "kicker":
        cohort_mask = modeling_table["model_position"].eq("K")  # (n_player_seasons,)
    else:
        raise ValueError(f"Unknown cohort: {cohort!r}")
    selected = modeling_table[cohort_mask].copy()  # (n_cohort_rows, c_modeling)
    return selected  # (n_cohort_rows, c_modeling)


def _minimum_target_season(tier: str) -> int:
    return 2013 if tier == "utilization" else 2002


def _split_masks(
    table: pd.DataFrame,
    validation_season: int,
    tier: str,
    training_window: int | None,
) -> tuple[pd.Series, pd.Series]:
    # table: (n_cohort_rows, c_modeling); train, valid: (n_cohort_rows,)
    lower_bound = _minimum_target_season(tier)
    if training_window is not None:
        lower_bound = max(lower_bound, validation_season - training_window)
    train = (  # (n_cohort_rows,)
        table["target_points"].notna()
        & table["target_season"].ge(lower_bound)
        & table["target_season"].lt(validation_season)
    )
    valid = table["target_points"].notna() & table[
        "target_season"
    ].eq(  # (n_cohort_rows,)
        validation_season
    )
    return train, valid  # each: (n_cohort_rows,)


def _fit_predict(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    spec: ModelSpec,
    validation_season: int,
    tier: str,
    training_window: int | None,
    atoms: tuple[str, ...] | None = None,
) -> tuple[Pipeline, pd.Series, pd.Series, pd.Series]:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    train_mask, valid_mask = _split_masks(  # each: (n_cohort_rows,)
        table,
        validation_season,
        tier,
        training_window,
    )
    selected_atoms = feature_set.atoms if atoms is None else atoms
    columns = feature_set.columns_for_atoms(selected_atoms)
    X_train = feature_set.frame.loc[train_mask, columns]  # (n_train, d_selected)
    y_train = table.loc[train_mask, "target_points"].to_numpy(  # (n_train,)
        dtype="float32"
    )
    X_valid = feature_set.frame.loc[valid_mask, columns]  # (n_valid, d_selected)

    estimator = make_estimator(spec)
    estimator.fit(X_train, y_train)
    predicted_values = estimator.predict(X_valid)  # (n_valid,)
    predictions = pd.Series(  # (n_valid,)
        np.clip(predicted_values, a_min=0.0, a_max=None),
        index=X_valid.index,
        name="predicted_points",
    )
    return (
        estimator,
        predictions,
        train_mask,
        valid_mask,
    )  # (n_valid,), masks: (n_cohort_rows,)


def _evaluate_spec(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    spec: ModelSpec,
    seasons: tuple[int, ...],
    tier: str,
    training_window: int | None,
    atoms: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    fold_results: list[dict[str, Any]] = []
    for season in seasons:
        _, predictions, _, valid_mask = (
            _fit_predict(  # predictions: (n_valid,); valid_mask: (n_cohort_rows,)
                table,
                feature_set,
                spec,
                season,
                tier,
                training_window,
                atoms,
            )
        )
        valid = table.loc[valid_mask]  # (n_valid, c_modeling)
        metrics = metric_summary(
            valid["target_points"],
            predictions,
            valid["model_position"],
            valid["target_season"],
        )
        fold_results.append(
            {
                "spec_id": spec.identifier,
                "family": spec.family,
                "parameters": json.dumps(spec.parameters, sort_keys=True),
                "tier": tier,
                "training_window": training_window or "all",
                "validation_season": season,
                "feature_atom_count": len(
                    feature_set.atoms if atoms is None else atoms
                ),
                **metrics,
            }
        )
    return fold_results


def _summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(results)  # (n_fold_results, c_metrics)
    group_columns = [
        "spec_id",
        "family",
        "parameters",
        "tier",
        "training_window",
        "feature_atom_count",
    ]
    metric_columns = [
        "spearman",
        "kendall_tau_b",
        "mae",
        "rmse",
        "r2",
        "rank_mae",
        "ndcg_at_roster_cutoff",
        "top_k_recall",
    ]
    summary = (  # (n_settings, c_summary)
        frame.groupby(group_columns, as_index=False, dropna=False)[metric_columns]
        .mean()
        .sort_values(["spearman", "mae"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary  # (n_settings, c_summary)


def _select_family_finalists(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    specs: list[ModelSpec],
    tier: str,
) -> tuple[list[ModelSpec], list[dict[str, Any]]]:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    recent_results: list[dict[str, Any]] = []
    latest_development = (max(DEVELOPMENT_SEASONS),)
    for spec in specs:
        recent_results.extend(
            _evaluate_spec(
                table,
                feature_set,
                spec,
                latest_development,
                tier,
                SEARCH_WINDOW,
            )
        )
    recent_summary = _summarize_results(recent_results)  # (n_specs, c_summary)
    finalist_ids = (
        recent_summary.sort_values(["spearman", "mae"], ascending=[False, True])
        .groupby("family", observed=True)
        .head(1)["spec_id"]
        .tolist()
    )
    finalists = [spec for spec in specs if spec.identifier in finalist_ids]
    return finalists, recent_results


def _search_models(
    table: pd.DataFrame,
    cohort: str,
    specs: list[ModelSpec],
) -> tuple[ModelSpec, str, int | None, pd.DataFrame, pd.DataFrame]:
    # table: (n_cohort_rows, c_modeling)
    base_features = build_features(  # frame: (n_cohort_rows, d_base)
        table,
        cohort,
        "base",
    )
    finalists, recent_results = _select_family_finalists(
        table,
        base_features,
        specs,
        "base",
    )

    development_results: list[dict[str, Any]] = []
    for spec in finalists:
        development_results.extend(
            _evaluate_spec(
                table,
                base_features,
                spec,
                DEVELOPMENT_SEASONS,
                "base",
                SEARCH_WINDOW,
            )
        )

    if cohort == "offense":
        utilization_features = build_features(  # frame: (n_cohort_rows, d_utilization)
            table,
            cohort,
            "utilization",
        )
        for spec in finalists:
            development_results.extend(
                _evaluate_spec(
                    table,
                    utilization_features,
                    spec,
                    DEVELOPMENT_SEASONS,
                    "utilization",
                    SEARCH_WINDOW,
                )
            )

    development_summary = _summarize_results(  # (n_finalists, c_summary)
        development_results
    )
    best_row = development_summary.iloc[0]  # (c_summary,)
    best_spec = next(
        spec for spec in finalists if spec.identifier == best_row["spec_id"]
    )
    best_tier = str(best_row["tier"])
    best_features = build_features(  # frame: (n_cohort_rows, d_best)
        table,
        cohort,
        best_tier,
    )

    window_results: list[dict[str, Any]] = []
    for window in (5, 8, None):
        window_results.extend(
            _evaluate_spec(
                table,
                best_features,
                best_spec,
                DEVELOPMENT_SEASONS,
                best_tier,
                window,
            )
        )
    window_summary = _summarize_results(window_results)  # (3, c_summary)
    window_row = window_summary.iloc[0]  # (c_summary,)
    best_window = (
        None
        if window_row["training_window"] == "all"
        else int(window_row["training_window"])
    )

    all_search = pd.DataFrame(  # (n_search_folds, c_metrics)
        [*recent_results, *development_results, *window_results]
    )
    return best_spec, best_tier, best_window, all_search, window_summary


def _permutation_importance(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    spec: ModelSpec,
    tier: str,
    training_window: int | None,
) -> pd.DataFrame:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    records: list[dict[str, Any]] = []
    for season in DEVELOPMENT_SEASONS:
        estimator, predictions, _, valid_mask = (
            _fit_predict(  # predictions: (n_valid,); valid_mask: (n_cohort_rows,)
                table,
                feature_set,
                spec,
                season,
                tier,
                training_window,
            )
        )
        valid = table.loc[valid_mask]  # (n_valid, c_modeling)
        baseline = macro_rank_metric(
            valid["target_points"],
            predictions,
            valid["model_position"],
            valid["target_season"],
        )
        columns = feature_set.columns_for_atoms(feature_set.atoms)
        X_valid = feature_set.frame.loc[valid_mask, columns]  # (n_valid, d_all)
        for atom_index, atom in enumerate(feature_set.atoms):
            atom_columns = feature_set.columns_for_atoms((atom,))
            atom_columns = [
                column
                for column in atom_columns
                if column not in feature_set.mandatory_features
            ]
            repeated_importance: list[float] = []  # (n_repeats,)
            for repeat in range(3):
                permuted = X_valid.copy()  # (n_valid, d_all)
                rng = np.random.default_rng(
                    RANDOM_SEED + season + atom_index * 17 + repeat
                )
                for position in valid["model_position"].unique():
                    position_mask = valid["model_position"].eq(position)  # (n_valid,)
                    row_index = valid.index[position_mask]  # (n_position,)
                    order = rng.permutation(len(row_index))  # (n_position,)
                    atom_values = X_valid.loc[
                        row_index, atom_columns
                    ].to_numpy()  # (n_position, d_atom)
                    permuted.loc[row_index, atom_columns] = (
                        atom_values[  # (n_position, d_atom)
                            order
                        ]
                    )
                shuffled_values = estimator.predict(permuted)  # (n_valid,)
                shuffled = pd.Series(  # (n_valid,)
                    np.clip(shuffled_values, a_min=0.0, a_max=None),
                    index=permuted.index,
                )
                score = macro_rank_metric(
                    valid["target_points"],
                    shuffled,
                    valid["model_position"],
                    valid["target_season"],
                )
                repeated_importance.append(baseline - score)
            records.append(
                {
                    "validation_season": season,
                    "atom": atom,
                    "permutation_importance": float(np.mean(repeated_importance)),
                    "positive_importance": float(np.mean(repeated_importance)) > 0,
                }
            )
    return pd.DataFrame(records)  # (n_seasons * n_atoms, 4)


def _elastic_importance(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    elastic_spec: ModelSpec,
    tier: str,
    training_window: int | None,
) -> dict[str, float]:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    validation_season = max(DEVELOPMENT_SEASONS) + 1
    estimator, _, train_mask, _ = _fit_predict(  # train_mask: (n_cohort_rows,)
        table,
        feature_set,
        elastic_spec,
        validation_season,
        tier,
        training_window,
    )
    coefficients = np.abs(estimator.named_steps["model"].coef_)  # (d,)
    columns = feature_set.columns_for_atoms(feature_set.atoms)
    importance = {atom: 0.0 for atom in feature_set.atoms}
    for column, coefficient in zip(columns, coefficients, strict=True):
        atom = feature_set.atom_by_feature.get(column)
        if atom is not None:
            importance[atom] += float(coefficient)
    assert int(train_mask.sum()) > 0
    return importance


def _univariate_importance(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    tier: str,
    training_window: int | None,
) -> dict[str, float]:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    cutoff = max(DEVELOPMENT_SEASONS) + 1
    lower = _minimum_target_season(tier)
    if training_window is not None:
        lower = max(lower, cutoff - training_window)
    mask = (  # (n_cohort_rows,)
        table["target_points"].notna()
        & table["target_season"].ge(lower)
        & table["target_season"].lt(cutoff)
    )
    importance: dict[str, float] = {}
    for atom in feature_set.atoms:
        raw_column = atom if atom in feature_set.frame else f"lag1_{atom}"
        association = macro_rank_metric(
            table.loc[mask, "target_points"],
            feature_set.frame.loc[mask, raw_column].fillna(0.0),
            table.loc[mask, "model_position"],
            table.loc[mask, "target_season"],
        )
        importance[atom] = abs(association)
    return importance


def _rank_atoms(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    champion_spec: ModelSpec,
    elastic_spec: ModelSpec,
    tier: str,
    training_window: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    permutation = _permutation_importance(  # (n_development_seasons * n_atoms, 4)
        table,
        feature_set,
        champion_spec,
        tier,
        training_window,
    )
    grouped = permutation.groupby("atom", as_index=False).agg(  # (n_atoms, 3)
        permutation_importance=("permutation_importance", "mean"),
        positive_fold_fraction=("positive_importance", "mean"),
    )
    elastic = _elastic_importance(
        table,
        feature_set,
        elastic_spec,
        tier,
        training_window,
    )
    univariate = _univariate_importance(table, feature_set, tier, training_window)
    grouped["elastic_abs_coefficient"] = (
        grouped["atom"]
        .map(elastic)
        .fillna(  # (n_atoms,)
            0.0
        )
    )
    grouped["univariate_abs_spearman"] = (
        grouped["atom"]
        .map(  # (n_atoms,)
            univariate
        )
        .fillna(0.0)
    )
    rank_columns = [
        "permutation_importance",
        "elastic_abs_coefficient",
        "univariate_abs_spearman",
    ]
    for column in rank_columns:
        grouped[f"rank_{column}"] = grouped[column].rank(  # (n_atoms,)
            method="average",
            ascending=False,
            pct=True,
        )
    component_ranks = grouped[  # (n_atoms, 3)
        [f"rank_{column}" for column in rank_columns]
    ]
    grouped["consensus_rank"] = component_ranks.mean(axis=1)  # (n_atoms,)
    grouped.sort_values(  # (n_atoms, c_ranking)
        ["consensus_rank", "atom"],
        inplace=True,
    )
    grouped.reset_index(  # (n_atoms, c_ranking)
        drop=True,
        inplace=True,
    )
    return grouped, permutation  # (n_atoms, c_ranking), (n_importance_rows, 4)


def _compact_decision(
    folds: pd.DataFrame,
    full_folds: pd.DataFrame,
) -> dict[str, float | bool]:
    """Compare one atom set with the full model on paired development folds."""
    # folds, full_folds: (n_development_seasons, c_fold_metrics)
    ordered_folds = folds.sort_values(  # (n_development_seasons, c_fold_metrics)
        "validation_season"
    )
    ordered_full = full_folds.sort_values(  # (n_development_seasons, c_fold_metrics)
        "validation_season"
    )
    differences = (
        ordered_full["spearman"].to_numpy() - ordered_folds["spearman"].to_numpy()
    )  # (v,)
    paired_standard_error = (
        float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        if len(differences) > 1
        else 0.0
    )
    mean_spearman = float(ordered_folds["spearman"].mean())
    mean_mae = float(ordered_folds["mae"].mean())
    full_spearman = float(ordered_full["spearman"].mean())
    full_mae = float(ordered_full["mae"].mean())
    rank_tolerance = mean_spearman >= full_spearman - 0.01
    point_tolerance = mean_mae <= full_mae * 1.02
    return {
        "mean_spearman": mean_spearman,
        "mean_mae": mean_mae,
        "spearman_loss_from_full": full_spearman - mean_spearman,
        "mae_ratio_to_full": mean_mae / full_mae,
        "paired_spearman_standard_error": paired_standard_error,
        "within_one_paired_se": mean_spearman >= full_spearman - paired_standard_error,
        "spearman_loss_within_0_01": rank_tolerance,
        "mae_within_two_percent": point_tolerance,
        "passes_compact_rule": rank_tolerance and point_tolerance,
    }


def _refine_atom_subset(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    spec: ModelSpec,
    tier: str,
    training_window: int | None,
    initial_atoms: tuple[str, ...],
    full_folds: pd.DataFrame,
    max_iterations: int | None,
) -> tuple[tuple[str, ...], pd.DataFrame, dict[str, float | bool]]:
    """Delete atoms greedily until no single removal passes the compact rule."""
    # table: (n_cohort_rows, c_modeling); full_folds: (n_development_seasons, c_fold_metrics)
    selected_atoms = initial_atoms
    records: list[dict[str, Any]] = []
    iteration = 1
    while 1 < len(selected_atoms) <= 15:
        candidates: list[tuple[tuple[str, ...], dict[str, float | bool]]] = []
        for omitted_atom in selected_atoms:
            candidate_atoms = tuple(
                atom for atom in selected_atoms if atom != omitted_atom
            )
            folds = pd.DataFrame(  # (n_development_seasons, c_fold_metrics)
                _evaluate_spec(
                    table,
                    feature_set,
                    spec,
                    DEVELOPMENT_SEASONS,
                    tier,
                    training_window,
                    candidate_atoms,
                )
            )
            decision = _compact_decision(folds, full_folds)
            records.append(
                {
                    "iteration": iteration,
                    "omitted_atom": omitted_atom,
                    "feature_atom_count": len(candidate_atoms),
                    "selected_atoms": ",".join(candidate_atoms),
                    **decision,
                }
            )
            candidates.append((candidate_atoms, decision))

        passing = [
            candidate for candidate in candidates if candidate[1]["passes_compact_rule"]
        ]
        if not passing:
            break
        selected_atoms, _ = sorted(
            passing,
            key=lambda candidate: (
                -float(candidate[1]["mean_spearman"]),
                float(candidate[1]["mean_mae"]),
                candidate[0],
            ),
        )[0]
        if max_iterations is not None and iteration >= max_iterations:
            break
        iteration += 1

    final_folds = pd.DataFrame(  # (n_development_seasons, c_fold_metrics)
        _evaluate_spec(
            table,
            feature_set,
            spec,
            DEVELOPMENT_SEASONS,
            tier,
            training_window,
            selected_atoms,
        )
    )
    final_decision = _compact_decision(final_folds, full_folds)
    refinement = pd.DataFrame(  # (n_refinement_trials, 13)
        records,
        columns=[
            "iteration",
            "omitted_atom",
            "feature_atom_count",
            "selected_atoms",
            "mean_spearman",
            "mean_mae",
            "spearman_loss_from_full",
            "mae_ratio_to_full",
            "paired_spearman_standard_error",
            "within_one_paired_se",
            "spearman_loss_within_0_01",
            "mae_within_two_percent",
            "passes_compact_rule",
        ],
    )
    return selected_atoms, refinement, final_decision  # refinement: (n_trials, 13)


def _select_minimum_atoms(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    spec: ModelSpec,
    tier: str,
    training_window: int | None,
    ranked_atoms: pd.DataFrame,
    max_refinement_iterations: int | None,
) -> tuple[
    tuple[str, ...],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | bool],
]:
    """Choose the smallest passing ranked prefix, then prune it locally."""
    # table: (n_cohort_rows, c_modeling); ranked_atoms: (n_atoms, c_ranking)
    atom_order = ranked_atoms["atom"].tolist()
    maximum_screened = min(15, len(atom_order))
    subset_sizes = list(range(1, maximum_screened + 1))
    if len(atom_order) not in subset_sizes:
        subset_sizes.append(len(atom_order))

    records: list[dict[str, Any]] = []
    for subset_size in subset_sizes:
        atoms = tuple(atom_order[:subset_size])
        records.extend(
            _evaluate_spec(
                table,
                feature_set,
                spec,
                DEVELOPMENT_SEASONS,
                tier,
                training_window,
                atoms,
            )
        )
    subset_results = pd.DataFrame(records)  # (n_subset_folds, c_fold_metrics)
    summary = _summarize_results(records)  # (n_subset_sizes, c_summary)

    full_size = len(atom_order)
    full_mask = subset_results["feature_atom_count"].eq(  # (n_subset_folds,)
        full_size
    )
    full_folds = subset_results[
        full_mask
    ].sort_values(  # (n_development_seasons, c_fold_metrics)
        "validation_season"
    )
    chosen_size = full_size
    decision_records: list[dict[str, Any]] = []
    for subset_size in sorted(subset_sizes):
        subset_mask = subset_results["feature_atom_count"].eq(  # (n_subset_folds,)
            subset_size
        )
        folds = subset_results[
            subset_mask
        ].sort_values(  # (n_development_seasons, c_fold_metrics)
            "validation_season"
        )
        decision = _compact_decision(folds, full_folds)
        decision_records.append(
            {
                "feature_atom_count": subset_size,
                "selected_atoms": ",".join(atom_order[:subset_size]),
                **decision,
            }
        )
        if chosen_size == full_size and decision["passes_compact_rule"]:
            chosen_size = subset_size

    decisions = pd.DataFrame(decision_records)  # (n_subset_sizes, 11)
    summary = summary.merge(  # (n_subset_sizes, c_decision_summary)
        decisions,
        on="feature_atom_count",
        how="left",
    )
    summary["selected_by_compact_rule"] = summary[
        "feature_atom_count"
    ].eq(  # (n_subset_sizes,)
        chosen_size
    )
    initial_atoms = tuple(atom_order[:chosen_size])
    selected_atoms, refinement, final_decision = _refine_atom_subset(
        table,
        feature_set,
        spec,
        tier,
        training_window,
        initial_atoms,
        full_folds,
        max_refinement_iterations,
    )
    return (
        selected_atoms,
        summary,
        refinement,
        final_decision,
    )  # frames: (n_sizes, c_summary), (n_trials, 13)


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    # p_values: (n_tests,); return: (n_tests,)
    values = p_values.to_numpy(dtype="float64")  # (n_tests,)
    order = np.argsort(values)  # (n_tests,)
    ranked = values[order]  # (n_tests,)
    adjusted = (  # (n_tests,)
        ranked * len(values) / np.arange(1, len(values) + 1)
    )
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(  # (n_tests,)
        0.0,
        1.0,
    )
    restored = np.empty_like(adjusted)  # (n_tests,)
    restored[order] = adjusted  # (n_tests,)
    return pd.Series(restored, index=p_values.index)  # (n_tests,)


def _correlation_tests(
    table: pd.DataFrame,
    feature_set: FeatureSet,
) -> pd.DataFrame:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    development_mask = table["target_season"].isin(  # (n_cohort_rows,)
        DEVELOPMENT_SEASONS
    )
    development = table[development_mask]  # (n_development_rows, c_modeling)
    records: list[dict[str, Any]] = []
    for atom in feature_set.atoms:
        raw_column = atom if atom in feature_set.frame else f"lag1_{atom}"
        fold_correlations: list[float] = []
        for _, group in development.groupby(
            ["target_season", "model_position"], observed=True
        ):
            # group: (n_group, c_modeling)
            values = feature_set.frame.loc[  # (n_group,)
                group.index,
                raw_column,
            ].fillna(0.0)
            if values.nunique() < 2 or group["target_points"].nunique() < 2:
                continue
            coefficient = spearmanr(values, group["target_points"]).statistic
            if not np.isnan(coefficient):
                fold_correlations.append(float(coefficient))
        if len(fold_correlations) >= 2:
            test = ttest_1samp(fold_correlations, popmean=0.0)
            p_value = float(test.pvalue)
        else:
            p_value = 1.0
        records.append(
            {
                "atom": atom,
                "mean_within_position_season_spearman": float(
                    np.mean(fold_correlations) if fold_correlations else 0.0
                ),
                "season_position_groups": len(fold_correlations),
                "exploratory_t_test_p": p_value,
            }
        )
    correlations = pd.DataFrame(records)  # (n_atoms, 4)
    correlations["benjamini_hochberg_q"] = _benjamini_hochberg(  # (n_atoms,)
        correlations["exploratory_t_test_p"]
    )
    correlations.sort_values(
        "mean_within_position_season_spearman",
        ascending=False,
        inplace=True,
    )  # (n_atoms, 5)
    return correlations  # (n_atoms, 5)


def _audit_predictions(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    champion: Champion,
    atoms: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    selected_atoms = champion.selected_atoms if atoms is None else atoms
    frames: list[pd.DataFrame] = []
    for season in AUDIT_SEASONS:
        _, predictions, train_mask, valid_mask = (
            _fit_predict(  # predictions: (n_valid,); masks: (n_cohort_rows,)
                table,
                feature_set,
                champion.spec,
                season,
                champion.tier,
                champion.training_window,
                selected_atoms,
            )
        )
        train = table.loc[train_mask]  # (n_train, c_modeling)
        valid = table.loc[valid_mask].copy()  # (n_valid, c_modeling)
        group_medians = train.groupby(  # (n_position_rookie_groups,)
            ["model_position", "is_rookie"], observed=True
        )["target_points"].median()
        position_medians = train.groupby(
            "model_position", observed=True
        )[  # (n_positions,)
            "target_points"
        ].median()
        median_predictions: list[float] = []  # (n_valid,)
        for position, is_rookie in zip(
            valid["model_position"], valid["is_rookie"], strict=True
        ):
            median_predictions.append(
                group_medians.get(
                    (position, is_rookie),
                    position_medians.get(
                        position, float(train["target_points"].median())
                    ),
                )
            )
        valid["predicted_points"] = predictions  # (n_valid,)
        valid["previous_points_baseline"] = valid[  # (n_valid,)
            "previous_points_baseline"
        ].clip(lower=0.0)
        valid["recency_baseline"] = (  # (n_valid,)
            0.67 * valid["lag1_target_points"] + 0.33 * valid["lag2_target_points"]
        ).clip(lower=0.0)
        valid["cohort_median_baseline"] = median_predictions  # (n_valid,)
        valid.rename(  # (n_valid, c_modeling + 4)
            columns={"target_points": "actual_points"},
            inplace=True,
        )
        audit_frame = valid[  # (n_valid, 11)
            [
                "player_id",
                "full_name",
                "team",
                "model_position",
                "target_season",
                "is_rookie",
                "actual_points",
                "predicted_points",
                "previous_points_baseline",
                "recency_baseline",
                "cohort_median_baseline",
            ]
        ]
        frames.append(audit_frame)
    predictions = pd.concat(frames, ignore_index=True)  # (n_audit_players, 11)

    metric_records: list[dict[str, Any]] = []
    season_groups: list[tuple[str, pd.DataFrame]] = []  # each frame: (n_season, 11)
    for season in AUDIT_SEASONS:
        season_mask = predictions["target_season"].eq(season)  # (n_audit_players,)
        season_group = predictions[season_mask]  # (n_season, 11)
        season_groups.append((str(season), season_group))
    for season_label, group in [*season_groups, ("combined", predictions)]:
        # group: (n_metric_scope, 11)
        for label, column in [
            ("model", "predicted_points"),
            ("previous_points", "previous_points_baseline"),
            ("two_year_recency", "recency_baseline"),
            ("cohort_median", "cohort_median_baseline"),
        ]:
            metrics = metric_summary(
                group["actual_points"],
                group[column],
                group["model_position"],
                group["target_season"],
            )
            metric_records.append(
                {"audit_season": season_label, "predictor": label, **metrics}
            )
    metric_frame = pd.DataFrame(metric_records)  # (n_metric_rows, c_metrics)
    return predictions, metric_frame  # (n_audit_players, 11), (n_metrics, c_metrics)


def _audit_sensitivity(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    champion: Champion,
    ranked_atoms: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate selected, top-15, and full atom sets on retrospective years."""
    # table: (n_cohort_rows, c_modeling); ranked_atoms: (n_atoms, c_ranking)
    ranked = tuple(ranked_atoms["atom"])
    requested_sets = [
        ("selected", champion.selected_atoms),
        ("top_15", ranked[: min(15, len(ranked))]),
        ("all", feature_set.atoms),
    ]
    seen: set[tuple[str, ...]] = set()
    records: list[dict[str, Any]] = []
    for label, atoms in requested_sets:
        if atoms in seen:
            continue
        seen.add(atoms)
        predictions, _ = _audit_predictions(  # (n_audit_players, 11)
            table,
            feature_set,
            replace(champion, selected_atoms=atoms),
            atoms,
        )
        groups = [("combined", predictions)]  # each frame: (n_scope_players, 11)
        groups.extend(
            (str(position), group)
            for position, group in predictions.groupby(
                "model_position",
                observed=True,
            )
        )
        for scope, group in groups:
            # group: (n_scope_players, c_audit)
            metrics = metric_summary(
                group["actual_points"],
                group["predicted_points"],
                group["model_position"],
                group["target_season"],
            )
            records.append(
                {
                    "atom_set": label,
                    "selected_atoms": ",".join(atoms),
                    "feature_atom_count": len(atoms),
                    "model_column_count": len(feature_set.columns_for_atoms(atoms)),
                    "scope": scope,
                    "rows": len(group),
                    **metrics,
                }
            )
    return pd.DataFrame(records)  # (n_atom_sets * n_scopes, c_metrics)


def _fit_final_model(
    table: pd.DataFrame,
    feature_set: FeatureSet,
    champion: Champion,
    current_season: int,
) -> tuple[Pipeline, pd.DataFrame, list[str]]:
    # table: (n_cohort_rows, c_modeling); feature_set.frame: (n_cohort_rows, d_all)
    lower_bound = _minimum_target_season(champion.tier)
    if champion.training_window is not None:
        lower_bound = max(lower_bound, current_season - champion.training_window)
    train_mask = (  # (n_cohort_rows,)
        table["target_points"].notna()
        & table["target_season"].ge(lower_bound)
        & table["target_season"].lt(current_season)
    )
    current_mask = table["target_season"].eq(current_season)  # (n_cohort_rows,)
    columns = feature_set.columns_for_atoms(champion.selected_atoms)
    X_train = feature_set.frame.loc[train_mask, columns]  # (n_train, d_selected)
    y_train = table.loc[train_mask, "target_points"].to_numpy(  # (n_train,)
        dtype="float32"
    )
    X_current = feature_set.frame.loc[current_mask, columns]  # (n_current, d_selected)
    estimator = make_estimator(champion.spec)
    estimator.fit(X_train, y_train)
    predicted_values = estimator.predict(X_current)  # (n_current,)

    current = table.loc[current_mask].copy()  # (n_current, c_modeling)
    current["predicted_points"] = np.clip(  # (n_current,)
        predicted_values,
        a_min=0.0,
        a_max=None,
    )
    current["model_name"] = champion.spec.identifier  # (n_current,)
    current["feature_tier"] = champion.tier  # (n_current,)
    current["selected_atoms"] = ",".join(champion.selected_atoms)  # (n_current,)
    current["model_feature_columns"] = ",".join(columns)  # (n_current,)

    training_min = X_train.min(axis=0)  # (d_selected,)
    training_max = X_train.max(axis=0)  # (d_selected,)
    below = X_current.lt(training_min, axis="columns")  # (n_current, d_selected)
    above = X_current.gt(training_max, axis="columns")  # (n_current, d_selected)
    current["out_of_range_feature_count"] = (
        (below | above)
        .sum(  # (n_current,)
            axis=1
        )
        .to_numpy()
    )
    current["out_of_distribution"] = current[
        "out_of_range_feature_count"
    ].gt(  # (n_current,)
        0
    )
    current["feature_coverage"] = (  # (n_current,)
        1.0 - X_current.isna().mean(axis=1).to_numpy()
    )

    feature_values = X_current.rename(  # (n_current, d_selected)
        columns=lambda column: f"feature_value__{column}"
    )
    current = pd.concat(  # (n_current, c_modeling + c_forecast + d_selected)
        [current, feature_values],
        axis=1,
    )
    return estimator, current, columns  # current: (n_current, c_current)


def _add_intervals_and_ranks(
    current: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    # current: (n_current, c_current); audit: (n_audit, c_audit)
    residuals = audit.assign(  # (n_audit, c_audit + 1)
        absolute_residual=(audit["actual_points"] - audit["predicted_points"]).abs()
    )
    position_quantiles = residuals.groupby(
        "model_position", observed=True
    )[  # (n_positions,)
        "absolute_residual"
    ].quantile(0.80)
    global_quantile = float(residuals["absolute_residual"].quantile(0.80))
    widths = (
        current["model_position"]
        .map(  # (n_current,)
            position_quantiles
        )
        .fillna(global_quantile)
    )
    current["prediction_interval_80_low"] = (  # (n_current,)
        current["predicted_points"] - widths
    ).clip(lower=0.0)
    current["prediction_interval_80_high"] = (  # (n_current,)
        current["predicted_points"] + widths
    )
    available = ~current["status"].isin({"CUT", "RET"})  # (n_current,)
    available_current = current.loc[available]  # (n_available, c_current)
    current["position_rank"] = pd.Series(  # (n_current,)
        pd.NA,
        index=current.index,
        dtype="Int32",
    )
    current.loc[available, "position_rank"] = (  # (n_available,)
        available_current.groupby("model_position", observed=True)["predicted_points"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    current["overall_point_rank"] = pd.Series(  # (n_current,)
        pd.NA,
        index=current.index,
        dtype="Int32",
    )
    current.loc[available, "overall_point_rank"] = (  # (n_available,)
        available_current["predicted_points"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    current["team_position_rank"] = pd.Series(  # (n_current,)
        pd.NA,
        index=current.index,
        dtype="Int32",
    )
    current.loc[available, "team_position_rank"] = (  # (n_available,)
        available_current.groupby(
            ["team", "model_position"],
            observed=True,
        )["predicted_points"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    return current  # (n_current, c_current)


def _run_cohort(
    modeling_table: pd.DataFrame,
    cohort: str,
    specs: list[ModelSpec],
    artifact_dir: Path,
    current_season: int,
    max_refinement_iterations: int | None,
) -> CohortOutput:
    # modeling_table: (n_player_seasons, c_modeling)
    table = _cohort_table(modeling_table, cohort)  # (n_cohort_rows, c_modeling)
    spec, tier, training_window, search_results, window_summary = (
        _search_models(  # search_results: (n_search_folds, c_metrics); window_summary: (3, c_summary)
            table,
            cohort,
            specs,
        )
    )
    features = build_features(  # frame: (n_cohort_rows, d_all)
        table,
        cohort,
        tier,
    )

    elastic_specs = [
        candidate for candidate in specs if candidate.family == "elastic_net"
    ]
    elastic_results: list[dict[str, Any]] = []
    for elastic_candidate in elastic_specs:
        elastic_results.extend(
            _evaluate_spec(
                table,
                features,
                elastic_candidate,
                DEVELOPMENT_SEASONS,
                tier,
                training_window,
            )
        )
    elastic_summary = _summarize_results(  # (n_elastic_specs, c_summary)
        elastic_results
    )
    elastic_id = elastic_summary.iloc[0]["spec_id"]  # scalar
    elastic_spec = next(
        candidate for candidate in elastic_specs if candidate.identifier == elastic_id
    )

    ranked_atoms, fold_importance = (
        _rank_atoms(  # (n_atoms, c_ranking), (n_importance_rows, 4)
            table,
            features,
            spec,
            elastic_spec,
            tier,
            training_window,
        )
    )
    (
        selected_atoms,
        subset_results,
        refinement_results,
        selected_decision,
    ) = (  # subset_results: (n_subset_sizes, c_summary); refinement: (n_trials, 13)
        _select_minimum_atoms(
            table,
            features,
            spec,
            tier,
            training_window,
            ranked_atoms,
            max_refinement_iterations,
        )
    )
    champion = Champion(
        cohort=cohort,
        spec=spec,
        tier=tier,
        training_window=training_window,
        selected_atoms=selected_atoms,
        development_spearman=float(selected_decision["mean_spearman"]),
        development_mae=float(selected_decision["mean_mae"]),
    )

    audit_predictions, audit_metrics = (
        _audit_predictions(  # (n_audit, c_audit), (n_metrics, c_metrics)
            table,
            features,
            champion,
        )
    )
    sensitivity = _audit_sensitivity(  # (n_sensitivity_rows, c_metrics)
        table,
        features,
        champion,
        ranked_atoms,
    )
    correlations = _correlation_tests(table, features)  # (n_atoms, c_tests)
    estimator, current, feature_columns = (
        _fit_final_model(  # current: (n_current, c_current)
            table,
            features,
            champion,
            current_season,
        )
    )
    current = _add_intervals_and_ranks(  # (n_current, c_current)
        current,
        audit_predictions,
    )
    intervals = {
        "model_spearman": clustered_spearman_interval(
            audit_predictions,
            "predicted_points",
        ),
        "difference_vs_previous_points": clustered_spearman_interval(
            audit_predictions,
            "predicted_points",
            "previous_points_baseline",
        ),
    }

    prefix = cohort
    search_results.to_csv(artifact_dir / f"{prefix}_model_search.csv", index=False)
    pd.DataFrame(elastic_results).to_csv(
        artifact_dir / f"{prefix}_elastic_ranking_search.csv",
        index=False,
    )
    window_summary.to_csv(artifact_dir / f"{prefix}_window_search.csv", index=False)
    ranked_atoms.to_csv(artifact_dir / f"{prefix}_feature_ranking.csv", index=False)
    fold_importance.to_csv(
        artifact_dir / f"{prefix}_fold_permutation_importance.csv", index=False
    )
    subset_results.to_csv(artifact_dir / f"{prefix}_subset_search.csv", index=False)
    refinement_results.to_csv(
        artifact_dir / f"{prefix}_backward_refinement.csv",
        index=False,
    )
    sensitivity.to_csv(
        artifact_dir / f"{prefix}_audit_sensitivity.csv",
        index=False,
    )
    correlations.to_csv(artifact_dir / f"{prefix}_correlations.csv", index=False)
    audit_predictions.to_parquet(
        artifact_dir / f"{prefix}_audit_predictions.parquet", index=False
    )
    audit_metrics.to_csv(artifact_dir / f"{prefix}_audit_metrics.csv", index=False)
    (artifact_dir / f"{prefix}_audit_intervals.json").write_text(
        json.dumps(intervals, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "estimator": estimator,
            "feature_columns": feature_columns,
            "champion": asdict(champion),
        },
        artifact_dir / f"{prefix}_model.joblib",
        compress=3,
    )
    return CohortOutput(
        champion=champion,
        ranking_elastic_spec=elastic_spec,
        current=current,
        audit_predictions=audit_predictions,
        audit_metrics=audit_metrics,
        feature_columns=tuple(feature_columns),
        intervals=intervals,
    )


def _combined_metrics(
    audit_metrics: pd.DataFrame,
    predictor: str,
) -> dict[str, float]:
    # audit_metrics: (n_metric_rows, c_metrics)
    combined_mask = (  # (n_metric_rows,)
        audit_metrics["audit_season"].eq("combined")
        & audit_metrics["predictor"].eq(predictor)
    )
    row = audit_metrics[combined_mask].iloc[0]  # (c_metrics,)
    return {
        column: float(value)
        for column, value in row.items()
        if column not in {"audit_season", "predictor"}
    }


def _write_result_summary(
    root: Path,
    artifact_dir: Path,
    offense: CohortOutput,
    kicker: CohortOutput,
    current: pd.DataFrame,
    run_mode: str,
    candidate_setting_count: int,
) -> None:
    # current: (n_current, c_current)
    build_summary = json.loads(
        (root / "data" / "processed" / "build_summary.json").read_text(encoding="utf-8")
    )
    summary = {
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_mode": run_mode,
        "data_and_artifact_bytes": 0,
        "scoring": {
            "profile": build_summary["scoring_profile"],
            "config_sha256": build_summary["scoring_config_sha256"],
            "target_period": build_summary["target_period"],
        },
        "data": {
            key: build_summary[key]
            for key in (
                "player_game_rows",
                "player_game_players",
                "player_season_rows",
                "modeling_rows",
                "current_roster_players",
                "current_fantasy_players",
                "snapshot_date",
            )
        },
        "search": {
            "candidate_model_settings": candidate_setting_count,
            "development_seasons": list(DEVELOPMENT_SEASONS),
            "retrospective_evaluation_seasons": list(AUDIT_SEASONS),
            "compact_rule": (
                "smallest consensus-ranked prefix with mean Spearman loss <= 0.01 "
                "and mean MAE <= 1.02 times the full model, followed by "
                + (
                    "one leave-one-atom-out deletion pass"
                    if run_mode == "smoke"
                    else "greedy leave-one-atom-out deletion until no removal passes"
                )
            ),
            "scope_caveat": (
                "This is a locally irreducible set on a greedy search path, not "
                "a global optimum over every atom combination."
            ),
        },
        "offense": {
            "champion": asdict(offense.champion),
            "ranking_elastic_spec": asdict(offense.ranking_elastic_spec),
            "model_column_count": len(offense.feature_columns),
            "retrospective_2024_2025": _combined_metrics(
                offense.audit_metrics,
                "model",
            ),
            "previous_points_baseline": _combined_metrics(
                offense.audit_metrics,
                "previous_points",
            ),
            "intervals": offense.intervals,
        },
        "kicker": {
            "champion": asdict(kicker.champion),
            "ranking_elastic_spec": asdict(kicker.ranking_elastic_spec),
            "model_column_count": len(kicker.feature_columns),
            "retrospective_2024_2025": _combined_metrics(
                kicker.audit_metrics,
                "model",
            ),
            "previous_points_baseline": _combined_metrics(
                kicker.audit_metrics,
                "previous_points",
            ),
            "intervals": kicker.intervals,
        },
        "current_predictions": {
            "rows": len(current),
            "position_counts": {
                str(position): int(count)
                for position, count in current["model_position"].value_counts().items()
            },
            "note": (
                "Ranks cover fantasy-position players in the nflverse snapshot; "
                "non-fantasy roster rows remain metadata-only in the catalog."
            ),
        },
    }
    (artifact_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_experiments(
    root: Path,
    current_season: int,
    fast: bool,
) -> ExperimentOutput:
    processed_dir = root / "data" / "processed"
    artifact_dir = root / "artifacts" / "smoke" if fast else root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    build_summary_path = processed_dir / "build_summary.json"
    build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
    required_build_fields = {
        "current_season",
        "scoring_profile",
        "scoring_config_sha256",
        "snapshot_date",
        "target_period",
    }
    missing_build_fields = sorted(required_build_fields - build_summary.keys())
    if missing_build_fields:
        raise ValueError(
            f"Build summary is missing required fields: {missing_build_fields}. "
            "Run the build stage before training."
        )
    if int(build_summary["current_season"]) != current_season:
        raise ValueError(
            f"Built season {build_summary['current_season']} does not match requested "
            f"season {current_season}."
        )

    model_table_path = processed_dir / "modeling_table.parquet"
    modeling_table = pd.read_parquet(  # (n_player_seasons, c_modeling)
        model_table_path
    )
    if not modeling_table["target_season"].eq(current_season).any():
        raise ValueError(f"Modeling table has no rows for season {current_season}.")
    specs = candidate_specs(fast=fast)
    started_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    offense = _run_cohort(
        modeling_table,
        "offense",
        specs,
        artifact_dir,
        current_season,
        1 if fast else None,
    )
    kicker = _run_cohort(
        modeling_table,
        "kicker",
        specs,
        artifact_dir,
        current_season,
        1 if fast else None,
    )
    current = pd.concat(
        [offense.current, kicker.current],
        ignore_index=True,
    )  # (n_current, c_current)
    audit = pd.concat(
        [offense.audit_predictions, kicker.audit_predictions],
        ignore_index=True,
    )  # (n_audit, c_audit)
    current["scoring_profile"] = build_summary["scoring_profile"]  # (n_current,)
    audit["scoring_profile"] = build_summary["scoring_profile"]  # (n_audit,)
    available = ~current["status"].isin({"CUT", "RET"})  # (n_current,)
    current["overall_point_rank"] = pd.Series(  # (n_current,)
        pd.NA,
        index=current.index,
        dtype="Int32",
    )
    current.loc[available, "overall_point_rank"] = (  # (n_available,)
        current.loc[available, "predicted_points"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    current.to_parquet(
        artifact_dir / f"predictions_{current_season}.parquet", index=False
    )
    audit.to_parquet(artifact_dir / "audit_predictions.parquet", index=False)

    intervals = {
        "model_spearman": clustered_spearman_interval(audit, "predicted_points"),
        "difference_vs_previous_points": clustered_spearman_interval(
            audit,
            "predicted_points",
            "previous_points_baseline",
        ),
    }
    (artifact_dir / "audit_intervals.json").write_text(
        json.dumps(intervals, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "started_at_utc": started_at,
        "run_mode": "smoke" if fast else "full",
        "current_season": current_season,
        "development_seasons": DEVELOPMENT_SEASONS,
        "retrospective_evaluation_seasons": AUDIT_SEASONS,
        "candidate_model_settings": len(specs),
        "offense_champion": asdict(offense.champion),
        "offense_ranking_elastic_spec": asdict(offense.ranking_elastic_spec),
        "offense_model_columns": list(offense.feature_columns),
        "kicker_champion": asdict(kicker.champion),
        "kicker_ranking_elastic_spec": asdict(kicker.ranking_elastic_spec),
        "kicker_model_columns": list(kicker.feature_columns),
        "prediction_rows": len(current),
        "maximum_cpu_threads": MAX_CPU_THREADS,
        "thread_limit": "estimator n_jobs plus threadpoolctl process limits",
        "random_seed": RANDOM_SEED,
        "scoring_profile": build_summary["scoring_profile"],
        "scoring_config_sha256": build_summary["scoring_config_sha256"],
        "snapshot_date": build_summary["snapshot_date"],
        "source_manifest_sha256": file_sha256(root / "data" / "raw" / "manifest.json"),
        "modeling_table_sha256": file_sha256(model_table_path),
        "package_versions": package_versions(),
        "note": "Overall point rank is not an optimal draft order; use position rank.",
    }
    (artifact_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_result_summary(
        root,
        artifact_dir,
        offense,
        kicker,
        current,
        "smoke" if fast else "full",
        len(specs),
    )
    synchronize_project_sizes(root, update_canonical=not fast)
    return ExperimentOutput(
        offense_champion=offense.champion,
        kicker_champion=kicker.champion,
        current_prediction_rows=len(current),
        artifact_dir=artifact_dir.as_posix(),
    )


def run_experiments(
    root: Path,
    current_season: int = CURRENT_SEASON,
    fast: bool = False,
) -> ExperimentOutput:
    """Search models, freeze audits, refit, and produce current projections."""
    with threadpool_limits(limits=MAX_CPU_THREADS):
        return _run_experiments(root, current_season, fast)
