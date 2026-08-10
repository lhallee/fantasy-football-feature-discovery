"""Run isolated, leakage-audited Phase 2 fantasy ranking experiments."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from threadpoolctl import threadpool_limits

from .metrics import clustered_spearman_interval, metric_summary
from .models import MAX_CPU_THREADS, RANDOM_SEED
from .phase2_features import RECENT_ROOM_COLUMNS, RECENT_TEAM_ENVIRONMENT_COLUMNS
from .phase2_features import RECENT_TRAJECTORY_COLUMNS, RECENT_WEEKLY_COLUMNS
from .phase2_features import REQUIRED_WEEKLY_COLUMNS, STABLE_ROOM_COLUMNS
from .phase2_features import STABLE_TEAM_ENVIRONMENT_COLUMNS
from .phase2_features import STABLE_TRAJECTORY_COLUMNS, STABLE_WEEKLY_COLUMNS
from .phase2_features import Phase2FeatureBundle, build_phase2_features
from .phase2_modeling import Phase2Candidate
from .phase2_modeling import blend_walk_forward_predictions
from .phase2_modeling import summarize_predictions, walk_forward_candidate
from .phase2_safety import hash_outputs, isolated_phase2_run
from .provenance import file_sha256, package_versions


OFFENSE_POSITIONS = ("QB", "RB", "WR", "TE")
VALID_STAGES = ("discovery", "retrospective")
TRAJECTORY_PREFIXES = ("lag1_late", "lag1_weekly_slope")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated leakage-safe Phase 2 experiment plan."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=VALID_STAGES, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        help="Discovery selection.json, required for retrospective stage.",
    )
    return parser.parse_args(argv)


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Phase 2 plan: {path}") from error
    if not isinstance(plan, dict):
        raise ValueError("Phase 2 plan must contain a JSON object.")
    required = {
        "plan_version",
        "plan_name",
        "discovery_seasons",
        "retrospective_seasons",
        "minimum_training_season",
        "candidates",
        "blends",
    }
    missing = sorted(required.difference(plan))
    if missing:
        raise ValueError(f"Phase 2 plan is missing fields: {missing!r}.")
    if not isinstance(plan["candidates"], list) or not plan["candidates"]:
        raise ValueError("Phase 2 plan must declare at least one candidate.")
    return plan


def _plan_hash(plan_path: Path) -> str:
    return file_sha256(plan_path)


def _weekly_feature_columns() -> tuple[str, ...]:
    declared = {
        *REQUIRED_WEEKLY_COLUMNS,
        *STABLE_WEEKLY_COLUMNS,
        *RECENT_WEEKLY_COLUMNS,
        *STABLE_TRAJECTORY_COLUMNS,
        *RECENT_TRAJECTORY_COLUMNS,
        *STABLE_TEAM_ENVIRONMENT_COLUMNS,
        *RECENT_TEAM_ENVIRONMENT_COLUMNS,
        *STABLE_ROOM_COLUMNS,
        *RECENT_ROOM_COLUMNS,
    }
    return tuple(sorted(declared))


def _weekly_input_paths(root: Path, maximum_source_season: int) -> list[Path]:
    paths = sorted((root / "data" / "raw" / "stats").glob("*.parquet"))
    selected: list[Path] = []
    for path in paths:
        try:
            season = int(path.stem.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            continue
        if season <= maximum_source_season:
            selected.append(path)
    if not selected:
        raise FileNotFoundError("No Phase 2 weekly-stat inputs were found.")
    return selected


def _load_weekly_stats(paths: Sequence[Path]) -> pd.DataFrame:
    declared = _weekly_feature_columns()
    frames: list[pd.DataFrame] = []
    for path in paths:
        available = set(pq.read_schema(path).names)
        columns = [column for column in declared if column in available]
        frame = pd.read_parquet(path, columns=columns)  # (n_file_rows, c_available)
        frame = frame.reindex(columns=declared)  # (n_file_rows, c_declared)
        frames.append(frame)
    weekly_stats = pd.concat(frames, ignore_index=True)  # (n_weekly, c_declared)
    return weekly_stats


def _feature_group(column: str) -> str:
    if column.startswith("room_"):
        return "room"
    if column.startswith(("team_", "prior_team_")):
        return "team"
    if column.startswith(TRAJECTORY_PREFIXES):
        return "trajectory"
    return "core"


def _candidate_feature_frame(
    bundle: Phase2FeatureBundle,
    candidate: Phase2Candidate,
    global_excluded_fragments: tuple[str, ...],
) -> pd.DataFrame:
    # bundle.frame: (n_player_seasons, d_tier_features)
    requested_groups = set(candidate.recipes)
    if not requested_groups or not requested_groups.issubset(
        {"all", "core", "trajectory", "room", "team"}
    ):
        raise ValueError(
            f"Candidate {candidate.identifier!r} has invalid feature groups."
        )
    excluded = (*global_excluded_fragments, *candidate.excluded_feature_fragments)
    columns = [
        column
        for column in bundle.frame.columns
        if ("all" in requested_groups or _feature_group(column) in requested_groups)
        and not any(fragment in column for fragment in excluded)
    ]
    if not columns:
        raise ValueError(f"Candidate {candidate.identifier!r} has no input features.")
    return bundle.frame.loc[:, columns]  # (n_player_seasons, d_candidate_pool)


def _table_for_bundle(
    modeling_table: pd.DataFrame,
    bundle: Phase2FeatureBundle,
) -> pd.DataFrame:
    metadata_columns = (
        "target_season",
        "player_id",
        "model_position",
        "target_points",
        "previous_points_baseline",
        "is_rookie",
        "team_changed",
    )
    source = modeling_table.loc[:, metadata_columns].copy()  # (n_targets, 7)
    if source.duplicated(["target_season", "player_id", "model_position"]).any():
        raise ValueError("Phase 2 modeling keys are not unique.")
    table = bundle.keys.merge(  # (n_bundle_rows, 7)
        source,
        on=["target_season", "player_id", "model_position"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    table.index = bundle.frame.index
    if table["target_points"].isna().all():
        raise ValueError("Phase 2 bundle has no observed targets.")
    return table


def _baseline_predictions(
    table: pd.DataFrame,
    validation_seasons: tuple[int, ...],
) -> pd.DataFrame:
    # table: (n_player_seasons, 7)
    mask = (
        table["target_season"].isin(validation_seasons) & table["target_points"].notna()
    )  # (n_player_seasons,)
    frame = table.loc[  # (n_validation_rows, 6)
        mask,
        [
            "player_id",
            "target_season",
            "model_position",
            "target_points",
            "previous_points_baseline",
        ],
    ].copy()
    frame["candidate"] = "previous_points_baseline"
    frame["predicted"] = frame["previous_points_baseline"].fillna(0.0)
    frame["fold_fit_seconds"] = 0.0
    frame.drop(columns="previous_points_baseline", inplace=True)
    return frame  # (n_validation_rows, 7)


def _fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    # predictions: (n_validation_rows, 7)
    records: list[dict[str, Any]] = []
    for season, frame in predictions.groupby("target_season", observed=True):
        # frame: (n_season_rows, 7)
        summary = metric_summary(
            frame["target_points"],
            frame["predicted"],
            frame["model_position"],
            frame["target_season"],
        )
        records.append(
            {
                "candidate": str(frame["candidate"].iloc[0]),
                "validation_season": int(season),
                "rows": len(frame),
                "fit_seconds": float(frame["fold_fit_seconds"].iloc[0]),
                **summary,
            }
        )
    return pd.DataFrame(records)  # (n_validation_folds, m_metrics)


def _summary_record(
    candidate: Phase2Candidate | None,
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    feature_pool_count: int,
    median_selected_count: float,
    *,
    stage: str,
    kind: str = "model",
) -> dict[str, Any]:
    summary = summarize_predictions(predictions)
    return {
        "stage": stage,
        "candidate": str(predictions["candidate"].iloc[0]),
        "kind": kind,
        "tier": "benchmark" if candidate is None else candidate.tier,
        "estimator": "direct" if candidate is None else candidate.estimator,
        "target": "points" if candidate is None else candidate.target,
        "position_specific": False
        if candidate is None
        else candidate.position_specific,
        "feature_pool_count": feature_pool_count,
        "median_selected_feature_count": median_selected_count,
        "mean_fold_fit_seconds": float(fold_metrics["fit_seconds"].mean()),
        **summary,
    }


def _candidate_mapping(plan: Mapping[str, Any]) -> dict[str, Phase2Candidate]:
    candidates = [Phase2Candidate.from_mapping(values) for values in plan["candidates"]]
    identifiers = [candidate.identifier for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Phase 2 candidate identifiers must be unique.")
    return {candidate.identifier: candidate for candidate in candidates}


def _blend_mapping(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    blends = {str(values["identifier"]): dict(values) for values in plan["blends"]}
    if len(blends) != len(plan["blends"]):
        raise ValueError("Phase 2 blend identifiers must be unique.")
    return blends


def _required_candidates(
    plan: Mapping[str, Any],
    stage: str,
    selection: Mapping[str, Any] | None,
) -> tuple[set[str], set[str]]:
    candidates = _candidate_mapping(plan)
    blends = _blend_mapping(plan)
    if stage == "discovery":
        return set(candidates), set(blends)
    if selection is None:
        raise ValueError("Retrospective stage requires a discovery selection.")
    identifier = str(selection["selected_candidate"])
    if identifier in candidates:
        return {identifier}, set()
    if identifier in blends:
        return set(str(value) for value in blends[identifier]["components"]), {
            identifier
        }
    raise ValueError(f"Selected candidate is not in the Phase 2 plan: {identifier!r}.")


def _selection_file(path: Path | None, plan_hash: str) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load Phase 2 selection: {path}") from error
    if selection.get("plan_sha256") != plan_hash:
        raise ValueError("Phase 2 selection and plan hashes do not match.")
    return selection


def _evaluate_models(
    plan: Mapping[str, Any],
    stage: str,
    validation_seasons: tuple[int, ...],
    bundles: Mapping[str, Phase2FeatureBundle],
    tables: Mapping[str, pd.DataFrame],
    required_identifiers: set[str],
) -> tuple[
    dict[str, pd.DataFrame],
    list[pd.DataFrame],
    list[pd.DataFrame],
    list[dict[str, Any]],
]:
    candidates = _candidate_mapping(plan)
    global_excluded = tuple(
        str(value) for value in plan.get("global_excluded_feature_fragments", ())
    )
    predictions_by_candidate: dict[str, pd.DataFrame] = {}
    fold_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for identifier in sorted(required_identifiers):
        candidate = candidates[identifier]
        bundle = bundles[candidate.tier]
        table = tables[candidate.tier]
        X = _candidate_feature_frame(bundle, candidate, global_excluded)
        minimum_training = max(
            int(plan["minimum_training_season"][candidate.tier]),
            int(table["target_season"].min()),
        )
        output = walk_forward_candidate(
            table,
            X,
            candidate,
            validation_seasons,
            minimum_training,
        )
        predictions_by_candidate[identifier] = output.predictions
        fold_frames.append(output.fold_metrics)
        selection_frames.append(output.selected_features)
        median_selected = float(
            output.selected_features.groupby(
                ["validation_season", "position_scope"], observed=True
            )["selected_feature_count"]
            .first()
            .median()
        )
        summaries.append(
            _summary_record(
                candidate,
                output.predictions,
                output.fold_metrics,
                len(X.columns),
                median_selected,
                stage=stage,
            )
        )
    return predictions_by_candidate, fold_frames, selection_frames, summaries


def _evaluate_blends(
    plan: Mapping[str, Any],
    stage: str,
    required_identifiers: set[str],
    predictions_by_candidate: dict[str, pd.DataFrame],
    fold_frames: list[pd.DataFrame],
    summaries: list[dict[str, Any]],
) -> None:
    blends = _blend_mapping(plan)
    for identifier in sorted(required_identifiers):
        blend = blends[identifier]
        components = tuple(str(value) for value in blend["components"])
        weights = tuple(float(value) for value in blend["weights"])
        predictions = blend_walk_forward_predictions(
            predictions_by_candidate,
            identifier,
            components,
            weights,
        )
        predictions_by_candidate[identifier] = predictions
        fold_metric = _fold_metrics(predictions)
        fold_frames.append(fold_metric)
        summaries.append(
            _summary_record(
                None,
                predictions,
                fold_metric,
                feature_pool_count=0,
                median_selected_count=float(
                    sum(
                        next(
                            row["median_selected_feature_count"]
                            for row in summaries
                            if row["candidate"] == component
                        )
                        for component in components
                    )
                ),
                stage=stage,
                kind="blend",
            )
        )


def _add_baseline(
    stage: str,
    validation_seasons: tuple[int, ...],
    table: pd.DataFrame,
    predictions_by_candidate: dict[str, pd.DataFrame],
    fold_frames: list[pd.DataFrame],
    summaries: list[dict[str, Any]],
) -> None:
    predictions = _baseline_predictions(table, validation_seasons)
    fold_metric = _fold_metrics(predictions)
    predictions_by_candidate["previous_points_baseline"] = predictions
    fold_frames.append(fold_metric)
    summaries.append(
        _summary_record(
            None,
            predictions,
            fold_metric,
            feature_pool_count=1,
            median_selected_count=1.0,
            stage=stage,
            kind="benchmark",
        )
    )


def _blend_is_eligible(
    blend: Mapping[str, Any],
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    minimum_gain: float,
) -> bool:
    identifier = str(blend["identifier"])
    components = [str(value) for value in blend["components"]]
    blend_rho = float(
        summary.loc[summary["candidate"].eq(identifier), "spearman"].iloc[0]
    )
    component_rho = float(
        summary.loc[summary["candidate"].isin(components), "spearman"].max()
    )
    if blend_rho < component_rho + minimum_gain:
        return False
    fold_pivot = fold_metrics.pivot(  # (n_validation_folds, n_candidates)
        index="validation_season",
        columns="candidate",
        values="spearman",
    )
    best_component = fold_pivot.loc[:, components].max(axis=1)  # (n_validation_folds,)
    improved = fold_pivot[identifier].gt(best_component)  # (n_validation_folds,)
    return int(improved.sum()) >= (len(improved) // 2 + 1)


def _select_discovery_candidate(
    plan: Mapping[str, Any],
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    plan_hash: str,
) -> dict[str, Any]:
    candidates = _candidate_mapping(plan)
    eligible = {
        identifier
        for identifier, candidate in candidates.items()
        if not candidate.benchmark_only
    }
    blend_minimum_gain = float(plan.get("blend_minimum_gain", 0.005))
    for blend in plan["blends"]:
        if _blend_is_eligible(blend, summary, fold_metrics, blend_minimum_gain):
            eligible.add(str(blend["identifier"]))
    if not eligible:
        raise ValueError("Phase 2 discovery has no selection-eligible candidates.")

    eligible_summary = summary.loc[summary["candidate"].isin(eligible)].copy()
    best_spearman = float(eligible_summary["spearman"].max())
    tolerance = float(plan.get("selection_tolerance", 0.005))
    near_best = eligible_summary.loc[
        eligible_summary["spearman"] >= best_spearman - tolerance
    ].copy()
    near_best.sort_values(
        ["median_selected_feature_count", "mean_fold_fit_seconds", "candidate"],
        inplace=True,
        kind="stable",
    )
    selected = near_best.iloc[0]
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": plan_hash,
        "selected_candidate": str(selected["candidate"]),
        "selection_rule": (
            f"within {tolerance:.3f} Spearman of the best eligible discovery "
            "candidate, then fewer selected columns, lower fold fit time, and ID"
        ),
        "best_discovery_spearman": best_spearman,
        "selected_discovery_spearman": float(selected["spearman"]),
        "selected_feature_count": float(selected["median_selected_feature_count"]),
        "eligible_candidates": sorted(eligible),
    }


def _season_block_interval(
    fold_metrics: pd.DataFrame, samples: int = 2000
) -> dict[str, float]:
    # fold_metrics: (n_validation_folds, m_metrics)
    values = fold_metrics["spearman"].to_numpy(dtype="float64")  # (n_seasons,)
    generator = np.random.default_rng(RANDOM_SEED)
    estimates = np.empty(samples, dtype="float64")  # (n_bootstrap_samples,)
    for index in range(samples):
        sampled = generator.choice(
            values, size=len(values), replace=True
        )  # (n_seasons,)
        estimates[index] = float(np.mean(sampled))
    return {
        "estimate": float(np.mean(values)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": samples,
    }


def _uncertainty(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> dict[str, Any]:
    interval_frame = predictions.rename(  # (n_validation_rows, 7)
        columns={"target_points": "actual_points"}
    )
    return {
        "player_cluster": clustered_spearman_interval(
            interval_frame,
            "predicted",
            samples=500,
            seed=RANDOM_SEED,
        ),
        "season_block": _season_block_interval(fold_metrics),
    }


def _position_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for position, frame in predictions.groupby("model_position", observed=True):
        # frame: (n_position_rows, 7)
        results[str(position)] = metric_summary(
            frame["target_points"],
            frame["predicted"],
            frame["model_position"],
            frame["target_season"],
        )
    return results


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def _write_run_ledger(
    path: Path,
    stage: str,
    plan: Mapping[str, Any],
    summary: pd.DataFrame,
    selection: Mapping[str, Any] | None,
    elapsed_seconds: float,
) -> None:
    ordered = summary.sort_values("spearman", ascending=False).head(10)
    table_lines = [
        "| Candidate | Spearman | MAE | Columns | Kind |",
        "|---|---:|---:|---:|---|",
    ]
    for row in ordered.itertuples(index=False):
        table_lines.append(
            f"| `{row.candidate}` | {row.spearman:.4f} | {row.mae:.2f} | "
            f"{row.median_selected_feature_count:.0f} | {row.kind} |"
        )
    selection_line = (
        "No selection was made."
        if selection is None
        else f"Selected procedure: `{selection['selected_candidate']}`."
    )
    text = "\n".join(
        [
            f"# Phase 2 run: {stage}",
            "",
            f"Plan: `{plan['plan_name']}`.",
            f"Elapsed time: {elapsed_seconds:.1f} seconds.",
            selection_line,
            "",
            *table_lines,
            "",
            "All folds train only on earlier target seasons. This run does not promote "
            "a model into Phase 1 artifacts or the player catalog.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _run_stage(
    root: Path,
    plan_path: Path,
    stage: str,
    selection_path: Path | None,
) -> Path:
    root = root.resolve()
    plan_path = plan_path.resolve()
    plan = _load_plan(plan_path)
    plan_hash = _plan_hash(plan_path)
    prior_selection = _selection_file(selection_path, plan_hash)
    if stage == "retrospective" and prior_selection is None:
        raise ValueError("Retrospective stage requires --selection.")

    validation_seasons = tuple(int(value) for value in plan[f"{stage}_seasons"])
    if tuple(sorted(validation_seasons)) != validation_seasons:
        raise ValueError("Phase 2 validation seasons must be sorted.")
    candidate_ids, blend_ids = _required_candidates(plan, stage, prior_selection)
    maximum_source_season = max(validation_seasons) - 1
    weekly_paths = _weekly_input_paths(root, maximum_source_season)
    input_paths: list[Path] = [
        plan_path,
        root / "data" / "processed" / "player_seasons.parquet",
        root / "data" / "processed" / "modeling_table.parquet",
        *weekly_paths,
    ]
    if selection_path is not None:
        input_paths.append(selection_path.resolve())

    label = f"{stage}-{plan_hash[:12]}"
    started = time.perf_counter()
    with isolated_phase2_run(root, label, input_paths=input_paths) as safety_run:
        weekly_stats = _load_weekly_stats(weekly_paths)  # (n_weekly, c_weekly)
        player_seasons = pd.read_parquet(  # (n_player_seasons, c_seasons)
            root / "data" / "processed" / "player_seasons.parquet"
        )
        player_seasons = player_seasons.loc[
            player_seasons["season"] <= maximum_source_season
        ].copy()  # (n_source_seasons, c_seasons)
        modeling_table = pd.read_parquet(  # (n_targets, c_modeling)
            root / "data" / "processed" / "modeling_table.parquet"
        )
        modeling_table = modeling_table.loc[
            modeling_table["model_position"].isin(OFFENSE_POSITIONS)
            & (modeling_table["target_season"] <= max(validation_seasons))
        ].copy()  # (n_offense_targets, c_modeling)

        needed_tiers = {
            _candidate_mapping(plan)[identifier].tier for identifier in candidate_ids
        }
        bundles: dict[str, Phase2FeatureBundle] = {}
        tables: dict[str, pd.DataFrame] = {}
        for tier in sorted(needed_tiers):
            bundle = build_phase2_features(
                weekly_stats,
                player_seasons,
                modeling_table,
                tier=tier,
                max_lag=int(plan.get("max_lag", 4)),
                late_weeks=int(plan.get("late_weeks", 6)),
            )
            bundles[tier] = bundle
            tables[tier] = _table_for_bundle(modeling_table, bundle)

        with threadpool_limits(limits=MAX_CPU_THREADS):
            predictions_by_candidate, fold_frames, selection_frames, summaries = (
                _evaluate_models(
                    plan,
                    stage,
                    validation_seasons,
                    bundles,
                    tables,
                    candidate_ids,
                )
            )
            _evaluate_blends(
                plan,
                stage,
                blend_ids,
                predictions_by_candidate,
                fold_frames,
                summaries,
            )
            baseline_table = (
                tables["stable"] if "stable" in tables else next(iter(tables.values()))
            )
            _add_baseline(
                stage,
                validation_seasons,
                baseline_table,
                predictions_by_candidate,
                fold_frames,
                summaries,
            )

        summary = pd.DataFrame(summaries).sort_values(  # (n_candidates, m_metrics)
            "spearman", ascending=False, kind="stable"
        )
        fold_metrics = pd.concat(fold_frames, ignore_index=True)  # (n_folds, m_metrics)
        selected_features = (
            pd.concat(selection_frames, ignore_index=True)
            if selection_frames
            else pd.DataFrame()
        )  # (n_selected_records, 7)
        prediction_table = pd.concat(  # (n_candidates * n_validation_rows, 7)
            predictions_by_candidate.values(),
            ignore_index=False,
        ).reset_index(names="source_row")

        current_selection = prior_selection
        if stage == "discovery":
            current_selection = _select_discovery_candidate(
                plan,
                summary,
                fold_metrics,
                plan_hash,
            )
            _write_json(safety_run.run_dir / "selection.json", current_selection)

        selected_identifier = str(current_selection["selected_candidate"])
        selected_predictions = predictions_by_candidate[selected_identifier]
        selected_fold_metrics = fold_metrics.loc[
            fold_metrics["candidate"].eq(selected_identifier)
        ].copy()  # (n_selected_folds, m_metrics)
        uncertainty = _uncertainty(selected_predictions, selected_fold_metrics)
        position_metrics = _position_metrics(selected_predictions)

        summary.to_csv(safety_run.run_dir / "candidate_summary.csv", index=False)
        fold_metrics.to_csv(safety_run.run_dir / "fold_metrics.csv", index=False)
        selected_features.to_csv(
            safety_run.run_dir / "selected_features.csv", index=False
        )
        prediction_table.to_parquet(
            safety_run.run_dir / "predictions.parquet", index=False
        )
        lineage = {
            tier: {
                column: {
                    "sources": list(record.sources),
                    "source_offsets": list(record.source_offsets),
                    "recipe": record.recipe,
                }
                for column, record in bundle.lineage.items()
            }
            for tier, bundle in bundles.items()
        }
        _write_json(safety_run.run_dir / "feature_lineage.json", lineage)
        _write_json(safety_run.run_dir / "uncertainty.json", uncertainty)
        _write_json(safety_run.run_dir / "position_metrics.json", position_metrics)

        elapsed_seconds = time.perf_counter() - started
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "stage": stage,
            "plan_name": plan["plan_name"],
            "plan_sha256": plan_hash,
            "freeze_id": safety_run.freeze_id,
            "validation_seasons": list(validation_seasons),
            "maximum_source_season": maximum_source_season,
            "input_hashes": safety_run.input_hashes,
            "protected_before": {
                key: {
                    "files": value.files,
                    "bytes": value.bytes,
                    "sha256": value.sha256,
                }
                for key, value in safety_run.protected_before.items()
            },
            "random_seed": RANDOM_SEED,
            "maximum_cpu_threads": MAX_CPU_THREADS,
            "python": platform.python_version(),
            "packages": package_versions(),
            "elapsed_seconds": elapsed_seconds,
            "selected_candidate": selected_identifier,
            "selected_metrics": next(
                row for row in summaries if row["candidate"] == selected_identifier
            ),
            "negative_control_spearman": next(
                (
                    float(row["spearman"])
                    for row in summaries
                    if "shuffle" in str(row["candidate"])
                ),
                None,
            ),
            "source_contract": (
                "performance inputs use source seasons strictly earlier than each "
                "target season; target-season metadata is limited to stable cutoff fields"
            ),
        }
        _write_json(safety_run.run_dir / "run_manifest.json", manifest)
        _write_run_ledger(
            safety_run.run_dir / "LEDGER.md",
            stage,
            plan,
            summary,
            current_selection,
            elapsed_seconds,
        )
        before_hash_manifest = hash_outputs(safety_run.run_dir, root=root)
        _write_json(
            safety_run.run_dir / "output_hashes.json",
            before_hash_manifest,
        )
        run_dir = safety_run.run_dir
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Phase 2 command-line entry point."""
    args = _parse_args(argv)
    try:
        run_dir = _run_stage(args.root, args.plan, args.stage, args.selection)
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"Phase 2 failed: {error}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
