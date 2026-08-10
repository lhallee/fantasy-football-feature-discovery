"""Run the locked August 9 Phase 2 production experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from threadpoolctl import threadpool_limits

from .metrics import clustered_spearman_interval, macro_rank_metric, metric_summary
from .models import MAX_CPU_THREADS, RANDOM_SEED
from .phase2_august import AugustCohort, build_august_cohort
from .phase2_features import RECENT_ROOM_COLUMNS, RECENT_TEAM_ENVIRONMENT_COLUMNS
from .phase2_features import RECENT_TRAJECTORY_COLUMNS, RECENT_WEEKLY_COLUMNS
from .phase2_features import REQUIRED_WEEKLY_COLUMNS, STABLE_ROOM_COLUMNS
from .phase2_features import STABLE_TEAM_ENVIRONMENT_COLUMNS
from .phase2_features import STABLE_TRAJECTORY_COLUMNS, STABLE_WEEKLY_COLUMNS
from .phase2_features import Phase2FeatureBundle, build_phase2_features
from .phase2_modeling import Phase2Candidate
from .phase2_modeling import _regression_pipeline, select_fold_features
from .phase2_modeling import summarize_predictions, walk_forward_candidate
from .phase2_safety import create_run_directory, enforce_size_limit, hash_inputs
from .phase2_safety import hash_outputs, load_freeze_manifest
from .phase2_safety import compute_protected_digests
from .provenance import file_sha256, package_versions


VALID_STAGES: Final = ("discovery", "audit", "fit", "promote")
OFFENSE_POSITIONS: Final = ("QB", "RB", "WR", "TE")
KICKER_POSITIONS: Final = ("K",)
SAFE_PROTECTED_SCOPES: Final = ("data/raw", "data/processed", "config")
TRAJECTORY_PREFIXES: Final = ("lag1_late", "lag1_weekly_slope")
OFFENSE_EXCLUDED_FRAGMENTS: Final = (
    "fg_",
    "field_goal",
    "gwfg",
    "kickoff_return",
    "pat_",
    "punt_return",
    "special_teams",
    "st_snaps",
    "team_changed",
)
KICKER_HISTORY_FRAGMENTS: Final = (
    "fg_",
    "field_goal",
    "gwfg",
    "pat_",
)
KICKER_CONTEXT_COLUMNS: Final = (
    "age",
    "years_exp",
    "height",
    "weight",
    "draft_number",
    "draft_round",
    "was_drafted",
    "is_rookie",
    "combine_forty",
    "combine_bench",
    "combine_vertical",
    "combine_broad_jump",
    "combine_cone",
    "combine_shuttle",
)
CANONICAL_PHASE2_FILES: Final = (
    "predictions_2026.parquet",
    "model_manifest.json",
    "result_summary.json",
    "audit_predictions.parquet",
    "audit_intervals.json",
    "offense_model.joblib",
    "kicker_model.joblib",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the August production experiment command."""
    parser = argparse.ArgumentParser(
        description="Run the fixed August 9 Phase 2 production protocol."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=VALID_STAGES, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--fit-run", type=Path)
    return parser.parse_args(argv)


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    """Write deterministic, standards-compliant JSON."""
    path.write_text(
        json.dumps(values, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    """Load one required JSON object."""
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {description}: {path}") from error
    if not isinstance(values, dict):
        raise ValueError(f"{description} must contain a JSON object.")
    return values


def load_august_plan(path: Path) -> dict[str, Any]:
    """Load and validate the predeclared August production plan."""
    plan = _load_json(path, "August Phase 2 plan")
    required = {
        "plan_version",
        "plan_name",
        "prefit_manifest_path",
        "forecast_origin",
        "current_season",
        "discovery_seasons",
        "audit_seasons",
        "minimum_training_season",
        "candidates",
        "selection",
        "null_controls",
        "promotion_gates",
    }
    missing = sorted(required.difference(plan))
    if missing:
        raise ValueError(f"August plan is missing fields: {missing!r}.")
    if plan["forecast_origin"] != "2026-08-09":
        raise ValueError("The production plan must use the fixed 2026-08-09 origin.")
    if int(plan["current_season"]) != 2026:
        raise ValueError("The production plan must target the 2026 season.")
    if not isinstance(plan["candidates"], list) or not plan["candidates"]:
        raise ValueError("The production plan must declare model candidates.")

    discovery = tuple(int(value) for value in plan["discovery_seasons"])
    audit = tuple(int(value) for value in plan["audit_seasons"])
    if discovery != tuple(sorted(discovery)) or audit != tuple(sorted(audit)):
        raise ValueError("Validation seasons must be sorted.")
    if not discovery or not audit or max(discovery) >= min(audit):
        raise ValueError("Discovery seasons must strictly precede audit seasons.")

    identifiers = [str(values["identifier"]) for values in plan["candidates"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("August candidate identifiers must be unique.")
    cohorts = {str(values.get("cohort")) for values in plan["candidates"]}
    if cohorts != {"offense", "kicker"}:
        raise ValueError("The candidate menu must cover offense and kicker cohorts.")
    return plan


def verify_prefit_manifest(
    root: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the plan, implementation, tests, and data lock recorded before fitting."""
    manifest_path = (root / str(plan["prefit_manifest_path"])).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise ValueError("The August pre-fit manifest escapes the repository.") from error
    manifest = _load_json(manifest_path, "August pre-fit manifest")
    if manifest.get("manifest_version") != 1:
        raise ValueError("The August pre-fit manifest must use version 1.")
    if manifest.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("The August plan changed after the pre-fit lock.")
    if manifest.get("code_hashes") != code_hashes(root):
        raise ValueError("August experiment code changed after the pre-fit lock.")
    if manifest.get("scientific_input_lock") != _verify_scientific_inputs(root):
        raise ValueError("August scientific inputs changed after the pre-fit lock.")
    return manifest


def _code_paths(root: Path) -> tuple[Path, ...]:
    """Return every first-party file that determines this experiment."""
    paths = (
        root / "fantasy_football" / "metrics.py",
        root / "fantasy_football" / "models.py",
        root / "fantasy_football" / "phase2_august.py",
        root / "fantasy_football" / "phase2_august_runner.py",
        root / "fantasy_football" / "phase2_features.py",
        root / "fantasy_football" / "phase2_modeling.py",
        root / "tests" / "test_phase2_august.py",
        root / "tests" / "test_phase2_august_runner.py",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"August protocol code is incomplete: {missing!r}.")
    return paths


def code_hashes(root: Path) -> dict[str, str]:
    """Hash the locked experiment implementation and its contract tests."""
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in _code_paths(root)
    }


def _verify_scientific_inputs(root: Path) -> dict[str, dict[str, Any]]:
    """Verify the frozen data and scoring trees while allowing document edits."""
    manifest = load_freeze_manifest(root)
    expected = manifest["trees"]
    actual = compute_protected_digests(root)
    records: dict[str, dict[str, Any]] = {}
    for scope in SAFE_PROTECTED_SCOPES:
        found = actual[scope]
        declared = expected[scope]
        if (
            found.files != int(declared["files"])
            or found.bytes != int(declared["bytes"])
            or found.sha256 != str(declared["sha256"])
        ):
            raise RuntimeError(f"Frozen scientific input changed: {scope}.")
        records[scope] = {
            "files": found.files,
            "bytes": found.bytes,
            "sha256": found.sha256,
        }
    return records


def _weekly_feature_columns() -> tuple[str, ...]:
    """Return the union of raw columns used by the Phase 2 feature builder."""
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
    """Select only weekly-stat files available before the last target season."""
    paths: list[Path] = []
    for path in sorted((root / "data" / "raw" / "stats").glob("*.parquet")):
        try:
            season = int(path.stem.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            continue
        if season <= maximum_source_season:
            paths.append(path)
    if not paths:
        raise FileNotFoundError("No weekly-stat inputs satisfy the August cutoff.")
    return paths


def _load_weekly_stats(paths: Sequence[Path]) -> pd.DataFrame:
    """Read the declared weekly columns across the selected source seasons."""
    declared = _weekly_feature_columns()
    frames: list[pd.DataFrame] = []  # each (n_file, d_declared)
    for path in paths:
        available = set(pq.read_schema(path).names)
        columns = [column for column in declared if column in available]
        frame = pd.read_parquet(path, columns=columns)  # (n_file, d_available)
        frames.append(frame.reindex(columns=declared))  # (n_file, d_declared)
    return pd.concat(frames, ignore_index=True)  # (n_weekly, d_declared)


def _all_input_paths(root: Path, plan_path: Path, maximum_source_season: int) -> list[Path]:
    """Enumerate every data, plan, and code input consumed by a run."""
    raw_dir = root / "data" / "raw"
    roster_inputs = []
    for path in sorted((raw_dir / "rosters").glob("roster_weekly_*.parquet")):
        try:
            season = int(path.stem.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            continue
        if season <= maximum_source_season:
            roster_inputs.append(path)
    weekly_inputs = _weekly_input_paths(root, maximum_source_season)
    inputs = [
        plan_path,
        *_code_paths(root),
        root / "data" / "processed" / "player_seasons.parquet",
        root / "data" / "processed" / "build_summary.json",
        raw_dir / "players.parquet",
        raw_dir / "combine.parquet",
        raw_dir / "draft_picks.parquet",
        *roster_inputs,
        *weekly_inputs,
    ]
    if maximum_source_season >= 2025:
        inputs.append(root / "data" / "processed" / "preseason_players.parquet")
    return inputs


def _feature_group(column: str) -> str:
    """Map a generated column to its predeclared recipe family."""
    if column.startswith("room_"):
        return "room"
    if column.startswith(("team_", "prior_team_")):
        return "team"
    if column.startswith(TRAJECTORY_PREFIXES):
        return "trajectory"
    return "core"


def _is_kicker_column(column: str) -> bool:
    """Return whether a feature is valid for the kicker-only model."""
    if column in KICKER_CONTEXT_COLUMNS or column.removeprefix(
        "missing_"
    ) in KICKER_CONTEXT_COLUMNS:
        return True
    if column.startswith("lag") and any(
        fragment in column for fragment in KICKER_HISTORY_FRAGMENTS
    ):
        return True
    return column.startswith("lag") and column.endswith(
        ("history_available", "games", "roster_weeks")
    )


def safe_feature_frame(
    bundle: Phase2FeatureBundle,
    candidate: Phase2Candidate,
    cohort: str,
) -> pd.DataFrame:
    """Return the candidate pool after the August-origin source allowlist."""
    # bundle.frame: (n_targets, d_all)
    requested = set(candidate.recipes)
    if not requested or not requested.issubset({"core", "trajectory"}):
        raise ValueError("August candidates may use only core and trajectory recipes.")
    columns = [
        column
        for column in bundle.frame.columns
        if _feature_group(column) in requested
    ]
    if cohort == "offense":
        columns = [
            column
            for column in columns
            if not any(fragment in column for fragment in OFFENSE_EXCLUDED_FRAGMENTS)
        ]
    elif cohort == "kicker":
        columns = [column for column in columns if _is_kicker_column(column)]
    else:
        raise ValueError(f"Unknown August cohort: {cohort!r}.")
    if not columns:
        raise ValueError(f"Candidate {candidate.identifier!r} has no safe features.")

    for column in columns:
        lineage = bundle.lineage[column]
        if any(source == "modeling_table.target_roster" for source in lineage.sources):
            raise RuntimeError(f"Target-roster feature passed the allowlist: {column}.")
        for source, offset in zip(
            lineage.sources,
            lineage.source_offsets,
            strict=True,
        ):
            if source in {"weekly_stats", "player_seasons"} and offset < 1:
                raise RuntimeError(f"Unlagged outcome source passed the allowlist: {column}.")
    return bundle.frame.loc[:, columns]  # (n_targets, d_safe)


def _candidate_records(plan: Mapping[str, Any], cohort: str) -> list[dict[str, Any]]:
    """Return the declared candidates for one production cohort."""
    return [
        dict(values)
        for values in plan["candidates"]
        if str(values["cohort"]) == cohort
    ]


def _candidate(values: Mapping[str, Any]) -> Phase2Candidate:
    """Build the shared model record after removing runner-only fields."""
    model_values = dict(values)
    model_values.pop("cohort", None)
    return Phase2Candidate.from_mapping(model_values)


def _table_for_bundle(
    cohort_table: pd.DataFrame,
    bundle: Phase2FeatureBundle,
) -> pd.DataFrame:
    """Align targets and audit metadata to one feature bundle."""
    columns = [
        "target_season",
        "player_id",
        "model_position",
        "team",
        "roster_status",
        "available_for_draft",
        "target_points",
        "previous_points_baseline",
        "is_rookie",
    ]
    source = cohort_table.loc[:, columns].copy()  # (n_targets, 9)
    table = bundle.keys.merge(
        source,
        on=["target_season", "player_id", "model_position"],
        how="left",
        validate="one_to_one",
        sort=False,
    )  # (n_bundle, 9)
    table.index = bundle.frame.index
    if table["target_points"].isna().all():
        raise ValueError("The aligned August table has no observed targets.")
    return table


def _baseline_predictions(
    table: pd.DataFrame,
    validation_seasons: tuple[int, ...],
    cohort: str,
) -> pd.DataFrame:
    """Return lag-one fantasy points as the declared benchmark."""
    # table: (n_targets, c_table)
    mask = table["target_season"].isin(validation_seasons) & table[
        "target_points"
    ].notna()  # (n_targets,)
    predictions = table.loc[
        mask,
        ["player_id", "target_season", "model_position", "target_points"],
    ].copy()  # (n_validation, 4)
    predictions["candidate"] = f"{cohort}_previous_points_baseline"  # (n_validation,)
    predictions["predicted"] = table.loc[
        mask, "previous_points_baseline"
    ].fillna(0.0)  # (n_validation,)
    predictions["fold_fit_seconds"] = 0.0  # (n_validation,)
    return predictions  # (n_validation, 7)


def _fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize one prediction procedure by validation season."""
    # predictions: (n_validation, 7)
    records: list[dict[str, Any]] = []
    for season, fold in predictions.groupby("target_season", observed=True):
        # fold: (n_fold, 7)
        records.append(
            {
                "candidate": str(fold["candidate"].iloc[0]),
                "validation_season": int(season),
                "rows": len(fold),
                "fit_seconds": float(fold["fold_fit_seconds"].iloc[0]),
                **metric_summary(
                    fold["target_points"],
                    fold["predicted"],
                    fold["model_position"],
                    fold["target_season"],
                ),
            }
        )
    return pd.DataFrame(records)  # (n_folds, m_metrics)


def _summary_record(
    identifier: str,
    cohort: str,
    kind: str,
    predictions: pd.DataFrame,
    feature_pool_count: int,
    selected_feature_count: float,
) -> dict[str, Any]:
    """Return one discovery or audit summary row."""
    return {
        "candidate": identifier,
        "cohort": cohort,
        "kind": kind,
        "rows": len(predictions),
        "feature_pool_count": feature_pool_count,
        "median_selected_feature_count": selected_feature_count,
        **summarize_predictions(predictions),
    }


def _evaluate_cohort(
    plan: Mapping[str, Any],
    cohort: str,
    validation_seasons: tuple[int, ...],
    bundles: Mapping[str, Phase2FeatureBundle],
    tables: Mapping[str, pd.DataFrame],
    selected_identifier: str | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate the declared menu or one locked candidate chronologically."""
    predictions_by_candidate: dict[str, pd.DataFrame] = {}
    fold_frames: list[pd.DataFrame] = []  # each (n_folds, m_metrics)
    feature_frames: list[pd.DataFrame] = []  # each (n_selected, 7)
    summaries: list[dict[str, Any]] = []
    for values in _candidate_records(plan, cohort):
        candidate = _candidate(values)
        if selected_identifier is not None and candidate.identifier != selected_identifier:
            continue
        bundle = bundles[candidate.tier]
        table = tables[candidate.tier]
        X = safe_feature_frame(bundle, candidate, cohort).loc[
            table.index
        ]  # (n_cohort_targets, d_pool)
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
        predictions_by_candidate[candidate.identifier] = (
            output.predictions
        )  # (n_validation, 7)
        fold_frames.append(output.fold_metrics)
        feature_frames.append(output.selected_features)
        selected_counts = output.selected_features.groupby(
            ["validation_season", "position_scope"], observed=True
        )["selected_feature_count"].first()  # (n_fold_scopes,)
        summaries.append(
            _summary_record(
                candidate.identifier,
                cohort,
                "model",
                output.predictions,
                len(X.columns),
                float(selected_counts.median()),
            )
        )

    baseline_tier = "stable" if "stable" in tables else next(iter(tables))
    baseline = _baseline_predictions(
        tables[baseline_tier], validation_seasons, cohort
    )  # (n_validation, 7)
    baseline_identifier = str(baseline["candidate"].iloc[0])
    predictions_by_candidate[baseline_identifier] = baseline  # (n_validation, 7)
    baseline_folds = _fold_metrics(baseline)  # (n_folds, m_metrics)
    fold_frames.append(baseline_folds)
    summaries.append(
        _summary_record(
            baseline_identifier,
            cohort,
            "benchmark",
            baseline,
            1,
            1.0,
        )
    )
    summary = pd.DataFrame(summaries)  # (n_candidates + 1, m_metrics)
    fold_metrics = pd.concat(fold_frames, ignore_index=True)  # (n_folds, m_metrics)
    selected_features = (
        pd.concat(feature_frames, ignore_index=True)
        if feature_frames
        else pd.DataFrame()
    )  # (n_selected_records, 7)
    return predictions_by_candidate, summary, fold_metrics, selected_features


def _select_candidate(
    plan: Mapping[str, Any],
    cohort: str,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    """Apply the locked near-best, then compactness, selection rule."""
    # summary: (n_candidates, m_metrics)
    selection = plan["selection"]
    eligible = summary["kind"].eq("model")  # (n_candidates,)
    if bool(selection["baseline_eligible"][cohort]):
        eligible |= summary["kind"].eq("benchmark")  # (n_candidates,)
    candidates = summary.loc[eligible].copy()  # (n_eligible, m_metrics)
    best_rho = float(candidates["spearman"].max())
    tolerance = float(selection["spearman_tolerance"])
    near_best = candidates.loc[
        candidates["spearman"].ge(best_rho - tolerance)
    ].copy()  # (n_near_best, m_metrics)
    near_best.sort_values(
        ["median_selected_feature_count", "mae", "candidate"],
        inplace=True,
        kind="stable",
    )
    selected = near_best.iloc[0]  # (m_metrics,)
    return {
        "identifier": str(selected["candidate"]),
        "best_discovery_spearman": best_rho,
        "selected_discovery_spearman": float(selected["spearman"]),
        "selected_feature_count": int(selected["median_selected_feature_count"]),
        "feature_pool_count": int(selected["feature_pool_count"]),
        "selection_rule": (
            f"within {tolerance:.3f} mean within-position-season Spearman of "
            "the best eligible procedure, then fewer columns, lower MAE, and ID"
        ),
    }


def _needed_tiers(
    plan: Mapping[str, Any],
    identifiers: set[str] | None,
) -> set[str]:
    """Return feature tiers needed for this stage."""
    return {
        str(values["tier"])
        for values in plan["candidates"]
        if identifiers is None or str(values["identifier"]) in identifiers
    }


def _build_inputs(
    root: Path,
    plan: Mapping[str, Any],
    maximum_target_season: int,
    needed_tiers: set[str],
) -> tuple[
    AugustCohort,
    dict[str, Phase2FeatureBundle],
    dict[str, pd.DataFrame],
]:
    """Build the target-independent cohort and requested feature tiers."""
    current_season = int(plan["current_season"])
    cohort = build_august_cohort(
        root,
        current_season=current_season,
        minimum_target_season=int(plan["minimum_candidate_season"]),
        maximum_target_season=maximum_target_season,
        roster_lookback=1,
    )
    modeling_table = cohort.table.copy()  # (n_targets, c_modeling)
    maximum_source_season = maximum_target_season - 1
    weekly_paths = _weekly_input_paths(root, maximum_source_season)
    weekly_stats = _load_weekly_stats(weekly_paths)  # (n_weekly, d_weekly)
    player_seasons = pd.read_parquet(
        root / "data" / "processed" / "player_seasons.parquet"
    )  # (n_player_seasons, c_player_seasons)
    player_seasons = player_seasons.loc[
        player_seasons["season"].le(maximum_source_season)
    ].copy()  # (n_source_seasons, c_player_seasons)

    bundles: dict[str, Phase2FeatureBundle] = {}  # each frame (n_tier, d_tier)
    tables: dict[str, pd.DataFrame] = {}  # each (n_tier, c_table)
    for tier in sorted(needed_tiers):
        bundle = build_phase2_features(
            weekly_stats,
            player_seasons,
            modeling_table,
            tier=tier,
            max_lag=int(plan["max_lag"]),
            late_weeks=int(plan["late_weeks"]),
        )
        bundles[tier] = bundle  # frame (n_tier, d_tier)
        tables[tier] = _table_for_bundle(modeling_table, bundle)  # (n_tier, c_table)
    return cohort, bundles, tables


def _lineage_payload(
    bundles: Mapping[str, Phase2FeatureBundle],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Serialize every feature recipe and temporal source offset."""
    return {
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


def _stage_input_hashes(
    root: Path,
    paths: Sequence[Path],
) -> dict[str, str]:
    """Hash a deduplicated, repository-relative input set."""
    unique = tuple(dict.fromkeys(path.resolve() for path in paths))
    return hash_inputs(root, unique)


def _run_discovery(root: Path, plan_path: Path, plan: Mapping[str, Any]) -> Path:
    """Evaluate and select procedures without opening the audit years."""
    validation_seasons = tuple(int(value) for value in plan["discovery_seasons"])
    maximum_target = max(validation_seasons)
    input_paths = _all_input_paths(root, plan_path, maximum_target - 1)
    before_hashes = _stage_input_hashes(root, input_paths)
    scientific_lock = _verify_scientific_inputs(root)
    run_dir = create_run_directory(root, f"august9-discovery-{file_sha256(plan_path)[:12]}")
    started = time.perf_counter()

    cohort, bundles, tables = _build_inputs(
        root,
        plan,
        maximum_target,
        _needed_tiers(plan, None),
    )
    summaries: list[pd.DataFrame] = []  # each (n_candidates, m_metrics)
    folds: list[pd.DataFrame] = []  # each (n_folds, m_metrics)
    features: list[pd.DataFrame] = []  # each (n_selected, 7)
    prediction_frames: list[pd.DataFrame] = []  # each (n_validation, 7)
    selections: dict[str, Any] = {}
    with threadpool_limits(limits=MAX_CPU_THREADS):
        for cohort_name, positions in (
            ("offense", OFFENSE_POSITIONS),
            ("kicker", KICKER_POSITIONS),
        ):
            cohort_tables = {
                tier: table.loc[table["model_position"].isin(positions)].copy()
                for tier, table in tables.items()
            }  # each (n_cohort_targets, c_table)
            output = _evaluate_cohort(
                plan,
                cohort_name,
                validation_seasons,
                bundles,
                cohort_tables,
            )
            predictions_by_candidate, summary, fold_metrics, selected_features = output
            summaries.append(summary)
            folds.append(fold_metrics)
            features.append(selected_features)
            prediction_frames.extend(predictions_by_candidate.values())
            selections[cohort_name] = _select_candidate(plan, cohort_name, summary)

    summary = pd.concat(summaries, ignore_index=True)  # (n_candidates, m_metrics)
    fold_metrics = pd.concat(folds, ignore_index=True)  # (n_folds, m_metrics)
    selected_features = pd.concat(features, ignore_index=True)  # (n_selected, 7)
    prediction_table = pd.concat(prediction_frames, ignore_index=True)  # (n_predictions, 7)
    elapsed = time.perf_counter() - started
    selection = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": file_sha256(plan_path),
        "code_hashes": code_hashes(root),
        "forecast_origin": plan["forecast_origin"],
        "discovery_seasons": list(validation_seasons),
        "cohorts": selections,
    }
    summary.to_csv(run_dir / "candidate_summary.csv", index=False)
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    selected_features.to_csv(run_dir / "selected_features.csv", index=False)
    prediction_table.to_parquet(run_dir / "predictions.parquet", index=False)
    cohort.audit.to_csv(run_dir / "cohort_audit.csv", index=False)
    _write_json(run_dir / "feature_lineage.json", _lineage_payload(bundles))
    _write_json(run_dir / "selection.json", selection)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "stage": "discovery",
            "plan_sha256": file_sha256(plan_path),
            "code_hashes": code_hashes(root),
            "scientific_input_lock": scientific_lock,
            "input_hashes": before_hashes,
            "validation_seasons": list(validation_seasons),
            "elapsed_seconds": elapsed,
            "maximum_cpu_threads": MAX_CPU_THREADS,
            "random_seed": RANDOM_SEED,
            "packages": package_versions(),
            "selections": selections,
        },
    )
    _write_json(run_dir / "output_hashes.json", hash_outputs(run_dir, root=root))
    if _stage_input_hashes(root, input_paths) != before_hashes:
        raise RuntimeError("A discovery input changed during fitting.")
    _verify_scientific_inputs(root)
    enforce_size_limit(root)
    return run_dir


def _load_selection(
    root: Path,
    plan_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    """Load a discovery decision and bind it to the unchanged implementation."""
    selection = _load_json(selection_path, "August discovery selection")
    if selection.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("The discovery selection does not match the August plan.")
    if selection.get("code_hashes") != code_hashes(root):
        raise ValueError("Experiment code changed after discovery selection.")
    cohorts = selection.get("cohorts")
    if not isinstance(cohorts, dict) or set(cohorts) != {"offense", "kicker"}:
        raise ValueError("The discovery selection must define both cohorts.")
    return selection


def _selected_identifiers(selection: Mapping[str, Any]) -> dict[str, str]:
    """Return the locked procedure name for each cohort."""
    return {
        cohort: str(selection["cohorts"][cohort]["identifier"])
        for cohort in ("offense", "kicker")
    }


def _permuted_target_table(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Break player-label association within each season-position group."""
    # table: (n_targets, c_table)
    shuffled = table.copy()  # (n_targets, c_table)
    generator = np.random.default_rng(seed)
    for _, group in table.groupby(
        ["target_season", "model_position"], observed=True
    ):
        # group: (n_group, c_table)
        indices = group.index.to_numpy()  # (n_group,)
        values = group["target_points"].to_numpy(dtype="float64")  # (n_group,)
        shuffled.loc[indices, "target_points"] = generator.permutation(values)
    return shuffled  # (n_targets, c_table)


def _fixed_prediction_null(
    predictions: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Score fixed forecasts against independently permuted validation labels."""
    # predictions: (n_validation, 7)
    values = np.empty(repeats, dtype="float64")  # (n_repeats,)
    for repeat in range(repeats):
        shuffled = _permuted_target_table(
            predictions,
            seed + repeat,
        )  # (n_validation, 7)
        values[repeat] = macro_rank_metric(
            shuffled["target_points"],
            predictions["predicted"],
            predictions["model_position"],
            predictions["target_season"],
        )
    return {
        "repeats": repeats,
        "mean_spearman": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _end_to_end_null(
    plan: Mapping[str, Any],
    cohort: str,
    identifier: str,
    validation_seasons: tuple[int, ...],
    bundles: Mapping[str, Phase2FeatureBundle],
    tables: Mapping[str, pd.DataFrame],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Repeat the locked fit after permuting all training and validation labels."""
    values = np.empty(repeats, dtype="float64")  # (n_repeats,)
    for repeat in range(repeats):
        permuted_tables = {
            tier: _permuted_target_table(table, seed + repeat)
            for tier, table in tables.items()
        }
        predictions_by_candidate, _, _, _ = _evaluate_cohort(
            plan,
            cohort,
            validation_seasons,
            bundles,
            permuted_tables,
            selected_identifier=identifier,
        )
        selected = predictions_by_candidate[identifier]  # (n_validation, 7)
        values[repeat] = summarize_predictions(selected)["spearman"]
    return {
        "repeats": repeats,
        "mean_spearman": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "spearman": values.tolist(),
    }


def _position_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Report every metric separately for each fantasy position."""
    results: dict[str, dict[str, float]] = {}
    for position, group in predictions.groupby("model_position", observed=True):
        # group: (n_position, 7)
        results[str(position)] = metric_summary(
            group["target_points"],
            group["predicted"],
            group["model_position"],
            group["target_season"],
        )
    return results


def _audit_gate_report(
    plan: Mapping[str, Any],
    selected_predictions: Mapping[str, pd.DataFrame],
    baseline_predictions: Mapping[str, pd.DataFrame],
    fixed_nulls: Mapping[str, Mapping[str, Any]],
    end_to_end_nulls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply every promotion threshold without tuning on the audit result."""
    gates = plan["promotion_gates"]
    offense_metrics = summarize_predictions(selected_predictions["offense"])
    offense_baseline = summarize_predictions(baseline_predictions["offense"])
    kicker_metrics = summarize_predictions(selected_predictions["kicker"])
    kicker_baseline = summarize_predictions(baseline_predictions["kicker"])
    position_metrics = _position_metrics(selected_predictions["offense"])

    checks = {
        "offense_minimum_spearman": offense_metrics["spearman"]
        >= float(gates["offense_minimum_spearman"]),
        "offense_minimum_gain_over_baseline": (
            offense_metrics["spearman"] - offense_baseline["spearman"]
        )
        >= float(gates["offense_minimum_gain_over_baseline"]),
        "offense_each_position_minimum_spearman": all(
            metrics["spearman"]
            >= float(gates["offense_each_position_minimum_spearman"])
            for metrics in position_metrics.values()
        ),
        "kicker_not_materially_below_baseline": kicker_metrics["spearman"]
        >= kicker_baseline["spearman"]
        - float(gates["kicker_allowed_loss_vs_baseline"]),
        "fixed_prediction_null_centered": all(
            abs(float(values["mean_spearman"]))
            <= float(gates["fixed_null_maximum_absolute_mean"])
            and float(values["lower_95"]) <= 0.0 <= float(values["upper_95"])
            for values in fixed_nulls.values()
        ),
        "end_to_end_null_centered": all(
            abs(float(values["mean_spearman"]))
            <= float(gates["end_to_end_null_maximum_absolute_mean"][cohort])
            for cohort, values in end_to_end_nulls.items()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "offense": {
            "selected": offense_metrics,
            "baseline": offense_baseline,
            "gain_over_baseline": offense_metrics["spearman"]
            - offense_baseline["spearman"],
            "by_position": position_metrics,
        },
        "kicker": {
            "selected": kicker_metrics,
            "baseline": kicker_baseline,
            "gain_over_baseline": kicker_metrics["spearman"]
            - kicker_baseline["spearman"],
        },
        "fixed_prediction_null": dict(fixed_nulls),
        "end_to_end_null": dict(end_to_end_nulls),
    }


def _run_audit(
    root: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    selection_path: Path,
) -> Path:
    """Evaluate the locked discovery decisions once on 2022 through 2025."""
    selection = _load_selection(root, plan_path, selection_path)
    identifiers = _selected_identifiers(selection)
    validation_seasons = tuple(int(value) for value in plan["audit_seasons"])
    maximum_target = max(validation_seasons)
    input_paths = [
        *_all_input_paths(root, plan_path, maximum_target - 1),
        selection_path,
    ]
    before_hashes = _stage_input_hashes(root, input_paths)
    scientific_lock = _verify_scientific_inputs(root)
    run_dir = create_run_directory(root, f"august9-audit-{file_sha256(plan_path)[:12]}")
    started = time.perf_counter()

    needed_ids = {
        identifier
        for identifier in identifiers.values()
        if not identifier.endswith("_previous_points_baseline")
    }
    needed_tiers = _needed_tiers(plan, needed_ids)
    needed_tiers.add("stable")
    cohort, bundles, tables = _build_inputs(
        root,
        plan,
        maximum_target,
        needed_tiers,
    )
    selected_predictions: dict[str, pd.DataFrame] = {}  # each (n_validation, 7)
    baseline_predictions: dict[str, pd.DataFrame] = {}  # each (n_validation, 7)
    summary_frames: list[pd.DataFrame] = []  # each (n_procedures, m_metrics)
    fold_frames: list[pd.DataFrame] = []  # each (n_folds, m_metrics)
    feature_frames: list[pd.DataFrame] = []  # each (n_selected, 7)
    output_predictions: list[pd.DataFrame] = []  # each (n_validation, 7)
    cohort_tables_by_name: dict[str, dict[str, pd.DataFrame]] = (
        {}
    )  # each table (n_cohort_targets, c_table)

    with threadpool_limits(limits=MAX_CPU_THREADS):
        for cohort_name, positions in (
            ("offense", OFFENSE_POSITIONS),
            ("kicker", KICKER_POSITIONS),
        ):
            cohort_tables = {
                tier: table.loc[table["model_position"].isin(positions)].copy()
                for tier, table in tables.items()
            }  # each (n_cohort_targets, c_table)
            cohort_tables_by_name[cohort_name] = cohort_tables  # same shapes
            output = _evaluate_cohort(
                plan,
                cohort_name,
                validation_seasons,
                bundles,
                cohort_tables,
                selected_identifier=identifiers[cohort_name],
            )
            predictions_by_candidate, summary, fold_metrics, selected_features = output
            selected = predictions_by_candidate[
                identifiers[cohort_name]
            ]  # (n_validation, 7)
            baseline_id = f"{cohort_name}_previous_points_baseline"
            selected_predictions[cohort_name] = selected  # (n_validation, 7)
            baseline_predictions[cohort_name] = predictions_by_candidate[
                baseline_id
            ]  # (n_validation, 7)
            summary_frames.append(summary)
            fold_frames.append(fold_metrics)
            feature_frames.append(selected_features)
            output_predictions.extend((selected, predictions_by_candidate[baseline_id]))

        null_plan = plan["null_controls"]
        fixed_nulls = {
            cohort_name: _fixed_prediction_null(
                predictions,
                int(null_plan["fixed_prediction_repeats"]),
                int(null_plan["seed"]) + 10_000 * index,
            )
            for index, (cohort_name, predictions) in enumerate(
                selected_predictions.items()
            )
        }
        end_to_end_nulls = {
            cohort_name: _end_to_end_null(
                plan,
                cohort_name,
                identifiers[cohort_name],
                validation_seasons,
                bundles,
                cohort_tables_by_name[cohort_name],
                int(null_plan["end_to_end_repeats"]),
                int(null_plan["seed"]) + 100_000 * index,
            )
            for index, cohort_name in enumerate(("offense", "kicker"))
        }

    gate_report = _audit_gate_report(
        plan,
        selected_predictions,
        baseline_predictions,
        fixed_nulls,
        end_to_end_nulls,
    )
    elapsed = time.perf_counter() - started
    summary = pd.concat(summary_frames, ignore_index=True)  # (n_procedures, m_metrics)
    fold_metrics = pd.concat(fold_frames, ignore_index=True)  # (n_folds, m_metrics)
    selected_features = pd.concat(
        feature_frames, ignore_index=True
    )  # (n_selected, 7)
    prediction_table = pd.concat(
        output_predictions, ignore_index=True
    )  # (2 * n_validation, 7)
    summary.to_csv(run_dir / "candidate_summary.csv", index=False)
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    selected_features.to_csv(run_dir / "selected_features.csv", index=False)
    prediction_table.to_parquet(run_dir / "predictions.parquet", index=False)
    pd.concat(
        selected_predictions.values(), ignore_index=True
    ).to_parquet(run_dir / "selected_audit_predictions.parquet", index=False)
    cohort.audit.to_csv(run_dir / "cohort_audit.csv", index=False)
    _write_json(run_dir / "feature_lineage.json", _lineage_payload(bundles))
    _write_json(run_dir / "audit_report.json", gate_report)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "stage": "audit",
            "plan_sha256": file_sha256(plan_path),
            "selection_sha256": file_sha256(selection_path),
            "code_hashes": code_hashes(root),
            "scientific_input_lock": scientific_lock,
            "input_hashes": before_hashes,
            "validation_seasons": list(validation_seasons),
            "retrospective_outcomes_blinded": False,
            "promotion_gates_passed": bool(gate_report["passed"]),
            "elapsed_seconds": elapsed,
            "maximum_cpu_threads": MAX_CPU_THREADS,
            "random_seed": RANDOM_SEED,
            "packages": package_versions(),
        },
    )
    _write_json(run_dir / "output_hashes.json", hash_outputs(run_dir, root=root))
    if _stage_input_hashes(root, input_paths) != before_hashes:
        raise RuntimeError("An audit input changed during fitting.")
    _verify_scientific_inputs(root)
    enforce_size_limit(root)
    return run_dir


def _candidate_by_identifier(
    plan: Mapping[str, Any],
    identifier: str,
) -> Phase2Candidate | None:
    """Return the declared model, or None for a selected baseline."""
    for values in plan["candidates"]:
        if str(values["identifier"]) == identifier:
            candidate = _candidate(values)
            if candidate.position_specific:
                raise ValueError("The August final fit does not support position-specific models.")
            return candidate
    if identifier.endswith("_previous_points_baseline"):
        return None
    raise ValueError(f"Selected identifier is absent from the plan: {identifier!r}.")


def _fit_final_cohort(
    plan: Mapping[str, Any],
    cohort_name: str,
    positions: tuple[str, ...],
    identifier: str,
    bundles: Mapping[str, Phase2FeatureBundle],
    tables: Mapping[str, pd.DataFrame],
    current_season: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """Fit one locked procedure on all pre-2026 candidate seasons."""
    candidate = _candidate_by_identifier(plan, identifier)
    if candidate is None:
        table = tables["stable"].loc[
            tables["stable"]["model_position"].isin(positions)
        ].copy()  # (n_cohort, c_table)
        current_mask = table["target_season"].eq(current_season)  # (n_cohort,)
        current = table.loc[current_mask].copy()  # (n_current, c_table)
        current["predicted_points"] = current[
            "previous_points_baseline"
        ].fillna(0.0)  # (n_current,)
        current["feature_value__previous_points_baseline"] = current[
            "previous_points_baseline"
        ].fillna(0.0)  # (n_current,)
        payload = {
            "kind": "baseline",
            "estimator": None,
            "feature_columns": ["previous_points_baseline"],
            "candidate": identifier,
        }
        return payload, current, ["previous_points_baseline"]

    bundle = bundles[candidate.tier]
    table = tables[candidate.tier].loc[
        tables[candidate.tier]["model_position"].isin(positions)
    ].copy()  # (n_cohort, c_table)
    X = safe_feature_frame(bundle, candidate, cohort_name).loc[table.index]  # (n_cohort, d_pool)
    minimum_training = max(
        int(plan["minimum_training_season"][candidate.tier]),
        int(table["target_season"].min()),
    )
    train_mask = (
        table["target_season"].ge(minimum_training)
        & table["target_season"].lt(current_season)
        & table["target_points"].notna()
    )  # (n_cohort,)
    current_mask = table["target_season"].eq(current_season)  # (n_cohort,)
    X_train = X.loc[train_mask]  # (n_train, d_pool)
    X_current = X.loc[current_mask]  # (n_current, d_pool)
    train_table = table.loc[train_mask]  # (n_train, c_table)
    columns, scores = select_fold_features(
        X_train,
        train_table["target_points"],
        train_table["target_season"],
        train_table["model_position"],
        candidate.feature_limit,
    )
    X_train_selected = X_train.loc[:, columns]  # (n_train, d_selected)
    X_current_selected = X_current.loc[:, columns]  # (n_current, d_selected)
    estimator = _regression_pipeline(candidate)
    estimator.fit(X_train_selected, train_table["target_points"])
    predicted = estimator.predict(X_current_selected)  # (n_current,)

    current = table.loc[current_mask].copy()  # (n_current, c_table)
    current["predicted_points"] = np.clip(predicted, 0.0, None)  # (n_current,)
    feature_values = X_current_selected.rename(
        columns=lambda column: f"feature_value__{column}"
    )  # (n_current, d_selected)
    current = pd.concat([current, feature_values], axis=1)  # (n_current, c_output)
    training_min = X_train_selected.min(axis=0)  # (d_selected,)
    training_max = X_train_selected.max(axis=0)  # (d_selected,)
    below = X_current_selected.lt(training_min, axis="columns")  # (n_current, d_selected)
    above = X_current_selected.gt(training_max, axis="columns")  # (n_current, d_selected)
    current["out_of_range_feature_count"] = (below | above).sum(
        axis=1
    )  # (n_current,)
    payload = {
        "kind": "model",
        "estimator": estimator,
        "feature_columns": columns,
        "feature_selection_scores": {
            column: float(scores[column]) for column in columns
        },
        "candidate": identifier,
        "tier": candidate.tier,
        "recipes": list(candidate.recipes),
        "parameters": dict(candidate.parameters),
        "training_rows": len(X_train_selected),
        "training_seasons": [
            int(train_table["target_season"].min()),
            int(train_table["target_season"].max()),
        ],
    }
    return payload, current, columns


def _add_intervals_and_ranks(
    current: pd.DataFrame,
    audit_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach audit-residual intervals and ranks among August-available players."""
    # current: (n_current, c_prediction); audit_predictions: (n_audit, 7)
    audit = audit_predictions.copy()  # (n_audit, 7)
    audit["absolute_residual"] = (
        audit["target_points"] - audit["predicted"]
    ).abs()  # (n_audit,)
    widths = audit.groupby("model_position", observed=True)[
        "absolute_residual"
    ].quantile(0.80)  # (n_positions,)
    global_width = float(audit["absolute_residual"].quantile(0.80))
    current_width = current["model_position"].map(widths).fillna(global_width)  # (n_current,)
    current["prediction_interval_80_low"] = (
        current["predicted_points"] - current_width
    ).clip(lower=0.0)  # (n_current,)
    current["prediction_interval_80_high"] = (
        current["predicted_points"] + current_width
    )  # (n_current,)

    available = current["available_for_draft"].astype("bool")  # (n_current,)
    current["position_rank"] = pd.Series(
        pd.NA, index=current.index, dtype="Int32"
    )  # (n_current,)
    current.loc[available, "position_rank"] = (
        current.loc[available]
        .groupby("model_position", observed=True)["predicted_points"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    current["overall_point_rank"] = pd.Series(
        pd.NA, index=current.index, dtype="Int32"
    )  # (n_current,)
    current.loc[available, "overall_point_rank"] = (
        current.loc[available, "predicted_points"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    current["team_position_rank"] = pd.Series(
        pd.NA, index=current.index, dtype="Int32"
    )  # (n_current,)
    current.loc[available, "team_position_rank"] = (
        current.loc[available]
        .groupby(["team", "model_position"], observed=True)["predicted_points"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    return current


def _canonical_audit_table(
    selected_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    scoring_profile: str,
) -> pd.DataFrame:
    """Return the canonical Phase 2 retrospective prediction table."""
    keys = ["player_id", "target_season", "model_position"]
    baseline = baseline_predictions.loc[:, [*keys, "predicted"]].rename(
        columns={"predicted": "previous_points_baseline"}
    )  # (n_audit, 4)
    audit = selected_predictions.merge(
        baseline,
        on=keys,
        how="left",
        validate="one_to_one",
    )  # (n_audit, 8)
    audit.rename(
        columns={
            "target_points": "actual_points",
            "predicted": "predicted_points",
        },
        inplace=True,
    )
    audit["scoring_profile"] = scoring_profile  # (n_audit,)
    return audit


def _validate_audit_run(
    root: Path,
    plan_path: Path,
    selection_path: Path,
    audit_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the locked audit decision and all run hashes before final fitting."""
    audit_dir = audit_dir.resolve(strict=True)
    report = _load_json(audit_dir / "audit_report.json", "August audit report")
    manifest = _load_json(audit_dir / "run_manifest.json", "August audit manifest")
    if manifest.get("stage") != "audit" or not report.get("passed"):
        raise RuntimeError("The August audit did not pass every promotion gate.")
    if manifest.get("plan_sha256") != file_sha256(plan_path):
        raise RuntimeError("The audit plan hash does not match the final-fit plan.")
    if manifest.get("selection_sha256") != file_sha256(selection_path):
        raise RuntimeError("The audit selection hash does not match final fitting.")
    if manifest.get("code_hashes") != code_hashes(root):
        raise RuntimeError("Experiment code changed after the locked audit.")
    declared = _load_json(audit_dir / "output_hashes.json", "audit output hashes")
    actual = hash_outputs(audit_dir, root=root)
    actual.pop("output_hashes.json", None)
    if actual != declared:
        raise RuntimeError("An audit output changed after the run completed.")
    return report, manifest


def _run_fit(
    root: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    selection_path: Path,
    audit_dir: Path,
) -> Path:
    """Fit the gated Phase 2 procedures and write isolated 2026 artifacts."""
    selection = _load_selection(root, plan_path, selection_path)
    audit_report, audit_manifest = _validate_audit_run(
        root, plan_path, selection_path, audit_dir
    )
    identifiers = _selected_identifiers(selection)
    current_season = int(plan["current_season"])
    input_paths = [
        *_all_input_paths(root, plan_path, current_season - 1),
        selection_path,
        audit_dir / "audit_report.json",
        audit_dir / "run_manifest.json",
        audit_dir / "output_hashes.json",
        audit_dir / "selected_audit_predictions.parquet",
        audit_dir / "predictions.parquet",
    ]
    before_hashes = _stage_input_hashes(root, input_paths)
    scientific_lock = _verify_scientific_inputs(root)
    run_dir = create_run_directory(root, f"august9-fit-{file_sha256(plan_path)[:12]}")
    started = time.perf_counter()

    needed_ids = {
        identifier
        for identifier in identifiers.values()
        if not identifier.endswith("_previous_points_baseline")
    }
    needed_tiers = _needed_tiers(plan, needed_ids)
    needed_tiers.add("stable")
    cohort, bundles, tables = _build_inputs(
        root,
        plan,
        current_season,
        needed_tiers,
    )
    audit_selected = pd.read_parquet(
        audit_dir / "selected_audit_predictions.parquet"
    )  # (n_audit, 7)
    audit_all = pd.read_parquet(audit_dir / "predictions.parquet")  # (n_audit_all, 7)
    build_summary = _load_json(
        root / "data" / "processed" / "build_summary.json",
        "processed-data summary",
    )
    scoring_profile = str(build_summary["scoring_profile"])
    outputs: list[pd.DataFrame] = []  # each (n_current_cohort, c_prediction)
    model_payloads: dict[str, dict[str, Any]] = {}
    feature_columns: dict[str, list[str]] = {}
    canonical_audits: list[pd.DataFrame] = []  # each (n_cohort_audit, c_audit)

    with threadpool_limits(limits=MAX_CPU_THREADS):
        for cohort_name, positions in (
            ("offense", OFFENSE_POSITIONS),
            ("kicker", KICKER_POSITIONS),
        ):
            payload, current, columns = _fit_final_cohort(
                plan,
                cohort_name,
                positions,
                identifiers[cohort_name],
                bundles,
                tables,
                current_season,
            )
            selected_audit = audit_selected.loc[
                audit_selected["model_position"].isin(positions)
            ].copy()  # (n_cohort_audit, 7)
            current = _add_intervals_and_ranks(current, selected_audit)
            current["model_name"] = identifiers[cohort_name]  # (n_current_cohort,)
            current["selected_atoms"] = (
                "previous_points_baseline"
                if payload["kind"] == "baseline"
                else ",".join(payload["recipes"])
            )  # (n_current_cohort,)
            current["model_feature_columns"] = ",".join(columns)  # (n_current_cohort,)
            current["feature_coverage"] = 1.0  # (n_current_cohort,)
            current["out_of_distribution"] = current.get(
                "out_of_range_feature_count", pd.Series(0, index=current.index)
            ).gt(0)  # (n_current_cohort,)
            current["scoring_profile"] = scoring_profile  # (n_current_cohort,)
            outputs.append(current)
            feature_columns[cohort_name] = columns

            baseline_id = f"{cohort_name}_previous_points_baseline"
            baseline_audit = audit_all.loc[
                audit_all["candidate"].eq(baseline_id)
            ].copy()  # (n_cohort_audit, 7)
            canonical_audits.append(
                _canonical_audit_table(
                    selected_audit,
                    baseline_audit,
                    scoring_profile,
                )
            )
            serializable_payload = dict(payload)
            estimator = serializable_payload.pop("estimator")
            joblib.dump(
                {
                    "estimator": estimator,
                    **serializable_payload,
                },
                run_dir / f"{cohort_name}_model.joblib",
                compress=3,
            )
            model_payloads[cohort_name] = serializable_payload

    current_predictions = pd.concat(outputs, ignore_index=True)  # (958, c_prediction)
    if len(current_predictions) != int(build_summary["current_fantasy_players"]):
        raise RuntimeError("Phase 2 did not predict every current fantasy player.")
    if current_predictions["predicted_points"].isna().any():
        raise RuntimeError("A current fantasy player has no Phase 2 prediction.")
    available = current_predictions["available_for_draft"].astype("bool")  # (958,)
    current_predictions["overall_point_rank"] = pd.Series(
        pd.NA, index=current_predictions.index, dtype="Int32"
    )  # (958,)
    current_predictions.loc[available, "overall_point_rank"] = (
        current_predictions.loc[available, "predicted_points"]
        .rank(method="min", ascending=False)
        .astype("Int32")
    )
    audit_predictions = pd.concat(canonical_audits, ignore_index=True)  # (n_audit, c_audit)
    audit_intervals = {
        cohort_name: {
            "model_spearman": clustered_spearman_interval(
                audit_predictions.loc[
                    audit_predictions["model_position"].isin(positions)
                ],
                "predicted_points",
            ),
            "difference_vs_previous_points": clustered_spearman_interval(
                audit_predictions.loc[
                    audit_predictions["model_position"].isin(positions)
                ],
                "predicted_points",
                "previous_points_baseline",
            ),
        }
        for cohort_name, positions in (
            ("offense", OFFENSE_POSITIONS),
            ("kicker", KICKER_POSITIONS),
        )
    }
    current_predictions.to_parquet(
        run_dir / "predictions_2026.parquet", index=False
    )
    audit_predictions.to_parquet(run_dir / "audit_predictions.parquet", index=False)
    _write_json(run_dir / "audit_intervals.json", audit_intervals)

    model_manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "production_generation": "phase2_august9_v1",
        "current_season": current_season,
        "forecast_origin": plan["forecast_origin"],
        "plan_sha256": file_sha256(plan_path),
        "selection_sha256": file_sha256(selection_path),
        "audit_manifest_sha256": file_sha256(audit_dir / "run_manifest.json"),
        "code_hashes": code_hashes(root),
        "selected_procedures": identifiers,
        "models": model_payloads,
        "feature_columns": feature_columns,
        "promotion_gates": audit_report,
        "random_seed": RANDOM_SEED,
        "maximum_cpu_threads": MAX_CPU_THREADS,
        "scoring_profile": scoring_profile,
        "scoring_config_sha256": build_summary["scoring_config_sha256"],
        "snapshot_date": build_summary["snapshot_date"],
        "source_contract": (
            "historical membership uses target-season minus one rosters plus "
            "target-year draft/combine entrants; every performance input is lagged"
        ),
        "retrospective_outcomes_blinded": False,
        "prospective_2026_outcomes_observed": False,
        "packages": package_versions(),
    }
    _write_json(run_dir / "model_manifest.json", model_manifest)
    result_summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": "phase2_august9_production",
        "data_and_artifact_bytes": 0,
        "forecast_origin": plan["forecast_origin"],
        "scoring": {
            "profile": scoring_profile,
            "config_sha256": build_summary["scoring_config_sha256"],
            "target_period": build_summary["target_period"],
        },
        "candidate_universe": {
            "historical": "prior-season roster plus target-year draft/combine entrants",
            "current": "all fantasy players in the frozen 2026-08-09 roster snapshot",
            "nonparticipants": "retained with explicit zero outcomes",
        },
        "discovery_seasons": plan["discovery_seasons"],
        "retrospective_evaluation_seasons": plan["audit_seasons"],
        "retrospective_outcomes_blinded": False,
        "promotion_gates": audit_report,
        "selected_procedures": identifiers,
        "model_column_counts": {
            cohort_name: len(columns)
            for cohort_name, columns in feature_columns.items()
        },
        "current_predictions": {
            "rows": len(current_predictions),
            "available_for_draft": int(available.sum()),
            "position_counts": {
                str(position): int(count)
                for position, count in current_predictions[
                    "model_position"
                ].value_counts().items()
            },
        },
        "claim_boundary": (
            "The retrospective is chronological but not analyst-blinded. The 2026 "
            "forecast is the sealed prospective test."
        ),
    }
    _write_json(run_dir / "result_summary.json", result_summary)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "stage": "fit",
            "plan_sha256": file_sha256(plan_path),
            "selection_sha256": file_sha256(selection_path),
            "audit_manifest_sha256": file_sha256(audit_dir / "run_manifest.json"),
            "code_hashes": code_hashes(root),
            "scientific_input_lock": scientific_lock,
            "input_hashes": before_hashes,
            "promotion_gates_passed": True,
            "prediction_rows": len(current_predictions),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_cpu_threads": MAX_CPU_THREADS,
            "random_seed": RANDOM_SEED,
            "packages": package_versions(),
            "audit_manifest": audit_manifest,
        },
    )
    _write_json(run_dir / "output_hashes.json", hash_outputs(run_dir, root=root))
    if _stage_input_hashes(root, input_paths) != before_hashes:
        raise RuntimeError("A final-fit input changed during fitting.")
    _verify_scientific_inputs(root)
    enforce_size_limit(root)
    return run_dir


def _verify_fit_run(root: Path, plan_path: Path, fit_dir: Path) -> dict[str, Any]:
    """Verify an isolated production fit before canonical replacement."""
    fit_dir = fit_dir.resolve(strict=True)
    manifest = _load_json(fit_dir / "run_manifest.json", "August fit manifest")
    if manifest.get("stage") != "fit" or not manifest.get("promotion_gates_passed"):
        raise RuntimeError("Only a gated Phase 2 fit can replace production artifacts.")
    if manifest.get("plan_sha256") != file_sha256(plan_path):
        raise RuntimeError("The fit-run plan hash does not match the promotion plan.")
    if manifest.get("code_hashes") != code_hashes(root):
        raise RuntimeError("Experiment code changed after the final fit.")
    declared = _load_json(fit_dir / "output_hashes.json", "fit output hashes")
    actual = hash_outputs(fit_dir, root=root)
    actual.pop("output_hashes.json", None)
    if actual != declared:
        raise RuntimeError("A final-fit artifact changed before promotion.")
    return manifest


def _promote_fit(root: Path, plan_path: Path, fit_dir: Path) -> Path:
    """Archive Phase 1 production files and promote the gated Phase 2 fit."""
    _verify_fit_run(root, plan_path, fit_dir)
    artifact_dir = root / "artifacts"
    archive_dir = root / "experiments" / "phase1" / "production_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_PHASE2_FILES:
        canonical = artifact_dir / name
        archived = archive_dir / name
        if canonical.is_file() and not archived.exists():
            shutil.copy2(canonical, archived)

    player_module = root / "fantasy_football" / "players_2026.py"
    archived_players = archive_dir / "players_2026.py"
    if player_module.is_file() and not archived_players.exists():
        shutil.copy2(player_module, archived_players)

    for name in CANONICAL_PHASE2_FILES:
        source = fit_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"The fit run is missing {name}.")
        shutil.copy2(source, artifact_dir / name)

    from .export import export_current_players

    exported = export_current_players(root, int(2026))
    enforce_size_limit(root)
    return exported


def main(argv: Sequence[str] | None = None) -> int:
    """Run one locked August production stage."""
    arguments = _parse_args(argv)
    root = arguments.root.resolve()
    plan_path = arguments.plan.resolve()
    try:
        plan = load_august_plan(plan_path)
        verify_prefit_manifest(root, plan_path, plan)
        if arguments.stage == "discovery":
            output = _run_discovery(root, plan_path, plan)
        elif arguments.stage == "audit":
            if arguments.selection is None:
                raise ValueError("Audit requires --selection.")
            output = _run_audit(
                root,
                plan_path,
                plan,
                arguments.selection.resolve(),
            )
        elif arguments.stage == "fit":
            if arguments.selection is None or arguments.audit is None:
                raise ValueError("Final fitting requires --selection and --audit.")
            output = _run_fit(
                root,
                plan_path,
                plan,
                arguments.selection.resolve(),
                arguments.audit.resolve(),
            )
        else:
            if arguments.fit_run is None:
                raise ValueError("Promotion requires --fit-run.")
            output = _promote_fit(root, plan_path, arguments.fit_run.resolve())
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"August Phase 2 failed: {error}", file=sys.stderr)
        return 1
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
