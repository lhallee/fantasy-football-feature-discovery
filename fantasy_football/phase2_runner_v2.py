"""Run the status-filtered Phase 2 v2 experiment without changing v1."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import numpy as np
import pandas as pd

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from threadpoolctl import threadpool_limits

from . import phase2_runner as v1
from .models import MAX_CPU_THREADS, RANDOM_SEED
from .phase2_cohort import EXCLUDED_STATUS_CODES
from .phase2_cohort import TEAM_CONTROLLED_STATUS_CODES
from .phase2_cohort import build_phase2_offense_cohort
from .phase2_features import Phase2FeatureBundle, build_phase2_features
from .phase2_layout import ExperimentLayoutError, verify_historical_file
from .phase2_safety import hash_inputs, hash_outputs, isolated_phase2_run
from .phase2_safety import load_freeze_manifest
from .provenance import file_sha256, package_versions


SCALE_DEPENDENT_METRICS = (
    "mae",
    "rmse",
    "r2",
    "calibration_intercept",
    "calibration_slope",
)
DRAFT_FEATURE_FRAGMENTS = ("draft_number", "draft_round", "was_drafted")
STATUS_DICTIONARY_URL = (
    "https://nflreadr.nflverse.com/articles/dictionary_roster_status.html"
)
EXPECTED_METRIC_CONTRACT = {
    "signed_log_predictions": "inverse_transform_before_point_metrics",
    "rank_hurdle_and_blend_point_metrics": "omit_as_scale_incompatible",
    "order_metrics": "retain_for_every_candidate",
}
EXPECTED_NONSELECTION_DIAGNOSTICS = {
    "end_to_end_null": (
        "clone the discovery-selected procedure and shuffle target points within "
        "training position-season groups before feature selection and fitting"
    ),
    "no_draft": (
        "clone the discovery-selected procedure with draft_number, draft_round, "
        "and was_drafted excluded at the same feature-limit setting"
    ),
}
EXPECTED_UPSTREAM_DEDUP_AUDIT = {
    "mixed_included_excluded_status_groups": 712,
    "deduplicated_to_included_status": 711,
    "deduplicated_to_excluded_status": 1,
    "known_excluded_player_id": "00-0029343",
    "known_excluded_target_season": 2014,
    "known_raw_statuses": ["RSN", "TRD"],
}
PHASE2_CODE_FILENAMES = (
    "phase2_cohort.py",
    "phase2_features.py",
    "phase2_layout.py",
    "phase2_modeling.py",
    "phase2_runner.py",
    "phase2_runner_v2.py",
    "phase2_safety.py",
)
PREFIT_AUDIT_RELATIVE_PATHS = (
    "EXPERIMENT_LEDGER_PHASE2.md",
    "experiments/phase2/README.md",
    "fantasy_football/phase2_cohort.py",
    "fantasy_football/phase2_features.py",
    "fantasy_football/phase2_modeling.py",
    "fantasy_football/phase2_runner.py",
    "fantasy_football/phase2_runner_v2.py",
    "fantasy_football/phase2_safety.py",
    "tests/test_phase2_artifacts.py",
    "tests/test_phase2_cohort.py",
    "tests/test_phase2_features.py",
    "tests/test_phase2_modeling.py",
    "tests/test_phase2_runner_v2.py",
    "tests/test_phase2_safety.py",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the status-filtered Phase 2 v2 experiment plan."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=v1.VALID_STAGES, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        help="V2 discovery selection.json, required for retrospective stage.",
    )
    return parser.parse_args(argv)


def _phase2_code_paths(root: Path) -> list[Path]:
    return [root / "fantasy_football" / filename for filename in PHASE2_CODE_FILENAMES]


def _validate_recorded_before_fit(value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError("Phase 2 v2 recorded_before_fit_utc must be a timestamp.")
    try:
        recorded = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "Phase 2 v2 recorded_before_fit_utc must be ISO-8601."
        ) from error
    if recorded.tzinfo is None:
        raise ValueError("Phase 2 v2 recorded_before_fit_utc must include a timezone.")
    if recorded.astimezone(UTC) > datetime.now(UTC):
        raise ValueError("Phase 2 v2 plan cannot be recorded in the future.")


def _load_amended_plan(
    root: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    try:
        amendment = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Phase 2 v2 plan: {plan_path}") from error
    if not isinstance(amendment, dict):
        raise ValueError("Phase 2 v2 plan must contain a JSON object.")

    required = {
        "plan_version",
        "plan_name",
        "recorded_before_fit_utc",
        "base_plan_path",
        "base_plan_sha256",
        "prefit_manifest_path",
        "cohort_contract",
        "metric_contract",
        "nonselecting_diagnostics",
    }
    missing = sorted(required.difference(amendment))
    if missing:
        raise ValueError(f"Phase 2 v2 plan is missing fields: {missing!r}.")
    if int(amendment["plan_version"]) != 2:
        raise ValueError("Phase 2 v2 runner requires plan_version 2.")
    _validate_recorded_before_fit(amendment["recorded_before_fit_utc"])

    base_path = (root / str(amendment["base_plan_path"])).resolve()
    try:
        base_path.relative_to(root)
    except ValueError as error:
        raise ValueError("Phase 2 v2 base plan escapes the project root.") from error
    if file_sha256(base_path) != str(amendment["base_plan_sha256"]):
        raise ValueError("Phase 2 v2 base-plan hash does not match.")

    cohort_contract = amendment["cohort_contract"]
    if not isinstance(cohort_contract, dict):
        raise ValueError("Phase 2 v2 cohort_contract must be an object.")
    declared_statuses = frozenset(
        str(value).strip().upper()
        for value in cohort_contract.get("included_statuses", [])
    )
    if declared_statuses != frozenset(TEAM_CONTROLLED_STATUS_CODES):
        raise ValueError("Phase 2 v2 status allowlist differs from the frozen helper.")
    declared_exclusions = frozenset(
        str(value).strip().upper()
        for value in cohort_contract.get("excluded_statuses", [])
    )
    if declared_exclusions != frozenset(EXCLUDED_STATUS_CODES):
        raise ValueError("Phase 2 v2 status exclusions differ from the frozen helper.")
    if cohort_contract.get("status_dictionary") != STATUS_DICTIONARY_URL:
        raise ValueError("Phase 2 v2 must cite the frozen roster-status dictionary.")
    if cohort_contract.get("missing_status_action") != "exclude_and_audit":
        raise ValueError("Phase 2 v2 must exclude and audit missing statuses.")
    if cohort_contract.get("unexpected_nonmissing_status_action") != "fail":
        raise ValueError("Phase 2 v2 must fail on unexpected nonmissing statuses.")
    if cohort_contract.get("apply_before_room_aggregation") is not True:
        raise ValueError("Phase 2 v2 must filter before room aggregation.")
    if (
        cohort_contract.get("status_application_grain")
        != "frozen_upstream_deduplicated_week_1_player_row"
    ):
        raise ValueError("Phase 2 v2 must declare its processed-row status grain.")
    if cohort_contract.get("upstream_deduplication_audit") != (
        EXPECTED_UPSTREAM_DEDUP_AUDIT
    ):
        raise ValueError("Phase 2 v2 upstream status-dedup audit differs.")
    if (
        cohort_contract.get("status_is_a_cohort_definition_not_a_model_feature")
        is not True
    ):
        raise ValueError("Phase 2 v2 status may define the cohort but not a feature.")
    if cohort_contract.get("historical_snapshot_timestamp_verified") is not False:
        raise ValueError("Phase 2 v2 must retain the historical timestamp caveat.")
    if amendment["metric_contract"] != EXPECTED_METRIC_CONTRACT:
        raise ValueError(
            "Phase 2 v2 metric contract differs from the executable contract."
        )
    if amendment["nonselecting_diagnostics"] != EXPECTED_NONSELECTION_DIAGNOSTICS:
        raise ValueError(
            "Phase 2 v2 diagnostic contract differs from the executable contract."
        )

    effective = v1._load_plan(base_path)
    effective["plan_version"] = 2
    effective["plan_name"] = str(amendment["plan_name"])
    effective["cohort_contract"] = cohort_contract
    return effective, amendment, base_path


def _verify_prefit_manifest(
    root: Path,
    plan_path: Path,
    amendment: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    manifest_path = (root / str(amendment["prefit_manifest_path"])).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "Phase 2 v2 pre-fit manifest escapes the project root."
        ) from error
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read Phase 2 v2 pre-fit manifest: {manifest_path}"
        ) from error
    manifest_version = (
        manifest.get("manifest_version") if isinstance(manifest, dict) else None
    )
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest_version, int)
        or isinstance(manifest_version, bool)
        or manifest_version != 1
    ):
        raise ValueError("Phase 2 v2 pre-fit manifest must use manifest_version 1.")
    if manifest.get("locked_before_fit_utc") != amendment["recorded_before_fit_utc"]:
        raise ValueError("Phase 2 v2 pre-fit timestamp does not match its plan.")
    if manifest.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("Phase 2 v2 pre-fit manifest has the wrong plan hash.")
    if manifest.get("base_plan_sha256") != amendment["base_plan_sha256"]:
        raise ValueError("Phase 2 v2 pre-fit manifest has the wrong base-plan hash.")
    phase1_freeze = load_freeze_manifest(root)
    valid_freeze_ids = {phase1_freeze["freeze_id"]}
    previous_freeze = phase1_freeze.get("previous_freeze")
    if isinstance(previous_freeze, Mapping):
        previous_freeze_id = previous_freeze.get("freeze_id")
        if isinstance(previous_freeze_id, str):
            valid_freeze_ids.add(previous_freeze_id)
    if manifest.get("phase1_freeze_id") not in valid_freeze_ids:
        raise ValueError("Phase 2 v2 pre-fit manifest has the wrong Phase 1 freeze ID.")

    locked_files = manifest.get("locked_files")
    if not isinstance(locked_files, dict):
        raise ValueError("Phase 2 v2 pre-fit manifest must contain locked_files.")
    plan_relative = plan_path.relative_to(root).as_posix()
    required_paths = {plan_relative, *PREFIT_AUDIT_RELATIVE_PATHS}
    if set(locked_files) != required_paths:
        missing = sorted(required_paths.difference(locked_files))
        extra = sorted(set(locked_files).difference(required_paths))
        raise ValueError(
            "Phase 2 v2 pre-fit locked-file set differs from the protocol: "
            f"missing={missing!r}, extra={extra!r}."
        )
    for relative_path, expected_hash in sorted(locked_files.items()):
        try:
            verify_historical_file(root, relative_path, str(expected_hash))
        except ExperimentLayoutError as error:
            raise ValueError(
                f"Phase 2 v2 pre-fit input hash differs: {relative_path}."
            ) from error
    return manifest_path, manifest


def _validate_selection_integrity(
    selection: Mapping[str, Any],
    current_code_hashes: Mapping[str, str],
    current_prefit_manifest_hash: str,
) -> None:
    selected_code_hashes = selection.get("phase2_code_hashes")
    if not isinstance(selected_code_hashes, dict):
        raise ValueError(
            "Phase 2 v2 selection does not bind its implementation hashes."
        )
    if selected_code_hashes != dict(current_code_hashes):
        raise ValueError(
            "Phase 2 v2 code changed between discovery and retrospective stages."
        )
    if selection.get("prefit_manifest_sha256") != current_prefit_manifest_hash:
        raise ValueError(
            "Phase 2 v2 pre-fit lock changed between discovery and retrospective "
            "stages."
        )


def _cohort_audit_payload(cohort: Any) -> dict[str, Any]:
    # cohort.status_counts: (n_statuses, 3)
    status_records = cohort.status_counts.to_dict(orient="records")
    return {
        "rows_before": int(cohort.rows_before),
        "rows_after": int(cohort.rows_after),
        "rows_removed": int(cohort.rows_before - cohort.rows_after),
        "status_counts": status_records,
        "included_statuses": sorted(TEAM_CONTROLLED_STATUS_CODES),
    }


def _repair_metric_scales(
    plan: Mapping[str, Any],
    predictions_by_candidate: dict[str, pd.DataFrame],
    fold_frames: list[pd.DataFrame],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = v1._candidate_mapping(plan)
    summary_by_candidate = {str(record["candidate"]): record for record in summaries}
    invalid_point_candidates: set[str] = set()
    inverse_transformed: list[str] = []

    for identifier, predictions in predictions_by_candidate.items():
        # predictions: (n_validation_rows, 7)
        candidate = candidates.get(identifier)
        target = None if candidate is None else candidate.target
        kind = str(summary_by_candidate[identifier]["kind"])
        if target == "signed_log_points":
            values = predictions["predicted"].to_numpy(  # (n_validation_rows,)
                dtype="float64"
            )
            predictions["predicted"] = (  # (n_validation_rows, 7)
                np.sign(values) * np.expm1(np.abs(values))
            )
            inverse_transformed.append(identifier)
        if target in {"rank_percentile", "hurdle_rank"} or kind == "blend":
            invalid_point_candidates.add(identifier)

        refreshed = v1.summarize_predictions(predictions)
        summary_by_candidate[identifier].update(refreshed)

    fold_frames.clear()
    for identifier, predictions in predictions_by_candidate.items():
        frame = v1._fold_metrics(predictions)  # (n_validation_folds, m_metrics)
        if identifier in invalid_point_candidates:
            for column in SCALE_DEPENDENT_METRICS:
                frame[column] = np.nan  # (n_validation_folds, m_metrics)
                summary_by_candidate[identifier][column] = np.nan
        fold_frames.append(frame)

    return {
        "signed_log_inverse_transformed": sorted(inverse_transformed),
        "point_metrics_omitted": sorted(invalid_point_candidates),
        "omitted_metric_columns": list(SCALE_DEPENDENT_METRICS),
        "order_metrics_retained_for_all_candidates": True,
    }


def _shuffled_target_table(table: pd.DataFrame) -> pd.DataFrame:
    # table: (n_targets, c_table)
    missing_target = table["target_points"].isna()  # (n_targets,)
    if missing_target.any():
        raise ValueError(
            "End-to-end null requires a non-null target for every eligible row."
        )
    shuffled = table.copy()  # (n_targets, c_table)
    grouped_indices = shuffled.groupby(  # g mappings to row-index vectors
        ["target_season", "model_position"],
        observed=True,
    ).groups
    for (season, position), row_index in sorted(grouped_indices.items()):
        position_seed = sum(ord(character) for character in str(position))
        generator = np.random.default_rng(
            RANDOM_SEED + 100 * int(season) + position_seed
        )
        values = shuffled.loc[row_index, "target_points"].to_numpy(  # (n_group,)
            dtype="float64"
        )
        shuffled.loc[row_index, "target_points"] = generator.permutation(
            values
        )  # (n_targets, c_table)
    return shuffled


def _candidate_uses_draft_inputs(
    plan: Mapping[str, Any],
    bundles: Mapping[str, Phase2FeatureBundle],
    identifier: str,
) -> bool:
    candidate = v1._candidate_mapping(plan)[identifier]
    bundle = bundles[candidate.tier]
    global_excluded = tuple(
        str(value) for value in plan.get("global_excluded_feature_fragments", ())
    )
    full_frame = v1._candidate_feature_frame(  # (n_targets, d_full)
        bundle,
        candidate,
        global_excluded,
    )
    no_draft_candidate = replace(
        candidate,
        excluded_feature_fragments=(
            *candidate.excluded_feature_fragments,
            *DRAFT_FEATURE_FRAGMENTS,
        ),
    )
    no_draft_frame = v1._candidate_feature_frame(  # (n_targets, d_no_draft)
        bundle,
        no_draft_candidate,
        global_excluded,
    )
    return tuple(full_frame.columns) != tuple(no_draft_frame.columns)


def _not_applicable_diagnostic(
    selected_identifier: str,
    diagnostic: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "identifier": f"{selected_identifier}__{diagnostic}",
        "applicable": False,
        "reason": reason,
        "summary": {
            "applicable": False,
            "reason": reason,
        },
    }


def _run_model_diagnostic(
    plan: Mapping[str, Any],
    stage: str,
    validation_seasons: tuple[int, ...],
    bundles: Mapping[str, Phase2FeatureBundle],
    tables: Mapping[str, pd.DataFrame],
    selected_identifier: str,
    diagnostic: str,
) -> dict[str, Any]:
    candidates = v1._candidate_mapping(plan)
    selected = candidates[selected_identifier]
    if selected.shuffle_target:
        raise ValueError("A shuffled benchmark cannot be the selected procedure.")

    bundle = bundles[selected.tier]
    source_table = tables[selected.tier]  # (n_targets, c_table)
    if diagnostic == "end_to_end_null":
        diagnostic_id = f"{selected_identifier}__end_to_end_null"
        candidate = replace(
            selected,
            identifier=diagnostic_id,
            shuffle_target=False,
            benchmark_only=True,
        )
        fit_table = _shuffled_target_table(source_table)  # (n_targets, c_table)
    elif diagnostic == "no_draft":
        diagnostic_id = f"{selected_identifier}__no_draft"
        candidate = replace(
            selected,
            identifier=diagnostic_id,
            excluded_feature_fragments=(
                *selected.excluded_feature_fragments,
                *DRAFT_FEATURE_FRAGMENTS,
            ),
            benchmark_only=True,
        )
        fit_table = source_table
    else:
        raise ValueError(f"Unknown Phase 2 v2 diagnostic: {diagnostic!r}.")

    global_excluded = tuple(
        str(value) for value in plan.get("global_excluded_feature_fragments", ())
    )
    X = v1._candidate_feature_frame(  # (n_targets, d_diagnostic_features)
        bundle,
        candidate,
        global_excluded,
    )
    minimum_training = max(
        int(plan["minimum_training_season"][candidate.tier]),
        int(source_table["target_season"].min()),
    )
    output = v1.walk_forward_candidate(
        fit_table,
        X,
        candidate,
        validation_seasons,
        minimum_training,
    )
    predictions = output.predictions.copy()  # (n_validation_rows, 7)
    predictions["target_points"] = source_table.loc[
        predictions.index,
        "target_points",
    ].to_numpy(dtype="float64")  # (n_validation_rows,)
    if selected.target == "signed_log_points":
        values = predictions["predicted"].to_numpy(  # (n_validation_rows,)
            dtype="float64"
        )
        predictions["predicted"] = (  # (n_validation_rows, 7)
            np.sign(values) * np.expm1(np.abs(values))
        )
    fold_metrics = v1._fold_metrics(predictions)  # (n_validation_folds, m_metrics)
    median_selected = float(
        output.selected_features.groupby(
            ["validation_season", "position_scope"],
            observed=True,
        )["selected_feature_count"]
        .first()
        .median()
    )
    summary = v1._summary_record(
        candidate,
        predictions,
        fold_metrics,
        len(X.columns),
        median_selected,
        stage=stage,
        kind="nonselecting_diagnostic",
    )
    if selected.target in {"rank_percentile", "hurdle_rank"}:
        for column in SCALE_DEPENDENT_METRICS:
            fold_metrics[column] = np.nan  # (n_validation_folds, m_metrics)
            summary[column] = None
    return {
        "identifier": diagnostic_id,
        "applicable": True,
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "selected_features": output.selected_features,
        "summary": summary,
    }


def _run_selected_diagnostic(
    plan: Mapping[str, Any],
    stage: str,
    validation_seasons: tuple[int, ...],
    bundles: Mapping[str, Phase2FeatureBundle],
    tables: Mapping[str, pd.DataFrame],
    selected_identifier: str,
    diagnostic: str,
) -> dict[str, Any]:
    candidates = v1._candidate_mapping(plan)
    blends = v1._blend_mapping(plan)
    if selected_identifier in candidates:
        component_identifiers = (selected_identifier,)
    elif selected_identifier in blends:
        component_identifiers = tuple(
            str(value) for value in blends[selected_identifier]["components"]
        )
    else:
        raise ValueError(
            f"Selected Phase 2 v2 procedure is unknown: {selected_identifier!r}."
        )

    if diagnostic == "no_draft" and not any(
        _candidate_uses_draft_inputs(plan, bundles, identifier)
        for identifier in component_identifiers
    ):
        return _not_applicable_diagnostic(
            selected_identifier,
            diagnostic,
            "The selected procedure's input pool already excludes every declared "
            "draft feature.",
        )

    if selected_identifier in candidates:
        return _run_model_diagnostic(
            plan,
            stage,
            validation_seasons,
            bundles,
            tables,
            selected_identifier,
            diagnostic,
        )

    blend = blends[selected_identifier]
    components = tuple(str(value) for value in blend["components"])
    weights = tuple(float(value) for value in blend["weights"])
    component_results = {
        identifier: _run_model_diagnostic(
            plan,
            stage,
            validation_seasons,
            bundles,
            tables,
            identifier,
            diagnostic,
        )
        for identifier in components
    }
    diagnostic_id = f"{selected_identifier}__{diagnostic}"
    diagnostic_components = tuple(
        str(component_results[identifier]["identifier"]) for identifier in components
    )
    component_predictions = {
        str(result["identifier"]): result["predictions"]
        for result in component_results.values()
    }
    predictions = v1.blend_walk_forward_predictions(  # (n_validation_rows, 7)
        component_predictions,
        diagnostic_id,
        diagnostic_components,
        weights,
    )
    fold_metrics = v1._fold_metrics(predictions)  # (n_validation_folds, m_metrics)
    selected_features = pd.concat(  # (n_component_selected_records, 7)
        [result["selected_features"] for result in component_results.values()],
        ignore_index=True,
    )
    feature_pool_count = int(
        sum(
            int(result["summary"]["feature_pool_count"])
            for result in component_results.values()
        )
    )
    median_selected = float(
        sum(
            float(result["summary"]["median_selected_feature_count"])
            for result in component_results.values()
        )
    )
    summary = v1._summary_record(
        None,
        predictions,
        fold_metrics,
        feature_pool_count,
        median_selected,
        stage=stage,
        kind="nonselecting_diagnostic_blend",
    )
    for column in SCALE_DEPENDENT_METRICS:
        fold_metrics[column] = np.nan  # (n_validation_folds, m_metrics)
        summary[column] = None
    return {
        "identifier": diagnostic_id,
        "applicable": True,
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "selected_features": selected_features,
        "summary": summary,
    }


def _json_safe_summary(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            None
            if isinstance(value, (float, np.floating)) and not np.isfinite(value)
            else value
        )
        for key, value in values.items()
    }


def _diagnostic_summary_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_safe_summary(result["summary"])
    payload["applicable"] = bool(result.get("applicable", True))
    if result.get("reason") is not None:
        payload["reason"] = str(result["reason"])
    return payload


def _position_metric_contract(
    position_metrics: Mapping[str, Mapping[str, float]],
    *,
    point_scale_valid: bool,
) -> dict[str, dict[str, float | None]]:
    contracted = {
        position: dict(metrics) for position, metrics in position_metrics.items()
    }
    if not point_scale_valid:
        for metrics in contracted.values():
            for column in SCALE_DEPENDENT_METRICS:
                metrics[column] = None
    return contracted


def _write_diagnostic(
    run_dir: Path,
    diagnostic: str,
    result: Mapping[str, Any],
) -> None:
    if result.get("applicable") is False:
        v1._write_json(
            run_dir / f"{diagnostic}_summary.json",
            _json_safe_summary(result["summary"]),
        )
        return
    predictions = result["predictions"]  # (n_validation_rows, 7)
    fold_metrics = result["fold_metrics"]  # (n_validation_folds, m_metrics)
    selected_features = result[  # (n_selected_feature_records, 7)
        "selected_features"
    ]
    predictions.to_parquet(
        run_dir / f"{diagnostic}_predictions.parquet",
        index=False,
    )
    fold_metrics.to_csv(run_dir / f"{diagnostic}_fold_metrics.csv", index=False)
    selected_features.to_csv(
        run_dir / f"{diagnostic}_selected_features.csv",
        index=False,
    )
    v1._write_json(
        run_dir / f"{diagnostic}_summary.json",
        _json_safe_summary(result["summary"]),
    )


def _run_stage_v2(
    root: Path,
    plan_path: Path,
    stage: str,
    selection_path: Path | None,
) -> Path:
    root = root.resolve()
    plan_path = plan_path.resolve()
    plan, amendment, base_plan_path = _load_amended_plan(root, plan_path)
    plan_hash = file_sha256(plan_path)
    code_paths = _phase2_code_paths(root)
    current_code_hashes = hash_inputs(root, code_paths)
    prefit_manifest_path, prefit_manifest = _verify_prefit_manifest(
        root,
        plan_path,
        amendment,
    )
    prior_selection = v1._selection_file(selection_path, plan_hash)
    if prior_selection is not None:
        _validate_selection_integrity(
            prior_selection,
            current_code_hashes,
            file_sha256(prefit_manifest_path),
        )
    if stage == "retrospective" and prior_selection is None:
        raise ValueError("Retrospective stage requires --selection.")

    validation_seasons = tuple(int(value) for value in plan[f"{stage}_seasons"])
    if tuple(sorted(validation_seasons)) != validation_seasons:
        raise ValueError("Phase 2 v2 validation seasons must be sorted.")
    candidate_ids, blend_ids = v1._required_candidates(
        plan,
        stage,
        prior_selection,
    )
    maximum_source_season = max(validation_seasons) - 1
    weekly_paths = v1._weekly_input_paths(root, maximum_source_season)
    input_paths: list[Path] = [
        plan_path,
        base_plan_path,
        prefit_manifest_path,
        *code_paths,
        root / "data" / "processed" / "player_seasons.parquet",
        root / "data" / "processed" / "modeling_table.parquet",
        *weekly_paths,
    ]
    if selection_path is not None:
        input_paths.append(selection_path.resolve())

    label = f"v2-{stage}-{plan_hash[:12]}"
    started = time.perf_counter()
    with isolated_phase2_run(root, label, input_paths=input_paths) as safety_run:
        weekly_stats = v1._load_weekly_stats(weekly_paths)  # (n_weekly, c_weekly)
        player_seasons = pd.read_parquet(  # (n_player_seasons, c_seasons)
            root / "data" / "processed" / "player_seasons.parquet"
        )
        player_seasons = player_seasons.loc[
            player_seasons["season"] <= maximum_source_season
        ].copy()  # (n_source_seasons, c_seasons)
        source_modeling = pd.read_parquet(  # (n_targets, c_modeling)
            root / "data" / "processed" / "modeling_table.parquet"
        )
        source_modeling = source_modeling.loc[
            (source_modeling["target_season"] <= max(validation_seasons))
            & source_modeling["model_position"].isin(v1.OFFENSE_POSITIONS)
        ].copy()  # (n_in_scope_offense_targets, c_modeling)
        cohort = build_phase2_offense_cohort(source_modeling)
        modeling_table = cohort.table  # (n_eligible_offense_targets, c_modeling)
        missing_target = modeling_table["target_points"].isna()  # (n_eligible_targets,)
        if missing_target.any():
            raise ValueError(
                "Phase 2 v2 historical cohort contains missing target points."
            )

        needed_tiers = {
            v1._candidate_mapping(plan)[identifier].tier for identifier in candidate_ids
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
            tables[tier] = v1._table_for_bundle(modeling_table, bundle)

        with threadpool_limits(limits=MAX_CPU_THREADS):
            predictions_by_candidate, fold_frames, selection_frames, summaries = (
                v1._evaluate_models(
                    plan,
                    stage,
                    validation_seasons,
                    bundles,
                    tables,
                    candidate_ids,
                )
            )
            v1._evaluate_blends(
                plan,
                stage,
                blend_ids,
                predictions_by_candidate,
                fold_frames,
                summaries,
            )
            metric_validity = _repair_metric_scales(
                plan,
                predictions_by_candidate,
                fold_frames,
                summaries,
            )
            baseline_table = (
                tables["stable"] if "stable" in tables else next(iter(tables.values()))
            )  # (n_baseline_targets, c_table)
            v1._add_baseline(
                stage,
                validation_seasons,
                baseline_table,
                predictions_by_candidate,
                fold_frames,
                summaries,
            )

        summary = pd.DataFrame(summaries).sort_values(  # (n_candidates, m_metrics)
            "spearman",
            ascending=False,
            kind="stable",
        )
        fold_metrics = pd.concat(  # (n_folds, m_metrics)
            fold_frames,
            ignore_index=True,
        )
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
            current_selection = v1._select_discovery_candidate(
                plan,
                summary,
                fold_metrics,
                plan_hash,
            )
            current_selection["phase2_code_hashes"] = current_code_hashes
            current_selection["prefit_manifest_sha256"] = file_sha256(
                prefit_manifest_path
            )
            v1._write_json(safety_run.run_dir / "selection.json", current_selection)
        if current_selection is None:
            raise AssertionError("Phase 2 v2 has no selected procedure.")

        selected_identifier = str(current_selection["selected_candidate"])
        with threadpool_limits(limits=MAX_CPU_THREADS):
            diagnostic_results = {
                diagnostic: _run_selected_diagnostic(
                    plan,
                    stage,
                    validation_seasons,
                    bundles,
                    tables,
                    selected_identifier,
                    diagnostic,
                )
                for diagnostic in ("end_to_end_null", "no_draft")
            }
        selected_predictions = predictions_by_candidate[
            selected_identifier
        ]  # (n_selected_validation_rows, 7)
        selected_fold_metrics = fold_metrics.loc[
            fold_metrics["candidate"].eq(selected_identifier)
        ].copy()  # (n_selected_folds, m_metrics)
        uncertainty = v1._uncertainty(
            selected_predictions,
            selected_fold_metrics,
        )
        selected_summary = next(
            row for row in summaries if row["candidate"] == selected_identifier
        )
        point_scale_valid = (
            str(selected_summary["target"]) not in {"rank_percentile", "hurdle_rank"}
            and str(selected_summary["kind"]) != "blend"
        )
        raw_position_metrics = v1._position_metrics(selected_predictions)
        position_metrics = _position_metric_contract(
            raw_position_metrics,
            point_scale_valid=point_scale_valid,
        )

        summary.to_csv(safety_run.run_dir / "candidate_summary.csv", index=False)
        fold_metrics.to_csv(safety_run.run_dir / "fold_metrics.csv", index=False)
        selected_features.to_csv(
            safety_run.run_dir / "selected_features.csv",
            index=False,
        )
        prediction_table.to_parquet(
            safety_run.run_dir / "predictions.parquet",
            index=False,
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
        v1._write_json(safety_run.run_dir / "feature_lineage.json", lineage)
        v1._write_json(safety_run.run_dir / "uncertainty.json", uncertainty)
        v1._write_json(safety_run.run_dir / "position_metrics.json", position_metrics)
        cohort_payload = _cohort_audit_payload(cohort)
        v1._write_json(safety_run.run_dir / "cohort_audit.json", cohort_payload)
        v1._write_json(safety_run.run_dir / "metric_validity.json", metric_validity)
        for diagnostic, result in diagnostic_results.items():
            _write_diagnostic(safety_run.run_dir, diagnostic, result)

        elapsed_seconds = time.perf_counter() - started
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "stage": stage,
            "plan_name": plan["plan_name"],
            "plan_sha256": plan_hash,
            "base_plan_path": str(amendment["base_plan_path"]),
            "base_plan_sha256": str(amendment["base_plan_sha256"]),
            "prefit_manifest_path": str(amendment["prefit_manifest_path"]),
            "prefit_manifest_sha256": file_sha256(prefit_manifest_path),
            "prefit_locked_before_fit_utc": prefit_manifest["locked_before_fit_utc"],
            "phase2_code_hashes": current_code_hashes,
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
            "cohort_contract": amendment["cohort_contract"],
            "cohort_audit": cohort_payload,
            "metric_contract": amendment["metric_contract"],
            "metric_validity": metric_validity,
            "selected_candidate": selected_identifier,
            "selected_metrics": _json_safe_summary(selected_summary),
            "end_to_end_negative_control_spearman": float(
                diagnostic_results["end_to_end_null"]["summary"]["spearman"]
            ),
            "legacy_shuffled_fit_control_spearman": next(
                (
                    float(row["spearman"])
                    for row in summaries
                    if "shuffle" in str(row["candidate"])
                ),
                None,
            ),
            "nonselecting_diagnostics": {
                diagnostic: _diagnostic_summary_payload(result)
                for diagnostic, result in diagnostic_results.items()
            },
            "source_contract": (
                "performance inputs use source seasons strictly earlier than each "
                "target season; status defines the cohort before room aggregation "
                "and is not a model input; historical publication timestamps remain "
                "unverified"
            ),
        }
        v1._write_json(safety_run.run_dir / "run_manifest.json", manifest)
        v1._write_run_ledger(
            safety_run.run_dir / "LEDGER.md",
            stage,
            plan,
            summary,
            current_selection,
            elapsed_seconds,
        )
        before_hash_manifest = hash_outputs(safety_run.run_dir, root=root)
        v1._write_json(
            safety_run.run_dir / "output_hashes.json",
            before_hash_manifest,
        )
        run_dir = safety_run.run_dir
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Phase 2 v2 command-line entry point."""
    args = _parse_args(argv)
    try:
        run_dir = _run_stage_v2(args.root, args.plan, args.stage, args.selection)
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"Phase 2 v2 failed: {error}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
