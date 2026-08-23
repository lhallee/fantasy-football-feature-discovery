"""Run the Phase 4 final ranking: points track, injury track, combination, workbook."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from .metrics import TOP_K, clustered_spearman_interval
from .phase4_labels import (
    LABEL_SEASONS,
    PRIMARY_LABEL,
    SECONDARY_LABEL,
    Phase4Bundle,
    build_phase4_bundle,
)
from .phase4_modeling import (
    INJURY_AUDIT_SEASONS,
    INJURY_DEVELOPMENT_SEASONS,
    INJURY_WEIGHTS,
    KEY_COLUMNS,
    MAX_THREADS,
    OFFENSE_POSITIONS,
    PHASE2_REFERENCE,
    POINTS_AUDIT_SEASONS,
    POINTS_DISCOVERY_SEASONS,
    RANDOM_SEED,
    SELECTION_TOLERANCE,
    PointsCandidate,
    apply_residual_intervals,
    assemble_candidate_predictions,
    auc_scorer,
    clustered_metric_interval,
    combined_score,
    evaluate_injury_weights,
    feature_association_table,
    fit_calibrated_injury_model,
    fit_points_learner,
    injury_learner_grid,
    permutation_importance,
    permute_within_groups,
    points_candidate_grid,
    points_fold_table,
    points_learner_grid,
    points_summary,
    predict_points,
    probability_metrics,
    residual_interval_table,
    select_injury_learner,
    select_injury_weight,
    select_points_candidate,
    spearman_scorer,
    walk_forward_injury,
    walk_forward_points,
)
from .phase4_workbook import write_final_workbook
from .phase5_features import extend_with_usage_features


PHASE4_CODE_PATHS = (
    "fantasy_football/phase4_labels.py",
    "fantasy_football/phase4_modeling.py",
    "fantasy_football/phase4_workbook.py",
    "fantasy_football/phase4_runner.py",
    "fantasy_football/phase5_features.py",
    "fantasy_football/phase3_injury.py",
    "fantasy_football/phase2_august.py",
    "fantasy_football/phase2_features.py",
    "fantasy_football/metrics.py",
    "fantasy_football/models.py",
)
PHASE2_DISCOVERY_PREDICTIONS = (
    "experiments/phase2/runs/"
    "20260810T174927548207Z-august9-discovery-3b07b79602c5-b96dfefdf4d91c71/"
    "predictions.parquet"
)  # relocated from experiments/phase2/production/runs during consolidation
PHASE2_AUDIT_PREDICTIONS = "artifacts/audit_predictions.parquet"
PHASE2_CURRENT_PREDICTIONS = "artifacts/predictions_2026.parquet"
PHASE2_OFFENSE_CANDIDATE = "off_stable_core_et_all"
CURRENT_INJURY_SCREEN = (
    "experiments/phase3/artifacts/current_injury_screen_2026-08-14.csv"
)
CURRENT_INJURY_AS_OF = "2026-08-14"
INJURY_DIRECTORY = "experiments/phase3/data/raw"
PRIMARY_INPUT_PATHS = (
    "data/processed/player_seasons.parquet",
    "data/processed/preseason_players.parquet",
    PHASE2_DISCOVERY_PREDICTIONS,
    PHASE2_AUDIT_PREDICTIONS,
    PHASE2_CURRENT_PREDICTIONS,
    CURRENT_INJURY_SCREEN,
    "experiments/phase4/plan_v1.json",
)
RISK_BAND_EDGES = (0.20, 0.35)
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#dddcd7",
    "surface": "#fcfcfb",
}


def _log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            values, indent=2, sort_keys=True, allow_nan=False, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 4 final ranking.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", default="final-ranking-v1")
    parser.add_argument("--null-seeds", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--skip-featureranker",
        action="store_true",
        help="Skip the FeatureRanker 3.0.4 evidence line (amendment 3 restores it by default).",
    )
    parser.add_argument(
        "--refresh-featureranker",
        action="store_true",
        help="With --rebuild-from: rerun FeatureRanker on the saved run and recompute the consensus ranks.",
    )
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--rebuild-from", type=Path, default=None)
    parser.add_argument(
        "--experiment",
        default="phase4",
        help="experiments/<name> directory for plan, runs, and artifacts",
    )
    parser.add_argument(
        "--feature-set",
        choices=("phase4", "phase5"),
        default="phase4",
        help="phase5 adds lag-1 per-game rates and prior-team usage shares",
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=None,
        help="Completed run whose adopted points forecast is the production reference; default is Phase 2",
    )
    parser.add_argument(
        "--workbook-name", default="fantasy_football_final_rankings_2026.xlsx"
    )
    parser.add_argument(
        "--points-min-season",
        type=int,
        default=2017,
        help="First training season for the extended points learners (plan amendment 1).",
    )
    return parser.parse_args(argv)


def _prefit_manifest(root: Path, run_dir: Path, experiment: str) -> dict[str, Any]:
    plan_path = root / "experiments" / experiment / "plan_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("recorded_before_fit_utc") is None:
        raise ValueError("Phase 4 plan lacks its pre-fit timestamp.")
    injury_dir = root / INJURY_DIRECTORY
    manifest = {
        "manifest_version": 1,
        "recorded_before_fit_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": _sha256(plan_path),
        "code_sha256": {
            relative: _sha256(root / relative) for relative in PHASE4_CODE_PATHS
        },
        "primary_input_sha256": {
            relative: _sha256(root / relative) for relative in PRIMARY_INPUT_PATHS
        },
        "injury_source_sha256": {
            path.name: _sha256(path)
            for path in sorted(injury_dir.glob("injuries_*.parquet"))
        },
        "roster_source_sha256": {
            path.name: _sha256(path)
            for path in sorted(
                (root / "data" / "raw" / "rosters").glob("roster_weekly_*.parquet")
            )
        },
    }
    _write_json(run_dir / "prefit_manifest.json", manifest)
    _write_json(root / "experiments" / experiment / "prefit_manifest_v1.json", manifest)
    return manifest


@dataclass(frozen=True, slots=True)
class PointsReference:
    """The production forecast that a new candidate must not fall below."""

    name: str
    discovery: pd.DataFrame
    audit: pd.DataFrame
    current: pd.DataFrame


def _load_reference(root: Path, reference_run: Path | None) -> PointsReference:
    """Load Phase 2 production or the adopted forecast of a completed Phase 4-style run."""
    if reference_run is None:
        return PointsReference(
            PHASE2_REFERENCE,
            _load_phase2_discovery(root),
            _load_phase2_audit(root, OFFENSE_POSITIONS),
            pd.read_parquet(root / PHASE2_CURRENT_PREDICTIONS).loc[
                :, [*KEY_COLUMNS, "predicted_points"]
            ],
        )
    summary = json.loads(
        (reference_run / "result_summary.json").read_text(encoding="utf-8")
    )
    adopted = summary["points"]["adopted"]
    name = f"{reference_run.parent.parent.name}_production"
    discovery = pd.read_parquet(reference_run / "points_discovery_predictions.parquet")
    audit = pd.read_parquet(reference_run / "points_audit_predictions.parquet")
    current = pd.read_parquet(reference_run / "points_predictions_2026.parquet")
    columns = [*KEY_COLUMNS, "actual_points", "predicted"]
    discovery = discovery.loc[discovery["candidate"].eq(adopted), columns].assign(
        learner=name
    )
    audit = audit.loc[audit["candidate"].eq(adopted), columns].assign(learner=name)
    return PointsReference(
        name,
        discovery.reset_index(drop=True),
        audit.reset_index(drop=True),
        current.loc[
            current["model_position"].isin(OFFENSE_POSITIONS),
            [*KEY_COLUMNS, "predicted_points"],
        ],
    )


def _load_phase2_discovery(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / PHASE2_DISCOVERY_PREDICTIONS)  # (n_rows, 7)
    frame = frame.loc[frame["candidate"].eq(PHASE2_OFFENSE_CANDIDATE)].copy()
    frame.rename(columns={"target_points": "actual_points"}, inplace=True)
    frame["learner"] = PHASE2_REFERENCE
    return frame.loc[:, [*KEY_COLUMNS, "actual_points", "predicted", "learner"]]


def _load_phase2_audit(root: Path, positions: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_parquet(root / PHASE2_AUDIT_PREDICTIONS)  # (n_rows, 9)
    frame = frame.loc[frame["model_position"].isin(positions)].copy()
    frame.rename(columns={"predicted_points": "predicted"}, inplace=True)
    frame["learner"] = PHASE2_REFERENCE
    return frame.loc[:, [*KEY_COLUMNS, "actual_points", "predicted", "learner"]]


def _candidate_table(
    component_predictions: Mapping[str, pd.DataFrame],
    candidates: Sequence[PointsCandidate],
    stage: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    assembled: dict[str, pd.DataFrame] = {}
    fold_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        predictions = assemble_candidate_predictions(component_predictions, candidate)
        summary = points_summary(predictions)
        records.append(
            {
                "stage": stage,
                "candidate": candidate.identifier,
                "components": len(candidate.components),
                "rows": int(len(predictions)),
                **summary,
            }
        )
        assembled[candidate.identifier] = predictions
        fold_frames.append(points_fold_table(predictions, candidate.identifier))
    return (
        pd.DataFrame.from_records(records),
        assembled,
        pd.concat(fold_frames, ignore_index=True),
    )


def _run_points_track(
    bundle: Phase4Bundle,
    root: Path,
    run_dir: Path,
    args: argparse.Namespace,
    reference: PointsReference,
) -> dict[str, Any]:
    """Discovery, audit, null control, and importance for the points track."""
    offense = bundle.keys["model_position"].isin(OFFENSE_POSITIONS)  # (n,)
    observed = bundle.metadata["target_points"].notna()  # (n,)
    rows = offense & observed & bundle.keys["target_season"].ge(2013)  # (n,)
    frame = bundle.frame.loc[rows].reset_index(drop=True)  # (n_pts, d)
    metadata = bundle.metadata.loc[rows].reset_index(drop=True)  # (n_pts, c_meta)
    targets = metadata["target_points"].astype("float64")  # (n_pts,)
    columns = list(frame.columns)
    learners = points_learner_grid()
    candidates = points_candidate_grid(reference.name)
    discovery_seasons = tuple(
        season
        for season in POINTS_DISCOVERY_SEASONS
        if season - int(args.points_min_season) >= 2
    )
    _log(
        f"points discovery seasons {discovery_seasons} with training from {args.points_min_season}"
    )

    _log("points discovery: walk-forward learners")
    discovery_components: dict[str, pd.DataFrame] = {}
    discovery_models: dict[str, dict[int, Any]] = {}
    for learner in learners:
        started = time.time()
        predictions, models = walk_forward_points(
            frame,
            metadata,
            targets,
            learner,
            discovery_seasons,
            minimum_training_season=args.points_min_season,
        )
        discovery_components[learner.identifier] = predictions
        discovery_models[learner.identifier] = models
        _log(f"  {learner.identifier}: {time.time() - started:.1f}s")
    discovery_components[reference.name] = reference.discovery.loc[
        reference.discovery["target_season"].isin(discovery_seasons)
    ].reset_index(drop=True)
    discovery_summary, discovery_predictions, discovery_folds = _candidate_table(
        discovery_components, candidates, "discovery"
    )
    selected = select_points_candidate(discovery_summary, reference.name)
    discovery_summary["selected"] = discovery_summary["candidate"].eq(selected)
    _log(f"points discovery selected: {selected}")
    _log(
        discovery_summary.loc[
            :, ["candidate", "spearman", "mae", "ndcg_at_roster_cutoff"]
        ].to_string(index=False)
    )

    _log("points audit: walk-forward learners")
    audit_components: dict[str, pd.DataFrame] = {}
    audit_models: dict[str, dict[int, Any]] = {}
    for learner in learners:
        predictions, models = walk_forward_points(
            frame,
            metadata,
            targets,
            learner,
            POINTS_AUDIT_SEASONS,
            minimum_training_season=args.points_min_season,
        )
        audit_components[learner.identifier] = predictions
        audit_models[learner.identifier] = models
    audit_components[reference.name] = reference.audit
    audit_summary, audit_predictions, audit_folds = _candidate_table(
        audit_components, candidates, "audit"
    )
    baseline = metadata.loc[
        metadata["target_season"].isin(POINTS_AUDIT_SEASONS),
        [*KEY_COLUMNS, "target_points", "previous_points_baseline"],
    ].rename(
        columns={
            "target_points": "actual_points",
            "previous_points_baseline": "predicted",
        }
    )
    baseline_summary = points_summary(baseline)
    audit_summary = pd.concat(
        [
            audit_summary,
            pd.DataFrame.from_records(
                [
                    {
                        "stage": "audit",
                        "candidate": "previous_points_baseline",
                        "components": 0,
                        "rows": int(len(baseline)),
                        **baseline_summary,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    selected_audit = float(
        audit_summary.loc[audit_summary["candidate"].eq(selected), "spearman"].iloc[0]
    )
    reference_audit = float(
        audit_summary.loc[
            audit_summary["candidate"].eq(reference.name), "spearman"
        ].iloc[0]
    )
    adopted = (
        selected
        if selected_audit >= reference_audit - SELECTION_TOLERANCE
        else reference.name
    )
    audit_summary["selected"] = audit_summary["candidate"].eq(selected)
    audit_summary["adopted"] = audit_summary["candidate"].eq(adopted)
    _log(
        f"points audit: selected {selected} rho={selected_audit:.4f}, {reference.name} rho={reference_audit:.4f}, adopted {adopted}"
    )
    _log(
        audit_summary.loc[
            :, ["candidate", "spearman", "mae", "ndcg_at_roster_cutoff", "top_k_recall"]
        ].to_string(index=False)
    )

    paired = (
        audit_predictions[adopted]
        .loc[:, [*KEY_COLUMNS, "actual_points", "predicted"]]
        .merge(
            audit_predictions[reference.name]
            .loc[:, [*KEY_COLUMNS, "predicted"]]
            .rename(columns={"predicted": "reference"}),
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        .merge(
            baseline.rename(columns={"predicted": "baseline"}).drop(
                columns="actual_points"
            ),
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
    )
    intervals = {
        "adopted_spearman": clustered_spearman_interval(
            paired, "predicted", samples=args.bootstrap_samples
        ),
        "adopted_minus_reference": clustered_spearman_interval(
            paired, "predicted", "reference", samples=args.bootstrap_samples
        ),
        "adopted_minus_previous_points": clustered_spearman_interval(
            paired, "predicted", "baseline", samples=args.bootstrap_samples
        ),
    }
    position_audit = (
        audit_folds.loc[audit_folds["candidate"].isin([adopted, reference.name])]
        .groupby(["candidate", "model_position"], observed=True)[["spearman", "mae"]]
        .mean()
        .reset_index()
    )

    _log("points null control")
    null_records: list[dict[str, Any]] = []
    null_learner = learners[0]
    for offset in range(args.null_seeds):
        seed = RANDOM_SEED + offset
        permuted = permute_within_groups(targets, metadata, seed)
        predictions, _ = walk_forward_points(
            frame,
            metadata,
            targets,
            null_learner,
            discovery_seasons,
            fit_targets=permuted,
            minimum_training_season=args.points_min_season,
        )
        null_records.append(
            {"seed": seed, "spearman": points_summary(predictions)["spearman"]}
        )
    null_table = pd.DataFrame.from_records(null_records)
    null_mean = float(null_table["spearman"].mean())
    _log(f"points null mean rho={null_mean:.4f}")

    importance = pd.DataFrame(
        columns=["learner", "target_season", "feature", "importance"]
    )
    if not args.skip_permutation:
        _log("points permutation importance")
        records: list[pd.DataFrame] = []
        for learner in learners:
            for season, model in discovery_models[learner.identifier].items():
                validation = metadata["target_season"].eq(season)
                validation_frame = frame.loc[validation].reset_index(drop=True)
                predictions = discovery_components[learner.identifier]
                fold_predictions = predictions.loc[
                    predictions["target_season"].eq(season)
                ].reset_index(drop=True)
                scorer = spearman_scorer(fold_predictions)
                table = permutation_importance(
                    lambda data, model=model: predict_points(model, data, columns),
                    validation_frame,
                    scorer,
                    columns,
                    repeats=args.permutation_repeats,
                    seed=RANDOM_SEED + season,
                )
                table["learner"] = learner.identifier
                table["target_season"] = season
                records.append(table)
            _log(f"  {learner.identifier} done")
        importance = pd.concat(records, ignore_index=True)

    ablation = pd.DataFrame(
        columns=["family", "columns", "spearman_without", "spearman_full", "delta"]
    )
    if not args.skip_ablation:
        _log("points family ablation")
        full = float(
            discovery_summary.loc[
                discovery_summary["candidate"].eq("ext_et_all"), "spearman"
            ].iloc[0]
        )
        families = pd.Series(bundle.families)
        records_ab: list[dict[str, Any]] = []
        for family in sorted(families.unique()):
            kept = [column for column in columns if bundle.families[column] != family]
            predictions, _ = walk_forward_points(
                frame,
                metadata,
                targets,
                null_learner,
                discovery_seasons,
                columns=kept,
                minimum_training_season=args.points_min_season,
            )
            score = points_summary(predictions)["spearman"]
            records_ab.append(
                {
                    "family": family,
                    "columns": int(len(columns) - len(kept)),
                    "spearman_without": score,
                    "spearman_full": full,
                    "delta": full - score,
                }
            )
            _log(f"  without {family}: rho={score:.4f}")
        ablation = pd.DataFrame.from_records(records_ab).sort_values(
            "delta", ascending=False
        )

    _log("points production fit")
    adopted_candidate = next(
        candidate for candidate in candidates if candidate.identifier == adopted
    )
    current_mask = bundle.keys["target_season"].eq(2026) & bundle.keys[
        "model_position"
    ].isin(OFFENSE_POSITIONS)
    current_frame = bundle.frame.loc[current_mask]
    current_keys = bundle.keys.loc[current_mask, list(KEY_COLUMNS)].reset_index(
        drop=True
    )
    phase2_current = pd.read_parquet(root / PHASE2_CURRENT_PREDICTIONS)
    component_current: dict[str, np.ndarray] = {}
    production_models: dict[str, Any] = {}
    train_mask = metadata["target_season"].between(args.points_min_season, 2025)
    for component in adopted_candidate.components:
        if component == reference.name:
            merged = current_keys.merge(
                reference.current,
                on=list(KEY_COLUMNS),
                how="left",
                validate="one_to_one",
            )
            if merged["predicted_points"].isna().any():
                raise ValueError(
                    "Reference current predictions do not cover every 2026 offense row."
                )
            component_current[component] = merged["predicted_points"].to_numpy(
                dtype="float64"
            )
            continue
        learner = next(item for item in learners if item.identifier == component)
        model = fit_points_learner(
            frame.loc[train_mask], targets.loc[train_mask], learner, columns
        )
        production_models[component] = model
        component_current[component] = predict_points(model, current_frame, columns)
    current = current_keys.copy()
    current["predicted_points"] = np.clip(
        np.mean(np.column_stack(list(component_current.values())), axis=1), 0.0, None
    )
    current["points_source"] = adopted
    kickers = phase2_current.loc[
        phase2_current["model_position"].eq("K"), [*KEY_COLUMNS, "predicted_points"]
    ].copy()
    kickers["points_source"] = "phase2_production_kicker"
    current = pd.concat([current, kickers], ignore_index=True)

    adopted_audit = audit_predictions[adopted].loc[
        :, [*KEY_COLUMNS, "actual_points", "predicted"]
    ]
    kicker_audit = _load_phase2_audit(root, ("K",)).loc[
        :, [*KEY_COLUMNS, "actual_points", "predicted"]
    ]
    interval_table = residual_interval_table(
        pd.concat([adopted_audit, kicker_audit], ignore_index=True)
    )
    current = apply_residual_intervals(current, interval_table)

    for name, table in (
        ("points_discovery_summary.csv", discovery_summary),
        ("points_discovery_folds.csv", discovery_folds),
        ("points_audit_summary.csv", audit_summary),
        ("points_audit_folds.csv", audit_folds),
        ("points_audit_by_position.csv", position_audit),
        ("points_null_control.csv", null_table),
        ("points_permutation_importance_raw.csv", importance),
        ("points_family_ablation.csv", ablation),
        ("points_residual_intervals.csv", interval_table),
    ):
        table.to_csv(run_dir / name, index=False)
    pd.concat(discovery_predictions.values(), ignore_index=True).to_parquet(
        run_dir / "points_discovery_predictions.parquet", index=False
    )
    pd.concat(audit_predictions.values(), ignore_index=True).to_parquet(
        run_dir / "points_audit_predictions.parquet", index=False
    )
    current.to_parquet(run_dir / "points_predictions_2026.parquet", index=False)
    for component, model in production_models.items():
        joblib.dump(model, run_dir / f"points_model_{component}.joblib")

    return {
        "selected": selected,
        "adopted": adopted,
        "adopted_components": list(adopted_candidate.components),
        "reference": reference.name,
        "discovery_summary": discovery_summary,
        "audit_summary": audit_summary,
        "audit_folds": audit_folds,
        "position_audit": position_audit,
        "intervals": intervals,
        "null_mean_spearman": null_mean,
        "null_table": null_table,
        "importance": importance,
        "ablation": ablation,
        "discovery_predictions": discovery_predictions,
        "audit_predictions": audit_predictions,
        "current": current,
        "interval_table": interval_table,
        "columns": columns,
        "discovery_frame": frame,
        "discovery_metadata": metadata,
        "discovery_targets": targets,
        "discovery_seasons": list(discovery_seasons),
    }


def _injury_fold_metrics(predictions: pd.DataFrame, identifier: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for season, group in predictions.groupby("target_season", observed=True):
        metrics = probability_metrics(
            group["label"], group["probability"].to_numpy(dtype="float64")
        )
        records.append(
            {
                "learner": identifier,
                "target_season": int(season),
                "rows": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(records)


def _exposure_table(
    predictions: pd.DataFrame, frame: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    lookup = metadata.loc[:, [*KEY_COLUMNS]].copy()
    lookup["lag1_games"] = frame["lag1_games"].to_numpy()
    merged = predictions.merge(
        lookup, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
    )
    records: list[dict[str, Any]] = []
    for label, mask in (
        ("all audit candidates", pd.Series(True, index=merged.index)),
        ("prior-year games > 0", merged["lag1_games"].gt(0)),
        ("prior-year games >= 4", merged["lag1_games"].ge(4)),
        ("prior-year games >= 8", merged["lag1_games"].ge(8)),
    ):
        subset = merged.loc[mask]
        records.append(
            {
                "subgroup": label,
                "rows": int(len(subset)),
                "positive_rate": float(subset["label"].mean()),
                "roc_auc": float(roc_auc_score(subset["label"], subset["probability"])),
                "prior_games_auc": float(
                    roc_auc_score(subset["label"], subset["lag1_games"])
                )
                if subset["lag1_games"].nunique() > 1
                else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records)


def _run_injury_track(
    bundle: Phase4Bundle,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Development, audit, null control, and importance for the injury track."""
    labeled = bundle.labeled_mask()  # (n,)
    frame = bundle.frame.loc[labeled].reset_index(drop=True)  # (n_inj, d)
    metadata = bundle.metadata.loc[labeled].reset_index(drop=True)  # (n_inj, c_meta)
    labels = (
        bundle.labels.loc[labeled, PRIMARY_LABEL].astype("int8").reset_index(drop=True)
    )  # (n_inj,)
    secondary = (
        bundle.labels.loc[labeled, SECONDARY_LABEL]
        .astype("int8")
        .reset_index(drop=True)
    )  # (n_inj,)
    columns = list(frame.columns)
    learners = injury_learner_grid()

    _log("injury development")
    development: dict[str, pd.DataFrame] = {}
    development_models: dict[str, dict[int, Any]] = {}
    fold_frames: list[pd.DataFrame] = []
    for learner in learners:
        started = time.time()
        predictions, models = walk_forward_injury(
            frame, metadata, labels, learner, INJURY_DEVELOPMENT_SEASONS
        )
        development[learner.identifier] = predictions
        development_models[learner.identifier] = models
        fold_frames.append(_injury_fold_metrics(predictions, learner.identifier))
        _log(f"  {learner.identifier}: {time.time() - started:.1f}s")
    development_folds = pd.concat(fold_frames, ignore_index=True)
    development_summary = (
        development_folds.groupby("learner", observed=True)[
            [
                "roc_auc",
                "average_precision",
                "log_loss",
                "brier_score",
                "calibration_slope",
            ]
        ]
        .mean()
        .reset_index()
    )
    development_summary["family"] = development_summary["learner"].map(
        {learner.identifier: learner.family for learner in learners}
    )
    selected = select_injury_learner(development_summary)
    development_summary["selected"] = development_summary["learner"].eq(selected)
    _log(f"injury selected: {selected}")
    _log(development_summary.to_string(index=False))
    selected_learner = next(item for item in learners if item.identifier == selected)

    _log("injury audit")
    audit: dict[str, pd.DataFrame] = {}
    audit_fold_frames: list[pd.DataFrame] = []
    audit_learners = [selected_learner] + [
        item
        for item in learners
        if item.family in {"age_position_baseline", "exposure_baseline"}
    ]
    for learner in audit_learners:
        predictions, _ = walk_forward_injury(
            frame, metadata, labels, learner, INJURY_AUDIT_SEASONS
        )
        audit[learner.identifier] = predictions
        audit_fold_frames.append(_injury_fold_metrics(predictions, learner.identifier))
    audit_folds = pd.concat(audit_fold_frames, ignore_index=True)
    pooled = {
        identifier: probability_metrics(
            table["label"], table["probability"].to_numpy(dtype="float64")
        )
        for identifier, table in audit.items()
    }
    audit_summary = pd.DataFrame.from_records(
        [
            {"learner": identifier, "rows": int(len(audit[identifier])), **metrics}
            for identifier, metrics in pooled.items()
        ]
    )
    audit_summary["selected"] = audit_summary["learner"].eq(selected)
    selected_audit = audit[selected]
    auc_interval = clustered_metric_interval(
        selected_audit,
        lambda table: float(roc_auc_score(table["label"], table["probability"])),
        samples=args.bootstrap_samples,
    )
    paired_auc = clustered_metric_interval(
        selected_audit.merge(
            audit["age_position_logistic"]
            .loc[:, [*KEY_COLUMNS, "probability"]]
            .rename(columns={"probability": "baseline"}),
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        ),
        lambda table: float(
            roc_auc_score(table["label"], table["probability"])
            - roc_auc_score(table["label"], table["baseline"])
        ),
        samples=args.bootstrap_samples,
    )
    position_auc = (
        selected_audit.groupby("model_position", observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "rows": int(len(group)),
                    "positive_rate": float(group["label"].mean()),
                    "roc_auc": float(
                        roc_auc_score(group["label"], group["probability"])
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    exposure = _exposure_table(selected_audit, frame, metadata)
    secondary_predictions, _ = walk_forward_injury(
        frame, metadata, secondary, selected_learner, INJURY_AUDIT_SEASONS
    )
    secondary_metrics = probability_metrics(
        secondary_predictions["label"],
        secondary_predictions["probability"].to_numpy(dtype="float64"),
    )
    success = bool(
        auc_interval["lower_95"] > 0.5
        and pooled[selected]["roc_auc"] - pooled["age_position_logistic"]["roc_auc"]
        >= 0.02
    )
    _log(
        audit_summary.loc[
            :,
            [
                "learner",
                "roc_auc",
                "average_precision",
                "log_loss",
                "brier_score",
                "calibration_intercept",
                "calibration_slope",
            ],
        ].to_string(index=False)
    )
    _log(f"injury audit interval: {auc_interval}")

    _log("injury null control")
    null_records: list[dict[str, Any]] = []
    for offset in range(args.null_seeds):
        seed = RANDOM_SEED + 100 + offset
        permuted = permute_within_groups(labels, metadata, seed)
        predictions, _ = walk_forward_injury(
            frame,
            metadata,
            labels,
            selected_learner,
            INJURY_DEVELOPMENT_SEASONS,
            fit_labels=permuted,
        )
        null_records.append(
            {
                "seed": seed,
                "pooled_roc_auc": float(
                    roc_auc_score(predictions["label"], predictions["probability"])
                ),
                "mean_season_roc_auc": float(
                    _injury_fold_metrics(predictions, selected)["roc_auc"].mean()
                ),
            }
        )
    null_table = pd.DataFrame.from_records(null_records)
    null_mean = float(null_table["pooled_roc_auc"].mean())
    _log(
        f"injury null mean pooled AUC={null_mean:.4f}, "
        f"mean within-season AUC={float(null_table['mean_season_roc_auc'].mean()):.4f}"
    )

    importance = pd.DataFrame(
        columns=["learner", "target_season", "feature", "importance"]
    )
    if not args.skip_permutation:
        _log("injury permutation importance")
        records: list[pd.DataFrame] = []
        for season, model in development_models[selected].items():
            validation = metadata["target_season"].eq(season)
            validation_frame = frame.loc[validation].reset_index(drop=True)
            fold_labels = labels.loc[validation].to_numpy(dtype="int8")
            table = permutation_importance(
                lambda data, model=model: model.predict_proba(data),
                validation_frame,
                auc_scorer(fold_labels),
                columns,
                repeats=args.permutation_repeats,
                seed=RANDOM_SEED + 200 + season,
            )
            table["learner"] = selected
            table["target_season"] = season
            records.append(table)
        importance = pd.concat(records, ignore_index=True)

    ablation = pd.DataFrame(
        columns=["family", "columns", "roc_auc_without", "roc_auc_full", "delta"]
    )
    if not args.skip_ablation:
        _log("injury family ablation")
        full = float(
            development_summary.loc[
                development_summary["learner"].eq(selected), "roc_auc"
            ].iloc[0]
        )
        records_ab: list[dict[str, Any]] = []
        for family in sorted(set(bundle.families.values())):
            kept = [column for column in columns if bundle.families[column] != family]
            if selected_learner.family in {
                "age_position_baseline",
                "exposure_baseline",
            }:
                break
            predictions, _ = walk_forward_injury(
                frame,
                metadata,
                labels,
                selected_learner,
                INJURY_DEVELOPMENT_SEASONS,
                columns=kept,
            )
            score = float(_injury_fold_metrics(predictions, selected)["roc_auc"].mean())
            records_ab.append(
                {
                    "family": family,
                    "columns": int(len(columns) - len(kept)),
                    "roc_auc_without": score,
                    "roc_auc_full": full,
                    "delta": full - score,
                }
            )
            _log(f"  without {family}: AUC={score:.4f}")
        ablation = pd.DataFrame.from_records(records_ab).sort_values(
            "delta", ascending=False
        )

    _log("injury production fit")
    seasons = metadata["target_season"]
    production = fit_calibrated_injury_model(
        frame.loc[seasons.le(LABEL_SEASONS[1] - 1)],
        labels.loc[seasons.le(LABEL_SEASONS[1] - 1)],
        frame.loc[seasons.eq(LABEL_SEASONS[1])],
        labels.loc[seasons.eq(LABEL_SEASONS[1])],
        selected_learner,
        columns,
    )
    current_mask = bundle.keys["target_season"].eq(2026)
    current = bundle.keys.loc[current_mask, list(KEY_COLUMNS)].reset_index(drop=True)
    current["injury_probability"] = production.predict_proba(
        bundle.frame.loc[current_mask]
    )
    joblib.dump(production, run_dir / "injury_model.joblib")

    for name, table in (
        ("injury_development_folds.csv", development_folds),
        ("injury_development_summary.csv", development_summary),
        ("injury_audit_folds.csv", audit_folds),
        ("injury_audit_summary.csv", audit_summary),
        ("injury_audit_by_position.csv", position_auc),
        ("injury_exposure_diagnostic.csv", exposure),
        ("injury_null_control.csv", null_table),
        ("injury_permutation_importance_raw.csv", importance),
        ("injury_family_ablation.csv", ablation),
    ):
        table.to_csv(run_dir / name, index=False)
    pd.concat(development.values(), ignore_index=True).to_parquet(
        run_dir / "injury_development_predictions.parquet", index=False
    )
    pd.concat(audit.values(), ignore_index=True).to_parquet(
        run_dir / "injury_audit_predictions.parquet", index=False
    )
    current.to_parquet(run_dir / "injury_predictions_2026.parquet", index=False)

    return {
        "selected": selected,
        "selected_family": selected_learner.family,
        "development_summary": development_summary,
        "audit_summary": audit_summary,
        "audit_folds": audit_folds,
        "pooled": pooled,
        "auc_interval": auc_interval,
        "paired_auc_interval": paired_auc,
        "position_auc": position_auc,
        "exposure": exposure,
        "secondary_label_metrics": secondary_metrics,
        "success_rule_passed": success,
        "null_mean_auc": null_mean,
        "null_table": null_table,
        "importance": importance,
        "ablation": ablation,
        "development_predictions": development[selected],
        "audit_predictions": selected_audit,
        "baseline_audit_predictions": audit["age_position_logistic"],
        "current": current,
        "labels_frame": frame,
        "labels_metadata": metadata,
        "labels": labels,
    }


def _run_combination(
    points: Mapping[str, Any], injury: Mapping[str, Any]
) -> dict[str, Any]:
    """Choose the default injury weight on development rows and audit it once."""

    def _join(points_frame: pd.DataFrame, injury_frame: pd.DataFrame) -> pd.DataFrame:
        return (
            points_frame.loc[:, [*KEY_COLUMNS, "actual_points", "predicted"]]
            .rename(columns={"predicted": "predicted_points"})
            .merge(
                injury_frame.loc[:, [*KEY_COLUMNS, "probability"]].rename(
                    columns={"probability": "injury_probability"}
                ),
                on=list(KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
            )
        )

    development = _join(
        points["discovery_predictions"][points["selected"]],
        injury["development_predictions"],
    )
    development_table = evaluate_injury_weights(development, INJURY_WEIGHTS)
    default_weight = select_injury_weight(development_table)
    audit = _join(
        points["audit_predictions"][points["adopted"]], injury["audit_predictions"]
    )
    audit_table = evaluate_injury_weights(audit, INJURY_WEIGHTS)
    development_table["stage"] = "development"
    audit_table["stage"] = "audit"
    table = pd.concat([development_table, audit_table], ignore_index=True)
    table["default"] = table["injury_weight"].eq(default_weight)
    _log(f"combination default weight: {default_weight}")
    _log(table.to_string(index=False))
    return {
        "default_weight": default_weight,
        "table": table,
        "development_rows": int(len(development)),
        "audit_rows": int(len(audit)),
    }


def _feature_ranking(
    bundle: Phase4Bundle,
    track: str,
    importance: pd.DataFrame,
    univariate: pd.DataFrame,
    panel: pd.DataFrame,
    featureranker: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine permutation, univariate, and FeatureRanker evidence into one table."""
    families = (
        pd.Series(bundle.families, name="family").rename_axis("feature").reset_index()
    )
    table = families.copy()
    if len(importance):
        per_learner = (
            importance.groupby(["learner", "feature"], observed=True)["importance"]
            .mean()
            .reset_index()
        )
        per_learner["rank"] = per_learner.groupby("learner")["importance"].rank(
            method="average", ascending=False
        )
        agg = (
            per_learner.groupby("feature", observed=True)
            .agg(
                permutation_importance=("importance", "mean"),
                permutation_rank=("rank", "mean"),
            )
            .reset_index()
        )
        sign = importance.groupby("feature", observed=True)["importance"].apply(
            lambda values: float((values > 0).mean())
        )
        agg["permutation_positive_share"] = agg["feature"].map(sign)
        table = table.merge(agg, on="feature", how="left")
    table = table.merge(univariate, on="feature", how="left")
    table = table.merge(panel, on="feature", how="left")
    if featureranker is not None and len(featureranker):
        table = table.merge(featureranker, on="feature", how="left")
    rank_columns = [
        column
        for column in (
            "permutation_rank",
            "univariate_rank",
            "panel_rank",
            "featureranker_rank",
        )
        if column in table.columns
    ]
    table["consensus_rank_score"] = table[rank_columns].mean(axis=1)
    table.sort_values(["consensus_rank_score", "feature"], inplace=True, kind="stable")
    table.insert(0, "consensus_rank", np.arange(1, len(table) + 1))
    table["track"] = track
    return table.reset_index(drop=True)


def _points_univariate(frame: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    y = targets.to_numpy(dtype="float64")
    for column in frame.columns:
        x = frame[column].to_numpy(dtype="float64")
        rho = 0.0 if np.unique(x).size < 2 else float(spearmanr(x, y).statistic)
        records.append(
            {"feature": column, "univariate_spearman": 0.0 if np.isnan(rho) else rho}
        )
    table = pd.DataFrame.from_records(records)
    table["univariate_rank"] = (
        table["univariate_spearman"].abs().rank(method="average", ascending=False)
    )
    return table


def _filter_panel(frame: pd.DataFrame, targets: pd.Series, task: str) -> pd.DataFrame:
    """Mutual information and random-forest impurity importance on discovery rows."""
    # frame: (n, d); targets: (n,)
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    X = frame.to_numpy(dtype="float64")  # (n, d)
    y = targets.to_numpy()  # (n,)
    started = time.time()
    with threadpool_limits(limits=MAX_THREADS):
        if task == "regression":
            information = mutual_info_regression(
                X, y, random_state=RANDOM_SEED, n_neighbors=5
            )
            forest = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                max_features=0.33,
                n_jobs=MAX_THREADS,
                random_state=RANDOM_SEED,
            )
        else:
            information = mutual_info_classif(
                X, y.astype("int8"), random_state=RANDOM_SEED, n_neighbors=5
            )
            forest = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=20,
                max_features=0.33,
                class_weight="balanced_subsample",
                n_jobs=MAX_THREADS,
                random_state=RANDOM_SEED,
            )
        forest.fit(X, y)
    panel = pd.DataFrame(
        {
            "feature": list(frame.columns),
            "mutual_information": information,
            "rf_impurity_importance": forest.feature_importances_,
        }
    )  # (d, 3)
    panel["panel_rank"] = (
        panel["mutual_information"].rank(method="average", ascending=False)
        + panel["rf_impurity_importance"].rank(method="average", ascending=False)
    ) / 2.0
    _log(f"filter panel {task}: {time.time() - started:.1f}s")
    return panel


def _run_featureranker(
    frame: pd.DataFrame,
    targets: pd.Series,
    task: str,
    run_dir: Path | None = None,
    track: str = "",
) -> pd.DataFrame | None:
    """FeatureRanker 3.0.4 probe-weighted reciprocal-rank consensus on discovery rows."""
    try:
        import featureranker
        from featureranker import feature_ranking, voting
    except ImportError:
        _log("featureranker unavailable")
        return None
    version = tuple(int(part) for part in featureranker.__version__.split(".")[:2])
    if version < (3, 0):
        _log(
            f"featureranker {featureranker.__version__} lacks the RankingResult API; skipped"
        )
        return None
    numeric = frame.astype("float64").reset_index(drop=True)
    constant = numeric.columns[numeric.nunique(dropna=False).le(1)]
    usable = numeric.drop(columns=constant)
    started = time.time()
    with threadpool_limits(limits=MAX_THREADS):
        result = feature_ranking(
            usable,
            targets.reset_index(drop=True),
            task=task,
            n_jobs=MAX_THREADS,
            random_state=RANDOM_SEED,
            probe=True,
        )
    primary = voting(result, weights="auto", method="reciprocal_rank").rename(
        columns={"score": "featureranker_score"}
    )
    primary = primary.loc[:, ["feature", "featureranker_score"]].copy()
    primary["featureranker_rank"] = np.arange(1, len(primary) + 1)
    _log(
        f"featureranker {featureranker.__version__} {task}: {time.time() - started:.1f}s"
    )
    if run_dir is not None:
        ranks = result.rank_matrix().rename_axis("feature").reset_index()
        ranks = ranks.rename(
            columns={
                method: f"featureranker_rank_{method}" for method in result.methods
            }
        )
        scores = result.score_matrix().rename_axis("feature").reset_index()
        scores = scores.rename(
            columns={
                method: f"featureranker_score_{method}" for method in result.methods
            }
        )
        detail = primary.merge(ranks, on="feature", how="left").merge(
            scores, on="feature", how="left"
        )
        detail.to_csv(run_dir / f"{track}_featureranker_ranking.csv", index=False)
        result.probe_table().rename_axis("method").reset_index().to_csv(
            run_dir / f"{track}_featureranker_probes.csv", index=False
        )
        equal = voting(result, weights=None, method="reciprocal_rank")
        borda = voting(result, weights="auto", method="borda")
        sensitivity = {
            "package_version": featureranker.__version__,
            "methods": list(result.methods),
            "rows": int(len(usable)),
            "features": int(usable.shape[1]),
            "constant_features_removed": list(map(str, constant)),
            "top_20_auto_vs_equal_overlap": int(
                len(set(primary.head(20)["feature"]) & set(equal.head(20)["feature"]))
            ),
            "top_20_reciprocal_vs_borda_overlap": int(
                len(set(primary.head(20)["feature"]) & set(borda.head(20)["feature"]))
            ),
            "probes": json.loads(
                result.probe_table()
                .rename_axis("method")
                .reset_index()
                .to_json(orient="records")
            ),
        }
        _write_json(run_dir / f"{track}_featureranker_summary.json", sensitivity)
    return primary


def _plot_top_features(
    table: pd.DataFrame, title: str, path: Path, top_n: int = 25
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    value_column = "permutation_importance"
    if value_column not in table.columns:
        value_column = next(
            column
            for column in ("univariate_spearman", "discriminative_auc")
            if column in table.columns
        )
    plotted = table.sort_values(value_column, ascending=False).head(top_n).iloc[::-1]
    figure = Figure(
        figsize=(10, 8.5), constrained_layout=True, facecolor=COLORS["surface"]
    )
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.set_facecolor(COLORS["surface"])
    values = plotted[value_column].fillna(0.0).abs()
    axis.barh(plotted["feature"], values, color=COLORS["blue"], height=0.62)
    axis.set_title(title, loc="left", color=COLORS["text"], fontsize=12)
    axis.set_xlabel(
        "Mean held-out permutation importance (metric drop)", color=COLORS["muted"]
    )
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=COLORS["muted"], labelsize=8.5)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(COLORS["grid"])
    for index, (value, rank) in enumerate(
        zip(values, plotted["consensus_rank"], strict=True)
    ):
        if index >= len(values) - 5:
            axis.text(
                value,
                index,
                f"  consensus #{int(rank)}",
                va="center",
                fontsize=8,
                color=COLORS["muted"],
            )
    figure.savefig(path, dpi=300)


def _plot_ablation(points: pd.DataFrame, injury: pd.DataFrame, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(11, 5.5), constrained_layout=True, facecolor=COLORS["surface"]
    )
    FigureCanvasAgg(figure)
    for index, (table, title, column, color) in enumerate(
        (
            (
                points,
                "Points track: Spearman drop when a family is removed",
                "delta",
                COLORS["blue"],
            ),
            (
                injury,
                "Injury track: ROC AUC drop when a family is removed",
                "delta",
                COLORS["orange"],
            ),
        ),
        start=1,
    ):
        axis = figure.add_subplot(1, 2, index)
        axis.set_facecolor(COLORS["surface"])
        if len(table):
            plotted = table.sort_values(column).copy()
            axis.barh(plotted["family"], plotted[column], color=color, height=0.62)
        axis.axvline(0.0, color=COLORS["muted"], linewidth=1.0)
        axis.set_title(title, loc="left", color=COLORS["text"], fontsize=10.5)
        axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=COLORS["muted"], labelsize=8.5)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color(COLORS["grid"])
    figure.savefig(path, dpi=300)


def _plot_injury_audit(
    selected: pd.DataFrame, baseline: pd.DataFrame, path: Path
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(11, 5.2), constrained_layout=True, facecolor=COLORS["surface"]
    )
    FigureCanvasAgg(figure)
    roc_axis = figure.add_subplot(1, 2, 1)
    roc_axis.set_facecolor(COLORS["surface"])
    for table, label, color in (
        (selected, "Selected model", COLORS["blue"]),
        (baseline, "Age + position baseline", COLORS["orange"]),
    ):
        fpr, tpr, _ = roc_curve(table["label"], table["probability"])
        auc = roc_auc_score(table["label"], table["probability"])
        roc_axis.plot(
            fpr, tpr, color=color, linewidth=2.0, label=f"{label} (AUC {auc:.3f})"
        )
    roc_axis.plot([0, 1], [0, 1], color=COLORS["grid"], linewidth=1.0, linestyle="--")
    roc_axis.set_xlabel("False positive rate", color=COLORS["muted"])
    roc_axis.set_ylabel("True positive rate", color=COLORS["muted"])
    roc_axis.set_title(
        "2022-2024 audit ROC: missed-time injury",
        loc="left",
        color=COLORS["text"],
        fontsize=10.5,
    )
    roc_axis.legend(frameon=False, fontsize=8.5)
    calibration_axis = figure.add_subplot(1, 2, 2)
    calibration_axis.set_facecolor(COLORS["surface"])
    observed, predicted = calibration_curve(
        selected["label"], selected["probability"], n_bins=10, strategy="quantile"
    )
    calibration_axis.plot(
        [0, 1], [0, 1], color=COLORS["grid"], linewidth=1.0, linestyle="--"
    )
    calibration_axis.plot(
        predicted,
        observed,
        color=COLORS["blue"],
        linewidth=2.0,
        marker="o",
        markersize=5,
    )
    calibration_axis.set_xlabel(
        "Predicted probability (decile mean)", color=COLORS["muted"]
    )
    calibration_axis.set_ylabel("Observed missed-time rate", color=COLORS["muted"])
    calibration_axis.set_title(
        "Calibration on the audit seasons",
        loc="left",
        color=COLORS["text"],
        fontsize=10.5,
    )
    for axis in (roc_axis, calibration_axis):
        axis.grid(color=COLORS["grid"], linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=COLORS["muted"], labelsize=8.5)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.savefig(path, dpi=300)


def _plot_weights(table: pd.DataFrame, default_weight: float, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(11, 4.8), constrained_layout=True, facecolor=COLORS["surface"]
    )
    FigureCanvasAgg(figure)
    for index, (metric, title) in enumerate(
        (
            ("spearman", "Rank accuracy vs realized points"),
            ("bust_rate_top_k", "Bust rate among top-k picks"),
        ),
        start=1,
    ):
        axis = figure.add_subplot(1, 2, index)
        axis.set_facecolor(COLORS["surface"])
        for stage, color in (
            ("development", COLORS["blue"]),
            ("audit", COLORS["orange"]),
        ):
            rows = table.loc[table["stage"].eq(stage)]
            axis.plot(
                rows["injury_weight"],
                rows[metric],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=5,
                label=stage,
            )
        axis.axvline(
            default_weight, color=COLORS["muted"], linewidth=1.0, linestyle="--"
        )
        axis.set_title(title, loc="left", color=COLORS["text"], fontsize=10.5)
        axis.set_xlabel("Injury weight in the draft score", color=COLORS["muted"])
        axis.grid(color=COLORS["grid"], linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=COLORS["muted"], labelsize=8.5)
        axis.legend(frameon=False, fontsize=8.5)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.savefig(path, dpi=300)


def _plot_points_audit(
    folds: pd.DataFrame, adopted: str, reference: str, path: Path
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(11, 4.8), constrained_layout=True, facecolor=COLORS["surface"]
    )
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.set_facecolor(COLORS["surface"])
    series = [(reference, COLORS["orange"])]
    if adopted != reference:
        series.insert(0, (adopted, COLORS["blue"]))
    positions = list(OFFENSE_POSITIONS)
    width = 0.36 if len(series) > 1 else 0.6
    for offset, (candidate, color) in enumerate(series):
        rows = (
            folds.loc[folds["candidate"].eq(candidate)]
            .groupby("model_position")["spearman"]
            .mean()
        )
        values = [rows.get(position, np.nan) for position in positions]
        x = np.arange(len(positions)) + (offset - (len(series) - 1) / 2) * width
        axis.bar(x, values, width=width, color=color, label=candidate)
        for xi, value in zip(x, values, strict=True):
            axis.text(
                xi,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                fontsize=8,
                color=COLORS["muted"],
            )
    axis.set_xticks(np.arange(len(positions)), positions)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(
        "Mean within-position-season Spearman, 2022-2025", color=COLORS["muted"]
    )
    axis.set_title(
        "Points track audit by position",
        loc="left",
        color=COLORS["text"],
        fontsize=10.5,
    )
    axis.legend(frameon=False, fontsize=8.5)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=COLORS["muted"], labelsize=8.5)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.savefig(path, dpi=300)


def _build_board(
    bundle: Phase4Bundle,
    root: Path,
    points: Mapping[str, Any],
    injury: Mapping[str, Any],
    default_weight: float,
) -> pd.DataFrame:
    """Join 2026 forecasts, injury probabilities, screen flags, and metadata."""
    current_mask = bundle.keys["target_season"].eq(2026)
    board = bundle.metadata.loc[
        current_mask,
        [
            *KEY_COLUMNS,
            "team",
            "candidate_name",
            "roster_status",
            "available_for_draft",
            "is_rookie",
        ],
    ].reset_index(drop=True)
    age = bundle.frame.loc[current_mask, "age"].to_numpy(dtype="float64")
    missing_age = bundle.frame.loc[current_mask, "missing_age"].to_numpy(
        dtype="float64"
    )
    board["age"] = np.where(missing_age > 0, np.nan, age)
    board = board.merge(
        points["current"].loc[
            :,
            [
                *KEY_COLUMNS,
                "predicted_points",
                "points_p10",
                "points_p90",
                "points_source",
            ],
        ],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    board = board.merge(
        injury["current"], on=list(KEY_COLUMNS), how="left", validate="one_to_one"
    )
    if (
        board["predicted_points"].isna().any()
        or board["injury_probability"].isna().any()
    ):
        raise ValueError(
            "Every 2026 candidate needs both a points forecast and an injury probability."
        )
    board = board.loc[board["available_for_draft"].astype("bool")].reset_index(
        drop=True
    )
    board["injury_probability"] = board["injury_probability"].clip(0.0, 1.0)
    board["health_probability"] = 1.0 - board["injury_probability"]
    board["points_percentile"] = board.groupby("model_position", observed=True)[
        "predicted_points"
    ].rank(method="average", pct=True)
    board["position_points_rank"] = (
        board.groupby("model_position", observed=True)["predicted_points"]
        .rank(method="min", ascending=False)
        .astype("int32")
    )
    board["draft_score"] = combined_score(
        board["points_percentile"], board["health_probability"], default_weight
    ).round(2)
    replacement_points: dict[str, float] = {}
    for position, group in board.groupby("model_position", observed=True):
        ordered = (
            group["predicted_points"].sort_values(ascending=False).to_numpy()
        )  # (n_position,)
        index = min(TOP_K.get(str(position), 12), len(ordered)) - 1
        replacement_points[str(position)] = float(ordered[index])
    board["replacement_points"] = board["model_position"].map(replacement_points)
    board["vorp"] = (board["predicted_points"] - board["replacement_points"]).round(2)
    board["overall_value"] = (
        board["vorp"].clip(lower=0.0) / float(board["vorp"].max())
    ).round(4)
    position_health = board.groupby("model_position", observed=True)[
        "health_probability"
    ].transform("mean")  # (n,)
    board["relative_health"] = (
        board["health_probability"] - position_health + 0.5
    ).clip(0.0, 1.0)
    board["overall_draft_score"] = combined_score(
        board["overall_value"], board["relative_health"], default_weight
    ).round(2)
    board["risk_band"] = pd.cut(
        board["injury_probability"],
        bins=[-0.001, RISK_BAND_EDGES[0], RISK_BAND_EDGES[1], 1.001],
        labels=["lower", "moderate", "higher"],
    ).astype("string")
    board["points_confidence"] = np.where(
        board["model_position"].eq("K"), "low (kicker)", "standard"
    )
    board["is_rookie"] = board["is_rookie"].astype("int8").map({0: "no", 1: "yes"})

    screen = pd.read_csv(root / CURRENT_INJURY_SCREEN)
    screen_columns = [
        "player_id",
        "current_injury_listing",
        "current_injury_or_designation",
        "current_injury_status",
        "expected_return",
        "evidence_date",
        "source_publisher",
        "source_tier",
        "source_url",
        "status_as_of",
    ]
    screen = screen.loc[:, screen_columns].drop_duplicates("player_id")
    board = board.merge(screen, on="player_id", how="left", validate="one_to_one")
    board["current_injury_listing"] = (
        board["current_injury_listing"].fillna(False).astype("bool")
    )
    if (board["status_as_of"].dropna().astype("string") != CURRENT_INJURY_AS_OF).any():
        raise ValueError(
            "Current injury screen dates do not match the declared as-of date."
        )
    board.sort_values(
        ["overall_draft_score", "predicted_points", "player_id"],
        ascending=[False, False, True],
        inplace=True,
        kind="stable",
    )
    board.reset_index(drop=True, inplace=True)
    return board


def _readme_rows(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    points = summary["points"]
    injury = summary["injury"]
    combination = summary["combination"]
    return [
        (
            "Purpose",
            "Full-season 2026 ESPN full-PPR point forecasts and a calibrated probability of a missed-time injury for every fantasy-position player on the August 9, 2026 roster snapshot, combined into a transparent draft score.",
        ),
        (
            "Forecast origin",
            "Inputs are frozen as of 2026-08-09. No 2026 statistics, depth charts, injury reports, projections, or ADP enter either model. The current-injury listing columns come from a separate dated screen (2026-08-14) and do not change the models.",
        ),
        (
            "Overall Draft Score",
            f"Cross-position ranking used on the ALL page. Points Over Replacement = projected points minus the projected points of the replacement-level player at the position (QB12, RB24, WR36, TE12, K12). Overall Value = points over replacement divided by the largest points over replacement on the board (negative values count as 0), so magnitudes are preserved across positions. It is blended with the player's healthy probability relative to position peers: 100 * ((1 - w) * overall value + w * (healthy probability - position mean healthy probability + 0.5)), w = {combination['default_weight']:.2f}. Centering on the position mean stops low-exposure positions such as kickers from gaining a blanket bonus. The custom-weight column on ALL uses the overall value and relative health columns.",
        ),
        (
            "Position Draft Score",
            f"Within-position ranking used on the QB, RB, WR, TE, and K pages: 100 * ((1 - w) * position points percentile + w * healthy probability) with the default w = {combination['default_weight']:.2f}, the largest weight whose discovery-fold rank accuracy against realized points stayed within 0.005 Spearman of the best weight. Edit SETTINGS!B2 to recompute the custom-weight column on every page.",
        ),
        (
            "Projected 2026 Points",
            f"Points track forecast: {points['adopted']} (components: {', '.join(points['adopted_components'])}). Locked 2022-2025 audit mean within-position-season Spearman {points['audit_spearman']:.3f} (95% player-cluster interval {points['audit_interval'][0]:.3f} to {points['audit_interval'][1]:.3f}); prior-season points baseline {points['baseline_spearman']:.3f}. Kicker forecasts retain the Phase 2 production model and are labeled low confidence.",
        ),
        (
            "Points P10 / P90",
            "Empirical 10th and 90th percentile bounds from 2022-2025 audit residuals of the adopted procedure within position and predicted-point tercile. About 80% of realized totals fell inside this band on the audit seasons; it is not a guarantee.",
        ),
        (
            "Missed-Time Injury Probability",
            f"Probability that the player appears on an official report as Out or Doubtful with a physical injury, or is placed on an injury reserve list, at least once during the 2026 regular season. Model: {injury['selected']}; 2022-2024 audit ROC AUC {injury['audit_auc']:.3f} (95% interval {injury['auc_interval'][0]:.3f} to {injury['auc_interval'][1]:.3f}), calibration slope {injury['calibration_slope']:.2f}, prevalence {injury['prevalence']:.1%}. The label is an observed absence event, not severity or games missed, and participation is part of the signal.",
        ),
        (
            "Injury Risk Band",
            f"lower < {RISK_BAND_EDGES[0]:.0%}, moderate {RISK_BAND_EDGES[0]:.0%} to {RISK_BAND_EDGES[1]:.0%}, higher > {RISK_BAND_EDGES[1]:.0%} predicted probability.",
        ),
        (
            "Current Injury Listing",
            "TRUE when the 2026-08-14 public screen matched an affirmative injury listing. FALSE is not evidence of health. The INJURED page repeats these players with their source details.",
        ),
        (
            "Sheets",
            "ALL and INJURED sort by the overall draft score; QB, RB, WR, TE, and K sort by the position draft score. MODEL_EVAL, WEIGHT_ANALYSIS, FEATURES_POINTS, FEATURES_INJURY, and FAMILY_ABLATION hold the evaluation evidence and feature rankings.",
        ),
        (
            "Leakage controls",
            "Historical candidates are prior-season roster members plus draft or combine entrants; nonparticipants keep zero points. Features use only seasons before the target. Report-based injury history uses offsets of two or more seasons because the official feed ends in 2024; roster reserve history uses offsets of one or more seasons. Model selection used 2016-2021 (points) and 2017-2021 (injury); audits ran once on 2022-2025 and 2022-2024. Negative controls: points null mean Spearman {:.4f}; injury null mean AUC {:.4f}.".format(
                points["null_mean_spearman"], injury["null_mean_auc"]
            ),
        ),
        (
            "Limits",
            "Chronological but not analyst-blinded audits. Historical August rosters are reconstructed. The 2026 season is the prospective test. The draft score is a preference index, not expected value or medical advice.",
        ),
        ("Generated", summary["generated_at_utc"]),
    ]


def _results_markdown(
    summary: Mapping[str, Any],
    points: Mapping[str, Any],
    injury: Mapping[str, Any],
    combination: Mapping[str, Any],
    rankings: Mapping[str, pd.DataFrame],
) -> str:
    discovery = points["discovery_summary"].loc[
        :,
        [
            "candidate",
            "components",
            "spearman",
            "mae",
            "ndcg_at_roster_cutoff",
            "top_k_recall",
        ],
    ]
    audit = points["audit_summary"].loc[
        :, ["candidate", "spearman", "mae", "ndcg_at_roster_cutoff", "top_k_recall"]
    ]
    position = points["position_audit"]
    dev = injury["development_summary"].loc[
        :, ["learner", "roc_auc", "average_precision", "log_loss", "brier_score"]
    ]
    iaudit = injury["audit_summary"].loc[
        :,
        [
            "learner",
            "roc_auc",
            "average_precision",
            "log_loss",
            "brier_score",
            "calibration_intercept",
            "calibration_slope",
            "prevalence",
        ],
    ]
    weights = combination["table"].loc[
        :,
        [
            "stage",
            "injury_weight",
            "spearman",
            "ndcg_at_roster_cutoff",
            "top_k_recall",
            "bust_rate_top_k",
        ],
    ]
    lines = [
        "# Phase 4 methods and results",
        "",
        "## Answer",
        "",
        f"The final 2026 ranking uses the `{points['adopted']}` points forecast and the `{injury['selected']}` missed-time injury classifier. "
        f"On the locked 2022-2025 audit the adopted points procedure reached mean within-position-season Spearman {summary['points']['audit_spearman']:.4f} "
        f"(player-cluster 95% interval {summary['points']['audit_interval'][0]:.4f} to {summary['points']['audit_interval'][1]:.4f}) against {summary['points']['reference_audit_spearman']:.4f} for the `{summary['points']['reference']}` reference "
        f"and {summary['points']['baseline_spearman']:.4f} for prior-season points. "
        f"The injury classifier reached pooled 2022-2024 ROC AUC {summary['injury']['audit_auc']:.4f} (95% interval {summary['injury']['auc_interval'][0]:.4f} to {summary['injury']['auc_interval'][1]:.4f}) "
        f"at {summary['injury']['prevalence']:.1%} prevalence with calibration slope {summary['injury']['calibration_slope']:.3f}.",
        "",
        f"The default injury weight in the draft score is {combination['default_weight']:.2f}. The workbook is `outputs/{summary['workbook_name']}`.",
        "",
        "## Injury label",
        "",
        "A player-season is positive when the player is listed Out or Doubtful with a physical injury on any regular-season official report, or spends any regular-season roster week on an injury-compatible reserve list. "
        "Questionable-only and practice-only designations are negative. COVID, retired, did-not-report, left-squad, future, and suspension reserve codes are excluded. "
        f"The label covers 2013-2024. On the audit seasons the selected learner reached ROC AUC {injury['secondary_label_metrics']['roc_auc']:.4f} against the Phase 3 any-designation label, for comparison.",
        "",
        "## Points track",
        "",
        f"Discovery ({points['discovery_seasons'][0]}-{points['discovery_seasons'][-1]}, training from "
        "the amended window) selected `{points['selected']}`; "
        f"the audit adoption rule adopted `{points['adopted']}`.",
        "",
        discovery.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Locked 2022-2025 audit:",
        "",
        audit.to_markdown(index=False, floatfmt=".4f"),
        "",
        position.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Paired gain of the adopted procedure over the `{points['reference']}` reference: {points['intervals']['adopted_minus_reference']['estimate']:.4f} "
        f"({points['intervals']['adopted_minus_reference']['lower_95']:.4f} to {points['intervals']['adopted_minus_reference']['upper_95']:.4f}). "
        f"Gain over prior-season points: {points['intervals']['adopted_minus_previous_points']['estimate']:.4f} "
        f"({points['intervals']['adopted_minus_previous_points']['lower_95']:.4f} to {points['intervals']['adopted_minus_previous_points']['upper_95']:.4f}). "
        f"Null control mean Spearman over {len(points['null_table'])} target permutations: {points['null_mean_spearman']:.4f}.",
        "",
        "## Injury track",
        "",
        "Development (2017-2021):",
        "",
        dev.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Locked 2022-2024 audit:",
        "",
        iaudit.to_markdown(index=False, floatfmt=".4f"),
        "",
        injury["position_auc"].to_markdown(index=False, floatfmt=".4f"),
        "",
        injury["exposure"].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Paired AUC gain over the age-position baseline: {injury['paired_auc_interval']['estimate']:.4f} "
        f"({injury['paired_auc_interval']['lower_95']:.4f} to {injury['paired_auc_interval']['upper_95']:.4f}). "
        f"Success rule passed: {injury['success_rule_passed']}. Null control mean AUC over {len(injury['null_table'])} label permutations: {injury['null_mean_auc']:.4f}.",
        "",
        "## Combination",
        "",
        weights.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Feature ranking",
        "",
        "Points track top 20 by consensus rank:",
        "",
        rankings["points"]
        .head(20)
        .loc[
            :,
            [
                "consensus_rank",
                "feature",
                "family",
                "permutation_importance",
                "univariate_spearman",
            ],
        ]
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "Injury track top 20 by consensus rank:",
        "",
        rankings["injury"]
        .head(20)
        .loc[
            :,
            [
                "consensus_rank",
                "feature",
                "family",
                "permutation_importance",
                "discriminative_auc",
            ],
        ]
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "Family ablation (points):",
        "",
        points["ablation"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Family ablation (injury):",
        "",
        injury["ablation"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Limits",
        "",
        "- Audits are chronological and procedure-locked but not analyst-blinded; earlier phases inspected the same seasons.",
        "- Historical August roster membership is reconstructed from the prior season, not observed.",
        "- The injury label records observed absence events; players who play through injuries or leave the league are negatives.",
        "- The official injury feed ends in 2024, so report-based history is at least two seasons old; roster reserve history fills the one-season gap.",
        "- The draft score is a preference index. Expected value also depends on lineup rules, replacement level, and league scoring.",
        "",
    ]
    return "\n".join(lines)


def _write_outputs(
    root: Path,
    run_dir: Path,
    bundle: Phase4Bundle,
    points_current: pd.DataFrame,
    injury_current: pd.DataFrame,
    summary: dict[str, Any],
    tables: Mapping[str, pd.DataFrame],
    results_markdown: str | None,
) -> Path:
    """Build the board, write the workbook, copy artifacts, and hash the run."""
    board = _build_board(
        bundle,
        root,
        {"current": points_current},
        {"current": injury_current},
        summary["combination"]["default_weight"],
    )
    board.to_parquet(run_dir / "final_rankings_2026.parquet", index=False)
    board.to_csv(run_dir / "final_rankings_2026.csv", index=False)
    summary["rows"]["current_players"] = int(len(board))
    summary["replacement_points"] = (
        board.drop_duplicates("model_position")
        .set_index("model_position")["replacement_points"]
        .round(2)
        .to_dict()
    )
    _write_json(run_dir / "result_summary.json", summary)
    workbook_path = root / "outputs" / summary["workbook_name"]
    write_final_workbook(
        board,
        workbook_path,
        default_weight=summary["combination"]["default_weight"],
        readme_rows=_readme_rows(summary),
        tables=tables,
    )
    _log(f"workbook {workbook_path}")
    experiment_dir = root / "experiments" / summary["experiment"]
    if results_markdown is not None:
        (experiment_dir / "RESULTS.md").write_text(
            results_markdown, encoding="utf-8", newline="\n"
        )
    artifacts = experiment_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_COPIES:
        shutil.copy2(run_dir / name, artifacts / name)
    hashes = {
        path.name: _sha256(path) for path in sorted(run_dir.iterdir()) if path.is_file()
    }
    _write_json(run_dir / "output_hashes.json", hashes)
    return workbook_path


def _consensus(table: pd.DataFrame) -> pd.DataFrame:
    """Recompute the consensus rank from whichever rank columns are present."""
    rank_columns = [
        column
        for column in (
            "permutation_rank",
            "univariate_rank",
            "panel_rank",
            "featureranker_rank",
        )
        if column in table.columns
    ]
    table = table.drop(
        columns=[
            column
            for column in ("consensus_rank", "consensus_rank_score")
            if column in table.columns
        ]
    )
    table["consensus_rank_score"] = table[rank_columns].mean(axis=1)
    table.sort_values(["consensus_rank_score", "feature"], inplace=True, kind="stable")
    table.insert(0, "consensus_rank", np.arange(1, len(table) + 1))
    return table.reset_index(drop=True)


def _refresh_featureranker(
    root: Path, run_dir: Path, bundle: Phase4Bundle, summary: dict[str, Any]
) -> None:
    """Rerun FeatureRanker on the saved run's discovery rows and rewrite the ranking tables."""
    minimum = int(summary["points_minimum_training_season"])
    last_discovery = int(summary["points_discovery_seasons"][-1])
    offense = bundle.keys["model_position"].isin(OFFENSE_POSITIONS)
    observed = bundle.metadata["target_points"].notna()
    points_rows = (
        offense
        & observed
        & bundle.keys["target_season"].between(minimum, last_discovery)
    )
    points_fr = _run_featureranker(
        bundle.frame.loc[points_rows],
        bundle.metadata.loc[points_rows, "target_points"].astype("float64"),
        "regression",
        run_dir,
        "points",
    )
    injury_rows = bundle.labeled_mask() & bundle.keys["target_season"].le(
        INJURY_DEVELOPMENT_SEASONS[-1]
    )
    injury_fr = _run_featureranker(
        bundle.frame.loc[injury_rows],
        bundle.labels.loc[injury_rows, PRIMARY_LABEL].astype("int8"),
        "classification",
        run_dir,
        "injury",
    )
    for track, fresh in (("points", points_fr), ("injury", injury_fr)):
        path = run_dir / f"{track}_feature_ranking.csv"
        table = pd.read_csv(path)
        table = table.drop(
            columns=[
                column
                for column in table.columns
                if column.startswith("featureranker_")
            ]
        )
        if fresh is not None:
            table = table.merge(fresh, on="feature", how="left")
        table = _consensus(table)
        table.to_csv(path, index=False)
        _plot_top_features(
            table,
            f"{track.capitalize()} track: top features by held-out permutation importance",
            run_dir / f"{track}_top_features.png",
        )
        summary["feature_ranking"][f"{track}_top_20"] = table.head(20)[
            "feature"
        ].tolist()
    summary["feature_ranking"]["featureranker_used"] = bool(
        points_fr is not None and injury_fr is not None
    )
    _write_json(run_dir / "result_summary.json", summary)


def _rebuild(
    root: Path,
    run_dir: Path,
    *,
    refresh_featureranker: bool = False,
    workbook_name: str | None = None,
) -> Path:
    """Regenerate the board and workbook from a completed run without refitting."""
    summary = json.loads((run_dir / "result_summary.json").read_text(encoding="utf-8"))
    summary.setdefault("experiment", "phase4")
    summary.setdefault("feature_set", "phase4")
    summary.setdefault("workbook_name", "fantasy_football_final_rankings_2026.xlsx")
    bundle = build_phase4_bundle(root, root / INJURY_DIRECTORY)
    if summary["feature_set"] == "phase5":
        bundle = extend_with_usage_features(bundle, root)
    if refresh_featureranker:
        _refresh_featureranker(root, run_dir, bundle, summary)
    points_current = pd.read_parquet(run_dir / "points_predictions_2026.parquet")
    injury_current = pd.read_parquet(run_dir / "injury_predictions_2026.parquet")
    tables = {
        "MODEL_EVAL": pd.read_csv(run_dir / "model_eval.csv"),
        "WEIGHT_ANALYSIS": pd.read_csv(run_dir / "weight_analysis.csv"),
        "FEATURES_POINTS": pd.read_csv(run_dir / "points_feature_ranking.csv"),
        "FEATURES_INJURY": pd.read_csv(run_dir / "injury_feature_ranking.csv"),
        "FAMILY_ABLATION": pd.read_csv(run_dir / "family_ablation.csv"),
    }
    summary["generated_at_utc"] = datetime.now(UTC).isoformat()
    if workbook_name is not None:
        summary["workbook_name"] = workbook_name
    return _write_outputs(
        root, run_dir, bundle, points_current, injury_current, summary, tables, None
    )


ARTIFACT_COPIES = (
    "result_summary.json",
    "points_feature_ranking.csv",
    "injury_feature_ranking.csv",
    "points_family_ablation.csv",
    "injury_family_ablation.csv",
    "points_audit_summary.csv",
    "injury_audit_summary.csv",
    "injury_exposure_diagnostic.csv",
    "label_prevalence_by_season.csv",
    "weight_analysis.csv",
    "model_eval.csv",
    "family_ablation.csv",
    "final_rankings_2026.csv",
    "final_rankings_2026.parquet",
    "points_top_features.png",
    "injury_top_features.png",
    "points_featureranker_ranking.csv",
    "injury_featureranker_ranking.csv",
    "points_featureranker_summary.json",
    "injury_featureranker_summary.json",
    "family_ablation.png",
    "injury_audit_roc_calibration.png",
    "injury_weight_tradeoff.png",
    "points_audit_by_position.png",
)


def main(argv: Sequence[str] | None = None) -> Path:
    """Run every Phase 4 stage and write the workbook and records."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = _parse_args(argv)
    root = args.root.resolve()
    if args.rebuild_from is not None:
        _write = _rebuild(
            root,
            args.rebuild_from.resolve(),
            refresh_featureranker=args.refresh_featureranker,
            workbook_name=args.workbook_name,
        )
        _log("done")
        return _write
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (
        root / "experiments" / args.experiment / "runs" / f"{stamp}-{args.run_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    _log(f"run directory {run_dir}")
    manifest = _prefit_manifest(root, run_dir, args.experiment)

    bundle = build_phase4_bundle(root, root / INJURY_DIRECTORY)
    if args.feature_set == "phase5":
        bundle = extend_with_usage_features(bundle, root)
    _log(f"bundle {bundle.frame.shape} feature set {args.feature_set}")
    reference = _load_reference(
        root, None if args.reference_run is None else args.reference_run.resolve()
    )
    _log(f"points reference {reference.name}")
    label_summary = (
        bundle.labels.loc[bundle.labeled_mask()]
        .groupby("target_season")[
            [PRIMARY_LABEL, SECONDARY_LABEL, "absence_reported", "reserve_injury"]
        ]
        .mean()
        .reset_index()
    )
    label_summary.to_csv(run_dir / "label_prevalence_by_season.csv", index=False)

    points = _run_points_track(bundle, root, run_dir, args, reference)
    injury = _run_injury_track(bundle, run_dir, args)
    combination = _run_combination(points, injury)

    _log("feature ranking: univariate and FeatureRanker")
    discovery_seasons = points["discovery_metadata"]["target_season"].between(
        args.points_min_season, POINTS_DISCOVERY_SEASONS[-1]
    )
    points_univariate = _points_univariate(
        points["discovery_frame"].loc[discovery_seasons],
        points["discovery_targets"].loc[discovery_seasons],
    )
    injury_window = injury["labels_metadata"]["target_season"].le(
        INJURY_DEVELOPMENT_SEASONS[-1]
    )
    injury_univariate = (
        feature_association_table(
            injury["labels_frame"].loc[injury_window],
            injury["labels"].loc[injury_window],
        )
        .loc[
            :,
            [
                "feature",
                "univariate_auc",
                "discriminative_auc",
                "direction",
                "spearman_rho",
                "q_value",
                "association_rank",
            ],
        ]
        .rename(columns={"association_rank": "univariate_rank"})
    )
    points_fr = (
        None
        if args.skip_featureranker
        else _run_featureranker(
            points["discovery_frame"].loc[discovery_seasons],
            points["discovery_targets"].loc[discovery_seasons],
            "regression",
            run_dir,
            "points",
        )
    )
    injury_fr = (
        None
        if args.skip_featureranker
        else _run_featureranker(
            injury["labels_frame"].loc[injury_window],
            injury["labels"].loc[injury_window],
            "classification",
            run_dir,
            "injury",
        )
    )
    points_panel = _filter_panel(
        points["discovery_frame"].loc[discovery_seasons],
        points["discovery_targets"].loc[discovery_seasons],
        "regression",
    )
    injury_panel = _filter_panel(
        injury["labels_frame"].loc[injury_window],
        injury["labels"].loc[injury_window],
        "classification",
    )
    rankings = {
        "points": _feature_ranking(
            bundle,
            "points",
            points["importance"],
            points_univariate,
            points_panel,
            points_fr,
        ),
        "injury": _feature_ranking(
            bundle,
            "injury",
            injury["importance"],
            injury_univariate,
            injury_panel,
            injury_fr,
        ),
    }
    rankings["points"].to_csv(run_dir / "points_feature_ranking.csv", index=False)
    rankings["injury"].to_csv(run_dir / "injury_feature_ranking.csv", index=False)

    _log("plots")
    _plot_top_features(
        rankings["points"],
        "Points track: top features by held-out permutation importance",
        run_dir / "points_top_features.png",
    )
    _plot_top_features(
        rankings["injury"],
        "Injury track: top features by held-out permutation importance",
        run_dir / "injury_top_features.png",
    )
    _plot_ablation(
        points["ablation"], injury["ablation"], run_dir / "family_ablation.png"
    )
    _plot_injury_audit(
        injury["audit_predictions"],
        injury["baseline_audit_predictions"],
        run_dir / "injury_audit_roc_calibration.png",
    )
    _plot_weights(
        combination["table"],
        combination["default_weight"],
        run_dir / "injury_weight_tradeoff.png",
    )
    _plot_points_audit(
        points["audit_folds"],
        points["adopted"],
        points["reference"],
        run_dir / "points_audit_by_position.png",
    )

    selected_points_audit = points["audit_summary"].set_index("candidate")
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_directory": run_dir.relative_to(root).as_posix(),
        "plan_sha256": manifest["plan_sha256"],
        "experiment": args.experiment,
        "feature_set": args.feature_set,
        "workbook_name": args.workbook_name,
        "points_minimum_training_season": int(args.points_min_season),
        "points_discovery_seasons": points["discovery_seasons"],
        "rows": {
            "labeled_player_seasons": int(bundle.labeled_mask().sum()),
            "current_players": 0,
        },
        "points": {
            "selected": points["selected"],
            "adopted": points["adopted"],
            "adopted_components": points["adopted_components"],
            "discovery_spearman": float(
                points["discovery_summary"]
                .set_index("candidate")
                .loc[points["selected"], "spearman"]
            ),
            "audit_spearman": float(
                selected_points_audit.loc[points["adopted"], "spearman"]
            ),
            "reference": points["reference"],
            "reference_audit_spearman": float(
                selected_points_audit.loc[points["reference"], "spearman"]
            ),
            "baseline_spearman": float(
                selected_points_audit.loc["previous_points_baseline", "spearman"]
            ),
            "audit_mae": float(selected_points_audit.loc[points["adopted"], "mae"]),
            "audit_interval": [
                points["intervals"]["adopted_spearman"]["lower_95"],
                points["intervals"]["adopted_spearman"]["upper_95"],
            ],
            "intervals": points["intervals"],
            "null_mean_spearman": points["null_mean_spearman"],
            "position_audit": points["position_audit"].to_dict(orient="records"),
        },
        "injury": {
            "selected": injury["selected"],
            "label": PRIMARY_LABEL,
            "development_auc": float(
                injury["development_summary"]
                .set_index("learner")
                .loc[injury["selected"], "roc_auc"]
            ),
            "audit_auc": injury["pooled"][injury["selected"]]["roc_auc"],
            "audit_average_precision": injury["pooled"][injury["selected"]][
                "average_precision"
            ],
            "audit_log_loss": injury["pooled"][injury["selected"]]["log_loss"],
            "audit_brier": injury["pooled"][injury["selected"]]["brier_score"],
            "calibration_intercept": injury["pooled"][injury["selected"]][
                "calibration_intercept"
            ],
            "calibration_slope": injury["pooled"][injury["selected"]][
                "calibration_slope"
            ],
            "prevalence": injury["pooled"][injury["selected"]]["prevalence"],
            "baseline_auc": injury["pooled"]["age_position_logistic"]["roc_auc"],
            "exposure_baseline_auc": injury["pooled"]["prior_games_logistic"][
                "roc_auc"
            ],
            "auc_interval": [
                injury["auc_interval"]["lower_95"],
                injury["auc_interval"]["upper_95"],
            ],
            "paired_auc_interval": injury["paired_auc_interval"],
            "secondary_label_audit": injury["secondary_label_metrics"],
            "success_rule_passed": injury["success_rule_passed"],
            "null_mean_auc": injury["null_mean_auc"],
            "null_mean_within_season_auc": float(
                injury["null_table"]["mean_season_roc_auc"].mean()
            ),
            "position_auc": injury["position_auc"].to_dict(orient="records"),
            "exposure": injury["exposure"].to_dict(orient="records"),
        },
        "combination": {
            "default_weight": combination["default_weight"],
            "development_rows": combination["development_rows"],
            "audit_rows": combination["audit_rows"],
            "table": combination["table"].to_dict(orient="records"),
        },
        "feature_ranking": {
            "points_top_20": rankings["points"].head(20)["feature"].tolist(),
            "injury_top_20": rankings["injury"].head(20)["feature"].tolist(),
            "featureranker_used": bool(points_fr is not None and injury_fr is not None),
        },
    }

    model_eval = pd.concat(
        [
            points["discovery_summary"]
            .assign(track="points")
            .loc[
                :,
                [
                    "track",
                    "stage",
                    "candidate",
                    "components",
                    "rows",
                    "spearman",
                    "mae",
                    "ndcg_at_roster_cutoff",
                    "top_k_recall",
                    "selected",
                ],
            ],
            points["audit_summary"]
            .assign(track="points")
            .loc[
                :,
                [
                    "track",
                    "stage",
                    "candidate",
                    "components",
                    "rows",
                    "spearman",
                    "mae",
                    "ndcg_at_roster_cutoff",
                    "top_k_recall",
                    "selected",
                    "adopted",
                ],
            ],
            injury["development_summary"]
            .assign(track="injury", stage="development")
            .rename(columns={"learner": "candidate"})
            .loc[
                :,
                [
                    "track",
                    "stage",
                    "candidate",
                    "roc_auc",
                    "average_precision",
                    "log_loss",
                    "brier_score",
                    "selected",
                ],
            ],
            injury["audit_summary"]
            .assign(track="injury", stage="audit")
            .rename(columns={"learner": "candidate"})
            .loc[
                :,
                [
                    "track",
                    "stage",
                    "candidate",
                    "rows",
                    "roc_auc",
                    "average_precision",
                    "log_loss",
                    "brier_score",
                    "calibration_intercept",
                    "calibration_slope",
                    "prevalence",
                    "selected",
                ],
            ],
        ],
        ignore_index=True,
    )
    ablation = pd.concat(
        [
            points["ablation"].assign(track="points"),
            injury["ablation"].assign(track="injury"),
        ],
        ignore_index=True,
    )
    model_eval.to_csv(run_dir / "model_eval.csv", index=False)
    combination["table"].to_csv(run_dir / "weight_analysis.csv", index=False)
    ablation.to_csv(run_dir / "family_ablation.csv", index=False)
    tables = {
        "MODEL_EVAL": model_eval,
        "WEIGHT_ANALYSIS": combination["table"],
        "FEATURES_POINTS": rankings["points"],
        "FEATURES_INJURY": rankings["injury"],
        "FAMILY_ABLATION": ablation,
    }
    results = _results_markdown(summary, points, injury, combination, rankings)
    _write_outputs(
        root,
        run_dir,
        bundle,
        points["current"],
        injury["current"],
        summary,
        tables,
        results,
    )
    _log("done")
    return run_dir


if __name__ == "__main__":
    main()
