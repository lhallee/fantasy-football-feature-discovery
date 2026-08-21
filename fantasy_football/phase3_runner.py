"""Run the locked Phase 3 injury-risk experiment and build 2026 draft outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import joblib
import numpy as np
import pandas as pd

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

from .phase2_safety import MAX_COMBINED_BYTES
from .phase3_draft import (
    DEFAULT_FANTASY_WEIGHT,
    combine_draft_scores,
    write_risk_adjusted_workbook,
)
from .phase3_injury import (
    InjuryFeatureBundle,
    build_injury_feature_bundle,
    download_injury_reports,
)
from .phase3_modeling import (
    AUDIT_SEASONS,
    DEVELOPMENT_SEASONS,
    RANDOM_SEED,
    InjuryCandidate,
    candidate_grid,
    clustered_auc_interval,
    evaluate_candidate,
    feature_association_table,
    fit_calibrated_model,
    permute_labels_within_season_position,
    probability_metrics,
    select_candidate,
    summarize_candidates,
)


PHASE3_CODE_PATHS = (
    "fantasy_football/phase3_injury.py",
    "fantasy_football/phase3_modeling.py",
    "fantasy_football/phase3_draft.py",
    "fantasy_football/phase3_runner.py",
    "fantasy_football/phase2_august.py",
    "fantasy_football/phase2_features.py",
    "fantasy_football/phase2_safety.py",
)
PRIMARY_INPUT_PATHS = (
    "data/processed/player_seasons.parquet",
    "data/processed/preseason_players.parquet",
    "artifacts/predictions_2026.parquet",
    "experiments/phase3/plan_v1.json",
)
NULL_SEEDS = tuple(RANDOM_SEED + offset for offset in range(11))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the August-origin physical injury-report experiment."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "command",
        choices=("download", "run", "all"),
        help="Download injury data, run from existing data, or do both.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _candidate_by_identifier(identifier: str) -> InjuryCandidate:
    matches = [candidate for candidate in candidate_grid() if candidate.identifier == identifier]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate Phase 3 candidate: {identifier!r}")
    return matches[0]


def _prefit_manifest(root: Path, injury_directory: Path) -> dict[str, Any]:
    """Record plan, code, and data hashes immediately before any fit."""
    plan_path = root / "experiments" / "phase3" / "plan_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("recorded_before_fit_utc") is None:
        raise ValueError("Phase 3 plan lacks its pre-fit timestamp.")
    code_hashes = {
        relative: _sha256(root / relative) for relative in PHASE3_CODE_PATHS
    }
    input_hashes = {
        relative: _sha256(root / relative) for relative in PRIMARY_INPUT_PATHS
    }
    injury_hashes = {
        path.name: _sha256(path)
        for path in sorted(injury_directory.glob("injuries_*.parquet"))
    }
    if len(injury_hashes) != 16:
        raise ValueError("Phase 3 requires all 16 injury-report seasons from 2009-2024.")
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "recorded_before_fit_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": _sha256(plan_path),
        "code_sha256": code_hashes,
        "primary_input_sha256": input_hashes,
        "injury_source_sha256": injury_hashes,
        "phase2_raw_input_sha256": _tree_hashes(root, root / "data" / "raw"),
        "phase2_processed_input_sha256": _tree_hashes(
            root,
            root / "data" / "processed",
        ),
        "inherited_phase1_freeze_verified": False,
        "inherited_freeze_note": (
            "The relocated repository contains committed post-Phase-1 documentation. "
            "Phase 3 therefore binds every input it uses instead of claiming the older "
            "whole-repository freeze still matches."
        ),
    }
    _write_json(root / "experiments" / "phase3" / "prefit_manifest_v1.json", manifest)
    return manifest


def _tree_hashes(root: Path, directory: Path) -> dict[str, str]:
    """Hash every regular file below one declared input directory."""
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _verify_prefit_inputs(
    root: Path,
    injury_directory: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Fail if any declared code or data input changed during the run."""
    current_code = {
        relative: _sha256(root / relative) for relative in PHASE3_CODE_PATHS
    }
    current_primary = {
        relative: _sha256(root / relative) for relative in PRIMARY_INPUT_PATHS
    }
    current_injury = {
        path.name: _sha256(path)
        for path in sorted(injury_directory.glob("injuries_*.parquet"))
    }
    comparisons = (
        ("code", current_code, manifest["code_sha256"]),
        ("primary input", current_primary, manifest["primary_input_sha256"]),
        ("injury source", current_injury, manifest["injury_source_sha256"]),
        (
            "Phase 2 raw input",
            _tree_hashes(root, root / "data" / "raw"),
            manifest["phase2_raw_input_sha256"],
        ),
        (
            "Phase 2 processed input",
            _tree_hashes(root, root / "data" / "processed"),
            manifest["phase2_processed_input_sha256"],
        ),
    )
    for label, current, expected in comparisons:
        if current != expected:
            raise RuntimeError(f"Phase 3 {label} hashes changed during the run.")


def _modeling_tables(
    bundle: InjuryFeatureBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return the labeled historical feature, metadata, and outcome partitions."""
    # bundle.frame: (n_all, d); bundle.metadata: (n_all, c_metadata)
    labeled_mask = bundle.metadata["target_season"].between(2013, 2024)  # (n_all,)
    frame = bundle.frame.loc[labeled_mask].copy()  # (n_labeled, d)
    metadata = bundle.metadata.loc[labeled_mask].copy()  # (n_labeled, c_metadata)
    labels = bundle.labels.loc[
        labeled_mask, "physical_injury_reported"
    ].copy()  # (n_labeled,)
    frame.reset_index(drop=True, inplace=True)
    metadata.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)
    return frame, metadata, labels


def _run_development(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Evaluate the complete predeclared grid and lock one candidate."""
    # frame: (n_labeled, d); metadata: (n_labeled, c_metadata); labels: (n_labeled,)
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    for candidate in candidate_grid():
        folds, predictions, features = evaluate_candidate(
            frame,
            metadata,
            labels,
            candidate,
            DEVELOPMENT_SEASONS,
        )
        fold_frames.append(folds)
        prediction_frames.append(predictions)
        feature_frames.append(features)
    fold_metrics = pd.concat(fold_frames, ignore_index=True)  # (n_candidates * f, c_metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)  # (sum_candidates n_dev, c_predictions)
    selected_features = pd.concat(feature_frames, ignore_index=True)  # (sum_selected, 4)
    summary = summarize_candidates(fold_metrics)  # (n_candidates, c_summary)
    selected_identifier = select_candidate(summary)
    summary["selected"] = summary["candidate"].eq(selected_identifier)  # (n_candidates,)
    return fold_metrics, predictions, selected_features, summary, selected_identifier


def _run_negative_control(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: pd.Series,
    candidate: InjuryCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the selected procedure on grouped target permutations."""
    # frame: (n_labeled, d); metadata: (n_labeled, c_metadata); labels: (n_labeled,)
    summary_records: list[dict[str, float | int]] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in NULL_SEEDS:
        permuted = permute_labels_within_season_position(labels, metadata, seed)  # (n_labeled,)
        folds, predictions, _ = evaluate_candidate(
            frame,
            metadata,
            labels,
            candidate,
            DEVELOPMENT_SEASONS,
            fit_labels=permuted,
        )
        pooled_auc = float(
            roc_auc_score(
                predictions["physical_injury_reported"],
                predictions["injury_probability"],
            )
        )
        summary_records.append(
            {
                "seed": int(seed),
                "pooled_roc_auc": pooled_auc,
                "mean_fold_roc_auc": float(folds["roc_auc"].mean()),
            }
        )
        predictions["seed"] = int(seed)  # (n_dev,)
        prediction_frames.append(predictions)
    summary = pd.DataFrame.from_records(summary_records)  # (11, 3)
    predictions = pd.concat(prediction_frames, ignore_index=True)  # (11 * n_dev, c_predictions + 1)
    null_mean = float(summary["pooled_roc_auc"].mean())
    if not 0.45 <= null_mean <= 0.55:
        raise RuntimeError(
            f"Phase 3 negative control failed: mean pooled ROC AUC={null_mean:.6f}."
        )
    return summary, predictions


def _run_audit(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: pd.Series,
    candidate: InjuryCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate the locked candidate and age-position baseline once on audit years."""
    # frame: (n_labeled, d); metadata: (n_labeled, c_metadata); labels: (n_labeled,)
    selected_folds, selected_predictions, selected_features = evaluate_candidate(
        frame,
        metadata,
        labels,
        candidate,
        AUDIT_SEASONS,
    )
    baseline = _candidate_by_identifier("age_position_logistic")
    baseline_folds, baseline_predictions, _ = evaluate_candidate(
        frame,
        metadata,
        labels,
        baseline,
        AUDIT_SEASONS,
    )
    selected_metrics = probability_metrics(
        selected_predictions["physical_injury_reported"],
        selected_predictions["injury_probability"].to_numpy(dtype="float64"),
    )
    baseline_metrics = probability_metrics(
        baseline_predictions["physical_injury_reported"],
        baseline_predictions["injury_probability"].to_numpy(dtype="float64"),
    )
    interval = clustered_auc_interval(selected_predictions)
    report: dict[str, Any] = {
        "validation_seasons": list(AUDIT_SEASONS),
        "selected_candidate": candidate.identifier,
        "selected_pooled": selected_metrics,
        "baseline_pooled": baseline_metrics,
        "roc_auc_gain_over_baseline": (
            selected_metrics["roc_auc"] - baseline_metrics["roc_auc"]
        ),
        "clustered_roc_auc_interval": interval,
        "success_rule_passed": bool(
            interval["lower_95"] > 0.5
            and selected_metrics["roc_auc"] - baseline_metrics["roc_auc"] >= 0.02
        ),
    }
    predictions = selected_predictions.copy()  # (n_audit, c_predictions)
    baseline_lookup = baseline_predictions.loc[
        :,
        ["target_season", "player_id", "injury_probability"],
    ].rename(columns={"injury_probability": "baseline_injury_probability"})  # (n_audit, 3)
    predictions = predictions.merge(
        baseline_lookup,
        on=["target_season", "player_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )  # (n_audit, c_predictions + 1)
    return selected_folds, predictions, selected_features, report


def _fit_production(
    root: Path,
    bundle: InjuryFeatureBundle,
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    labels: pd.Series,
    candidate: InjuryCandidate,
) -> tuple[Any, pd.DataFrame]:
    """Fit through 2023, calibrate on 2024, and score the 2026 candidates."""
    # bundle.frame: (n_all, d); frame: (n_labeled, d); labels: (n_labeled,)
    seasons = metadata["target_season"]  # (n_labeled,)
    base_mask = seasons.le(2023)  # (n_labeled,)
    calibration_mask = seasons.eq(2024)  # (n_labeled,)
    model = fit_calibrated_model(
        frame.loc[base_mask],
        labels.loc[base_mask],
        frame.loc[calibration_mask],
        labels.loc[calibration_mask],
        candidate,
    )
    current_mask = bundle.metadata["target_season"].eq(2026)  # (n_all,)
    injury_probability = model.predict_proba(bundle.frame.loc[current_mask])  # (n_current,)
    risk = bundle.metadata.loc[
        current_mask,
        [
            "target_season",
            "player_id",
            "model_position",
            "team",
            "candidate_name",
            "roster_status",
            "available_for_draft",
            "is_rookie",
        ],
    ].copy()  # (n_current, 8)
    risk["injury_probability"] = injury_probability  # (n_current,)
    fantasy = pd.read_parquet(
        root / "artifacts" / "predictions_2026.parquet",
        columns=["target_season", "player_id", "model_position", "predicted_points"],
    )  # (n_current, 4)
    combined = risk.merge(
        fantasy,
        on=["target_season", "player_id", "model_position"],
        how="left",
        validate="one_to_one",
        sort=False,
    )  # (n_current, 10)
    if combined["predicted_points"].isna().any() or len(combined) != len(fantasy):
        raise ValueError("Phase 3 and production Phase 2 current-player keys do not match.")
    board = combine_draft_scores(
        combined,
        fantasy_weight=DEFAULT_FANTASY_WEIGHT,
    )  # (n_current, c_board)
    for fantasy_weight in (0.7, 0.9):
        label = int(round(100 * fantasy_weight))
        board[f"combined_draft_score_{label}_{100 - label}"] = 100.0 * (
            fantasy_weight * board["fantasy_score_percentile"]
            + (1.0 - fantasy_weight) * board["health_probability"]
        )  # (n_current,)
    return model, board


def _plot_roc(audit_predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot selected and baseline audit ROC curves at 300 dpi."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 6.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    y = audit_predictions["physical_injury_reported"].to_numpy(dtype="int8")  # (n_audit,)
    for column, label, color in (
        ("injury_probability", "Selected model", "#17365D"),
        ("baseline_injury_probability", "Age + position baseline", "#D97904"),
    ):
        p = audit_predictions[column].to_numpy(dtype="float64")  # (n_audit,)
        false_positive, true_positive, _ = roc_curve(y, p)  # (t,), (t,), (t,)
        auc = roc_auc_score(y, p)
        axis.plot(
            false_positive,
            true_positive,
            color=color,
            linewidth=2.4,
            label=f"{label} (AUC={auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.2)
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="Physical injury-report classification, 2022-2024 audit",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="lower right", frameon=False)
    axis.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, format="png")


def _plot_calibration(audit_predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot quantile-binned audit calibration at 300 dpi."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 6.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    y = audit_predictions["physical_injury_reported"].to_numpy(dtype="int8")  # (n_audit,)
    for column, label, color in (
        ("injury_probability", "Selected model", "#17365D"),
        ("baseline_injury_probability", "Age + position baseline", "#D97904"),
    ):
        p = audit_predictions[column].to_numpy(dtype="float64")  # (n_audit,)
        observed, predicted = calibration_curve(
            y,
            p,
            n_bins=10,
            strategy="quantile",
        )  # (b,), (b,)
        axis.plot(predicted, observed, marker="o", color=color, linewidth=2.0, label=label)
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.2)
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed physical injury-report rate",
        title="Probability calibration, 2022-2024 audit",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="upper left", frameon=False)
    axis.grid(alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, format="png")


def _plot_associations(associations: pd.DataFrame, output_path: Path) -> None:
    """Plot the strongest discovery-only univariate associations at 300 dpi."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    top = associations.head(20).sort_values(
        "discriminative_auc", ascending=True
    )  # (20, c_associations)
    effect = top["discriminative_auc"].to_numpy(dtype="float64") - 0.5  # (20,)
    colors = np.where(top["direction"].eq("higher_risk"), "#D97904", "#2F6B9A")  # (20,)
    figure = Figure(figsize=(9.5, 7.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.barh(top["feature"], effect, color=colors)
    axis.set(
        xlabel="Absolute univariate ROC AUC distance from 0.5",
        ylabel="",
        title="Strongest discovery-period injury-report associations",
    )
    axis.grid(axis="x", alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, format="png")


def _output_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "output_hashes.json"
    }


def _project_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


def run_phase3(root: Path) -> Path:
    """Execute the predeclared discovery, controls, audit, and 2026 fit."""
    root = root.resolve()
    phase3 = root / "experiments" / "phase3"
    injury_directory = phase3 / "data" / "raw"
    if not (injury_directory / "manifest.json").is_file():
        raise FileNotFoundError("Run Phase 3 download before fitting.")
    prefit = _prefit_manifest(root, injury_directory)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_directory = phase3 / "runs" / f"{timestamp}-injury-risk-v1"
    run_directory.mkdir(parents=True, exist_ok=False)

    bundle = build_injury_feature_bundle(root, injury_directory)
    frame, metadata, labels = _modeling_tables(bundle)
    fold_metrics, development_predictions, development_features, candidate_summary, selected_id = (
        _run_development(frame, metadata, labels)
    )
    selected_candidate = _candidate_by_identifier(selected_id)
    null_summary, null_predictions = _run_negative_control(
        frame,
        metadata,
        labels,
        selected_candidate,
    )
    audit_folds, audit_predictions, audit_features, audit_report = _run_audit(
        frame,
        metadata,
        labels,
        selected_candidate,
    )

    discovery_mask = metadata["target_season"].le(max(DEVELOPMENT_SEASONS))  # (n_labeled,)
    associations = feature_association_table(
        frame.loc[discovery_mask],
        labels.loc[discovery_mask],
    )  # (d, 13)
    production_model, board = _fit_production(
        root,
        bundle,
        frame,
        metadata,
        labels,
        selected_candidate,
    )
    available_board = board.loc[board["available_for_draft"]].copy()  # (n_available, c_board)

    fold_metrics.to_csv(run_directory / "development_fold_metrics.csv", index=False)
    candidate_summary.to_csv(run_directory / "candidate_summary.csv", index=False)
    development_predictions.loc[
        development_predictions["candidate"].eq(selected_id)
    ].to_parquet(run_directory / "development_predictions.parquet", index=False)
    development_features.loc[
        development_features["candidate"].eq(selected_id)
    ].to_csv(run_directory / "development_selected_features.csv", index=False)
    null_summary.to_csv(run_directory / "negative_control_summary.csv", index=False)
    null_predictions.to_parquet(run_directory / "negative_control_predictions.parquet", index=False)
    audit_folds.to_csv(run_directory / "audit_fold_metrics.csv", index=False)
    audit_predictions.to_parquet(run_directory / "audit_predictions.parquet", index=False)
    audit_features.to_csv(run_directory / "audit_selected_features.csv", index=False)
    associations.to_csv(run_directory / "feature_associations.csv", index=False)
    board.to_parquet(run_directory / "injury_adjusted_players_2026.parquet", index=False)
    available_board.to_csv(run_directory / "injury_adjusted_draft_board_2026.csv", index=False)
    joblib.dump(production_model, run_directory / "injury_model.joblib", compress=3)
    _write_json(run_directory / "audit_report.json", audit_report)
    _write_json(
        run_directory / "feature_lineage.json",
        {
            column: {
                "sources": list(bundle.lineage[column].sources),
                "source_offsets": list(bundle.lineage[column].source_offsets),
                "recipe": bundle.lineage[column].recipe,
            }
            for column in production_model.feature_columns
        },
    )
    _plot_roc(audit_predictions, run_directory / "audit_roc.png")
    _plot_calibration(audit_predictions, run_directory / "audit_calibration.png")
    _plot_associations(associations, run_directory / "feature_associations.png")

    output_workbook = root / "outputs" / "injury_adjusted_draft_board_2026.xlsx"
    write_risk_adjusted_workbook(available_board, output_workbook)
    artifact_directory = phase3 / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "injury_adjusted_players_2026.parquet",
        "injury_adjusted_draft_board_2026.csv",
        "audit_roc.png",
        "audit_calibration.png",
        "feature_associations.png",
        "feature_associations.csv",
        "audit_report.json",
    ):
        shutil.copy2(run_directory / name, artifact_directory / name)

    result_summary: dict[str, Any] = {
        "phase": 3,
        "run_directory": run_directory.relative_to(root).as_posix(),
        "plan_sha256": prefit["plan_sha256"],
        "selected_candidate": selected_id,
        "selected_feature_count": len(production_model.feature_columns),
        "development_mean_roc_auc": float(
            candidate_summary.loc[
                candidate_summary["candidate"].eq(selected_id), "mean_roc_auc"
            ].iloc[0]
        ),
        "negative_control_mean_pooled_roc_auc": float(
            null_summary["pooled_roc_auc"].mean()
        ),
        "audit": audit_report,
        "current_players": int(len(board)),
        "available_current_players": int(len(available_board)),
        "feature_count_available": int(bundle.frame.shape[1]),
        "injury_source_latest_season": 2024,
        "injury_history_offsets": [2, 3, 4, 5],
        "combined_score_fantasy_weight": DEFAULT_FANTASY_WEIGHT,
        "combined_score_interpretation": (
            "Draft preference index, not expected fantasy points or medical advice."
        ),
        "workbook": output_workbook.relative_to(root).as_posix(),
    }
    _write_json(run_directory / "result_summary.json", result_summary)
    _write_json(phase3 / "result_summary.json", result_summary)
    _verify_prefit_inputs(root, injury_directory, prefit)
    output_hashes = _output_hashes(run_directory)
    _write_json(run_directory / "output_hashes.json", output_hashes)
    total_bytes = _project_bytes(root)
    if total_bytes >= 1_000_000_000 or total_bytes >= MAX_COMBINED_BYTES:
        raise RuntimeError(f"Project storage exceeds its stricter size cap: {total_bytes:,} bytes.")
    return run_directory


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected Phase 3 workflow stage."""
    arguments = _parse_args(argv)
    root = arguments.root.resolve()
    injury_directory = root / "experiments" / "phase3" / "data" / "raw"
    if arguments.command in {"download", "all"}:
        manifest = download_injury_reports(
            injury_directory,
            force=bool(arguments.force),
        )
        print(json.dumps(manifest, indent=2))
    if arguments.command in {"run", "all"}:
        run_directory = run_phase3(root)
        print(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
