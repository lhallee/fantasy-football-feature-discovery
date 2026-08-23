"""Render the report figures from the Phase 4 and Phase 5 artifacts."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[2]
PHASE4 = ROOT / "experiments" / "phase4"
FIGURES = Path(__file__).resolve().parent / "figures"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dddcd7"
POSITIONS = ("QB", "RB", "WR", "TE")


def _run_dir() -> Path:
    summary = json.loads(
        (PHASE4 / "artifacts" / "result_summary.json").read_text(encoding="utf-8")
    )
    return ROOT / summary["run_directory"]


def _figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), constrained_layout=True, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _style(axis) -> None:
    axis.set_facecolor("white")
    axis.grid(color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)


def _save(figure: Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf")
    figure.savefig(FIGURES / f"{name}.png", dpi=300)


def points_audit_by_position(run_dir: Path) -> None:
    """Audit Spearman per position: adopted model, Phase 2, and prior-season points."""
    audit = pd.read_parquet(run_dir / "points_audit_predictions.parquet")  # (n_rows, 7)
    baseline = pd.read_parquet(
        ROOT / "artifacts" / "audit_predictions.parquet"
    )  # (n_phase2, 9)
    summary = json.loads((run_dir / "result_summary.json").read_text(encoding="utf-8"))
    adopted = summary["points"]["adopted"]
    series = {
        f"Adopted ({adopted})": audit.loc[audit["candidate"].eq(adopted)].rename(
            columns={"predicted": "score"}
        ),
        "Phase 2 production": audit.loc[
            audit["candidate"].eq("phase2_production")
        ].rename(columns={"predicted": "score"}),
        "Prior-season points": baseline.rename(
            columns={"previous_points_baseline": "score"}
        ),
    }
    from fantasy_football.metrics import macro_rank_metric

    figure = _figure(6.4, 3.0)
    axis = figure.add_subplot(1, 1, 1)
    width = 0.26
    for offset, (label, table) in enumerate(series.items()):
        values = []
        for position in POSITIONS:
            rows = table.loc[table["model_position"].eq(position)]
            values.append(
                macro_rank_metric(
                    rows["actual_points"],
                    rows["score"],
                    rows["model_position"],
                    rows["target_season"],
                )
            )
        x = np.arange(len(POSITIONS)) + (offset - 1) * width
        color = (BLUE, ORANGE, AQUA)[offset]
        axis.bar(x, values, width=width, color=color, label=label)
        for xi, value in zip(x, values, strict=True):
            axis.text(
                xi, value + 0.01, f"{value:.2f}", ha="center", fontsize=6.5, color=MUTED
            )
    axis.set_xticks(np.arange(len(POSITIONS)), POSITIONS)
    axis.set_ylim(0.0, 0.95)
    axis.set_ylabel("Mean within-position-season Spearman", color=MUTED, fontsize=8)
    axis.set_title(
        "Points forecast, locked 2022-2025 audit", loc="left", fontsize=9, color=TEXT
    )
    axis.legend(frameon=False, fontsize=7, loc="lower right")
    _style(axis)
    _save(figure, "points_audit_by_position")


def injury_roc_calibration(run_dir: Path) -> None:
    """Audit ROC and decile calibration for the selected injury classifier."""
    audit = pd.read_parquet(run_dir / "injury_audit_predictions.parquet")  # (n_rows, 6)
    summary = json.loads((run_dir / "result_summary.json").read_text(encoding="utf-8"))
    selected = summary["injury"]["selected"]
    figure = _figure(6.4, 2.9)
    roc_axis = figure.add_subplot(1, 2, 1)
    for learner, label, color in (
        (selected, "Selected Extra Trees", BLUE),
        ("prior_games_logistic", "Prior-year games", AQUA),
        ("age_position_logistic", "Age and position", ORANGE),
    ):
        rows = audit.loc[audit["learner"].eq(learner)]
        fpr, tpr, _ = roc_curve(rows["label"], rows["probability"])
        auc = roc_auc_score(rows["label"], rows["probability"])
        roc_axis.plot(
            fpr, tpr, color=color, linewidth=1.6, label=f"{label} ({auc:.3f})"
        )
    roc_axis.plot([0, 1], [0, 1], color=GRID, linewidth=0.8, linestyle="--")
    roc_axis.set_xlabel("False positive rate", color=MUTED, fontsize=8)
    roc_axis.set_ylabel("True positive rate", color=MUTED, fontsize=8)
    roc_axis.set_title("ROC, 2022-2024 audit", loc="left", fontsize=9, color=TEXT)
    roc_axis.legend(frameon=False, fontsize=6.5, loc="lower right")
    calibration_axis = figure.add_subplot(1, 2, 2)
    rows = audit.loc[audit["learner"].eq(selected)]
    observed, predicted = calibration_curve(
        rows["label"], rows["probability"], n_bins=10, strategy="quantile"
    )
    calibration_axis.plot([0, 0.7], [0, 0.7], color=GRID, linewidth=0.8, linestyle="--")
    calibration_axis.plot(
        predicted, observed, color=BLUE, linewidth=1.6, marker="o", markersize=3.5
    )
    calibration_axis.set_xlabel(
        "Predicted probability (decile mean)", color=MUTED, fontsize=8
    )
    calibration_axis.set_ylabel("Observed missed-time rate", color=MUTED, fontsize=8)
    calibration_axis.set_title("Calibration", loc="left", fontsize=9, color=TEXT)
    for axis in (roc_axis, calibration_axis):
        _style(axis)
    _save(figure, "injury_roc_calibration")


def feature_importance() -> None:
    """Top features per track by held-out permutation importance, with consensus rank."""
    figure = _figure(6.6, 4.2)
    for index, (track, title, color) in enumerate(
        (("points", "Points track", BLUE), ("injury", "Injury track", ORANGE)), start=1
    ):
        table = pd.read_csv(PHASE4 / "artifacts" / f"{track}_feature_ranking.csv")
        plotted = (
            table.sort_values("permutation_importance", ascending=False)
            .head(15)
            .iloc[::-1]
        )
        axis = figure.add_subplot(1, 2, index)
        axis.barh(
            plotted["feature"],
            plotted["permutation_importance"],
            color=color,
            height=0.6,
        )
        for y, (value, rank) in enumerate(
            zip(
                plotted["permutation_importance"],
                plotted["consensus_rank"],
                strict=True,
            )
        ):
            axis.text(value, y, f" #{int(rank)}", va="center", fontsize=6, color=MUTED)
        axis.set_title(title, loc="left", fontsize=9, color=TEXT)
        axis.set_xlabel("Permutation importance", color=MUTED, fontsize=8)
        axis.tick_params(axis="y", labelsize=6.5)
        _style(axis)
        axis.spines["left"].set_visible(False)
    _save(figure, "feature_importance")


def family_ablation() -> None:
    """Metric change when one feature family is removed, both tracks."""
    table = pd.read_csv(PHASE4 / "artifacts" / "family_ablation.csv")
    figure = _figure(6.6, 3.2)
    for index, (track, title, color) in enumerate(
        (
            ("points", "Points: Spearman drop", BLUE),
            ("injury", "Injury: ROC AUC drop", ORANGE),
        ),
        start=1,
    ):
        rows = table.loc[table["track"].eq(track)].sort_values("delta")
        axis = figure.add_subplot(1, 2, index)
        axis.barh(rows["family"], rows["delta"], color=color, height=0.6)
        axis.axvline(0.0, color=MUTED, linewidth=0.8)
        axis.set_title(title, loc="left", fontsize=9, color=TEXT)
        axis.tick_params(axis="y", labelsize=6.5)
        _style(axis)
        axis.spines["left"].set_visible(False)
    _save(figure, "family_ablation")


def null_controls() -> None:
    """Negative-control diagnostics that changed the protocol."""
    era = json.loads(
        (PHASE4 / "artifacts" / "points_null_era_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    position = json.loads(
        (PHASE4 / "artifacts" / "injury_null_position_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    figure = _figure(6.4, 2.8)
    left = figure.add_subplot(1, 2, 1)
    for offset, (start, color) in enumerate((("2013", ORANGE), ("2017", BLUE))):
        values = era[start]["nulls"]
        left.scatter(
            np.full(len(values), offset) + np.linspace(-0.12, 0.12, len(values)),
            values,
            color=color,
            s=16,
            zorder=3,
        )
        left.bar(offset, era[start]["null_mean"], width=0.5, color=color, alpha=0.35)
        left.text(
            offset,
            era[start]["real"] + 0.01,
            f"real {era[start]['real']:.3f}",
            ha="center",
            fontsize=6.5,
            color=MUTED,
        )
        left.scatter(
            [offset], [era[start]["real"]], marker="_", s=400, color=color, zorder=4
        )
    left.axhline(0.0, color=MUTED, linewidth=0.8)
    left.set_xticks([0, 1], ["train from 2013", "train from 2017"])
    left.set_ylabel("Spearman, target-permuted fits", color=MUTED, fontsize=8)
    left.set_title(
        "Points null control by training window", loc="left", fontsize=9, color=TEXT
    )
    right = figure.add_subplot(1, 2, 2)
    pooled = [seed["pooled"] for seed in position["seeds"]]
    within = [seed["within_season_position"] for seed in position["seeds"]]
    for offset, (values, label, color) in enumerate(
        ((pooled, "pooled", ORANGE), (within, "within season and position", BLUE))
    ):
        right.scatter(
            np.full(len(values), offset) + np.linspace(-0.12, 0.12, len(values)),
            values,
            color=color,
            s=16,
            zorder=3,
        )
        right.bar(offset, float(np.mean(values)), width=0.5, color=color, alpha=0.35)
    right.axhline(0.5, color=MUTED, linewidth=0.8)
    right.set_xticks([0, 1], ["pooled AUC", "within season\nand position"])
    right.set_ylabel("AUC, label-permuted fits", color=MUTED, fontsize=8)
    right.set_title(
        "Injury null control by evaluation grain", loc="left", fontsize=9, color=TEXT
    )
    for axis in (left, right):
        _style(axis)
    _save(figure, "null_controls")


def weight_tradeoff() -> None:
    """Rank accuracy and bust rate as the injury weight rises."""
    table = pd.read_csv(PHASE4 / "artifacts" / "weight_analysis.csv")
    default = float(table.loc[table["default"], "injury_weight"].iloc[0])
    figure = _figure(6.4, 2.7)
    for index, (metric, title) in enumerate(
        (
            ("spearman", "Spearman vs realized points"),
            ("bust_rate_top_k", "Bust rate among top-k picks"),
        ),
        start=1,
    ):
        axis = figure.add_subplot(1, 2, index)
        for stage, color in (("development", BLUE), ("audit", ORANGE)):
            rows = table.loc[table["stage"].eq(stage)]
            axis.plot(
                rows["injury_weight"],
                rows[metric],
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=3,
                label=stage,
            )
        axis.axvline(default, color=MUTED, linewidth=0.8, linestyle="--")
        axis.set_title(title, loc="left", fontsize=9, color=TEXT)
        axis.set_xlabel("Injury weight w", color=MUTED, fontsize=8)
        axis.legend(frameon=False, fontsize=7)
        _style(axis)
    _save(figure, "weight_tradeoff")


def label_prevalence() -> None:
    """Prevalence of the two injury labels by season."""
    table = pd.read_csv(PHASE4 / "artifacts" / "label_prevalence_by_season.csv")
    figure = _figure(6.4, 2.5)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(
        table["target_season"],
        table["physical_injury_reported"],
        color=ORANGE,
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="Any physical designation (Phase 3)",
    )
    axis.plot(
        table["target_season"],
        table["missed_time_injury"],
        color=BLUE,
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="Missed-time injury (Phase 4)",
    )
    axis.plot(
        table["target_season"],
        table["absence_reported"],
        color=AQUA,
        marker="o",
        markersize=3,
        linewidth=1.2,
        linestyle="--",
        label="Out or Doubtful report",
    )
    axis.plot(
        table["target_season"],
        table["reserve_injury"],
        color=MUTED,
        marker="o",
        markersize=3,
        linewidth=1.2,
        linestyle="--",
        label="Injury reserve week",
    )
    axis.set_ylabel("Share of candidates", color=MUTED, fontsize=8)
    axis.set_title(
        "Injury label prevalence by season", loc="left", fontsize=9, color=TEXT
    )
    axis.legend(frameon=False, fontsize=6.5, ncol=2)
    _style(axis)
    _save(figure, "label_prevalence")


if __name__ == "__main__":
    run_dir = _run_dir()
    points_audit_by_position(run_dir)
    injury_roc_calibration(run_dir)
    feature_importance()
    family_ablation()
    null_controls()
    weight_tradeoff()
    label_prevalence()
    print(f"figures written to {FIGURES}")
