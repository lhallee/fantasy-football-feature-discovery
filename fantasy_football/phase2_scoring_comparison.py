"""Compare common reception-based fantasy scoring formats."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from collections.abc import Mapping, Sequence
from pathlib import Path


SCHEME_COLUMNS: Mapping[str, str] = {
    "Non-PPR": "non_ppr_points",
    "Half PPR": "half_ppr_points",
    "Full PPR\n(ours)": "full_ppr_points",
}
POSITION_MAP: Mapping[str, str] = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "HB": "RB",
    "WR": "WR",
    "TE": "TE",
    "K": "K",
    "PK": "K",
}
REQUIRED_COLUMNS = (
    "player_id",
    "position",
    "season",
    "season_type",
    "recomputed_non_ppr_points",
    "recomputed_half_ppr_points",
    "recomputed_ppr_points",
    "recomputed_kicker_points",
)


def aggregate_player_seasons(player_games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate comparable score totals at player-season grain."""
    # player_games: (n_games, c_game)
    missing = sorted(set(REQUIRED_COLUMNS) - set(player_games.columns))
    if missing:
        raise ValueError(f"Player-game table is missing columns: {missing}.")

    model_position = player_games["position"].map(POSITION_MAP)  # (n_games,)
    eligible = (  # (n_games,)
        player_games["player_id"].notna()
        & model_position.notna()
        & player_games["season_type"].eq("REG")
    )
    games = player_games.loc[eligible].copy()  # (n_eligible_games, c_game)
    games["model_position"] = model_position.loc[eligible]  # (n_eligible_games,)

    kicker_bonus = games["recomputed_kicker_points"].where(  # (n_eligible_games,)
        games["model_position"].eq("K"),
        0.0,
    )
    games["non_ppr_points"] = (  # (n_eligible_games,)
        games["recomputed_non_ppr_points"] + kicker_bonus
    )
    games["half_ppr_points"] = (  # (n_eligible_games,)
        games["recomputed_half_ppr_points"] + kicker_bonus
    )
    games["full_ppr_points"] = (  # (n_eligible_games,)
        games["recomputed_ppr_points"] + kicker_bonus
    )

    scoring_columns = list(SCHEME_COLUMNS.values())
    season_scores = (  # (n_player_seasons, 3 + n_schemes)
        games.groupby(
            ["season", "player_id", "model_position"],
            as_index=False,
            observed=True,
        )[scoring_columns]
        .sum()
        .sort_values(
            ["season", "model_position", "player_id"],
            kind="stable",
            ignore_index=True,
        )
    )
    return season_scores  # (n_player_seasons, 3 + n_schemes)


def scoring_correlations(
    season_scores: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return pooled Pearson and Spearman scoring correlations."""
    # season_scores: (n_player_seasons, 3 + n_schemes)
    score_matrix = season_scores.loc[
        :, list(SCHEME_COLUMNS.values())
    ].rename(  # (n_player_seasons, n_schemes)
        columns={column: label for label, column in SCHEME_COLUMNS.items()}
    )
    return {
        "Pearson": score_matrix.corr(method="pearson"),  # (n_schemes, n_schemes)
        "Spearman": score_matrix.corr(method="spearman"),  # (n_schemes, n_schemes)
    }


def correlation_records(
    correlations: Mapping[str, pd.DataFrame],
    n_player_seasons: int,
) -> pd.DataFrame:
    """Convert correlation matrices to an auditable long table."""
    records: list[dict[str, str | float | int]] = []
    for method, matrix in correlations.items():
        # matrix: (n_schemes, n_schemes)
        for first_scheme in matrix.index:
            for second_scheme in matrix.columns:
                records.append(
                    {
                        "method": method,
                        "scheme_a": str(first_scheme).replace("\n", " "),
                        "scheme_b": str(second_scheme).replace("\n", " "),
                        "correlation": float(matrix.loc[first_scheme, second_scheme]),
                        "player_seasons": n_player_seasons,
                    }
                )
    return pd.DataFrame(records)  # (2 * n_schemes**2, 5)


def plot_scoring_correlations(
    correlations: Mapping[str, pd.DataFrame],
    *,
    n_player_seasons: int,
    first_season: int,
    last_season: int,
    output_path: Path,
) -> None:
    """Save a two-panel 300 dpi correlation heatmap."""
    try:
        import matplotlib

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires the optional matplotlib dependency."
        ) from error

    methods = ("Pearson", "Spearman")
    figure = Figure(figsize=(11.4, 5.5), constrained_layout=False)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)  # (2,)
    color_map = matplotlib.colormaps["Blues"]

    for axis, method in zip(axes, methods, strict=True):
        matrix = correlations[method]  # (n_schemes, n_schemes)
        values = matrix.to_numpy(dtype="float64")  # (n_schemes, n_schemes)
        axis.imshow(values, vmin=0.75, vmax=1.0, cmap=color_map)
        axis.set_title(f"{method} correlation", fontsize=12, fontweight="bold", pad=12)
        axis.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
        axis.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
        axis.tick_params(axis="x", labelrotation=25, labelsize=9)
        axis.tick_params(axis="y", labelsize=9)
        axis.spines[:].set_visible(False)

        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                text_color = "white" if value >= 0.94 else "#17233C"
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                )

    figure.suptitle(
        "Common reception-format score correlations",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#17233C",
    )
    figure.text(
        0.06,
        0.905,
        (
            f"Pooled regular-season fantasy-position player totals, {first_season} to {last_season} "
            f"(n={n_player_seasons:,} player-seasons)"
        ),
        ha="left",
        fontsize=10,
        color="#4B5563",
    )
    figure.text(
        0.06,
        0.02,
        (
            "All non-reception coefficients use the ESPN 2026 profile. Kicker points are included. "
            "Source: local nflverse-derived player-game table."
        ),
        ha="left",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.18, wspace=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")


def generate_scoring_comparison(
    player_games_path: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Generate the correlation table and figure from processed player games."""
    player_games = pd.read_parquet(player_games_path)  # (n_games, c_game)
    season_scores = aggregate_player_seasons(player_games)  # (n_player_seasons, 6)
    correlations = scoring_correlations(season_scores)
    records = correlation_records(correlations, len(season_scores))  # (18, 5)

    output_directory.mkdir(parents=True, exist_ok=True)
    table_path = output_directory / "scoring_scheme_correlations.csv"
    figure_path = output_directory / "scoring_scheme_correlations.png"
    records.to_csv(table_path, index=False)
    plot_scoring_correlations(
        correlations,
        n_player_seasons=len(season_scores),
        first_season=int(season_scores["season"].min()),
        last_season=int(season_scores["season"].max()),
        output_path=figure_path,
    )
    return table_path, figure_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standalone scoring-comparison command."""
    parser = argparse.ArgumentParser(
        description="Plot full-, half-, and non-PPR score correlations."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("experiments/phase2/artifacts"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scoring-comparison command."""
    args = _parse_args(argv)
    root = args.root.resolve()
    output_directory = (
        args.output_directory
        if args.output_directory.is_absolute()
        else root / args.output_directory
    )
    table_path, figure_path = generate_scoring_comparison(
        root / "data" / "processed" / "player_games.parquet",
        output_directory,
    )
    print(table_path)
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
