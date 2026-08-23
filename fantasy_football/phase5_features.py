"""Add literature-motivated lag-1 usage shares and per-game rates to the Phase 4 bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from .phase2_features import FeatureLineage
from .phase3_injury import _load_weekly_stats
from .phase4_labels import Phase4Bundle


PER_GAME_COUNTS = (
    "targets",
    "carries",
    "receptions",
    "attempts",
    "receiving_yards",
    "rushing_yards",
    "passing_yards",
    "offense_snaps",
)
SHARE_COUNTS = ("targets", "carries", "receptions", "attempts")


def prior_team_totals(root: Path) -> pd.DataFrame:
    """Sum regular-season team counts per season from the weekly stat files."""
    weekly = _load_weekly_stats(root)  # (n_player_weeks, c_weekly)
    regular = weekly.loc[
        weekly["season_type"].astype("string").str.upper().eq("REG")
        & weekly["team"].notna()
    ].copy()  # (n_regular, c_weekly)
    regular["team"] = regular["team"].astype("string").str.upper()  # (n_regular,)
    for column in SHARE_COUNTS:
        regular[column] = pd.to_numeric(regular[column], errors="coerce").fillna(
            0.0
        )  # (n_regular,)
    totals = (
        regular.groupby(["season", "team"], observed=True)[list(SHARE_COUNTS)]
        .sum()
        .add_prefix("team_")
        .reset_index()
    )  # (n_team_seasons, 2 + len(SHARE_COUNTS))
    totals["season"] = totals["season"].astype("int32")
    return totals


def extend_with_usage_features(bundle: Phase4Bundle, root: Path) -> Phase4Bundle:
    """Attach lag-1 per-game rates and prior-team usage shares with lineage."""
    frame = bundle.frame.copy()  # (n, d)
    lineage = dict(bundle.lineage)
    families = dict(bundle.families)
    games = frame["lag1_games"].astype("float64")  # (n,)

    for count in PER_GAME_COUNTS:
        source = f"lag1_{count}"
        column = f"{source}_per_game"
        frame[column] = (
            frame[source]
            .astype("float64")
            .div(games.where(games.gt(0.0)))
            .fillna(0.0)
            .astype("float32")
        )  # (n,)
        lineage[column] = FeatureLineage(
            column=column,
            sources=("weekly_stats", "player_seasons"),
            source_offsets=(1, 1),
            recipe=f"lag1_{count} / lag1_games; zero when no target_season-1 games",
        )
        families[column] = "lag1_per_game"

    player_seasons = pd.read_parquet(
        root / "data" / "processed" / "player_seasons.parquet",
        columns=["season", "player_id", "last_team"],
    )  # (n_player_seasons, 3)
    player_seasons["last_team"] = (
        player_seasons["last_team"].astype("string").str.upper()
    )
    player_seasons["target_season"] = (
        player_seasons["season"].astype("int32") + 1
    )  # (n_player_seasons,)
    totals = prior_team_totals(root).rename(
        columns={"team": "last_team"}
    )  # (n_team_seasons, 6)
    prior = player_seasons.merge(
        totals, on=["season", "last_team"], how="left"
    )  # (n_player_seasons, 9)
    keyed = bundle.keys.loc[:, ["target_season", "player_id"]].merge(
        prior.drop(columns=["season", "last_team"]),
        on=["target_season", "player_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )  # (n, 2 + len(SHARE_COUNTS))
    keyed.index = bundle.keys.index
    for count in SHARE_COUNTS:
        column = f"lag1_share_{count}_prior_team"
        team_total = pd.to_numeric(keyed[f"team_{count}"], errors="coerce")  # (n,)
        frame[column] = (
            frame[f"lag1_{count}"]
            .astype("float64")
            .div(team_total.where(team_total.gt(0.0)))
            .fillna(0.0)
            .clip(0.0, 1.0)
            .astype("float32")
        )  # (n,)
        lineage[column] = FeatureLineage(
            column=column,
            sources=("weekly_stats", "player_seasons"),
            source_offsets=(1, 1),
            recipe=(
                f"lag1_{count} divided by the regular-season {count} of the player's final "
                "target_season-1 team; zero when the team or count is unavailable"
            ),
        )
        families[column] = "lag1_usage_share"
    assert np.isfinite(frame.to_numpy(dtype="float64")).all()
    return Phase4Bundle(
        keys=bundle.keys,
        metadata=bundle.metadata,
        labels=bundle.labels,
        frame=frame,
        lineage=lineage,
        families=families,
    )
