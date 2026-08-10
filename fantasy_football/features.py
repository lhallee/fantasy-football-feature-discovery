"""Mechanical lag and transform generation with raw-atom lineage."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import COMBINE_ATOMS, KICKER_STAT_ATOMS, METADATA_ATOMS
from .constants import OFFENSE_STAT_ATOMS, UTILIZATION_ATOMS


MANDATORY_COLUMNS = ("position_QB", "position_RB", "position_WR", "position_TE")


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Numeric features and their source-atom lineage."""

    frame: pd.DataFrame  # (n_player_seasons, d_model_features)
    atom_by_feature: dict[str, str]
    mandatory_features: tuple[str, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return columns in model order."""
        return tuple(self.frame.columns)

    @property
    def atoms(self) -> tuple[str, ...]:
        """Return declared raw atoms in first-use order."""
        return tuple(dict.fromkeys(self.atom_by_feature.values()))

    def columns_for_atoms(self, atoms: set[str] | tuple[str, ...]) -> list[str]:
        """Return features derived from selected atoms plus mandatory adjustments."""
        selected = set(atoms)
        return [
            column
            for column in self.frame.columns
            if column in self.mandatory_features
            or self.atom_by_feature.get(column) in selected
        ]


def declared_atoms(cohort: str, tier: str) -> tuple[str, ...]:
    """Return raw atoms for an offense or kicker feature tier."""
    if cohort == "offense":
        atoms = [*METADATA_ATOMS, *COMBINE_ATOMS, *OFFENSE_STAT_ATOMS]
        if tier == "utilization":
            atoms.extend(UTILIZATION_ATOMS)
        return tuple(atoms)
    if cohort == "kicker":
        return tuple([*METADATA_ATOMS, *KICKER_STAT_ATOMS])
    raise ValueError(f"Unknown cohort: {cohort!r}")


def _add_feature(
    features: dict[str, pd.Series],
    lineage: dict[str, str],
    name: str,
    values: pd.Series,
    atom: str,
) -> None:
    # values: (n_player_seasons,)
    features[name] = pd.to_numeric(
        values, errors="coerce"
    ).astype(  # (n_player_seasons,)
        "float32"
    )
    lineage[name] = atom


def build_features(
    modeling_table: pd.DataFrame,
    cohort: str,
    tier: str = "base",
) -> FeatureSet:
    """Generate raw lags, logs, and trends without fantasy-derived inputs."""
    # modeling_table: (n_player_seasons, c_modeling)
    atoms = declared_atoms(cohort, tier)
    features: dict[str, pd.Series] = {}
    lineage: dict[str, str] = {}

    for atom in METADATA_ATOMS:
        if atom not in atoms:
            continue
        values = modeling_table[atom]  # (n_player_seasons,)
        _add_feature(features, lineage, atom, values, atom)
        if atom in {"age", "draft_number", "depth_tier"}:
            _add_feature(
                features,
                lineage,
                f"log1p_{atom}",
                np.log1p(values.clip(lower=0)),
                atom,
            )
        if atom == "age":
            _add_feature(features, lineage, "age_squared", values**2, atom)

    for atom in COMBINE_ATOMS:
        if atom in atoms:
            _add_feature(features, lineage, atom, modeling_table[atom], atom)
            _add_feature(
                features,
                lineage,
                f"missing_{atom}",
                modeling_table[atom].isna().astype("int8"),
                atom,
            )

    stat_atoms = [
        atom
        for atom in atoms
        if atom not in METADATA_ATOMS and atom not in COMBINE_ATOMS
    ]
    for atom in stat_atoms:
        lag1_name = f"lag1_{atom}"
        lag2_name = f"lag2_{atom}"
        lag1 = modeling_table[lag1_name]  # (n_player_seasons,)
        lag2 = modeling_table[lag2_name]  # (n_player_seasons,)
        _add_feature(features, lineage, lag1_name, lag1, atom)
        _add_feature(features, lineage, lag2_name, lag2, atom)
        _add_feature(
            features,
            lineage,
            f"log1p_{lag1_name}",
            np.log1p(lag1.clip(lower=0)),
            atom,
        )
        _add_feature(
            features,
            lineage,
            f"log1p_{lag2_name}",
            np.log1p(lag2.clip(lower=0)),
            atom,
        )
        _add_feature(features, lineage, f"trend_{atom}", lag1 - lag2, atom)
    position_flags = pd.get_dummies(  # (n_player_seasons, p_present)
        modeling_table["model_position"],
        prefix="position",
        dtype="float32",
    )
    mandatory = tuple(
        column for column in MANDATORY_COLUMNS if column in position_flags
    )
    for column in mandatory:
        features[column] = position_flags[column]  # (n_player_seasons,)

    feature_frame = pd.DataFrame(  # (n_player_seasons, d_model_features)
        features,
        index=modeling_table.index,
    )
    return FeatureSet(  # frame: (n_player_seasons, d_model_features)
        frame=feature_frame,
        atom_by_feature=lineage,
        mandatory_features=mandatory,
    )
