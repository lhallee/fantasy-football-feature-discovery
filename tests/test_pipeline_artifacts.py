"""Integration checks for temporal features, artifacts, and reproducibility."""

import hashlib
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from fantasy_football.constants import CURRENT_SEASON, MAX_PROJECT_DATA_BYTES
from fantasy_football.features import build_features
from fantasy_football.provenance import project_data_bytes, synchronize_project_sizes


ROOT = Path(__file__).resolve().parents[1]


def test_download_manifest_hashes_match_files() -> None:
    manifest = json.loads((ROOT / "data" / "raw" / "manifest.json").read_text())
    assert manifest["manifest_version"] == 2
    assert manifest["data_cutoff_utc"]
    assert manifest["verified_at_utc"]
    assert manifest["total_bytes"] < MAX_PROJECT_DATA_BYTES
    for record in manifest["files"]:
        path = ROOT / "data" / "raw" / record["path"]
        assert path.is_file()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"]
        assert record["retrieved_at_utc"]
        assert record["verified_at_utc"]


def test_current_features_have_no_target_values() -> None:
    modeling = pd.read_parquet(  # (n_player_seasons, c_modeling)
        ROOT / "data" / "processed" / "modeling_table.parquet"
    )
    current_mask = modeling["target_season"].eq(CURRENT_SEASON)  # (n_player_seasons,)
    current = modeling[current_mask]  # (n_current_players, c_modeling)
    features = build_features(  # frame: (n_current_players, d_base)
        current,
        "offense",
        "base",
    )

    assert current["target_points"].isna().all()
    assert not any("target_points" in column for column in features.feature_names)
    assert not any("fantasy" in column for column in features.feature_names)
    assert "status_active" not in features.atoms
    assert "depth_tier" not in features.atoms
    assert not any(column.endswith("_per_game") for column in features.feature_names)


def test_model_manifest_identifies_inputs_scoring_and_thread_limit() -> None:
    manifest = json.loads((ROOT / "artifacts" / "model_manifest.json").read_text())
    build_summary = json.loads(
        (ROOT / "data" / "processed" / "build_summary.json").read_text()
    )

    assert manifest["production_generation"] == "phase2_august9_v1"
    assert manifest["current_season"] == CURRENT_SEASON
    assert manifest["forecast_origin"] == "2026-08-09"
    assert manifest["scoring_profile"] == build_summary["scoring_profile"]
    assert manifest["scoring_config_sha256"] == build_summary["scoring_config_sha256"]
    assert manifest["snapshot_date"] == build_summary["snapshot_date"]
    assert manifest["maximum_cpu_threads"] == 4
    assert manifest["promotion_gates"]["passed"] is True
    assert len(manifest["feature_columns"]["offense"]) == 228
    assert len(manifest["feature_columns"]["kicker"]) == 32
    lowered = " ".join(
        (*manifest["feature_columns"]["offense"], *manifest["feature_columns"]["kicker"])
    ).lower()
    for forbidden in ("target_points", "fantasy", "status", "depth", "room_"):
        assert forbidden not in lowered


def test_saved_predictions_match_saved_estimators() -> None:
    saved = pd.read_parquet(  # (n_current_players, c_prediction)
        ROOT / "artifacts" / f"predictions_{CURRENT_SEASON}.parquet"
    )
    for cohort, position_mask in (
        ("offense", saved["model_position"].ne("K")),
        ("kicker", saved["model_position"].eq("K")),
    ):
        # position_mask: (n_current_players,)
        current = saved.loc[position_mask].copy()  # (n_current, c_prediction)
        bundle = joblib.load(ROOT / "artifacts" / f"{cohort}_model.joblib")
        feature_columns = bundle["feature_columns"]
        stored_columns = [f"feature_value__{column}" for column in feature_columns]
        X_current = current.loc[:, stored_columns].copy()  # (n_current, d_selected)
        X_current.columns = feature_columns
        predicted = np.clip(
            bundle["estimator"].predict(X_current),
            a_min=0.0,
            a_max=None,
        )  # (n_current,)
        expected = current["predicted_points"].to_numpy()  # (n_current,)
        np.testing.assert_allclose(predicted, expected, rtol=0.0, atol=1e-12)


def test_complete_project_data_remains_below_limit() -> None:
    paths = list((ROOT / "data").rglob("*")) + list((ROOT / "artifacts").rglob("*"))
    canonical_bytes = project_data_bytes(ROOT)
    result_summary = json.loads(
        (ROOT / "artifacts" / "result_summary.json").read_text()
    )

    assert result_summary["data_and_artifact_bytes"] == canonical_bytes

    paths.append(ROOT / "fantasy_football" / f"players_{CURRENT_SEASON}.py")
    total_bytes = sum(path.stat().st_size for path in paths if path.is_file())
    assert total_bytes < MAX_PROJECT_DATA_BYTES


def test_smoke_size_sync_preserves_canonical_summaries(tmp_path: Path) -> None:
    build_path = tmp_path / "data" / "processed" / "build_summary.json"
    result_path = tmp_path / "artifacts" / "result_summary.json"
    build_path.parent.mkdir(parents=True)
    result_path.parent.mkdir(parents=True)
    build_path.write_text('{"total_project_data_bytes": 0}\n', encoding="utf-8")
    result_path.write_text('{"data_and_artifact_bytes": 0}\n', encoding="utf-8")
    synchronize_project_sizes(tmp_path)
    canonical_build = build_path.read_bytes()
    canonical_result = result_path.read_bytes()

    smoke_path = tmp_path / "artifacts" / "smoke" / "result_summary.json"
    smoke_path.parent.mkdir(parents=True)
    smoke_path.write_text('{"data_and_artifact_bytes": 0}\n', encoding="utf-8")
    (smoke_path.parent / "search.csv").write_text("metric\n1\n", encoding="utf-8")
    synchronize_project_sizes(tmp_path, update_canonical=False)

    assert build_path.read_bytes() == canonical_build
    assert result_path.read_bytes() == canonical_result
    smoke_summary = json.loads(smoke_path.read_text(encoding="utf-8"))
    assert smoke_summary["data_and_artifact_bytes"] == project_data_bytes(
        tmp_path,
        include_smoke=True,
    )
