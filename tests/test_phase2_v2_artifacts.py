"""Validate the corrected Phase 2 v2 experiment and post-fit claims."""

import json
import pandas as pd
import pytest

from pathlib import Path

from fantasy_football.phase2_modeling import summarize_predictions
from fantasy_football.phase2_safety import combined_project_bytes
from fantasy_football.phase2_safety import hash_outputs, verify_phase1_freeze
from fantasy_football.provenance import file_sha256


ROOT = Path(__file__).parents[1]
PHASE2 = ROOT / "experiments" / "phase2"
MANIFEST = json.loads(
    (PHASE2 / "experiment_manifest_v2.json").read_text(encoding="utf-8")
)


def _run_directory(stage: str) -> Path:
    return ROOT / MANIFEST["runs"][stage]["directory"]


def test_v2_plan_prefit_code_documents_and_size_are_bound() -> None:
    assert len(verify_phase1_freeze(ROOT)) == 7
    assert MANIFEST["phase1_freeze_id"] == (
        "366b32e659f67cbbcb81a68ee20a66bbe79cdefc004db5c971c23e89b35f17ae"
    )
    assert file_sha256(ROOT / MANIFEST["plan"]["path"]) == MANIFEST["plan"]["sha256"]
    assert (
        file_sha256(ROOT / MANIFEST["prefit_lock"]["path"])
        == MANIFEST["prefit_lock"]["sha256"]
    )
    for relative_path, expected_hash in MANIFEST["code"].items():
        assert file_sha256(ROOT / relative_path) == expected_hash
    documents = MANIFEST["documents"]
    assert (
        file_sha256(ROOT / documents["postfit_ledger"])
        == documents["postfit_ledger_sha256"]
    )
    assert (
        file_sha256(ROOT / documents["results_guide"])
        == documents["results_guide_sha256"]
    )
    assert MANIFEST["combined_phase1_phase2_bytes"] == combined_project_bytes(ROOT)
    assert combined_project_bytes(ROOT) < MANIFEST["size_limit_bytes"]


def test_v2_run_outputs_and_cross_stage_selection_hashes_match() -> None:
    for stage in ("discovery", "retrospective"):
        run_dir = _run_directory(stage)
        declared = json.loads(
            (run_dir / "output_hashes.json").read_text(encoding="utf-8")
        )
        actual = hash_outputs(run_dir, root=ROOT)
        actual.pop("output_hashes.json")
        assert actual == declared
        assert (
            file_sha256(run_dir / "run_manifest.json")
            == MANIFEST["runs"][stage]["run_manifest_sha256"]
        )
        assert (
            file_sha256(run_dir / "output_hashes.json")
            == MANIFEST["runs"][stage]["output_hashes_sha256"]
        )

    discovery_dir = _run_directory("discovery")
    retrospective_dir = _run_directory("retrospective")
    selection = json.loads(
        (discovery_dir / "selection.json").read_text(encoding="utf-8")
    )
    retrospective = json.loads(
        (retrospective_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        file_sha256(discovery_dir / "selection.json")
        == MANIFEST["runs"]["discovery"]["selection_sha256"]
    )
    assert selection["phase2_code_hashes"] == MANIFEST["code"]
    assert selection["prefit_manifest_sha256"] == MANIFEST["prefit_lock"]["sha256"]
    assert retrospective["phase2_code_hashes"] == selection["phase2_code_hashes"]
    assert retrospective["selected_candidate"] == selection["selected_candidate"]


@pytest.mark.parametrize("stage", ["discovery", "retrospective"])
def test_v2_saved_predictions_reproduce_selected_metrics(stage: str) -> None:
    run_dir = _run_directory(stage)
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    selected = predictions.loc[  # (n_selected_rows, 8)
        predictions["candidate"].eq(MANIFEST["selected_procedure"]["identifier"])
    ].copy()
    metrics = summarize_predictions(selected)

    expected = MANIFEST["runs"][stage]
    assert metrics["spearman"] == pytest.approx(expected["selected_spearman"])
    if stage == "retrospective":
        assert len(selected) == expected["evaluated_rows"]
        assert metrics["mae"] == pytest.approx(expected["selected_mae"])
    assert metrics["spearman"] < 0.9


def test_v2_selected_feature_vector_and_lineage_match_claims() -> None:
    run_dir = _run_directory("discovery")
    selected_features = pd.read_csv(  # (n_selection_records, 7)
        run_dir / "selected_features.csv"
    )
    selected_features = selected_features.loc[  # (6 * 248, 7)
        selected_features["candidate"].eq(MANIFEST["selected_procedure"]["identifier"])
    ].copy()
    feature_names = set(selected_features["feature"])
    assert len(feature_names) == MANIFEST["selected_procedure"]["columns"] == 248
    fold_counts = selected_features.groupby(  # (6,)
        ["validation_season", "position_scope"],
        observed=True,
    )["selected_feature_count"].first()
    assert fold_counts.eq(248).all()

    room_features = {name for name in feature_names if name.startswith("room_")}
    position_features = {name for name in feature_names if name.startswith("position_")}
    lagged_features = {
        name
        for name in feature_names
        if name.startswith(("lag1_", "lag2_", "lag3_", "lag4_"))
    }
    metadata_features = feature_names.difference(
        room_features | position_features | lagged_features
    )
    assert len(room_features) == 19
    assert len(position_features) == 4
    assert len(lagged_features) == 196
    assert len(metadata_features) == 29

    lineage = json.loads((run_dir / "feature_lineage.json").read_text())["stable"]
    for feature in feature_names:
        record = lineage[feature]
        for source, offset in zip(
            record["sources"], record["source_offsets"], strict=True
        ):
            if source in {"weekly_stats", "player_seasons"}:
                assert offset >= 1
    lowered = " ".join(feature_names).lower()
    for forbidden in ("target_points", "fantasy", "status", "depth", "epa"):
        assert forbidden not in lowered


def test_v2_cohort_diagnostics_and_failure_decision_match_artifacts() -> None:
    discovery = json.loads(
        (_run_directory("discovery") / "cohort_audit.json").read_text()
    )
    retrospective = json.loads(
        (_run_directory("retrospective") / "cohort_audit.json").read_text()
    )
    assert (discovery["rows_before"], discovery["rows_after"]) == (12893, 11615)
    assert (retrospective["rows_before"], retrospective["rows_after"]) == (
        16726,
        14669,
    )
    assert all(
        record["rows_after"] == 0
        for record in retrospective["status_counts"]
        if record["status"]
        in {"CUT", "RET", "UFA", "RFA", "NWT", "RSR", "E01", "TRD", "MISSING"}
    )

    uncertainty = json.loads(
        (_run_directory("retrospective") / "uncertainty.json").read_text()
    )
    assert uncertainty["player_cluster"]["lower_95"] == pytest.approx(
        MANIFEST["runs"]["retrospective"]["player_cluster_lower_95"]
    )
    assert max(MANIFEST["runs"]["retrospective"]["position_spearman"].values()) < 0.85
    assert max(MANIFEST["runs"]["retrospective"]["season_spearman"].values()) < 0.88
    assert MANIFEST["validity"]["exceeds_0_90"] is False


def test_v2_post_selection_null_diagnostic_is_explicitly_nonconfirmatory() -> None:
    diagnostic = json.loads(
        (ROOT / MANIFEST["null_diagnostic"]["path"]).read_text(encoding="utf-8")
    )
    assert (
        file_sha256(ROOT / MANIFEST["null_diagnostic"]["path"])
        == MANIFEST["null_diagnostic"]["sha256"]
    )
    raw_null = diagnostic["raw_grouped_target_permutations"]
    assert len(raw_null["spearman"]) == 11
    assert raw_null["mean"] == pytest.approx(
        MANIFEST["null_diagnostic"]["raw_grouped_label_null_mean"]
    )
    assert all(value > 0 for value in raw_null["spearman"])
    assert diagnostic["fixed_prediction_validation_label_permutations"][
        "mean_spearman"
    ] == pytest.approx(-0.0003889642199600843)
    assert diagnostic["confirmatory_use"] is False
    assert MANIFEST["validity"]["raw_grouped_label_zero_null_passed"] is False
