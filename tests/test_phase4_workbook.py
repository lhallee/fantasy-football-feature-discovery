"""Test the final workbook layout and live custom-score formula."""

import pandas as pd

from openpyxl import load_workbook

from fantasy_football.phase4_workbook import SHEET_ORDER, write_final_workbook


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_name": ["Alpha", "Beta", "Gamma"],
            "team": ["BUF", "BUF", "NYJ"],
            "model_position": ["QB", "QB", "K"],
            "draft_score": [92.0, 55.0, 70.0],
            "overall_draft_score": [95.0, 40.0, 60.0],
            "vorp": [100.0, 0.0, 20.0],
            "overall_value": [1.0, 0.0, 0.2],
            "predicted_points": [300.0, 200.0, 120.0],
            "points_p10": [200.0, 100.0, 60.0],
            "points_p90": [380.0, 280.0, 150.0],
            "position_points_rank": [1, 2, 1],
            "points_percentile": [1.0, 0.5, 1.0],
            "injury_probability": [0.2, 0.4, 0.1],
            "health_probability": [0.8, 0.6, 0.9],
            "relative_health": [0.6, 0.4, 0.5],
            "risk_band": ["lower", "higher", "lower"],
            "current_injury_listing": [False, True, False],
            "current_injury_status": [None, "Questionable", None],
            "is_rookie": ["no", "no", "yes"],
            "age": [28.1, 31.4, 23.0],
            "points_confidence": ["standard", "standard", "low (kicker)"],
            "roster_status": ["ACT", "ACT", "ACT"],
            "player_id": ["a", "b", "c"],
            "current_injury_or_designation": [None, "Hamstring", None],
            "expected_return": [None, "Week 1", None],
            "evidence_date": [None, "2026-08-12", None],
            "source_publisher": [None, "MFL", None],
            "source_tier": [None, "official", None],
            "source_url": [None, "https://example.test", None],
        }
    )  # (3, 25)


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "MODEL_EVAL": pd.DataFrame(
            {"track": ["points"], "candidate": ["x"], "spearman": [0.7]}
        ),
        "WEIGHT_ANALYSIS": pd.DataFrame(
            {"stage": ["audit"], "injury_weight": [0.2], "spearman": [0.7]}
        ),
        "FEATURES_POINTS": pd.DataFrame(
            {"consensus_rank": [1], "feature": ["lag1_games"]}
        ),
        "FEATURES_INJURY": pd.DataFrame(
            {"consensus_rank": [1], "feature": ["lag1_games"]}
        ),
        "FAMILY_ABLATION": pd.DataFrame(
            {"track": ["points"], "family": ["lag1_production"], "delta": [0.1]}
        ),
    }


def test_workbook_sheets_rank_and_formula(tmp_path) -> None:
    path = write_final_workbook(
        _board(),
        tmp_path / "final.xlsx",
        default_weight=0.2,
        readme_rows=[("Purpose", "test")],
        tables=_tables(),
    )
    workbook = load_workbook(path)
    assert tuple(workbook.sheetnames) == SHEET_ORDER
    sheet = workbook["ALL"]
    assert sheet["A1"].value == "Rank"
    assert sheet["B2"].value == "Alpha"
    assert sheet["A2"].value == 1 and sheet["A4"].value == 3
    assert sheet["B3"].value == "Gamma"
    assert sheet["G2"].value == "=ROUND(100*((1-SETTINGS!$B$2)*J2+SETTINGS!$B$2*Q2),2)"
    assert (
        workbook["QB"]["G2"].value
        == "=ROUND(100*((1-SETTINGS!$B$2)*N2+SETTINGS!$B$2*P2),2)"
    )
    assert workbook["SETTINGS"]["B2"].value == 0.2
    assert workbook["QB"].max_row == 3
    assert workbook["K"].max_row == 2
    assert workbook["INJURED"].max_row == 2
    assert workbook["INJURED"]["B2"].value == "Beta"
