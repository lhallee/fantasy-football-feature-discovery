"""Write the final 2026 ranking workbook with a tunable injury weight."""

from __future__ import annotations

import pandas as pd

from pathlib import Path
from typing import Mapping, Sequence

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet


POSITIONS = ("QB", "RB", "WR", "TE", "K")
BOARD_SHEETS = ("ALL", *POSITIONS)
SHEET_ORDER = (
    "README",
    "SETTINGS",
    *BOARD_SHEETS,
    "INJURED",
    "MODEL_EVAL",
    "WEIGHT_ANALYSIS",
    "FEATURES_POINTS",
    "FEATURES_INJURY",
    "FAMILY_ABLATION",
)
CUSTOM_SCORE = "__custom_score_formula__"
BOARD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("display_rank", "Rank"),
    ("candidate_name", "Player"),
    ("team", "Team"),
    ("model_position", "Pos"),
    ("overall_draft_score", "Overall Draft Score (replacement-adjusted)"),
    ("draft_score", "Position Draft Score (default weight)"),
    (CUSTOM_SCORE, "Custom-Weight Score (weight from SETTINGS)"),
    ("predicted_points", "Projected 2026 Points"),
    ("vorp", "Points Over Replacement"),
    ("overall_value", "Overall Value (points over replacement / max)"),
    ("points_p10", "Points P10"),
    ("points_p90", "Points P90"),
    ("position_points_rank", "Pos Points Rank"),
    ("points_percentile", "Pos Points Percentile"),
    ("injury_probability", "Missed-Time Injury Probability"),
    ("health_probability", "Healthy Probability"),
    ("relative_health", "Healthy Probability vs Position Mean (+0.5)"),
    ("risk_band", "Injury Risk Band"),
    ("current_injury_listing", "Current Injury Listing (screen date)"),
    ("current_injury_status", "Current Listing Status"),
    ("is_rookie", "Rookie"),
    ("age", "Age"),
    ("points_confidence", "Points Model Confidence"),
    ("roster_status", "Roster Status (Aug 9)"),
    ("player_id", "Player ID"),
)
INJURED_EXTRA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("current_injury_or_designation", "Current Injury or Designation"),
    ("expected_return", "Expected Return"),
    ("evidence_date", "Evidence Date"),
    ("source_publisher", "Source Publisher"),
    ("source_tier", "Evidence Level"),
    ("source_url", "Source URL"),
)
PERCENT_COLUMNS = {
    "points_percentile",
    "overall_value",
    "injury_probability",
    "health_probability",
    "relative_health",
}
TWO_DECIMAL_COLUMNS = {
    "draft_score",
    "overall_draft_score",
    "vorp",
    "predicted_points",
    "points_p10",
    "points_p90",
    "age",
}
NAVY = "17365D"
BAND = "EAF1FA"


def _style_header(sheet: Worksheet, column_count: int) -> None:
    for index in range(1, column_count + 1):
        cell = sheet.cell(1, index)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=True
        )
    sheet.row_dimensions[1].height = 42
    sheet.freeze_panes = "A2"
    sheet.sheet_view.showGridLines = False


def _autosize(sheet: Worksheet, frame: pd.DataFrame, headers: Sequence[str]) -> None:
    for index, (column, header) in enumerate(
        zip(frame.columns, headers, strict=True), start=1
    ):
        values = frame[column].astype("string").fillna("")  # (n,)
        longest = int(values.str.len().max()) if len(values) else 0
        width = min(max(12, longest + 2, min(len(header), 28) + 2), 60)
        sheet.column_dimensions[get_column_letter(index)].width = width


def _write_table(
    workbook: Workbook,
    name: str,
    frame: pd.DataFrame,
    headers: Sequence[str] | None = None,
    *,
    percent_columns: Sequence[str] = (),
    decimal_columns: Sequence[str] = (),
) -> Worksheet:
    """Write one plain data table with a styled header and autofilter."""
    # frame: (n, c)
    sheet = workbook.create_sheet(name)
    used_headers = list(frame.columns) if headers is None else list(headers)
    sheet.append(used_headers)
    for row in frame.itertuples(index=False, name=None):
        sheet.append([_cell_value(value) for value in row])
    _style_header(sheet, len(used_headers))
    if len(frame):
        sheet.auto_filter.ref = (
            f"A1:{get_column_letter(len(used_headers))}{len(frame) + 1}"
        )
    _autosize(sheet, frame, used_headers)
    for column in percent_columns:
        if column in frame.columns:
            letter = get_column_letter(list(frame.columns).index(column) + 1)
            for row in range(2, len(frame) + 2):
                sheet[f"{letter}{row}"].number_format = "0.0%"
    for column in decimal_columns:
        if column in frame.columns:
            letter = get_column_letter(list(frame.columns).index(column) + 1)
            for row in range(2, len(frame) + 2):
                sheet[f"{letter}{row}"].number_format = "0.000"
    return sheet


def _cell_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_readme(workbook: Workbook, rows: Sequence[tuple[str, str]]) -> None:
    sheet = workbook.create_sheet("README")
    sheet.column_dimensions["A"].width = 34
    sheet.column_dimensions["B"].width = 120
    sheet.sheet_view.showGridLines = False
    sheet.append(["Section", "Content"])
    for section, text in rows:
        sheet.append([section, text])
    _style_header(sheet, 2)
    for row in range(2, len(rows) + 2):
        sheet.cell(row, 1).font = Font(bold=True)
        sheet.cell(row, 1).alignment = Alignment(vertical="top", wrap_text=True)
        sheet.cell(row, 2).alignment = Alignment(vertical="top", wrap_text=True)


def _write_settings(workbook: Workbook, default_weight: float) -> None:
    sheet = workbook.create_sheet("SETTINGS")
    sheet.column_dimensions["A"].width = 58
    sheet.column_dimensions["B"].width = 16
    sheet.column_dimensions["C"].width = 90
    sheet.sheet_view.showGridLines = False
    sheet.append(["Setting", "Value", "Note"])
    sheet.append(
        [
            "Injury weight used by the custom score column (0 to 1)",
            float(default_weight),
            "Edit this cell. Position pages recompute their custom score as "
            "100 * ((1 - weight) * position points percentile + weight * healthy probability); "
            "ALL and INJURED use the overall value percentile instead. "
            "Rows stay in default order; use the column filter to re-sort.",
        ]
    )
    sheet.append(
        [
            "Default injury weight selected by the predeclared rule",
            float(default_weight),
            "Largest weight whose discovery-fold rank accuracy against realized points "
            "stayed within 0.005 Spearman of the best weight.",
        ]
    )
    _style_header(sheet, 3)
    sheet["B2"].number_format = "0.00"
    sheet["B3"].number_format = "0.00"
    sheet["B2"].fill = PatternFill("solid", fgColor="FFF2CC")
    sheet["B2"].font = Font(bold=True)


def _sorted_board(board: pd.DataFrame, score_key: str) -> pd.DataFrame:
    rows = board.sort_values(
        [score_key, "predicted_points", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    ).copy()  # (n, c)
    rows["display_rank"] = range(1, len(rows) + 1)
    return rows


def _write_board(
    workbook: Workbook,
    name: str,
    board: pd.DataFrame,
    *,
    extra_columns: Sequence[tuple[str, str]] = (),
    overall: bool = False,
) -> None:
    """Write one ranked page with a live custom-score formula column."""
    # board: (n, c)
    sheet = workbook.create_sheet(name)
    columns = (*BOARD_COLUMNS, *extra_columns)
    headers = [header for _, header in columns]
    sheet.append(headers)
    score_key = "overall_draft_score" if overall else "draft_score"
    page = _sorted_board(board, score_key)  # (n, c + 1)
    letters = {
        key: get_column_letter(index) for index, (key, _) in enumerate(columns, start=1)
    }
    percentile_letter = letters["overall_value" if overall else "points_percentile"]
    health_letter = letters["relative_health" if overall else "health_probability"]
    for offset, row in enumerate(page.itertuples(index=False), start=2):
        values: list[object] = []
        for key, _ in columns:
            if key == CUSTOM_SCORE:
                values.append(
                    f"=ROUND(100*((1-SETTINGS!$B$2)*{percentile_letter}{offset}"
                    f"+SETTINGS!$B$2*{health_letter}{offset}),2)"
                )
            else:
                values.append(_cell_value(getattr(row, key)))
        sheet.append(values)
    _style_header(sheet, len(columns))
    n = len(page)
    if n:
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{n + 1}"
    widths = {
        "display_rank": 7,
        "candidate_name": 24,
        "team": 7,
        "model_position": 6,
        "overall_draft_score": 15,
        "draft_score": 14,
        CUSTOM_SCORE: 16,
        "predicted_points": 13,
        "vorp": 12,
        "overall_value": 12,
        "points_p10": 10,
        "points_p90": 10,
        "position_points_rank": 10,
        "points_percentile": 12,
        "injury_probability": 14,
        "health_probability": 12,
        "relative_health": 14,
        "risk_band": 11,
        "current_injury_listing": 14,
        "current_injury_status": 16,
        "is_rookie": 8,
        "age": 7,
        "points_confidence": 16,
        "roster_status": 11,
        "player_id": 13,
    }
    for key, _ in columns:
        sheet.column_dimensions[letters[key]].width = widths.get(key, 24)
    for key, _ in columns:
        letter = letters[key]
        if key in PERCENT_COLUMNS:
            number_format = "0.0%"
        elif key in TWO_DECIMAL_COLUMNS or key == CUSTOM_SCORE:
            number_format = "0.00"
        else:
            continue
        for row in range(2, n + 2):
            sheet[f"{letter}{row}"].number_format = number_format
    for row in range(2, n + 2):
        if row % 2 == 0:
            for key, _ in columns:
                sheet[f"{letters[key]}{row}"].fill = PatternFill("solid", fgColor=BAND)
    if n:
        sheet.conditional_formatting.add(
            f"{letters[score_key]}2:{letters[score_key]}{n + 1}",
            ColorScaleRule(
                start_type="min",
                start_color="F8CBAD",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFF2CC",
                end_type="max",
                end_color="C6E0B4",
            ),
        )
        sheet.conditional_formatting.add(
            f"{letters['injury_probability']}2:{letters['injury_probability']}{n + 1}",
            ColorScaleRule(
                start_type="min",
                start_color="C6E0B4",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFF2CC",
                end_type="max",
                end_color="F8CBAD",
            ),
        )


def write_final_workbook(
    board: pd.DataFrame,
    output_path: Path,
    *,
    default_weight: float,
    readme_rows: Sequence[tuple[str, str]],
    tables: Mapping[str, pd.DataFrame],
) -> Path:
    """Write the README, settings, ranked pages, and evidence tables."""
    # board: (n, c)
    required = {
        key for key, _ in BOARD_COLUMNS if key not in {CUSTOM_SCORE, "display_rank"}
    }
    required.update(key for key, _ in INJURED_EXTRA_COLUMNS)
    missing = sorted(required.difference(board.columns))
    if missing:
        raise ValueError(f"Board is missing columns: {missing!r}.")
    if board["player_id"].duplicated().any():
        raise ValueError("Board player IDs must be unique.")
    expected_tables = (
        "MODEL_EVAL",
        "WEIGHT_ANALYSIS",
        "FEATURES_POINTS",
        "FEATURES_INJURY",
        "FAMILY_ABLATION",
    )
    missing_tables = [name for name in expected_tables if name not in tables]
    if missing_tables:
        raise ValueError(f"Workbook tables are missing: {missing_tables!r}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_readme(workbook, readme_rows)
    _write_settings(workbook, default_weight)
    _write_board(workbook, "ALL", board, overall=True)
    for position in POSITIONS:
        _write_board(
            workbook, position, board.loc[board["model_position"].eq(position)]
        )
    injured = board.loc[
        board["current_injury_listing"].astype("bool")
    ]  # (n_injured, c)
    _write_board(
        workbook, "INJURED", injured, extra_columns=INJURED_EXTRA_COLUMNS, overall=True
    )
    for name in expected_tables:
        table = tables[name]
        percent_columns = [
            column
            for column in table.columns
            if column.endswith("_auc")
            or column in {"roc_auc", "prevalence", "bust_rate_top_k", "top_k_recall"}
        ]
        decimal_columns = [
            column
            for column in table.columns
            if table[column].dtype.kind == "f" and column not in percent_columns
        ]
        _write_table(
            workbook,
            name,
            table,
            percent_columns=percent_columns,
            decimal_columns=decimal_columns,
        )
    if tuple(workbook.sheetnames) != SHEET_ORDER:
        raise AssertionError(
            f"Sheet order {workbook.sheetnames!r} does not match the contract."
        )
    workbook.properties.title = "2026 Fantasy Football Final Rankings"
    workbook.properties.subject = "Full-season ESPN full-PPR point forecasts with a calibrated missed-time injury probability"
    workbook.save(output_path)
    return output_path
