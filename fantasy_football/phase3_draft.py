"""Combine fantasy projections and injury risk into a draft preference index."""

from __future__ import annotations

import pandas as pd

from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


POSITIONS = ("QB", "RB", "WR", "TE", "K")
SHEET_ORDER = ("ALL", *POSITIONS, "INJURED")
DEFAULT_FANTASY_WEIGHT = 0.8
BOARD_COLUMNS = (
    "display_rank",
    "candidate_name",
    "team",
    "model_position",
    "combined_draft_score",
    "predicted_points",
    "injury_probability",
    "health_probability",
    "fantasy_score_percentile",
    "position_raw_points_rank",
    "risk_band",
    "player_id",
)
BOARD_HEADERS = (
    "Rank",
    "Player",
    "Team",
    "Position",
    "Risk-Adjusted Draft Score",
    "Projected 2026 Fantasy Points",
    "Predicted Physical Injury-Report Probability",
    "No-Injury-Report Probability",
    "Within-Position Fantasy Percentile",
    "Original Position Fantasy Points Rank",
    "Risk Band",
    "Player ID",
)
INJURY_COLUMNS = (
    "current_injury_or_designation",
    "current_injury_status",
    "expected_return",
    "evidence_date",
    "status_as_of",
    "source_publisher",
    "source_tier",
    "source_url",
    "source_player_url",
)
INJURY_HEADERS = (
    "Current Injury or Designation",
    "Current Status",
    "Expected Return",
    "Evidence Date",
    "Checked As Of",
    "Source Publisher",
    "Evidence Level",
    "Source URL",
    "Player Source URL",
)


def combine_draft_scores(
    predictions: pd.DataFrame,
    *,
    fantasy_weight: float = DEFAULT_FANTASY_WEIGHT,
) -> pd.DataFrame:
    """Return an explicit fantasy-value and health-probability preference index."""
    # predictions: (n, c_predictions)
    if not 0.0 <= fantasy_weight <= 1.0:
        raise ValueError("fantasy_weight must lie from 0 through 1.")
    required = {
        "player_id",
        "model_position",
        "predicted_points",
        "injury_probability",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Draft predictions lack columns: {sorted(missing)}")
    board = predictions.copy()  # (n, c_predictions)
    board["injury_probability"] = pd.to_numeric(
        board["injury_probability"], errors="raise"
    ).clip(0.0, 1.0)  # (n,)
    board["health_probability"] = 1.0 - board["injury_probability"]  # (n,)
    board["fantasy_score_percentile"] = board.groupby(
        "model_position",
        observed=True,
    )["predicted_points"].rank(method="average", pct=True)  # (n,)
    board["combined_draft_score"] = 100.0 * (
        fantasy_weight * board["fantasy_score_percentile"]
        + (1.0 - fantasy_weight) * board["health_probability"]
    )  # (n,)
    board["position_raw_points_rank"] = board.groupby(
        "model_position",
        observed=True,
    )["predicted_points"].rank(method="min", ascending=False).astype("int32")  # (n,)
    board["position_combined_rank"] = board.groupby(
        "model_position",
        observed=True,
    )["combined_draft_score"].rank(method="min", ascending=False).astype("int32")  # (n,)
    board["risk_band"] = pd.cut(
        board["injury_probability"],
        bins=[-0.001, 0.25, 0.50, 1.001],
        labels=["lower", "moderate", "higher"],
    ).astype("string")  # (n,)
    board.sort_values(
        ["model_position", "combined_draft_score", "predicted_points", "player_id"],
        ascending=[True, False, False, True],
        inplace=True,
        kind="stable",
    )
    board.reset_index(drop=True, inplace=True)
    return board  # (n, c_predictions + 7)


def query_risk_adjusted_players(
    board: pd.DataFrame,
    *,
    position: str | None = None,
    team: str | None = None,
    limit: int | None = 25,
) -> pd.DataFrame:
    """Filter a completed board and preserve combined-score order."""
    # board: (n, c_board)
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative or None.")
    selected = board.copy()  # (n, c_board)
    if position is not None:
        wanted_position = position.strip().upper()
        if wanted_position == "PK":
            wanted_position = "K"
        selected = selected.loc[
            selected["model_position"].astype("string").str.upper().eq(wanted_position)
        ]  # (n_position, c_board)
    if team is not None:
        wanted_team = team.strip().upper()
        selected = selected.loc[
            selected["team"].astype("string").str.upper().eq(wanted_team)
        ]  # (n_filtered, c_board)
    selected = selected.sort_values(
        ["combined_draft_score", "predicted_points", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    )  # (n_filtered, c_board)
    if limit is not None:
        selected = selected.head(limit)  # (min(n_filtered, limit), c_board)
    return selected.reset_index(drop=True)  # (n_returned, c_board)


def _style_sheet(sheet, row_count: int, column_count: int) -> None:
    """Apply one compact, readable draft-board style."""
    navy = "17365D"
    blue = "D9EAF7"
    sheet.freeze_panes = "A2"
    last_column = get_column_letter(column_count)
    sheet.auto_filter.ref = f"A1:{last_column}{max(row_count + 1, 1)}"
    sheet.sheet_view.showGridLines = False
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor=navy)
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in range(2, row_count + 2):
        if row % 2 == 0:
            for cell in sheet[row]:
                cell.fill = PatternFill("solid", fgColor=blue)
        for column in (1, 5, 6, 7, 8, 9, 10):
            sheet.cell(row, column).alignment = Alignment(horizontal="center")
    widths = (8, 24, 9, 10, 24, 26, 33, 27, 27, 28, 12, 18)
    injury_widths = (28, 20, 18, 14, 14, 18, 22, 42, 42)
    widths = widths + injury_widths[: max(column_count - len(widths), 0)]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    if row_count:
        for column in (5, 6):
            for row in range(2, row_count + 2):
                sheet.cell(row, column).number_format = "0.00"
        for column in (7, 8, 9):
            for row in range(2, row_count + 2):
                sheet.cell(row, column).number_format = "0.0%"
        sheet.conditional_formatting.add(
            f"E2:E{row_count + 1}",
            ColorScaleRule(
                start_type="min",
                start_color="F8696B",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFEB84",
                end_type="max",
                end_color="63BE7B",
            ),
        )
        sheet.conditional_formatting.add(
            f"G2:G{row_count + 1}",
            ColorScaleRule(
                start_type="min",
                start_color="63BE7B",
                mid_type="percentile",
                mid_value=50,
                mid_color="FFEB84",
                end_type="max",
                end_color="F8696B",
            ),
        )
    sheet.column_dimensions["L"].hidden = True


def _sorted_sheet_rows(board: pd.DataFrame) -> pd.DataFrame:
    """Sort one workbook page and assign its contiguous visible rank."""
    # board: (n, c_board)
    rows = board.sort_values(
        ["combined_draft_score", "predicted_points", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    ).copy()  # (n, c_board)
    rows["display_rank"] = range(1, len(rows) + 1)  # (n,)
    return rows  # (n, c_board + 1)


def _write_sheet(
    workbook: Workbook,
    name: str,
    rows: pd.DataFrame,
    *,
    injury_evidence: bool,
) -> None:
    """Write one sorted draft or current-injury worksheet."""
    # rows: (n, c_rows)
    sheet = workbook.create_sheet(name)
    headers = BOARD_HEADERS + (INJURY_HEADERS if injury_evidence else ())
    columns = BOARD_COLUMNS + (INJURY_COLUMNS if injury_evidence else ())
    sheet.append(list(headers))
    page = _sorted_sheet_rows(rows)  # (n, c_rows + 1)
    page_values = page.loc[:, columns]  # (n, len(columns))
    for row in page_values.itertuples(index=False, name=None):
        sheet.append(list(row))
    _style_sheet(sheet, len(page_values), len(columns))


def empty_current_injuries() -> pd.DataFrame:
    """Return a typed, empty injury-evidence table for undated rebuilds."""
    return pd.DataFrame(columns=["player_id", *INJURY_COLUMNS])  # (0, 10)


def write_risk_adjusted_workbook(
    board: pd.DataFrame,
    output_path: Path,
    *,
    current_injuries: pd.DataFrame | None = None,
    injury_as_of: str = "",
) -> Path:
    """Write active and injured combined-score draft pages without a read-me tab."""
    # board: (n, c_board)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = (
        empty_current_injuries()
        if current_injuries is None
        else current_injuries.copy()
    )  # (n_injured, 10)
    required_injury_columns = {"player_id", *INJURY_COLUMNS}
    missing = sorted(required_injury_columns.difference(evidence.columns))
    if missing:
        raise ValueError(f"Current-injury table is missing columns: {missing!r}.")
    if evidence["player_id"].duplicated().any():
        raise ValueError("Current-injury table requires unique player IDs.")
    unknown_ids = sorted(set(evidence["player_id"]).difference(board["player_id"]))
    if unknown_ids:
        raise ValueError(f"Current-injury table contains unknown player IDs: {unknown_ids!r}.")
    if len(evidence) and not injury_as_of:
        raise ValueError("A dated injury screen requires injury_as_of.")
    if len(evidence) and not evidence["status_as_of"].astype("string").eq(injury_as_of).all():
        raise ValueError("Current-injury evidence does not match the workbook as-of date.")

    injured_ids = set(evidence["player_id"])
    active_board = board.loc[~board["player_id"].isin(injured_ids)].copy()  # (n_active, c_board)
    injured_board = board.loc[board["player_id"].isin(injured_ids)].merge(
        evidence.loc[:, ["player_id", *INJURY_COLUMNS]],
        on="player_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )  # (n_injured, c_board + len(INJURY_COLUMNS))
    workbook = Workbook()
    workbook.remove(workbook.active)
    _write_sheet(workbook, "ALL", active_board, injury_evidence=False)
    for position in POSITIONS:
        position_rows = active_board.loc[
            active_board["model_position"].eq(position)
        ].copy()  # (n_position, c_board)
        _write_sheet(workbook, position, position_rows, injury_evidence=False)
    _write_sheet(workbook, "INJURED", injured_board, injury_evidence=True)
    if tuple(workbook.sheetnames) != SHEET_ORDER:
        raise AssertionError("Workbook sheet order does not match its public contract.")
    workbook.properties.title = "2026 Injury-Adjusted Fantasy Football Draft Board"
    workbook.properties.subject = (
        "Risk-adjusted draft rankings with a dated current-injury listing screen"
    )
    workbook.save(output_path)
    return output_path
