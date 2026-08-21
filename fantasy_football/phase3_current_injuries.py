"""Freeze and match a dated public current-injury screen to the 2026 board."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import pandas as pd
import requests

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .phase3_draft import write_risk_adjusted_workbook


MFL_INJURY_URL = "https://api.myfantasyleague.com/2026/export"
MFL_DOCUMENTATION_URL = "https://myfantasyleague.wordpress.com/2008/08/06/developer-api/"
SNAPSHOT_DATE = "2026-08-14"
FANTASY_POSITIONS = frozenset({"QB", "RB", "FB", "WR", "TE", "PK", "K"})
INJURY_STATUSES = frozenset(
    {"QUESTIONABLE", "DOUBTFUL", "OUT", "IR", "IR-PUP", "IR-NFI", "PUP", "NFI"}
)
NON_INJURY_DETAILS = frozenset({"ILLNESS", "COVID", "PERSONAL", "SUSPENSION"})
REQUIRED_BOARD_COLUMNS = frozenset(
    {"player_id", "candidate_name", "team", "model_position"}
)
OPERATIONAL_CODE_PATHS = (
    "fantasy_football/phase3_current_injuries.py",
    "fantasy_football/phase3_draft.py",
)
TEAM_ALIASES = {
    "ARI": "ARI",
    "AZ": "ARI",
    "GBP": "GB",
    "JAC": "JAX",
    "JAX": "JAX",
    "KCC": "KC",
    "LA": "LAR",
    "LAR": "LAR",
    "LVR": "LV",
    "NEP": "NE",
    "NOS": "NO",
    "SFO": "SF",
    "TBB": "TB",
    "WAS": "WAS",
    "WSH": "WAS",
}
SUPPLEMENTAL_EVIDENCE = (
    {
        "source_id": "official-nfl-jalen-walthall-2026-07-31",
        "source_team": "NYJ",
        "source_player_name": "Jalen Walthall",
        "source_position": "WR",
        "current_injury_or_designation": "Undisclosed",
        "current_injury_status": "Reserve/Injured",
        "expected_return": "",
        "evidence_date": "2026-07-31",
        "source_updated_at_utc": "",
        "source_url": "https://www.nfl.com/transactions/league/reserve-list/2026/7",
        "source_player_url": "",
        "source_publisher": "NFL",
        "source_tier": "Official reserve transaction",
        "current_injury_listing": True,
    },
    {
        "source_id": "official-nfl-gee-scott-jr-2026-08-01",
        "source_team": "NYJ",
        "source_player_name": "Gee Scott Jr.",
        "source_position": "WR",
        "current_injury_or_designation": "Undisclosed",
        "current_injury_status": "Reserve/Injured",
        "expected_return": "",
        "evidence_date": "2026-08-01",
        "source_updated_at_utc": "",
        "source_url": "https://www.nfl.com/transactions/league/reserve-list/2026/8",
        "source_player_url": "",
        "source_publisher": "NFL",
        "source_tier": "Official reserve transaction",
        "current_injury_listing": True,
    },
    {
        "source_id": "cbs-corey-rucker-2026-07-23",
        "source_team": "LV",
        "source_player_name": "Corey Rucker",
        "source_position": "WR",
        "current_injury_or_designation": "Undisclosed",
        "current_injury_status": "IR",
        "expected_return": "",
        "evidence_date": "2026-07-23",
        "source_updated_at_utc": "",
        "source_url": "https://www.cbssports.com/nfl/injuries/",
        "source_player_url": "",
        "source_publisher": "CBS Sports",
        "source_tier": "Secondary public tracker, independently reviewed",
        "current_injury_listing": True,
    },
    {
        "source_id": "cbs-dan-chisena-2026-07-29",
        "source_team": "CAR",
        "source_player_name": "Dan Chisena",
        "source_position": "WR",
        "current_injury_or_designation": "Undisclosed",
        "current_injury_status": "IR",
        "expected_return": "",
        "evidence_date": "2026-07-29",
        "source_updated_at_utc": "",
        "source_url": "https://www.cbssports.com/nfl/injuries/",
        "source_player_url": "",
        "source_publisher": "CBS Sports",
        "source_tier": "Secondary public tracker, independently reviewed",
        "current_injury_listing": True,
    },
    {
        "source_id": "cbs-jeremiah-webb-2026-07-26",
        "source_team": "NE",
        "source_player_name": "Jeremiah Webb",
        "source_position": "WR",
        "current_injury_or_designation": "Arm",
        "current_injury_status": "IR",
        "expected_return": "",
        "evidence_date": "2026-07-26",
        "source_updated_at_utc": "",
        "source_url": "https://www.cbssports.com/nfl/injuries/",
        "source_player_url": "",
        "source_publisher": "CBS Sports",
        "source_tier": "Secondary public tracker, independently reviewed",
        "current_injury_listing": True,
    },
    {
        "source_id": "cbs-jimmy-kibble-2026-07-25",
        "source_team": "NE",
        "source_player_name": "Jimmy Kibble",
        "source_position": "WR",
        "current_injury_or_designation": "Undisclosed",
        "current_injury_status": "IR",
        "expected_return": "",
        "evidence_date": "2026-07-25",
        "source_updated_at_utc": "",
        "source_url": "https://www.cbssports.com/nfl/injuries/",
        "source_player_url": "",
        "source_publisher": "CBS Sports",
        "source_tier": "Secondary public tracker, independently reviewed",
        "current_injury_listing": True,
    },
    {
        "source_id": "cbs-kye-robichaux-2026-07-31",
        "source_team": "DET",
        "source_player_name": "Kye Robichaux",
        "source_position": "RB",
        "current_injury_or_designation": "Undisclosed",
        "current_injury_status": "Questionable",
        "expected_return": "",
        "evidence_date": "2026-07-31",
        "source_updated_at_utc": "",
        "source_url": "https://www.cbssports.com/nfl/injuries/",
        "source_player_url": "",
        "source_publisher": "CBS Sports",
        "source_tier": "Secondary public tracker, independently reviewed",
        "current_injury_listing": True,
    },
    {
        "source_id": "cbs-kyron-hudson-2026-08-11",
        "source_team": "CHI",
        "source_player_name": "Kyron Hudson",
        "source_position": "WR",
        "current_injury_or_designation": "Undisclosed",
        "current_injury_status": "IR",
        "expected_return": "",
        "evidence_date": "2026-08-11",
        "source_updated_at_utc": "",
        "source_url": "https://www.cbssports.com/nfl/injuries/",
        "source_player_url": "",
        "source_publisher": "CBS Sports",
        "source_tier": "Secondary public tracker, independently reviewed",
        "current_injury_listing": True,
    },
    {
        "source_id": "cbs-malick-meiga-2026-08-12",
        "source_team": "CAR",
        "source_player_name": "Malick Meiga",
        "source_position": "WR",
        "current_injury_or_designation": "Undisclosed",
        "current_injury_status": "Questionable",
        "expected_return": "",
        "evidence_date": "2026-08-12",
        "source_updated_at_utc": "",
        "source_url": "https://www.cbssports.com/nfl/injuries/",
        "source_player_url": "",
        "source_publisher": "CBS Sports",
        "source_tier": "Secondary public tracker, independently reviewed",
        "current_injury_listing": True,
    },
    {
        "source_id": "cbs-princeton-fant-2026-08-06",
        "source_team": "DAL",
        "source_player_name": "Princeton Fant",
        "source_position": "TE",
        "current_injury_or_designation": "ACL and MCL knee",
        "current_injury_status": "IR",
        "expected_return": "",
        "evidence_date": "2026-08-06",
        "source_updated_at_utc": "",
        "source_url": "https://www.cbssports.com/nfl/injuries/",
        "source_player_url": "",
        "source_publisher": "CBS Sports",
        "source_tier": "Secondary public tracker, independently reviewed",
        "current_injury_listing": True,
    },
    {
        "source_id": "cbs-rj-maryland-2026-08-13",
        "source_team": "GB",
        "source_player_name": "R.J. Maryland",
        "source_position": "TE",
        "current_injury_or_designation": "Hamstring",
        "current_injury_status": "Questionable",
        "expected_return": "",
        "evidence_date": "2026-08-13",
        "source_updated_at_utc": "",
        "source_url": "https://www.cbssports.com/nfl/injuries/",
        "source_player_url": "",
        "source_publisher": "CBS Sports",
        "source_tier": "Secondary public tracker, independently reviewed",
        "current_injury_listing": True,
    },
)


@dataclass(frozen=True, slots=True)
class CurrentInjuryScreen:
    """Frozen source rows and their exact player-board reconciliation."""

    source_rows: pd.DataFrame
    player_screen: pd.DataFrame
    current_injuries: pd.DataFrame
    unmatched_source_rows: pd.DataFrame
    retrieved_at_utc: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    return "".join(token for token in tokens if token not in suffixes)


def _mfl_name(value: object) -> str:
    text = str(value).strip()
    if "," not in text:
        return text
    family_name, given_name = text.split(",", maxsplit=1)
    return f"{given_name.strip()} {family_name.strip()}"


def _normalized_team(value: object) -> str:
    team = str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)


def _response_json(session: requests.Session, parameters: Mapping[str, str]) -> Mapping[str, Any]:
    response = session.get(
        MFL_INJURY_URL,
        params=parameters,
        headers={"User-Agent": "fantasy-football-feature-discovery/0.1"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _player_details(
    session: requests.Session,
    player_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for start in range(0, len(player_ids), 100):
        batch = player_ids[start : start + 100]
        payload = _response_json(
            session,
            {
                "TYPE": "players",
                "DETAILS": "1",
                "JSON": "1",
                "PLAYERS": ",".join(batch),
            },
        )
        batch_records = payload.get("players", {}).get("player", [])
        if isinstance(batch_records, Mapping):
            batch_records = [batch_records]
        records.extend(batch_records)
    return records


def parse_mfl_payloads(
    injury_payload: Mapping[str, Any],
    player_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Combine documented MFL injury and player exports into source rows."""
    injury_block = injury_payload.get("injuries", {})
    injury_records = injury_block.get("injury", [])
    if isinstance(injury_records, Mapping):
        injury_records = [injury_records]
    players_by_id = {str(record["id"]): record for record in player_records}
    source_timestamp = int(injury_block["timestamp"])
    source_updated_at_utc = datetime.fromtimestamp(source_timestamp, UTC).isoformat()
    records: list[dict[str, Any]] = []
    for injury in injury_records:
        source_id = str(injury["id"])
        player = players_by_id.get(source_id)
        if player is None:
            raise ValueError(f"MFL player lookup omitted injury ID {source_id!r}.")
        status = str(injury.get("status", "")).strip()
        details = str(injury.get("details", "")).strip()
        position = str(player.get("position", "")).strip().upper()
        current_listing = (
            position in FANTASY_POSITIONS
            and status.upper() in INJURY_STATUSES
            and details.upper() not in NON_INJURY_DETAILS
        )
        records.append(
            {
                "source_id": source_id,
                "source_team": str(player.get("team", "")).strip(),
                "source_player_name": _mfl_name(player.get("name", "")),
                "source_position": "K" if position == "PK" else position,
                "current_injury_or_designation": details,
                "current_injury_status": status,
                "expected_return": str(injury.get("exp_return", "")).strip(),
                "evidence_date": source_updated_at_utc[:10],
                "source_updated_at_utc": source_updated_at_utc,
                "source_url": f"{MFL_INJURY_URL}?TYPE=injuries&JSON=1",
                "source_player_url": "",
                "source_publisher": "MyFantasyLeague",
                "source_tier": "Documented public injury API",
                "current_injury_listing": current_listing,
            }
        )
    source_rows = pd.DataFrame.from_records(records)  # (n_source, 14)
    if source_rows.empty:
        raise ValueError("MFL injury feed did not contain player rows.")
    source_rows["source_team_key"] = source_rows["source_team"].map(
        _normalized_team
    )  # (n_source,)
    source_rows["player_name_key"] = source_rows["source_player_name"].map(
        _normalized_name
    )  # (n_source,)
    if source_rows.duplicated(["source_team_key", "player_name_key"]).any():
        raise ValueError("MFL injury snapshot contains duplicate team-player rows.")
    return source_rows.sort_values(
        ["current_injury_listing", "source_team", "source_player_name"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)  # (n_source, 16)


def fetch_mfl_injury_snapshot() -> tuple[pd.DataFrame, str]:
    """Download the documented MFL injury and player exports once."""
    with requests.Session() as session:
        injury_payload = _response_json(
            session,
            {"TYPE": "injuries", "JSON": "1"},
        )
        injury_records = injury_payload.get("injuries", {}).get("injury", [])
        if isinstance(injury_records, Mapping):
            injury_records = [injury_records]
        player_ids = [str(record["id"]) for record in injury_records]
        player_records = _player_details(session, player_ids)
    retrieved_at_utc = datetime.now(UTC).isoformat()
    source_rows = parse_mfl_payloads(injury_payload, player_records)  # (n_mfl, 16)
    supplemental_rows = pd.DataFrame.from_records(SUPPLEMENTAL_EVIDENCE)  # (11, 14)
    supplemental_rows["source_team_key"] = supplemental_rows["source_team"].map(
        _normalized_team
    )  # (11,)
    supplemental_rows["player_name_key"] = supplemental_rows["source_player_name"].map(
        _normalized_name
    )  # (11,)
    source_rows = pd.concat(
        [source_rows, supplemental_rows.loc[:, source_rows.columns]],
        ignore_index=True,
    )  # (n_mfl + 11, 16)
    if source_rows.duplicated(["source_team_key", "player_name_key"]).any():
        raise ValueError("Combined injury snapshot contains duplicate team-player rows.")
    return source_rows, retrieved_at_utc


def match_current_injuries(
    board: pd.DataFrame,
    source_rows: pd.DataFrame,
    *,
    as_of_date: str,
    retrieved_at_utc: str,
) -> CurrentInjuryScreen:
    """Match every available board player to one dated injury-screen result."""
    # board: (n_players, c_board); source_rows: (n_source, c_source)
    missing_board = sorted(REQUIRED_BOARD_COLUMNS.difference(board.columns))
    if missing_board:
        raise ValueError(f"Current-injury screen board is missing columns: {missing_board!r}.")
    if board["player_id"].duplicated().any():
        raise ValueError("Current-injury screen requires unique player IDs.")
    board_keys = board.loc[
        :, ["player_id", "candidate_name", "team", "model_position"]
    ].copy()  # (n_players, 4)
    board_keys["source_team_key"] = board_keys["team"].map(_normalized_team)  # (n_players,)
    board_keys["player_name_key"] = board_keys["candidate_name"].map(
        _normalized_name
    )  # (n_players,)
    eligible_source = source_rows.loc[
        source_rows["current_injury_listing"]
    ].copy()  # (n_source_eligible, c_source)
    evidence_columns = [
        "source_team_key",
        "player_name_key",
        "source_player_name",
        "current_injury_or_designation",
        "current_injury_status",
        "expected_return",
        "evidence_date",
        "source_url",
        "source_player_url",
        "source_publisher",
        "source_tier",
    ]
    player_screen = board_keys.merge(
        eligible_source.loc[:, evidence_columns],
        on=["source_team_key", "player_name_key"],
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )  # (n_players, 16)
    player_screen["current_injury_listing"] = player_screen["_merge"].eq(
        "both"
    )  # (n_players,)
    player_screen["screen_result"] = player_screen["current_injury_listing"].map(
        {True: "Current public injury listing", False: "No current listing found"}
    )  # (n_players,)
    player_screen["status_as_of"] = as_of_date  # (n_players,)
    player_screen["retrieved_at_utc"] = retrieved_at_utc  # (n_players,)
    player_screen["match_method"] = player_screen["current_injury_listing"].map(
        {True: "normalized_name_and_team", False: "screened_no_match"}
    )  # (n_players,)
    player_screen.drop(columns=["_merge"], inplace=True)  # (n_players, 20)
    current_injuries = player_screen.loc[
        player_screen["current_injury_listing"]
    ].copy()  # (n_injured, 20)

    matched_source_keys = set(
        zip(
            current_injuries["source_team_key"],
            current_injuries["player_name_key"],
            strict=True,
        )
    )
    source_keys = list(
        zip(
            eligible_source["source_team_key"],
            eligible_source["player_name_key"],
            strict=True,
        )
    )
    unmatched_mask = pd.Series(
        [key not in matched_source_keys for key in source_keys],
        index=eligible_source.index,
        dtype="bool",
    )  # (n_source_eligible,)
    unmatched_source_rows = eligible_source.loc[
        unmatched_mask
    ].copy()  # (n_unmatched_source, c_source)
    return CurrentInjuryScreen(
        source_rows=source_rows,
        player_screen=player_screen,
        current_injuries=current_injuries,
        unmatched_source_rows=unmatched_source_rows,
        retrieved_at_utc=retrieved_at_utc,
    )


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_current_injury_artifacts(
    root: Path,
    screen: CurrentInjuryScreen,
    *,
    as_of_date: str,
) -> dict[str, Path]:
    """Persist the source snapshot, player audit, matches, and their hashes."""
    phase3 = root / "experiments" / "phase3"
    paths = {
        "source_snapshot": (
            phase3
            / "data"
            / "current_injuries"
            / f"mfl_and_official_injuries_{as_of_date}.csv"
        ),
        "player_screen": (
            phase3 / "artifacts" / f"current_injury_screen_{as_of_date}.csv"
        ),
        "current_injuries": (
            phase3 / "artifacts" / f"current_injuries_{as_of_date}.csv"
        ),
        "unmatched_source": (
            phase3 / "artifacts" / f"unmatched_source_injuries_{as_of_date}.csv"
        ),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    screen.source_rows.to_csv(paths["source_snapshot"], index=False)
    screen.player_screen.to_csv(paths["player_screen"], index=False)
    screen.current_injuries.to_csv(paths["current_injuries"], index=False)
    screen.unmatched_source_rows.to_csv(paths["unmatched_source"], index=False)
    manifest_path = phase3 / "current_injury_manifest_v1.json"
    manifest = {
        "manifest_version": 1,
        "status_as_of": as_of_date,
        "retrieved_at_utc": screen.retrieved_at_utc,
        "sources": {
            "primary": {
                "publisher": "MyFantasyLeague",
                "url": f"{MFL_INJURY_URL}?TYPE=injuries&JSON=1",
                "documentation_url": MFL_DOCUMENTATION_URL,
            },
            "supplemental_evidence": [
                {
                    "publisher": record["source_publisher"],
                    "url": record["source_url"],
                    "player": record["source_player_name"],
                    "evidence_date": record["evidence_date"],
                }
                for record in SUPPLEMENTAL_EVIDENCE
            ],
            "interpretation": (
                "Affirmative current public injury listings. Absence means no listing was "
                "found, not confirmed health."
            ),
        },
        "available_players_screened": int(len(screen.player_screen)),
        "current_injury_listings": int(len(screen.current_injuries)),
        "players_without_current_listing": int(
            (~screen.player_screen["current_injury_listing"]).sum()
        ),
        "unmatched_fantasy_source_rows": int(len(screen.unmatched_source_rows)),
        "operational_code_sha256": {
            relative: _sha256(root / relative) for relative in OPERATIONAL_CODE_PATHS
        },
        "files": {
            label: {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for label, path in paths.items()
        },
    }
    _write_json(manifest_path, manifest)
    paths["manifest"] = manifest_path
    return paths


def rebuild_current_injury_workbook(
    root: Path,
    *,
    as_of_date: str = SNAPSHOT_DATE,
) -> tuple[Path, CurrentInjuryScreen]:
    """Refresh the dated screen and rebuild the risk-adjusted workbook."""
    board = pd.read_parquet(
        root / "experiments" / "phase3" / "artifacts" / "injury_adjusted_players_2026.parquet"
    )  # (958, c_board)
    available_board = board.loc[board["available_for_draft"]].copy()  # (951, c_board)
    source_rows, retrieved_at_utc = fetch_mfl_injury_snapshot()
    screen = match_current_injuries(
        available_board,
        source_rows,
        as_of_date=as_of_date,
        retrieved_at_utc=retrieved_at_utc,
    )
    write_current_injury_artifacts(root, screen, as_of_date=as_of_date)
    output_path = root / "outputs" / "injury_adjusted_draft_board_2026.xlsx"
    write_risk_adjusted_workbook(
        available_board,
        output_path,
        current_injuries=screen.current_injuries,
        injury_as_of=as_of_date,
    )
    return output_path, screen


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze current injury listings and rebuild the 2026 draft workbook."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--as-of", default=SNAPSHOT_DATE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dated current-injury screen and workbook rebuild."""
    arguments = _parse_args(argv)
    output_path, screen = rebuild_current_injury_workbook(
        arguments.root.resolve(),
        as_of_date=str(arguments.as_of),
    )
    print(
        f"Wrote {output_path} with {len(screen.current_injuries)} current injury listings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
