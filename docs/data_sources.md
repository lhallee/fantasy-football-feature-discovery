# Public data sources and coverage

## Source of record

The automated pipeline uses public nflverse Parquet releases. nflverse documents nightly player-stat updates after games, daily roster and depth updates, and frequent in-season snap updates. [nflverse update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)

The frozen source cutoff is 2026-08-10 03:11:22 UTC, or August 9 at 11:11 PM Eastern. The files were reverified on 2026-08-10 at 04:40:53 UTC. [manifest.json](../data/raw/manifest.json) version 2 separates retrieval time from later verification time and records each URL, local path, size, and SHA-256.

A live GitHub release audit found all 94 declared assets present with identical sizes. GitHub published SHA-256 digests for 48 current or newer assets, and every published digest matched. Release assets can change in place, so the local digest identifies the analyzed bytes more precisely than the mutable URL.

| Source | Coverage | Files | Bytes | Project use |
|---|---:|---:|---:|---|
| Weekly player stats | 1999-2025 | 27 | 20,206,597 | Raw game-level passing, rushing, receiving, return, fumble, and kicking counts |
| Weekly rosters | 2002-2026 | 25 | 14,157,966 | Player cohorts, backups, zero scorers, status, team, and biography |
| Depth charts | 2002-2026 | 25 | 14,979,097 | Catalog metadata and a sensitivity source; excluded from primary model inputs |
| Player ID master | Historical-current | 1 | 3,404,185 | GSIS key, crosswalks, measurements, and draft fields |
| Snap counts | 2012-2025 | 14 | 2,918,884 | Optional recent-history utilization tier |
| Draft picks | 1980-2026 | 1 | 699,050 | Draft and rookie audit source |
| Combine | 2000-2026 | 1 | 374,318 | Candidate athletic measurements |

Endpoint templates are:

```text
https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{YEAR}.parquet
https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{YEAR}.parquet
https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{YEAR}.parquet
https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{YEAR}.parquet
https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet
https://github.com/nflverse/nflverse-data/releases/download/combine/combine.parquet
https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.parquet
```

The nflverse data dictionary defines the weekly-stat fields. [Player-stat data dictionary](https://nflreadr.nflverse.com/articles/dictionary_player_stats.html)

## Compiled data

| Artifact | Rows or records | Purpose |
|---|---:|---|
| `player_games.parquet` | 475,626 | Identified player-game rows, raw counts, and reconstructed scores |
| `player_seasons.parquet` | 66,907 | Regular-season raw aggregates, roster availability, and snaps |
| `preseason_players.parquet` | 56,505 | Historical Week 1 proxies for Phase 1 plus current snapshot metadata |
| `modeling_table.parquet` | 18,630 | Archived Phase 1 player-season table with one- and two-year lags |
| `players_2026.py` | 2,930 | Typed current-snapshot catalog, including 958 fantasy-eligible players |

The player-game table covers every season from 1999-2025 and 11,372 identified players. Historical roster proxies cover 2002-2026. The current raw roster, processed table, and generated catalog contain identical GSIS ID sets across all 32 teams. Backups are retained; current depth ranks extend to 15.

The current `data/` and production `artifacts/` directories use 83,376,799 bytes. The richer Phase 2 Python catalog adds 9,082,576 bytes, for 92,459,375 bytes total. The complete size-cap scope, including isolated Phase 2 research runs, is recorded in the production experiment manifest and remains below the project's conservative 900 MiB guard.

## Cohort and schema decisions

Weekly rosters begin in 2002, so 1999-2001 statistics are lag history but cannot define complete roster cohorts. Historical files do not carry reliable as-of timestamps. The Phase 1 `preseason_players` cohort is therefore a Week 1 proxy, not a proven preseason snapshot.

Phase 2 production does not use historical target-year Week 1 membership. For target season `t`, it constructs candidates from the `t - 1` regular-season roster and target-year draft or combine entrants. A candidate without a box-score row in `t` receives zero target points. This avoids conditioning on eventual participation while matching the August 9 information origin more closely.

The source statistics include 530 unresolved rows with no player ID, display name, or position. Of those, 156 use the generic `player_name="Team"`. They contribute 6.68 PPR points across 2001-2025 and cannot represent individual players, so the builder excludes them and records the count.

Depth charts change schema after 2024. The builder can freeze timestamped recent files at an explicit date and keeps normalized depth metadata in the catalog. Depth was removed from the primary model candidate menu because historical Week 1 depth and an August current depth snapshot are not time-comparable, and because depth rank is a human-produced role ranking rather than a raw player measurement. Current source depth rank is available for 909 of 958 fantasy-position players; 49 unranked players receive tier 3 for catalog display only.

Current status is also excluded from modeling. About half of historical Week 1 fantasy cohorts are marked active, while 936 of 958 current fantasy-position players were active in the August camp file. Using that mismatched field inflated fringe-player forecasts.

## Current-snapshot scope

"All players" means every player in the nflverse August 9 team-roster file, not every unattached free agent. The source contains 2,852 `ACT`, 36 `RES`, 28 `E14`, 11 `RET`, and 3 `CUT` records. The full catalog preserves them; default queries exclude cut and retired records.

The current snapshot has no 2026 game statistics, which is expected before the regular season. It also lacks a current injury feed, schedule effects, and team-context features. A post-cut refresh would answer a later information-origin question and must be versioned separately rather than overwriting the August 9 prospective forecast.

## Public sources considered but not used

- nflverse play-by-play covers 1999 onward and would add about 488 MB. It remains under the 1 GB cap, but the player-week tables already answer this first season-level question with simpler lineage and lower compute.
- Next Gen Stats begins in 2016 and excludes players below attempt thresholds. Its missingness is selected, so it needs a separate recent-era experiment.
- nflverse injuries stop after 2024 because the upstream feed ended. They cannot support a current 2026 injury input.
- Schedules, PFR advanced statistics, and team context are plausible future additions, but are not evaluated here.
- Sleeper documents a public player endpoint, but its terms updated July 24, 2026 prohibit automated querying or extraction without written consent. This project does not call it. [Sleeper API documentation](https://docs.sleeper.com/#players), [Sleeper terms](https://sleeper.com/terms)

## License and attribution

nflverse-data is published under CC BY 4.0. [NOTICE.md](../NOTICE.md) identifies the source, license, and project modifications. nflreadr notes that underlying NFL and third-party data remain subject to their owners' terms. This project is not a blanket commercial-use clearance. [nflverse-data license](https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md)
