# Fantasy Football Raw-Feature Discovery

This project builds a compact NFL player history, reconstructs ESPN full-PPR scoring, predicts next-season player totals, and searches for a small set of raw inputs that preserves within-position ranking. The August 9, 2026 nflverse snapshot contains 2,930 team-roster players, including 958 fantasy-eligible quarterbacks, running backs, wide receivers, tight ends, and kickers.

The offense result is a locally irreducible five-atom Extra Trees model:

- NFL draft number
- prior receiving yards
- prior rushing yards
- prior passing touchdowns
- preseason age

Those atoms expand into 24 fitted columns: raw values, one- and two-year lags, `log1p` transforms, one-year trends, age transforms, and four position indicators. The model excludes imported fantasy scores, projections, ADP, depth ranks, roster-status flags, and other composite fantasy features.

On chronological 2024-2025 retrospective evaluation, the model reached mean within-position Spearman \(\rho=0.757\), compared with 0.669 for prior-year points. MAE was 29.46 versus 29.58 points. A 500-sample player-cluster bootstrap placed the model correlation at 0.725 to 0.782 and the paired correlation gain at 0.060 to 0.116. These are retrospective results; 2026 is the first prospective season.

Kicker ranking did not beat prior-year points. Kicker forecasts remain available with a low-confidence label. An explicit K or PK query uses prior-season points by default; pass `sort_by="prediction"` to inspect model order.

## Query the 2026 catalog

```python
from fantasy_football.players_2026 import get_player, query_players

top_running_backs = query_players(position="RB", limit=12)
atlanta_options = query_players(team="ATL", limit=20)
pit_quarterbacks = query_players(team="PIT", position="QB", limit=None)
kickers_by_baseline = query_players(position="K", limit=12)
josh_allen = get_player("00-0034857")
```

`query_players` also filters by roster status, name, college, confidence, depth tier, maximum position rank, minimum predicted points, and out-of-distribution status. Team aliases such as `ARI`, `LAR`, and `WSH` are accepted. Cut and retired players remain in `PLAYERS` but are excluded from default queries.

Each `PlayerProjection` includes roster, biography, draft, identifier, prediction, interval, rank, confidence, selected-atom, prior-points, and exact fitted-feature fields. Set `fantasy_only=False` to inspect every roster position.

## Reproduce the frozen snapshot

Run from this source-project root with Python 3.11 or later. CUDA is not required. Estimator and native numerical pools are capped at four CPU threads.

```powershell
python -m pip install -e ".[test]"
python -m fantasy_football download --workers 4
python -m fantasy_football build --snapshot-date 2026-08-09
python -m fantasy_football train
python -m fantasy_football export
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -v --tb=short
python -m ruff check .
```

Existing local downloads are hash-verified and reused by default. The completed full search evaluated 30 settings across eight model families and took 743.4 seconds on this laptop. The reduced integration search writes only below `artifacts/smoke`:

```powershell
python -m fantasy_football train --fast
```

## Refresh near draft day

`download` does not check for changed upstream assets unless `--force` is supplied. Choose the intended information cutoff explicitly:

```powershell
python -m fantasy_football all --force --snapshot-date 2026-08-30
```

Archive the current `artifacts/` directory and `fantasy_football/players_2026.py` first if the August 9 forecast must remain available for prospective scoring. nflverse release files can change in place, so the manifest digest identifies the analyzed bytes more reliably than the URL alone.

## Main artifacts

- [Fantasy football and scoring](docs/fantasy_football.md)
- [Data sources and coverage](docs/data_sources.md)
- [Methods and results](docs/methods_and_results.md)
- [Experiment and excitement ledger](EXPERIMENT_LEDGER.md)
- [Completion audit](COMPLETION_AUDIT.md)
- [Generated 2026 Python catalog](fantasy_football/players_2026.py)
- [Model decision manifest](artifacts/model_manifest.json)
- [Machine-readable result summary](artifacts/result_summary.json)
- [Processed modeling table](data/processed/modeling_table.parquet)
- [Source hash manifest](data/raw/manifest.json)

## Interpretation limits

These are full regular-season point forecasts, including Week 18, not optimal draft values. Draft decisions also depend on lineup rules, replacement level, positional scarcity, injuries, and the league's exact scoring profile. The model discovers associations, not causal effects.

The compact set is locally irreducible on a declared greedy search path. It is not a proven global minimum across every possible atom combination. Draft number is a raw scalar but embeds NFL teams' scouting decisions. Current injuries, schedule context, and team environment are absent. August rosters remain volatile until final cuts.
