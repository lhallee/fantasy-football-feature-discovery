# Fantasy Football Raw-Feature Discovery

This project predicts full-season ESPN full-PPR points from elementary public NFL data. Phase 2 produces the raw fantasy forecast. Phase 3 adds a calibrated estimate of whether each player will receive any physical injury-report designation and combines it with the raw forecast in a draft preference score.

The August 9 snapshot contains 2,930 roster records. The generated catalog includes all of them, with forecasts for 958 fantasy-position players and 951 players available for the draft board.

## Injury-report risk overlay

The Phase 3 Extra Trees classifier reached pooled 2022-2024 ROC AUC 0.868, with a player-cluster 95% interval of 0.854 to 0.880. Average precision was 0.763 at 34.9% prevalence. The model predicts any physical designation on an official regular-season injury or practice report. It does not predict season-ending injury or games missed.

Prior participation is a major part of the result. Prior-year games alone reached audit AUC 0.772. The selected model reached 0.789 among players with at least one prior-year game and 0.729 among players with at least eight. Treat the output as report risk among preseason candidates, not a pure measure of biological fragility.

The default combined score gives 80% weight to the within-position percentile of raw Phase 2 predicted points and 20% to the calibrated probability of no physical injury report. This is a preference index, not expected points or medical advice.

- [2026 injury-adjusted draft workbook](outputs/injury_adjusted_draft_board_2026.xlsx)
- [Phase 3 methods and results](experiments/phase3/RESULTS.md)
- [Phase 3 machine-readable player table](experiments/phase3/artifacts/injury_adjusted_players_2026.parquet)
- [FeatureRanker comparison of points and injury-report inputs](experiments/feature_ranker/README.md)

The current workbook has `ALL`, `QB`, `RB`, `WR`, `TE`, `K`, and `INJURED` pages and no read-me page. Its three decision columns are the risk-adjusted draft score, projected 2026 fantasy points, and predicted physical injury-report probability. A dated August 14, 2026 screen moved 109 current public injury listings to `INJURED`; the other 842 rows have no matched current listing, which is not proof of health.

## Result

The selected offense model reached mean within-position-season Spearman ρ = 0.751 on chronological 2022-2025 evaluation. Prior-season points reached ρ = 0.647, for a gain of 0.104. The player-cluster bootstrap interval for model ρ was 0.726 to 0.770.

| Position | Spearman ρ | MAE |
|---|---:|---:|
| QB | 0.709 | 37.77 |
| RB | 0.748 | 29.28 |
| WR | 0.759 | 23.01 |
| TE | 0.787 | 17.38 |

The model did not reach the exploratory 0.90 target. The selected offense procedure needed all 228 leakage-controlled columns; the best 128-column alternative was outside the predeclared 0.005 selection tolerance. This is evidence against claiming a small sufficient feature set from the current public inputs.

The selected kicker model reached ρ = 0.625 versus 0.600 for prior-season points. Its MAE and NDCG were worse, and the bootstrap interval for the rank gain included zero, so kicker projections remain labeled low confidence.

## What enters the Extra Trees model

The offense vector contains 228 columns:

- 192 raw lagged football columns, 48 each from one through four seasons before the target year
- 14 missingness indicators
- 8 age, size, experience, rookie, and draft fields
- 6 combine measurements
- 4 lag-availability indicators
- 4 position indicators

The raw lagged columns include games, prior roster weeks, passing, rushing, receiving, first-down, explosive-play, and fumble counts, plus elementary rates such as completion rate, yards per attempt, yards per carry, and yards per reception. The kicker model uses 32 lag-one and lag-two field-goal and PAT columns.

Every missing numeric input is median-imputed inside the training fold. Fold-local feature scores use training labels only. Extra Trees then fits 180 randomized regression trees to raw full-season point totals. The offense trees use maximum depth 14, minimum leaf size 5, and 70% of columns per split. The kicker trees use maximum depth 10, minimum leaf size 4, and 80% of columns per split. Predictions are the mean across trees and are clipped at zero.

The model excludes target-season roster membership, roster-room aggregates, team context, status, depth charts, injuries without a verified historical August timestamp, imported fantasy points, projections, ADP, EPA, WOPR, and other composite efficiency features.

## Leakage controls

For historical target season `t`, the candidate set is the union of players on a regular-season roster in `t - 1` and entrants found in the target-year draft or combine. Players who never appeared in season `t` remain in the cohort with zero points. No historical target-year Week 1 roster file is used to decide membership or construct features.

Candidate and feature choices used only the 2016-2021 discovery folds. The chosen procedures were then evaluated once on 2022-2025 chronological folds. Those later outcomes had been viewed in earlier project phases, so this is not an analyst-blinded holdout. The 2026 forecast is the first prospective test.

Two negative controls passed their locked gates:

- 499 fixed-prediction validation-label permutations produced mean ρ = 0.0004 for offense.
- 11 end-to-end target permutations produced mean ρ = 0.0009 for offense.

These controls test the implementation for a direct label path. They do not prove that every public source had a historically verified August 9 publication timestamp. That remaining source-timing limitation is explicit in the result record.

## Use the 2026 catalog

```python
from fantasy_football.players_2026 import get_player, query_players

top_running_backs = query_players(position="RB", limit=12)
atlanta_options = query_players(team="ATL", limit=20)
pit_quarterbacks = query_players(team="PIT", position="QB", limit=None)
josh_allen = get_player("00-0034857")
```

`query_players` filters by position, team, roster status, name, college, confidence, rank, minimum predicted points, and out-of-distribution status. Team aliases such as `ARI`, `LAR`, and `WSH` are accepted. Cut and retired players remain in `PLAYERS` but are excluded from default queries.

The draft workbook has Overall, QB, RB, WR, TE, and K sheets. Every sheet is sorted only by predicted points. It also includes editable draft status, fantasy team, round, pick, and notes columns:

- [2026 draft workbook](outputs/fantasy_football_draft_board_2026.xlsx)
- [Workbook-neutral draft data](experiments/phase2/artifacts/draft_board_2026.json)

## Reproduce the locked Phase 2 experiment

Run from the source-project root with Python 3.11 or later. CUDA is not required. Numerical and estimator pools are capped at four CPU threads.

```powershell
python -m pip install -e ".[test]"
python -m fantasy_football.phase2_august_runner `
  --plan experiments/phase2/production/august9_plan_v1.json `
  --stage discovery
```

The command prints a unique run directory. Supply its `selection.json` to the audit stage, then supply both locked outputs to final fitting. Promotion is a separate gated action. Exact commands and paths are in the [Phase 2 production guide](experiments/phase2/production/README.md).

For local verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
python -m ruff check .
```

Do not use the original `python -m fantasy_football train` command to recreate the production forecast. That command belongs to the archived Phase 1 pipeline.

## Research record

- [Fantasy football and scoring](docs/fantasy_football.md)
- [Data sources and coverage](docs/data_sources.md)
- [Production methods and results](docs/methods_and_results.md)
- [Experiment index](experiments/README.md)
- [Phase 2 production record](experiments/phase2/production/README.md)
- [Phase 3 injury-report risk record](experiments/phase3/README.md)
- [Scoring-scheme correlations](experiments/phase2/SCORING_CORRELATIONS.md)
- [Generated Python catalog](fantasy_football/players_2026.py)
- [Model manifest](artifacts/model_manifest.json)
- [Machine-readable result summary](artifacts/result_summary.json)

## Interpretation limits

These are full regular-season point forecasts, including Week 18, not optimal draft values. Draft decisions also depend on lineup rules, replacement level, positional scarcity, injuries, and the league's exact scoring profile. The analysis discovers predictive associations, not causal effects.

The 2022-2025 evaluation was chronological but not analyst-blinded. Public historical files do not reconstruct exact August 9 injury, transaction, depth-chart, or roster states. August 2026 rosters can still change before final cuts. The 2026 season must finish before prospective accuracy can be measured.
