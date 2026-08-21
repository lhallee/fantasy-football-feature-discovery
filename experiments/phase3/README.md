# Phase 3: Injury-Report Risk

Phase 3 adds a leakage-controlled injury-report classifier to the Phase 2 fantasy projection. It predicts whether a player will receive at least one physical designation on an official regular-season injury or practice report. It then combines that calibrated probability with the raw Phase 2 point forecast in a transparent draft preference index.

## Result

The locked Extra Trees classifier reached pooled ROC AUC 0.868 on the 2022-2024 chronological audit. The player-cluster bootstrap interval was 0.854 to 0.880. Average precision was 0.763 at 34.9% prevalence, log loss was 0.430, and Brier score was 0.140. An age-and-position logistic baseline reached ROC AUC 0.551.

The result predicts an observed report event, not hidden biological fragility or games missed. Prior participation is a major part of the signal. Prior-year games alone reached audit AUC 0.772. The selected model reached 0.789 among players with at least one prior-year game and 0.729 among players with at least eight. These restrictions use only pre-cutoff information, but they were added after selection and are diagnostic rather than confirmatory.

## Start here

- [Methods and results](RESULTS.md)
- [Detailed experiment ledger](EXPERIMENT_LEDGER.md)
- [Pre-fit plan](plan_v1.json)
- [Pre-fit code and input hashes](prefit_manifest_v1.json)
- [Experiment release manifest](experiment_manifest_v1.json)
- [Machine-readable result summary](result_summary.json)
- [Current-injury screen manifest](current_injury_manifest_v1.json)
- [Player-by-player current-injury screen](artifacts/current_injury_screen_2026-08-14.csv)
- [Completed run](runs/20260814T143014705993Z-injury-risk-v1/)
- [Exposure diagnostic](artifacts/posthoc_exposure_diagnostic.json)
- [2026 injury-adjusted draft workbook](../../outputs/injury_adjusted_draft_board_2026.xlsx)

The workbook has `ALL`, `QB`, `RB`, `WR`, `TE`, `K`, and `INJURED` sheets. It has no read-me sheet. The `ALL` and position pages contain 842 players with no matched current public injury listing. The `INJURED` page contains the 109 matched listings as of August 14, 2026. All pages sort by the risk-adjusted draft score. A missing listing is not evidence that a player is healthy.

## Outcome

A player-season is positive when any of four fields contains a physical designation during the regular season:

- primary or secondary game-report injury
- primary or secondary practice-report injury

Blank values, illness, COVID, personal matters, rest, coaching decisions, and explicit non-injury designations are excluded. Mixed entries such as `Illness, Shoulder` remain physical positives. The official nflverse feed covers 2009-2024. It does not contain 2025, so every modeled season uses injury-history offsets of two through five years. No 2025 label is imputed for the 2026 prediction.

The source and field definitions come from the [nflreadr injury loader](https://nflreadr.nflverse.com/reference/load_injuries.html) and [injury data dictionary](https://nflreadr.nflverse.com/articles/dictionary_injuries.html).

## Model contract

The model uses the Phase 2 August-origin candidate universe and 500 numeric inputs:

- age, size, experience, draft, combine, rookie, and position fields known at the origin
- raw performance, opportunity, rate, and late-season trajectory measurements lagged one through four seasons
- physical injury-report events, report weeks, and designation counts lagged two through five seasons
- explicit missingness and history-availability indicators

Target-season statistics, fantasy points, injuries, roster status, depth ranks, active-roster counts, roster-room aggregates, and target-team aggregates are forbidden. Every validation year recomputes feature ranking from its earlier base-training seasons only.

The selected model has 300 Extra Trees, minimum leaf size 20, and 70% random feature sampling per split. It uses all 500 available columns. For validation year `y`, the base model trains through `y - 2`, a sigmoid calibrator fits on `y - 1`, and evaluation uses `y`. Model selection used 2017-2021. The locked audit used 2022-2024.

Load and filter the completed player table with pandas:

```python
from pathlib import Path

import pandas as pd

from fantasy_football.phase3_draft import query_risk_adjusted_players

root = Path.cwd()
board = pd.read_parquet(
    root / "experiments/phase3/artifacts/injury_adjusted_players_2026.parquet"
)
top_running_backs = query_risk_adjusted_players(board, position="RB", limit=15)
```

## Combined score

For each position, let `f` be the percentile of raw Phase 2 predicted points and let `h = 1 - p` be the calibrated probability of no physical injury report. The default score is:

```text
combined_draft_score = 100 * (0.80 * f + 0.20 * h)
```

This is a preference index, not expected points or medical advice. The 80/20 weight was declared before fitting and was not tuned on audit outcomes. The Parquet artifact also contains 70/30 and 90/10 sensitivity scores. Within-position Spearman correlations between the default index and raw points range from 0.990 to 0.997, so injury risk adjusts the Phase 2 order without replacing fantasy value.

## Reproduce

Run from the repository root with Python 3.11 or later. CUDA is not required. Numerical and estimator pools use at most four threads.

```powershell
python -m pip install -e ".[test]"
python -m pip install -r experiments/phase3/requirements_extensions.txt
python -m fantasy_football.phase3_runner --root . download
python -m fantasy_football.phase3_runner --root . run
```

A rerun reproduces the locked procedure but is not a new blinded audit. The 2022-2024 outcomes have now been inspected. The 2026 labels are unavailable and remain the prospective evaluation.

Refresh the separate current-status screen with the documented public MyFantasyLeague API, official NFL reserve-list additions, and the frozen secondary-source cross-check:

```powershell
python -m fantasy_football.phase3_current_injuries --root . --as-of 2026-08-14
```

This operational screen does not refit either model or replace the prospective injury probability. It only moves players with affirmative current injury evidence from the active draft pages to `INJURED`.
