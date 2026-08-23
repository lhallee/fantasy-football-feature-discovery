# Fantasy Football Feature Discovery

Full-season 2026 ESPN full-PPR point forecasts and a calibrated probability of a missed-time injury for every fantasy-position player on the August 9, 2026 roster snapshot, built from public nflverse data with no target-season information in any input.

## Deliverable

- [`outputs/fantasy_football_final_rankings_2026.xlsx`](outputs/fantasy_football_final_rankings_2026.xlsx): README, SETTINGS, ALL, QB, RB, WR, TE, K, INJURED, and evidence sheets. ALL ranks by points over replacement with a peer-relative health adjustment; position pages rank by a within-position score; `SETTINGS!B2` changes the injury weight live.
- [`docs/report/report.pdf`](docs/report/report.pdf): short technical report with methods, leakage controls, experiments, and figures.
- [`experiments/LEDGER.md`](experiments/LEDGER.md): consolidated record of every phase, including negative results and protocol amendments.

## Result

| Track | Model | Locked audit | Baselines |
|---|---|---|---|
| Points (QB, RB, WR, TE) | Extra Trees on 512 leakage-controlled inputs, trained from 2017 | 2022-2025 mean within-position-season Spearman 0.756 (0.732 to 0.775) | Phase 2 production 0.751; prior-season points 0.647 |
| Missed-time injury (all five positions) | Calibrated Extra Trees, 300 trees, min leaf 50 | 2022-2024 ROC AUC 0.749 (0.731 to 0.767), calibration slope 1.02, prevalence 24.7% | Prior-year games 0.677; age and position 0.586 |

The injury label is an Out or Doubtful listing with a physical injury on an official report, or an injury-reserve roster week, during the regular season. The default injury weight in the draft score is 0.20, the largest grid weight that kept development-fold rank accuracy within 0.005 Spearman of the best weight. Kicker point forecasts retain the Phase 2 production model and are labeled low confidence.

Adding literature-motivated usage shares and per-game rates (Phase 5) did not improve either track under the same protocol, so Phase 4 is the final model.

## Layout

```
fantasy_football/   data build, Phase 2 features and cohort, injury label, Phase 4 modeling and workbook
experiments/        LEDGER.md plus the Phase 2 production record, Phase 3 injury data and screen, Phase 4 and 5 runs
artifacts/          Phase 2 production models and predictions (points reference and kicker forecast)
data/               raw nflverse releases (ignored except the manifest) and processed tables
docs/               data sources, scoring notes, and the report
outputs/            the final workbook
```

## Reproduce

Python 3.11 or later, no GPU.

```powershell
python -m pip install -e ".[test]"
python -m fantasy_football download --workers 4
python -m fantasy_football build
python -m fantasy_football.phase4_runner --run-name final-ranking
python -m fantasy_football.phase4_runner --experiment phase5 --feature-set phase5 --reference-run experiments/phase4/runs/<run> --run-name usage-features
```

The Phase 4 run takes about twelve minutes on eight threads and writes a run directory, the artifacts, `RESULTS.md`, and the workbook. `--rebuild-from <run>` regenerates the board and workbook without refitting; add `--refresh-featureranker` to rerun the FeatureRanker 3.0.4 evidence line. The injury-report download is `python -c "from pathlib import Path; from fantasy_football.phase3_injury import download_injury_reports; download_injury_reports(Path('experiments/phase3/data/raw'))"`.

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## Leakage controls

Historical candidates for target season `t` are players on a regular-season roster in `t - 1` plus target-year draft or combine entrants; nonparticipants keep zero points. Every input is a `t - 1` or earlier quantity with recorded lineage. Report-based injury history uses offsets of two or more seasons because the official feed ends in 2024; roster reserve history uses offsets of one or more. Model selection used development folds only (points 2019-2021, injury 2017-2021); audits ran once on 2022-2025 and 2022-2024. Negative controls permute targets or labels within season and position; the Phase 4 record documents two diagnostics that changed the protocol.

## Limits

The audits are chronological and procedure-locked but not analyst-blinded. Historical August rosters are reconstructed. The injury label records observed absence events, so participation is part of the signal. The draft scores are preference indexes, not expected value. The 2026 season is the prospective test.

## Data and license

nflverse-data is published under CC BY 4.0; see [`NOTICE.md`](NOTICE.md) and [`docs/data_sources.md`](docs/data_sources.md).
