# Phase 2 production methods and results

## Prediction question

For every fantasy-position player who could reasonably be identified by August 9, predict full regular-season ESPN full-PPR points using only elementary information from before the target season. Rank quality is evaluated within position and season because draft decisions compare players who can fill the same roster slot.

The primary metric is the arithmetic mean of Spearman correlation across position-season groups. Secondary metrics include Kendall τ-b, MAE, RMSE, R², position-rank MAE, NDCG at QB12/RB24/WR36/TE12/K12, top-k recall, and linear calibration.

## Fixed forecast origin

The production origin is August 9, 2026. Historical target-season roster membership cannot be used because equivalent timestamped August snapshots are unavailable for every training year.

For historical target season `t`, the candidate universe is constructed from:

1. Every fantasy-position player on a regular-season roster in `t - 1`.
2. Every target-year draft pick or combine participant that can be mapped to a player.

The union is deduplicated by player and position. A candidate who records no regular-season statistics in `t` stays in the table with zero points. This preserves backups, cuts, retirements, and unsuccessful entrants instead of conditioning the sample on later participation.

The 2026 inference cohort is different only where observable information permits it to be: all 958 fantasy-position players in the frozen August 9 roster snapshot receive forecasts. Seven CUT or RET records remain in the catalog for provenance but are excluded from the default draft board.

## Target and exclusions

The target is the sum of all regular-season ESPN full-PPR player-game points, including Week 18 since 2021. The executable scoring profile is [`scoring_espn_full_ppr_2026.json`](../config/scoring_espn_full_ppr_2026.json), with SHA-256 `744c0cea0e70966966d5446595b0858304c5e941a1951fff8feb078e78792d7e`.

The feature contract excludes:

- target-season roster membership in historical rows
- target-season roster-room or team aggregates
- current status, depth rank, and depth tier
- injuries without a historically verified August timestamp
- `team_changed`, because the historical August team is unknown
- imported fantasy scores, projections, ADP, betting lines, EPA, WOPR, and other composite efficiency features

Prior-season fantasy points are retained only as a benchmark. They are not model inputs.

## Raw feature construction

All performance sources have offsets from one through four seasons. No target-season performance column can enter a model vector.

### Offense

The selected offense vector has 228 columns:

| Feature group | Columns | Examples |
|---|---:|---|
| Lag-one raw counts and rates | 48 | games, passing yards, carries, receptions, touchdowns, first downs, explosive plays, fumbles, yards per attempt |
| Lag-two raw counts and rates | 48 | same elementary statistics from `t - 2` |
| Lag-three raw counts and rates | 48 | same elementary statistics from `t - 3` |
| Lag-four raw counts and rates | 48 | same elementary statistics from `t - 4` |
| Missingness indicators | 14 | missing age, draft, size, experience, or combine result |
| Metadata and draft | 8 | age, height, weight, experience, rookie flag, drafted flag, round, pick |
| Combine measurements | 6 | forty, bench, vertical, broad jump, shuttle, cone |
| History-availability indicators | 4 | whether each lag season exists |
| Position indicators | 4 | QB, RB, WR, TE |

Rates are direct arithmetic transforms of lagged raw counts, such as completion rate, passing yards per attempt, rushing yards per carry, receiving yards per reception, first-down rate, and field-goal make rate. There are no fantasy-point transforms or target-derived aggregates.

The candidate search also tested late-season trajectory features and a recent-era source tier with snaps, targets, and air-yard inputs. Neither improved the locked discovery objective enough to be selected.

### Kicker

The selected kicker vector has 32 columns from `t - 1` and `t - 2`: games, roster weeks, PAT attempts and makes, field-goal attempts and makes, distance buckets, total made and missed distance, mean made and missed distance, and elementary make rates.

## Chronological protocol

The experimental plan and code hashes were recorded before fitting. Discovery and audit are separate commands with separately hashed outputs.

- Discovery folds: 2016 through 2021.
- Locked retrospective folds: 2022 through 2025.
- Final fit: all eligible history through 2025, followed by 2026 inference.

For every validation year, training uses only earlier target seasons. Feature scoring, median imputation, and estimator fitting occur inside that training fold. The discovery build is bounded to its maximum source season, so 2022-2025 outcomes are not loaded into discovery feature construction.

The later outcomes had been inspected during Phase 1 and the superseded source-proxy studies. The audit is therefore chronological and procedure-locked, but not analyst-blinded. The 2026 season is the first prospective test.

## Candidate and feature-count search

The pre-fit menu contained 16 offense procedures and 9 kicker procedures. It compared Extra Trees with histogram gradient boosting, XGBoost, and ridge, stable and recent source tiers, core and trajectory recipes, and explicit column limits.

Selection maximized discovery Spearman. Any procedure within 0.005 of the best was resolved by fewer columns, lower MAE, and lexicographic identifier, in that order. Prior-season points was eligible to win the kicker cohort but not offense.

### Offense discovery sensitivity

| Procedure | Columns | Discovery ρ | MAE |
|---|---:|---:|---:|
| Stable core Extra Trees, all | 228 | 0.6988 | 27.86 |
| Stable core plus trajectory Extra Trees, all | 264 | 0.6970 | 27.94 |
| Recent core plus trajectory Extra Trees, all | 327 | 0.6918 | 28.08 |
| Stable core Extra Trees, 128 | 128 | 0.6814 | 29.34 |
| Stable core Extra Trees, 64 | 64 | 0.6775 | 29.62 |
| Stable core Extra Trees, 32 | 32 | 0.6515 | 31.40 |
| Stable core Extra Trees, 16 | 16 | 0.5336 | 35.23 |

The 128-column model was 0.0174 below the best, so it failed the 0.005 compactness tolerance. Phase 2 does not support a claim that a small combination preserves the available rank signal.

### Kicker discovery sensitivity

The 32-column stable-core Extra Trees model led discovery at ρ = 0.6094. The 164-column version reached 0.6018, the 16-column version reached 0.5933, and prior-season points reached 0.5571. The 32-column model was selected.

## Final estimators

Each model is a scikit-learn pipeline with median imputation and `ExtraTreesRegressor`.

| Setting | Offense | Kicker |
|---|---:|---:|
| Trees | 180 | 180 |
| Maximum depth | 14 | 10 |
| Minimum leaf size | 5 | 4 |
| Columns considered per split | 70% | 80% |
| Selected columns | 228 | 32 |
| Random seed | 20260809 | 20260809 |

Training columns are ordered by the absolute within-position-season association computed from training rows only. Missing values are median-imputed inside the pipeline. Each tree is fit to a bootstrap-free randomized sample of split thresholds and feature subsets, and the forest prediction is the mean of its trees. Negative totals are clipped to zero after prediction.

Numerical and estimator thread pools are capped at four. CUDA is not required.

## Retrospective results

### Offense

| Predictor | Spearman ρ | Kendall τ-b | MAE | RMSE | R² | NDCG | Top-k recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 2 Extra Trees | 0.7510 | 0.5928 | 25.25 | 43.54 | 0.659 | 0.7895 | 0.5651 |
| Prior-season points | 0.6472 | 0.5372 | 27.18 | 52.70 | 0.500 | 0.7850 | 0.5547 |

The Spearman gain is 0.1038. A 500-sample player-cluster bootstrap gives a 95% interval of 0.7263 to 0.7703 for model ρ and 0.0843 to 0.1244 for the paired gain.

| Position | Rows | Model ρ | Model MAE |
|---|---:|---:|---:|
| QB | 497 | 0.7092 | 37.77 |
| RB | 1,076 | 0.7485 | 29.28 |
| WR | 1,900 | 0.7595 | 23.01 |
| TE | 891 | 0.7867 | 17.38 |

| Validation season | Rows | Model ρ | Prior-points ρ |
|---|---:|---:|---:|
| 2022 | 1,050 | 0.7413 | 0.6330 |
| 2023 | 1,114 | 0.7441 | 0.6332 |
| 2024 | 1,079 | 0.7595 | 0.6676 |
| 2025 | 1,121 | 0.7589 | 0.6548 |

No season or position approaches 0.90. The stable values across four chronological years are more informative than a single pooled correlation, but they do not establish future accuracy.

### Kicker

The kicker model reached ρ = 0.6255 versus 0.5995 for prior-season points. Its MAE was 37.88 versus 35.96, and NDCG was 0.7752 versus 0.8008. The bootstrap interval for the Spearman gain was -0.0255 to 0.0766. Kicker ordering is therefore exploratory even though the locked discovery rule selected the model.

## Negative controls

The plan fixed both controls before fitting.

| Control | Offense mean ρ | Kicker mean ρ |
|---|---:|---:|
| Fixed predictions, 499 independently permuted validation labels | 0.0004 | 0.0008 |
| End-to-end fit, 11 independently permuted training and validation labels | 0.0009 | 0.0287 |

The offense end-to-end values ranged from -0.0187 to 0.0376. These controls passed the predeclared centering gates. They are implementation sentinels, not confirmatory permutation tests over the full model-selection process.

## Leading August 9 outputs

| Position | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| QB | Josh Allen, 315.9 | Jared Goff, 270.9 | Baker Mayfield, 264.8 |
| RB | Jahmyr Gibbs, 257.8 | Bijan Robinson, 240.5 | De'Von Achane, 228.8 |
| WR | Puka Nacua, 274.2 | Amon-Ra St. Brown, 270.4 | Ja'Marr Chase, 249.9 |
| TE | Trey McBride, 222.9 | Harold Fannin Jr., 183.2 | Tyler Warren, 167.2 |

Kicker pages in the workbook are also sorted by model-predicted points, as requested. The Python query interface retains the conservative prior-points default for explicit K or PK queries because kicker evidence is mixed; pass `sort_by="prediction"` for model order.

## Claim boundary

Phase 2 is the production generation because it has the strongest forecast-origin contract in this repository, not because it achieved ρ > 0.90. The result is conditional on the available public sources, candidate-universe rule, feature menu, model menu, and selection metric.

Historical public data do not reconstruct exact August 9 roster, injury, transaction, or depth-chart states. The retrospective was not analyst-blinded. The analysis does not prove causal effects, a globally minimal feature set, optimal draft value, or prospective 2026 accuracy.
