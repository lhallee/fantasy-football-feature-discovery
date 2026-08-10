# Phase 2 Experiment Ledger: Post-Cut Roster-Proxy Rank Study

## Question

Can a post-final-cuts, pre-Week-1 model built from public raw NFL data rank the next season's QB, RB, WR, and TE fantasy totals with mean within-position Spearman correlation above 0.90 without target leakage or material selection overfit?

The threshold is aspirational. A lower, stable result is preferable to an inflated result produced by changing the cohort, inspecting confirmation outcomes during tuning, or using information that was unavailable at prediction time.

## Entry 0: protocol lock

**Recorded:** 2026-08-10 12:12 UTC, before any Phase 2 candidate was fit.

### Frozen Phase 1 reference

Phase 1 remains unchanged. Its compact five-atom offense model reached development Spearman 0.7386 and retrospective 2024-2025 Spearman 0.7575. Its full 32-atom sensitivity reached retrospective Spearman 0.7694. Phase 2 run outputs write only below `experiments/phase2/` and must not change `data/processed/`, `artifacts/`, or `fantasy_football/players_2026.py`.

A machine-readable freeze manifest will record Phase 1 file hashes before Phase 2 fitting.

### Prediction contract

Phase 2 v1 has one declared forecast origin: after final roster cuts and before the first regular-season game. Historical player membership comes from nflverse Week 1 roster files, so it is a proxy for that origin. Those old files do not provide reliable publication timestamps. The performance lineage is source-season safe, but the roster cutoff is not a timestamp-proven as-of archive. The result is not valid evidence for the August 9 camp snapshot. Applying the room-context features in August would import later roster-survival information into the historical folds while withholding it from 2026.

For target season \(t\), every numeric performance input must be computable by that forecast origin. Performance inputs must come from seasons strictly earlier than \(t\). Stable target-season metadata may include age, draft and combine measurements, and post-cut team assignment. The primary model excludes target-season statistics, imported fantasy scores, expert projections, ADP, betting data, depth rank, and roster status.

Derived features are allowed only when their lineage is explicit. Examples include lagged per-game rates, exponentially weighted history, career totals through \(t-1\), and teammate or team aggregates built from pre-\(t\) player statistics. Any target encoding must be fit inside the training fold.

The scored population remains every QB, RB, WR, or TE in the historical Week 1 roster proxy, including rookies, backups, and eventual zero scorers. Segment results may diagnose the problem but cannot replace the full-cohort primary metric.

### Temporal selection protocol

1. **Discovery period:** rolling target seasons 2016-2021. A candidate trained for season \(t\) may use only rows with `target_season < t`.
2. **Fixed candidate evaluation:** feature recipes, feature-count limits, model parameters, target forms, and blend formulas are declared before fitting. Imputation, scaling, and any adaptive top-k selector are fit separately inside each training fold.
3. **Procedure selection:** one Phase 2 procedure is selected from aggregate discovery results. Ties within 0.005 Spearman favor fewer input columns, then lower compute. The retrospective years do not choose a feature count or parameter.
4. **Retrospective stress test:** the selected procedure is run on 2022-2025, with no later manual tuning based on those results.
5. **Prospective test:** 2026 is the only genuinely untouched season. Its outcomes are not available.

The 2022-2025 period is not blinded: Phase 1 already exposed its outcomes and baseline performance. It cannot support the same claim as a sealed external test.

### Declared candidate families

The discovery stage may compare these predefined ideas:

1. Phase 1 compact and full-atom baselines.
2. Position-specific versions of the raw-feature models.
3. One- through four-season raw lags, trends, recency-weighted histories, and career summaries.
4. Opportunity and efficiency decompositions derived from raw counts, with denominators formed only from lagged data.
5. Post-cut roster-context and prior-team context built only from lagged player totals.
6. Point regression, within-position percentile regression, two-stage zero-versus-positive models, and pairwise rank learning.
7. Convex blends of independently generated out-of-fold predictions.
8. Fold-local feature ranking at fixed declared counts. The procedure is chosen from aggregate 2016-2021 discovery-fold results; there is no nested inner temporal loop.

Unlisted feature sources or model classes require a dated ledger amendment before their results are computed. An amendment cannot redefine the retrospective seasons or primary cohort.

### Primary metric and success rule

The primary metric is the unweighted mean of Spearman correlations across position-season groups. Secondary metrics are Kendall \(\tau_b\), MAE, RMSE, NDCG at roster cutoffs, top-k recall, and position-rank MAE.

A cross-validated `Spearman > 0.90` result must satisfy all of the following, but it remains provisional until the prospective 2026 test:

- Full-cohort retrospective mean Spearman exceeds 0.90.
- Player-cluster bootstrap 95% lower bound exceeds 0.85.
- Each position's retrospective mean exceeds 0.85.
- Discovery-to-retrospective degradation is no more than 0.03.
- At least three of four retrospective seasons exceed 0.88.
- Shuffled-target and forbidden-column sentinels pass.
- The lineage audit finds no feature whose source season is \(t\) or later.

If the point estimate exceeds 0.90 but these checks fail, the ledger will label it unstable or contaminated rather than successful.

### Leakage and overfit gates

- Assert `source_season <= target_season - 1` for every lagged performance feature.
- Require target-season roster membership, team assignment, and roster-room membership to be observable at the declared post-cut origin.
- Reject any feature name or lineage containing current target points, source fantasy points, or target-season game statistics.
- Fit imputers, scalers, encoders, feature selectors, and blend weights inside training data only.
- Recompute position ranks independently inside each training fold.
- Keep repeated players across chronological folds because the deployment problem predicts returning players; never allow a later player-season to train an earlier prediction.
- Report rookies, returning players, zero scorers, and each position separately.
- Run a shuffled-label negative control and an intentionally forbidden-feature rejection test.
- Preserve every tried candidate and do not report only the winner.
- Use player-cluster uncertainty intervals and season-level stability, not random row splits.

### Initial diagnostic, before Phase 2 fitting

The offense cohort contains 16,726 historical player-seasons from 2002-2025. Zero-point rates are high: 27.8% for QB, 21.8% for RB, 30.8% for TE, and 31.2% for WR. Across 2020-2025, the mean univariate Spearman correlation of prior-season points with next-season points is 0.645. This establishes a difficult ranking problem and makes a 0.90 post-cut result unlikely without substantially better role or team-context information.

## Entry 1: outcome-exposure correction and oracle quarantine

**Recorded:** 2026-08-10, before the reproducible Phase 2 candidate run.

The parallel design audit inspected 2024-2025 while assessing the likely ceiling. Those seasons were already visible in Phase 1, but this additional inspection removes any basis for calling them a Phase 2 confirmation set. They remain useful retrospective stress tests. No later feature or parameter choice may be justified as if their result were blind.

Two deliberately forbidden discovery-period diagnostics locate the missing signal:

| Realized target-season input | 2016-2021 Spearman |
|---|---:|
| Games played | 0.9221 |
| Offensive snaps | 0.9467 |

These are oracle diagnostics, not model candidates. They use information unavailable at the post-cut forecast origin and are prohibited by the lineage gate. They show that realized target-season participation and workload are strongly co-ranked with realized fantasy points. They neither estimate an attainable preseason ceiling nor establish that availability or role can be predicted accurately at the forecast cutoff.

The read-only design audit also found that the source-season-controlled information set remains near 0.77 across model forms. This observation fixes the next experiment menu around raw history, weekly workload trajectory, first downs, explosive-play counts, roster-room competition, prior-team context, and an explicit availability decomposition. It does not relax the split or success rule.

## Running log

## Entry 2: isolated implementation and fixed screen

**Recorded:** 2026-08-10, before model fitting.

The Phase 1 freeze ID is `366b32e659f67cbbcb81a68ee20a66bbe79cdefc004db5c971c23e89b35f17ae`. It covers 160 protected files and 86,490,574 bytes. The Phase 2 safety wrapper verified the seven protected tree digests before and after each run. The safety wrapper writes run outputs only below `experiments/phase2/runs/` and cannot promote a result into the production catalog.

The fixed [plan_v1.json](experiments/phase2/plan_v1.json) hash is `ace43d1ce173302f401f7616f25a4c7a74bb0317064a2d5ece395ab635cd9b17`. It declares 27 model configurations, three rank blends, and the prior-points benchmark. The candidates cover Extra Trees, histogram boosting, XGBoost, ridge, a hurdle model, pooled and position-specific fits, three target forms, and feature limits from 8 through the full pool.

The stable builder generated 471 columns. The recent builder generated 539. The offense-only exclusions reduced the full candidate pools to 305 and 373 columns. Every column has a source table, source offset, and recipe. Runtime assertions reject fantasy or target values, target-season performance, EPA, CPOE, PACR, RACR, WOPR, provider-created shares, depth, status, active-roster features, and identifiers.

The plan, code hashes, and run directories are recorded in [experiment_manifest_v1.json](experiments/phase2/experiment_manifest_v1.json). Each run's `run_manifest.json` records its exact source-input hashes, and its `output_hashes.json` records every generated output. The test suite had 39 passing tests before fitting.

## Entry 3: 2016-2021 discovery

**Run:** `20260810T124403492462Z-discovery-ace43d1ce173-af1988c772f320f6`

The discovery run trained each fold only on earlier target seasons. Its persisted runner elapsed time was 177.2 seconds, with numerical and estimator pools configured for at most four threads. Its legacy shuffled-fit control reached Spearman -0.0531, but a later audit found that feature selection saw the real labels before fitting labels were shuffled. It was not an end-to-end null control. The forbidden-column and source-offset sentinels passed.

### Feature-count curve

| Fold-local column limit | Spearman | MAE |
|---:|---:|---:|
| 8 | 0.5025 | 45.44 |
| 16 | 0.6766 | 35.53 |
| 32 | 0.6864 | 34.56 |
| 64 | 0.6941 | 33.78 |
| 128 | 0.7019 | 33.07 |
| 305, no selector | 0.7157 | 32.35 |

More columns improved this discovery period. None of the evaluated compact limits preserved the full model within 0.005 Spearman.

### Feature-family ablation

| Stable feature family | Columns | Spearman | MAE |
|---|---:|---:|---:|
| Core lagged counts and rates | 229 | 0.7077 | 33.36 |
| Core plus weekly trajectory | 265 | 0.7060 | 33.47 |
| Core plus prior-team environment | 250 | 0.7085 | 33.18 |
| Core plus position-room competition | 248 | 0.7138 | 32.32 |
| All stable families | 305 | 0.7157 | 32.35 |

Room competition added 0.0062 Spearman to the core. Weekly trajectory and prior-team totals did not help enough. Recent-era targets, air-yard, YAC, and snap columns reached 0.7102 with all 373 columns, below the stable full model. Rank targets, signed-log targets, hurdle modeling, position-specific fits, and ET/HGB/XGB blends also failed to improve the point estimate.

The best point estimate used all 305 stable columns. The locked selection rule chose `stable_core_room_et_all_points` because its 248 columns were within 0.005 of that result. This is the smallest passing configuration among the declared candidates, not a global minimum or proof that every one of its columns is necessary.

The 248 fitted inputs consist of:

- 14 stable metadata values and 15 missingness indicators;
- four position indicators and four history-availability flags;
- four lags for 38 elementary count atoms, or 152 columns;
- four lags for ten raw ratios, or 40 columns;
- 19 position-room competition columns.

The leading fold-local associations were prior receptions, receiving gains of at least 10 yards, receiving first downs, receiving yards, gains of at least 16 and 20 yards, room-relative receptions, games, room-relative receiving yards, receiving touchdowns, draft number, and carries. These are rankings for diagnosis; the selected Extra Trees model retained all 248 inputs.

## Entry 4: locked 2022-2025 retrospective stress test

**Run:** `20260810T124722038967Z-retrospective-ace43d1ce173-f27b159113508c81`

Only the discovery-selected 248-column procedure was fit. Its persisted runner elapsed time was 22.2 seconds.

| Predictor | Spearman | Kendall | MAE | RMSE | NDCG | Top-k recall |
|---|---:|---:|---:|---:|---:|---:|
| Phase 2 selected model | 0.7661 | 0.5982 | 28.15 | 44.92 | 0.8078 | 0.5816 |
| Prior-season points | 0.6585 | 0.5372 | 29.51 | 55.01 | 0.7891 | 0.5573 |

The descriptive player-cluster interval for Spearman is 0.7413-0.7876. A descriptive resampling of only four season estimates gives 0.7569-0.7744. These conditional intervals do not include candidate-selection, cohort-definition, source-timing, or future-regime uncertainty. Fold Spearman values were 0.7721, 0.7518, 0.7639, and 0.7767 for 2022 through 2025.

| Position | Spearman |
|---|---:|
| QB | 0.7361 |
| RB | 0.7429 |
| WR | 0.7771 |
| TE | 0.8085 |

### Success-rule decision

The 0.90 goal was not achieved.

- Overall Spearman 0.7661 is below 0.90.
- The player-cluster lower bound 0.7413 is below 0.85.
- No position exceeded 0.85.
- No retrospective season exceeded 0.88.
- Discovery-to-retrospective performance improved rather than degraded.
- The legacy shuffled-fit control was near zero but did not test supervised feature selection. The forbidden-column, temporal-lineage, Phase 1 freeze, and storage guards passed.

The result supports a leakage-controlled estimate near 0.77 for this public post-cut Week 1 roster-proxy information set. It is not a formal irreducible ceiling or a timestamp-proven prospective estimate. It does not support further hyperparameter searching as a credible route to 0.90. The deliberately forbidden oracle tests show that realized games and snaps can cross 0.90; legitimate progress likely requires better pre-cutoff predictors of injuries, availability, and role.

No Phase 2 v1 result should be promoted into the August 9 catalog. A prospective 2026 use must either wait for the equivalent post-cut roster snapshot or start a separately declared August-origin study with timestamp-aligned historical August snapshots. High-value raw additions are injury/PUP/IR events known by the chosen cutoff, preseason participation, transaction and roster-tenure history, vacated teammate opportunities, and college production for rookies. Those sources must carry publication timestamps and pass the same as-of lineage gate before modeling.

## Entry 5: cutoff audit and release gate

**Recorded:** 2026-08-10, after the locked retrospective run.

The final cutoff audit identified that historical cohort and roster-room membership come from Week 1 roster proxies without verified historical publication timestamps. Phase 2 v1 is therefore a post-final-cuts, pre-Week-1 proxy experiment, not a timestamp-proven as-of study. This correction does not change any fitted value or metric. It narrows the valid use of the result and prevents an invalid comparison with the August 9 camp roster.

The release gate passed 44 tests, Ruff linting, Python compilation, Phase 1 freeze verification, exact plan and code hashes, exact run-output hashes, and the declared storage check. The size-cap-scoped project data and isolated Phase 2 tree total 90,542,518 bytes. No Phase 2 prediction or model was written into the Phase 1 artifacts or player catalog.

Phase 2 v1 is complete as a retrospective research result. Its prospective 2026 evaluation remains sealed until an equivalent, timestamped post-cut snapshot exists.

## Entry 6: v1 invalidation and pre-fit v2 cohort amendment

**Recorded:** 2026-08-10, before any corrected cohort was fit.

The final model audit found that v1 did not implement the intended post-cut population. The Week 1 source files retain rows whose status is `CUT` or `RET`. V1 included 778 such rows in the 2022-2025 retrospective cohort and in its same-team position-room aggregates. Of those rows, 94.9% eventually scored zero. Their inclusion changed the room count for 80.1% of otherwise retained 2022-2025 rows. V1's saved Spearman 0.7661 reconstructs exactly, but it is invalid evidence for a post-cut cohort and is superseded.

The correction is declared in [plan_v2.json](experiments/phase2/plan_v2.json) before fitting. It makes no change to the v1 candidate menu, temporal splits, hyperparameters, selection tolerance, blend rule, metric, or success rule. It changes only the cohort definition.

V2 uses the official [nflreadr roster-status dictionary](https://nflreadr.nflverse.com/articles/dictionary_roster_status.html). It includes `ACT`, `DEV`, `E14`, `EXE`, `INA`, `PUP`, `RES`, `RSN`, and `SUS`, which represent active, practice-squad, exempt, inactive-under-contract, physically-unable-to-perform, reserve, non-football-injury reserve, or suspended players whose team retains their rights. It excludes `CUT`, `RET`, `RFA`, `UFA`, likely-waived `NWT`, released-from-injured-reserve `RSR`, and practice-squad-release codes `TRC`, `TRD`, and `TRT`. The observed `E01` code has no authoritative mapping and is conservatively excluded.

Two offense rows have missing source status: Phil Bates in 2014 and Nick Harwell in 2016. The pre-fit rule conservatively excludes and audits missing status rather than inferring it from later participation. Any previously unseen nonmissing code fails the run for review. Status defines membership only and is not a model input. Filtering occurs before both outcome scoring and roster-room aggregation.

The status rule applies to the frozen modeling table's upstream-deduplicated Week 1 player row, not to every duplicate raw row independently. Of 712 raw player groups containing both included and excluded codes, the upstream deterministic priority retained an included code for 711. It retained `TRD` rather than duplicate `RSN` for Laron Byrd in 2014, so v2 conservatively excludes that one ambiguous player-season. Both raw rows list Cleveland. This fixed source-processing edge was identified before fitting and does not use the player's target outcome.

This correction is driven by the cutoff contract, not by model performance. The 2022-2025 outcomes remain exposed and can only provide a retrospective stress test. A timestamped 2026 post-cut snapshot remains the sole prospective test.

The same pre-fit amendment repairs three audit controls without changing model selection. First, the end-to-end null clones the discovery-selected procedure and shuffles labels within training position-season groups before both feature selection and fitting. Second, signed-log predictions are inverse-transformed before point-scale metrics; point-scale metrics are omitted for percentile, hurdle-rank, and rank-blend outputs. Third, an apples-to-apples no-draft diagnostic clones the selected procedure while excluding draft number, draft round, and drafted status. Both diagnostics are reported after selection and cannot choose the champion.

The discovery procedure comparison is ordinary aggregate temporal cross-validation, not nested model-selection validation. Its winning score is selection-biased. The outcome-exposed retrospective years provide a stress test, and only 2026 can provide a sealed prospective estimate.

Before fitting, `prefit_manifest_v2.json` locks the amended plan, all six Phase 2 implementation files, the Phase 2 tests, this ledger, and the reproduction guide. The discovery selection records the same six code hashes and the pre-fit manifest hash. The retrospective runner fails before fitting if either differs. This makes implementation drift detectable across stages; it does not make the historical outcomes blinded.
