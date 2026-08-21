# Phase 3 Experiment Ledger

## Entry 0: Request and scope

Date: 2026-08-14

Goal: predict whether each fantasy-position player receives any physical injury designation during the season, identify the strongest raw-feature associations, and combine calibrated injury-report risk with the Phase 2 raw fantasy forecast.

The analysis is a new layer. It does not retrain or replace the Phase 2 point projection.

## Entry 1: Outcome decision

The official nflverse injury feed contains weekly game-report and practice-report fields from 2009-2024. Phase 3 defines `physical_injury_reported = 1` when any of four injury fields contains a physical designation during the regular season.

Excluded values are blank, illness, COVID, personal matters, rest, coaching decisions, and explicit non-injury designations. Mixed values with a physical term remain positive. The label is an observed report event. It is not equivalent to games missed or confirmed tissue damage.

The 16 Parquet files contain 84,684 source rows and 1,881,309 bytes. Their URLs, sizes, and SHA-256 hashes are stored in [`data/raw/manifest.json`](data/raw/manifest.json).

## Entry 2: Protocol lock

The pre-fit plan was written at 2026-08-14T14:20:00Z. Its SHA-256 is `b3f74279771213fb3c99e95c7a9e83c9f2ec14958914d4b8180174edebf04dcb`.

The plan locked model selection on 2017-2021, audit seasons 2022-2024, base training through `y - 2`, calibration on `y - 1`, 21 candidates, fold-local feature ranking, an 11-seed negative-control gate, a player-cluster bootstrap, and a fixed 80/20 combined-score weight.

The [pre-fit manifest](prefit_manifest_v1.json) hashes the plan, relevant code, every restored Phase 2 raw and processed input, the Phase 2 2026 prediction artifact, and every injury source. The inherited Phase 1 whole-repository freeze does not match the relocated repository's later committed documentation. Phase 3 therefore binds the files it reads and verifies them again after the run.

## Entry 3: Feature assembly

The August-origin cohort contains 14,631 rows from 2013-2026. The labeled 2013-2024 table contains 12,434 rows. The 2026 inference table contains 958 players.

The feature matrix has 500 columns. Every weekly or seasonal performance source is lagged by at least one season. Injury history is lagged two through five seasons for every target, matching the unavailable 2025 injury feed at the 2026 origin. Target-season roster-room, team, status, depth, activity, injury, and outcome fields are excluded.

Historical positive rates range from 31.4% to 47.2% by season. Across the labeled sample, rates are 21.9% for K, 30.9% for QB, 38.5% for RB, 39.4% for TE, and 37.0% for WR.

## Entry 4: Development search

The run evaluated an age-and-position logistic baseline, elastic-net logistic regression, Extra Trees, and histogram gradient boosting. Candidate feature counts were 16, 32, 64, or all where declared.

The selected candidate was `extra_trees_all_leaf20`:

- mean 2017-2021 ROC AUC: 0.842364
- mean average precision: 0.713670
- mean log loss: 0.452478
- mean Brier score: 0.147776
- 500 features

The all-feature Extra Trees candidate with minimum leaf size 50 reached AUC 0.842225 but had worse log loss. The best 64-feature candidate reached 0.832963, outside the 0.005 eligibility band. The selected procedure therefore uses all inputs.

## Entry 5: Negative control

Eleven controls shuffled fitting labels within season and position, refit fold-local feature ranking, base classifiers, and calibrators, then evaluated predictions against the real validation labels.

Mean pooled ROC AUC was 0.521981, inside the locked 0.45 to 0.55 gate. Seed results ranged from 0.478708 to 0.556505. Ten of 11 were above 0.5. The gate passed, but the right shift is retained as a warning about group prevalence and feature geometry. It does not prove that every possible label path is absent.

## Entry 6: Locked audit

The selected procedure was evaluated once on 2022, 2023, and 2024.

- pooled ROC AUC: 0.867580
- player-cluster 95% interval: 0.853669 to 0.880367
- average precision: 0.763161 at 0.349429 prevalence
- log loss: 0.429674
- Brier score: 0.140231
- calibration intercept: 0.041688
- calibration slope: 1.028054

The age-and-position baseline reached ROC AUC 0.551359. The predeclared success rule passed.

Position AUC was 0.857440 for QB, 0.850512 for RB, 0.898154 for WR, 0.847289 for TE, and 0.709334 for K.

## Entry 7: Feature interpretation

The strongest discovery association was prior-year games: univariate AUC 0.741750 and Spearman correlation 0.416413. Prior targets, receptions, receiving yards, yards after catch, first downs, offense snaps, and late-season receiving volume followed.

This pattern is exposure-sensitive. High-usage players have more opportunities to receive a report designation. It should not be interpreted as a causal claim that receiving volume damages players.

The lag-two injury designation count and report weeks each reached univariate AUC about 0.622 and ranked 42nd and 43rd. The lag-two event flag reached audit AUC 0.622812 by itself.

## Entry 8: Post-selection exposure diagnostic

The selected model's AUC was recomputed in subgroups defined only by prior-year games:

- all audit candidates: 0.867580
- prior-year games > 0: 0.789206
- prior-year games >= 4: 0.749866
- prior-year games >= 8: 0.729471

Prior-year games alone reached audit AUC 0.772430. The diagnostic confirms that candidate persistence and exposure explain a substantial part of the headline. It also shows residual discrimination among established players. Because this check was added after selection, it does not change the champion or the locked audit claim.

## Entry 9: Production fit and combined score

The final base model fits labeled seasons through 2023. The sigmoid calibrator uses 2024. It produces calibrated injury-report probabilities for all 958 Phase 2 current players.

For each position, the combined score is:

```text
100 * (0.80 * raw-points percentile + 0.20 * no-injury-report probability)
```

This index is not expected fantasy points. It expresses the declared preference that fantasy value receives four times the weight of report risk. Before the dated current-status overlay, the workbook candidate set includes 951 available players.

## Entry 10: Verification

The completed model reproduces all 958 stored injury probabilities with maximum absolute error `4.44e-16`. The saved model and lineage each contain the same 500 ordered feature names. All run outputs match [`output_hashes.json`](runs/20260814T143014705993Z-injury-risk-v1/output_hashes.json).

The pre-overlay workbook contained Read Me, QB, RB, WR, TE, and K sheets. That layout was superseded by the dated current-status overlay in Entry 11.

A broader repository run with an isolated temporary directory passed 100 tests, failed 12, and deselected one older freeze test. All 12 failures are Phase 1 or Phase 2 byte-hash, historical-layout, or stored-size contracts that are stale in the relocated checkout. Phase 3 does not rewrite those historical manifests after their results were observed. Its independent pre-fit manifest, release hash chain, and 15 focused tests pass exactly.

## Entry 11: Dated current-injury screen and workbook revision

Date: 2026-08-14

The operational draft workbook was revised after the frozen model run. This did not refit the fantasy model or injury classifier, alter their predictions, or change the 80/20 score. It added a dated current-status filter from the documented MyFantasyLeague injury API, with two official NFL reserve-list additions and nine independently reviewed secondary-source additions that the feed or strict team match omitted.

All 951 available players were screened. One hundred nine matched affirmative current injury evidence: 75 Questionable, 9 Out, 23 IR, and 2 official Reserve/Injured. They were removed from `ALL` and the position pages and preserved on `INJURED` with their original risk-adjusted score, fantasy projection, modeled injury probability, status, evidence date, expected return, and source. The other 842 rows are described as `No current listing found`, not healthy.

The workbook now contains `ALL`, `QB`, `RB`, `WR`, `TE`, `K`, and `INJURED`, with no read-me sheet. Every sheet uses contiguous ranks and sorts by risk-adjusted score, then projected points, then player ID. The three core measures have separate columns named `Risk-Adjusted Draft Score`, `Projected 2026 Fantasy Points`, and `Predicted Physical Injury-Report Probability`.

This is a conservative training-camp screen. A `Questionable` row is a current public injury listing, not a standardized Week 1 game designation or proof of unavailability. Absence from the feed is not proof of health. Exact source rows, the complete player screen, unmatched source rows, retrieval time, hashes, and counts are stored in [`current_injury_manifest_v1.json`](current_injury_manifest_v1.json).

The original pre-fit manifest remains unchanged and continues to identify the exact modeling implementation. Because the workbook writer changed after fitting, the current-injury manifest separately hashes the two operational modules. Verification checks all unchanged model and data inputs against the pre-fit record, checks the revised operational code against the overlay manifest, and replays the saved model probabilities exactly.

The final focused Phase 3 overlay gate passed 13 tests. A repository-wide run passed 105 tests and failed 13 older Phase 1 or Phase 2 freeze, layout, plan-hash, and stored-size contracts. Those records describe earlier immutable states and were not rewritten to make a later operational workbook pass. Ruff and source compilation pass for the changed implementation and tests.
