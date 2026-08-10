# Experiment and excitement ledger

This ledger records the work behind the frozen August 9, 2026 forecast. It preserves failed checks, design changes, runtimes, and negative results because they constrain the claims. Commands are relative to the source-project root.

## 1. Objective and constraints

Status: complete.

- Predict next-season ESPN full-PPR total and within-position rank for individual players.
- Use raw source fields and self-generated mechanical transforms, not imported fantasy features or projections.
- Retain backups and rostered zero scorers.
- Train and infer on this laptop without requiring CUDA.
- Cap estimators and native numerical pools at four CPU threads.
- Keep source data, compiled tables, model artifacts, and the generated catalog below 1 GB.
- Produce a queryable 2026 Python data structure and durable evidence for every headline result.

The final environment records Python-compatible package versions in `artifacts/model_manifest.json`: NumPy 1.26.4, pandas 2.3.1, PyArrow 22.0.0, scikit-learn 1.6.1, SciPy 1.13.1, XGBoost 2.1.3, joblib 1.3.2, and threadpoolctl 3.2.0.

## 2. Scoring and popularity research

Status: complete with source limitations.

Full PPR was selected before modeling. RotoWire reported full PPR in 47.6% of 36,599 imported 2025 leagues, compared with 38.4% half PPR and 13.4% non-PPR. This is a convenience sample, not a population census.

ESPN's July 2026 coefficients were encoded in `config/scoring_espn_full_ppr_2026.json`. The build now loads that file directly and records SHA-256 `744c0cea0e70966966d5446595b0858304c5e941a1951fff8feb078e78792d7e`.

Scoring audit findings and fixes:

- Reception points had been hardcoded, so custom profile values were ignored. The scorer now uses the loaded coefficient for full and half PPR.
- Kicker targets omitted reception points. They now combine full-PPR offense points with kicking points.
- Missing required source columns were silently padded with zeros. The loader now checks each Parquet schema before optional padding.
- Individual defensive touchdowns are supported through a curated field, but the production nflverse input lacks a trustworthy individual source. This remains a documented two-way-player limitation.
- The target is every NFL regular-season row, including Week 18 since 2021.

Reconstructed offense PPR matches nflverse's comparison column exactly on 99.3303% of eligible player-game rows. Mean absolute difference is 0.0153 points. The comparison column is never a feature.

## 3. Source selection

Status: complete.

nflverse was selected because it provides downloadable Parquet files, stable GSIS identifiers, documented refresh schedules, and a CC BY 4.0 data release. Sleeper was considered for current players, but its July 24, 2026 terms prohibit automated extraction without written consent. The pipeline does not call Sleeper.

Sources considered but deferred:

- Play-by-play would add about 488 MB and remains under the cap, but is not required for this first player-season experiment.
- Next Gen Stats begins in 2016 and has threshold-selected missingness.
- nflverse injury data stop after 2024.
- Schedule, PFR advanced, and team-context fields remain future candidates.

## 4. Download and provenance

Status: complete.

Frozen-source command:

```powershell
python -m fantasy_football download --workers 4
```

The local source set contains 94 Parquet files and 56,740,097 source bytes:

| Source | Seasons | Files | Bytes |
|---|---:|---:|---:|
| Weekly player statistics | 1999-2025 | 27 | 20,206,597 |
| Weekly rosters | 2002-2026 | 25 | 14,157,966 |
| Depth charts | 2002-2026 | 25 | 14,979,097 |
| Snap counts | 2012-2025 | 14 | 2,918,884 |
| Player master | Historical-current | 1 | 3,404,185 |
| Draft picks | 1980-2026 | 1 | 699,050 |
| Combine | 2000-2026 | 1 | 374,318 |

An audit found that manifest version 1 labeled a later verification time as a download cutoff. Manifest version 2 now separates `retrieved_at_utc`, `verified_at_utc`, and `data_cutoff_utc`. The frozen cutoff is 2026-08-10 03:11:22 UTC; the local bytes were reverified at 04:40:53 UTC.

A live release audit found all 94 assets present with identical sizes. All 48 assets for which GitHub published digests matched those digests. The other 46 older assets match the local manifest but lack an upstream digest.

Default download behavior reuses local files. A current refresh must use `--force`.

## 5. Table construction

Status: complete.

Command:

```powershell
python -m fantasy_football build --snapshot-date 2026-08-09
```

Observed wall time was about 8.4 seconds.

| Table | Rows | Grain |
|---|---:|---|
| Player games | 475,626 | player, game |
| Player seasons | 66,907 | player, season |
| Roster proxies and snapshot | 56,505 | player, season |
| Modeling table | 18,630 | player, target season |

Quality findings:

- 11,372 identified players appear in 1999-2025 game data.
- All processed primary keys are non-null and unique at their declared grain.
- PFR or GSIS identifiers match 99.9295% of available snap rows.
- 530 unresolved stat rows have no player ID, display name, or position. Of those, 156 have generic name `Team`. They contribute 6.68 PPR points and are excluded from individual-player data.
- Week 1 historical roster files do not prove a preseason as-of date. Documentation now calls them Week 1 roster proxies.

The current nflverse roster has 2,930 unique GSIS IDs across all 32 teams. The processed snapshot and Python catalog contain the same IDs. Of these, 958 are fantasy-position rows: 119 QB, 213 RB, 388 WR, 195 TE, and 43 K.

## 6. Cutoff mismatch discovered

Status: fixed before final training.

The first model used `status_active` and normalized `depth_tier`. Audit evidence showed both were unsuitable for the August prediction contract:

- Historical fantasy-position active rates were about 47-52%, but the August camp file marked 936 of 958 current fantasy-position players active.
- The status atom had material permutation importance and inflated deep camp players by about 17 points on average in a counterfactual check.
- Historical depth uses Week 1 assignments, while current depth is an August daily file.
- Depth is also a human-produced role ranking rather than a raw player measurement.

Both atoms were removed from the candidate menu. They remain catalog metadata. The final model therefore gives up some retrospective accuracy in exchange for a defensible and time-comparable input contract.

## 7. Hidden-input audit

Status: fixed.

The first feature generator created per-game rates by dividing selected statistical atoms by `games`, even when games was not counted. This made the reported atom set understate its raw inputs.

Per-game transforms were removed. Each selected statistical atom now owns only its two lags, two log transforms, and one trend. The exact fitted vector is stored for every eligible current player, including `None` for missing values.

## 8. Temporal validation

Status: complete.

- Development seasons: 2020-2023.
- Split rule: all training target seasons precede the validation season.
- Window candidates: five seasons, eight seasons, and all history.
- Retrospective evaluation: 2024 and 2025 rolling fits.
- Prospective output: refit through 2025 and predict the August 9, 2026 snapshot.
- Primary metric: mean within-position, within-season Spearman correlation.
- Secondary metrics: Kendall tau-b, MAE, RMSE, R-squared, rank MAE, NDCG, top-k recall, and calibration.

The 2024-2025 results were visible during pipeline development. They are retrospective evidence, not a sealed holdout. Feature and hyperparameter comparisons also reuse the four development years, so uncertainty is broader than one fixed-model confidence interval suggests.

## 9. Model search

Status: complete.

Command:

```powershell
python -m fantasy_football train
```

The final full run took 743.4 seconds. A reduced seven-setting integration run with one deletion pass took 147.6 seconds. Observed process memory remained near or below 500 MB RAM in both runs. CUDA was unused.

Thirty settings covered eight families: ridge, high-L1 elastic net, random forest, extra trees, histogram gradient boosting, XGBoost, RBF SVR, and small MLPs. The winning offense point estimate was `extra_trees_02` with 240 trees, depth 12, minimum leaf size 5, all features at each split, the base tier, and all historical training seasons.

The full 32-atom model reached development Spearman 0.74325 and MAE 31.435. The high-L1 elastic model used for coefficient ranking was a true lasso with alpha 0.2. One rejected elastic candidate emitted a convergence warning; the selected offense and kicker ranking elastic models converge cleanly.

## 10. Compact-set search

Status: complete with a bounded claim.

Atom order combines three development-only ranks:

1. Grouped permutation Spearman loss.
2. Absolute coefficient mass from the selected elastic model.
3. Absolute univariate within-position, within-season Spearman.

The pipeline tested ranked prefixes of size 1-15 and the full set. A prefix passes if mean Spearman loss versus full is at most 0.01 and mean MAE is no more than 2% worse. The smallest passing prefix contained 11 atoms.

Greedy backward deletion evaluated every one-atom removal at each step:

| Iteration | Removed | Remaining | Development Spearman | Development MAE |
|---:|---|---:|---:|---:|
| 1 | receptions | 10 | 0.73898 | 31.686 |
| 2 | receiving YAC | 9 | 0.73940 | 31.674 |
| 3 | receiving TDs | 8 | 0.74056 | 31.753 |
| 4 | carries | 7 | 0.74000 | 31.729 |
| 5 | passing yards | 6 | 0.73953 | 31.785 |
| 6 | games | 5 | 0.73865 | 31.770 |
| 7 | no deletion passed | 5 | 0.73865 | 31.770 |

The final five atoms are draft number, prior receiving yards, prior rushing yards, prior passing touchdowns, and target-season age. They expand to 24 model columns. The full-model Spearman loss is 0.00460 and the MAE increase is 1.07%. Every four-atom deletion fails at least one threshold.

This is locally irreducible along a deterministic greedy path. It is not the global minimum across every possible subset. Correlated substitutes outside the chosen path may exist.

## 11. Retrospective offense evidence

Status: complete.

| Predictor | Spearman | MAE | RMSE | NDCG | Top-k recall |
|---|---:|---:|---:|---:|---:|
| Five-atom Extra Trees | 0.7575 | 29.46 | 46.01 | 0.7940 | 0.5747 |
| Prior-year points | 0.6688 | 29.58 | 56.10 | 0.7642 | 0.5382 |
| Two-year recency points | 0.6437 | 29.57 | 54.41 | 0.7648 | 0.5313 |

The 500-sample player-cluster interval for model Spearman is 0.7249-0.7822. The paired gain over prior points is 0.0887 with interval 0.0602-0.1157.

Important counterevidence:

- Running-back MAE is 35.25 versus 31.94 for prior points.
- Position-rank MAE is slightly worse overall despite higher Spearman.
- The full 32-atom retrospective model reaches Spearman 0.7694 and MAE 28.83. Compactness has a measurable accuracy cost.
- The five-atom model improves aggregate NDCG over the 15- and 32-atom variants, but not every position-specific metric.

Rookie-only retrospective results are Spearman 0.646 and MAE 20.53. Many deep rookies score zero, so low point error does not prove accurate top-rookie order.

## 12. Kicker negative result

Status: retained rather than hidden.

Greedy pruning reduces the kicker model to prior PAT makes, represented by five lag/log/trend columns and fit with an RBF SVR. Retrospective Spearman is 0.466 versus 0.512 for prior-year points. Model MAE is 43.51 versus 43.67, but NDCG is 0.702 versus 0.810.

The kicker model does not support a superior rank claim. All 43 current kickers carry `low_kicker_rank_evidence`. An explicit K or PK query defaults to the stronger prior-points order, while `sort_by="prediction"` exposes the model order.

## 13. Catalog export

Status: complete.

Command:

```powershell
python -m fantasy_football export
```

The generated `fantasy_football/players_2026.py` contains all 2,930 snapshot players. All 958 fantasy-position rows have predictions and exact fitted vectors. The remaining 1,972 records are metadata-only.

Each record contains:

- GSIS, ESPN, Sleeper, and PFR identifiers when available
- team, raw position, model position, status, jersey, and depth metadata
- birth date, age, height, weight, college, experience, and draft fields
- prediction, prior-season points, interval, and three ranks
- scoring profile, selected atoms, exact fitted feature vector, feature coverage, and out-of-distribution flag
- confidence label

Default queries exclude seven fantasy-position rows marked cut or retired. The full catalog preserves them. Team aliases and filters for status, name, college, confidence, depth, rank, point threshold, and out-of-distribution state are supported.

## 14. Current leading outputs

Status: forecast only.

- QB: Lamar Jackson 286.2, Josh Allen 281.4, Joe Burrow 276.3.
- RB: Bijan Robinson 269.5, Jahmyr Gibbs 264.1, Ashton Jeanty 249.8.
- WR: Puka Nacua 284.0, Amon-Ra St. Brown 248.8, Ja'Marr Chase 244.5.
- TE: Trey McBride 197.7, Brock Bowers 180.2, Kyle Pitts 177.9.
- K model order: Jake Elliott 121.2, Wil Lutz 120.6, Jason Myers 119.4.

These are August 9 predictions, not observed outcomes or optimal draft values.

## 15. Implementation and audit fixes

Status: complete.

- Added missing `Any` import that had broken the CLI after a provenance edit.
- Made snapshot date explicit and validated against the requested season.
- Added scoring-profile loading, hash lineage, strict required-column checks, kicker receptions, and configurable reception scoring.
- Removed time-mismatched status and human depth rank from model inputs.
- Removed hidden games denominators from per-game transforms.
- Added `threadpoolctl` so BLAS and OpenMP pools honor the four-thread limit.
- Isolated smoke output below `artifacts/smoke`; full output remains under `artifacts/`.
- Made result summaries, cohort intervals, sensitivity tables, elastic searches, and greedy-refinement evidence pipeline outputs.
- Added source-manifest, modeling-table, scoring-config, package-version, seed, and run-mode provenance.
- Added exact fitted vectors and richer filters to the player catalog.
- Fixed NDCG by clipping negative relevance to zero, as required by scikit-learn.
- Added an XGBoost compatibility wrapper for scikit-learn estimator tags.
- Added explicit package discovery so a flat-layout wheel builds correctly.
- Removed stale smoke and packaging build directories after verification.

The global environment autoloads a broken third-party `torchtyping` pytest plugin before project tests. The project does not use Torch. Verification disables external plugin autoload.

## 16. Verification

Status: clean after the final documentation and standards pass.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -v --tb=short
python -m ruff check .
python -m compileall fantasy_football tests
python -m fantasy_football --help
```

Twenty-five tests pass. They cover scoring configuration and arithmetic, schema failure, source hashes and manifest v2, table grain, snapshot cutoff, no-target features, excluded status/depth/per-game signals, exact saved-estimator predictions, compact thresholds and refinement, retrospective metric reconstruction, input/scoring hashes, thread limits, smoke-output isolation, data cap, full roster parity, exact fitted vectors, query aliases and ordering, and kicker warnings.

The repository-wide coding pass inspected all 18 eligible first-party Python files and excluded only the generated catalog. Numerical modules now trace DataFrame, Series, mask, vector, model-call, reduction, mutation, concatenation, and return shapes. The writing audit found no actionable prose issues; its two repeated-opening warnings refer to command lines inside code blocks.

Data plus model artifacts use 82,802,977 bytes. The generated Python catalog uses 3,468,732 bytes. Combined use is 86,271,709 bytes, or 8.63% of decimal 1 GB.

## 17. What was most informative

Status: recorded as the "excitement" part of the ledger.

- The strongest early result was also the least trustworthy: status and depth produced high retrospective rank correlation but failed the current-cutoff contract.
- Removing those fields reduced the headline correlation but made the result scientifically defensible.
- High-L1 elastic net was not the final offense predictor, but it was useful as one of three independent atom-ranking signals.
- Greedy deletion reduced an 11-atom passing prefix to five without exceeding development tolerances.
- The simple final atoms are interpretable, but the result still needs 24 transformed and position-adjustment columns.
- Kicker rank failure was clear and worth exposing rather than smoothing over.
- Exact provenance and refresh behavior mattered as much as model choice because August roster files change rapidly.

## 18. Refresh checklist

Status: ready for use.

1. Archive the August `artifacts/` directory and `players_2026.py` if prospective scoring matters.
2. Run `python -m fantasy_football all --force --snapshot-date YYYY-MM-DD` after final cuts.
3. Inspect manifest hash changes and current status counts.
4. Review `limited_rookie_history`, `low_kicker_rank_evidence`, `not_available`, and `out_of_range` labels.
5. Convert position ranks to league-specific draft value only after adding lineup slots, replacement level, and scoring settings.
6. After the 2026 regular season, score the frozen August snapshot as the first prospective evaluation.
