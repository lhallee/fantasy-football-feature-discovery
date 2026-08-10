# Methods and results

## Prediction question

For each fantasy-position player in a target-season roster cohort, predict full regular-season ESPN PPR points using only information available before that season. Same-season statistics construct the label. Imported fantasy points, expert projections, ADP, betting lines, EPA, WOPR, target share, depth rank, and other prebuilt indices are excluded from model inputs. Prior fantasy points appear only as benchmarks.

The primary metric is mean Spearman correlation across position-season groups. Secondary metrics are Kendall \(\tau_b\), MAE, RMSE, \(R^2\), position-rank MAE, NDCG at QB12/RB24/WR36/TE12/K12, top-k recall, and calibration slope and intercept.

## Cohort and temporal protocol

- Unit: one `(player_id, target_season)` row.
- Cohort: every fantasy-position player in the historical Week 1 roster proxy, including backups and eventual zero scorers.
- Inputs: target-season metadata plus raw one- and two-season lags.
- Development: chronological 2020, 2021, 2022, and 2023 folds.
- Retrospective evaluation: train before 2024 and predict 2024, then train before 2025 and predict 2025.
- Final forecast: refit through 2025 and predict the August 9, 2026 roster snapshot.

The development folds support model, window, and feature decisions. Their reuse makes the comparison exploratory rather than a formal multiple-comparison experiment. Aggregate 2024-2025 results were inspected during pipeline development, so those years are retrospective evaluation, not a sealed audit. No 2026 outcomes were available.

## Raw feature contract

The 32-atom offense base menu contains direct metadata, combine measurements, and prior box-score counts. An optional 33rd atom adds offense snaps. Current status and depth are retained as catalog metadata but excluded from modeling because their historical and current cutoffs are not comparable. Draft number remains because it is a direct source scalar, although it embeds NFL teams' scouting decisions rather than pure player performance.

Each statistical atom generates five columns: lag one, lag two, `log1p` of each nonnegative lag, and lag-one minus lag-two trend. Age generates raw, `log1p`, and squared values. Draft number generates raw and `log1p` values. The pooled offense model also requires four position indicators. There are no hidden per-game denominators or fantasy-derived features.

## Model and hyperparameter search

Thirty deterministic settings cover ridge, elastic net with L1 ratios 0.8-1.0, random forest, extra trees, histogram gradient boosting, XGBoost, RBF SVR, and small MLPs. A recent development fold retains one setting per family. Those finalists are compared across all four development seasons, the base and utilization tiers, and five-season, eight-season, and all-history windows.

Estimator `n_jobs` and process-native numerical pools are capped at four threads. CUDA is not required. ConvNets and transformers were not tested because this is a small tabular panel without a justified spatial or token order; the tree, kernel, linear, boosting, and small-neural candidates already cover nonlinear interactions at lower cost.

The point-estimate offense champion is `extra_trees_02`:

- 240 trees
- maximum depth 12
- minimum leaf size 5
- all features considered per split
- all available training seasons
- base feature tier

The full 32-atom model reached development \(\rho=0.74325\), MAE 31.435, NDCG 0.8091, and top-k recall 0.5851. The selected high-L1 ranking model was a lasso with alpha 0.2. The model-family comparison does not establish that Extra Trees is meaningfully superior to every near-tied finalist; it is the winner under the declared point-estimate rule.

## Compact feature search

Atoms are ranked by the mean percentile rank of three development-only signals:

1. Grouped permutation loss in within-position Spearman.
2. Summed absolute coefficients from the selected high-L1 elastic net.
3. Absolute within-position, within-season univariate Spearman association.

The pipeline refits ranked prefixes of size 1-15 and the full set across all four development folds. A set passes when mean Spearman loss versus the full model is at most 0.01 and mean MAE is at most 1.02 times the full-model MAE. The smallest passing prefix had 11 atoms.

Greedy backward deletion then removed, in order, receptions, yards after catch, receiving touchdowns, carries, passing yards, and games. At each step, every one-atom deletion was evaluated on the four development folds, and the passing subset with highest Spearman was retained. No deletion from the final five-atom set passed. This proves local irreducibility on that greedy path, not a global minimum across all \(2^{32}\) atom subsets.

The final five atoms are:

| Atom | Direct meaning | Mean univariate Spearman | Mean grouped permutation loss |
|---|---|---:|---:|
| `draft_number` | Overall NFL draft pick; undrafted is coded 300 | -0.561 | 0.152 |
| `receiving_yards` | Prior raw receiving yards | 0.495 | 0.081 |
| `rushing_yards` | Prior raw rushing yards | 0.432 | 0.030 |
| `passing_tds` | Prior raw passing touchdowns | 0.333 | 0.010 |
| `age` | Age on September 1 of the target season | 0.116 | 0.008 |

All five have positive grouped permutation importance in each development year. Their exploratory Benjamini-Hochberg adjusted correlation tests range from below \(10^{-15}\) to 0.0051. Repeated players and correlated atoms violate simple independent-test assumptions, so those p-values describe consistency, not causal evidence or a selection guarantee.

The five atoms expand to 24 estimator columns: 3 age columns, 2 draft columns, 15 lag/log/trend columns from the three statistics, and 4 position indicators. Development \(\rho\) is 0.73865 and MAE is 31.770. Relative to the full model, the losses are 0.00460 Spearman and 1.07% MAE. Every four-atom deletion fails at least one threshold.

## Retrospective offense results

| Predictor | Spearman | Kendall \(\tau_b\) | MAE | RMSE | \(R^2\) | NDCG | Top-k recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Five-atom Extra Trees | 0.7575 | 0.5917 | 29.46 | 46.01 | 0.654 | 0.7940 | 0.5747 |
| Prior-year points | 0.6688 | 0.5522 | 29.58 | 56.10 | 0.486 | 0.7642 | 0.5382 |
| Two-year recency points | 0.6437 | 0.5231 | 29.57 | 54.41 | 0.516 | 0.7648 | 0.5313 |
| Position and rookie median | 0.2119 | 0.1836 | 48.55 | 80.53 | -0.060 | 0.2055 | 0.1181 |

A 500-sample player-cluster bootstrap gives a 95% interval of 0.7249-0.7822 for model Spearman. The paired improvement over prior-year points is 0.0887, with interval 0.0602-0.1157. The model improves ranking and RMSE; its MAE advantage is only 0.12 points, and position-rank MAE is slightly worse than the baseline.

| Position | Rows | Model Spearman | Prior-points Spearman | Model MAE | Prior-points MAE | Model top-k recall |
|---|---:|---:|---:|---:|---:|---:|
| QB | 257 | 0.756 | 0.635 | 41.25 | 45.17 | 0.542 |
| RB | 458 | 0.707 | 0.616 | 35.25 | 31.94 | 0.646 |
| WR | 805 | 0.765 | 0.685 | 27.18 | 27.68 | 0.611 |
| TE | 412 | 0.802 | 0.739 | 20.14 | 20.93 | 0.500 |

The running-back point error is worse than prior points despite better rank correlation, so the model should not be treated as uniformly better on every objective.

### Feature-count sensitivity

| Atom set | Columns | Spearman | MAE | NDCG | Top-k recall |
|---|---:|---:|---:|---:|---:|
| Selected five | 24 | 0.7575 | 29.46 | 0.7940 | 0.5747 |
| Top 15 | 70 | 0.7590 | 29.17 | 0.7833 | 0.5885 |
| Full 32 | 117 | 0.7694 | 28.83 | 0.7853 | 0.5764 |

The compact model gives up retrospective correlation and MAE relative to all atoms, while slightly improving NDCG. The compact threshold was chosen on development folds; the retrospective period did not choose the final set.

Rookie-only retrospective rows have \(\rho=0.646\) and MAE 20.53. Many deep rookies score zero, which makes point error look favorable. Without NFL history, rookies rely heavily on draft number and age; the catalog labels 224 current offense rookies `limited_rookie_history`.

## Kicker result

The greedy kicker search reduces to prior `pat_made`, expanded into five lag/log/trend columns and fit by an all-history RBF SVR. Development \(\rho=0.509\). Retrospective \(\rho=0.466\) is below prior-year points at 0.512. MAE is 43.51 versus 43.67, but NDCG is 0.702 versus 0.810 and top-12 recall is 0.458 versus 0.500.

This does not support a better kicker ranking. All kicker rows carry `low_kicker_rank_evidence`, and an explicit K or PK query defaults to the stronger prior-points order. Pass `sort_by="prediction"` to inspect model order.

## Leading August 9 outputs

| Position | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| QB | Lamar Jackson, 286.2 | Josh Allen, 281.4 | Joe Burrow, 276.3 |
| RB | Bijan Robinson, 269.5 | Jahmyr Gibbs, 264.1 | Ashton Jeanty, 249.8 |
| WR | Puka Nacua, 284.0 | Amon-Ra St. Brown, 248.8 | Ja'Marr Chase, 244.5 |
| TE | Trey McBride, 197.7 | Brock Bowers, 180.2 | Kyle Pitts, 177.9 |
| K | Jake Elliott, 121.2 | Wil Lutz, 120.6 | Jason Myers, 119.4 |

These are August 9 model outputs, not observed 2026 results or optimal draft order. The catalog contains every nflverse roster player, 791 eligible backups, approximate intervals, prior-season points, exact fitted vectors, and confidence labels.

## Claim boundary

The result is conditional on the public sources, season cohort, raw atom menu, mechanical transforms, model families, metric, and greedy search described above. It does not establish a universal minimum, causal drivers, prospective accuracy, or optimal draft value. Correlated atoms can substitute for one another. Draft number carries human judgment. Current injuries, schedule context, and final roster decisions are absent.
