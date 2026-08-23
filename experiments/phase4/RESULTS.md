# Phase 4 methods and results

## Answer

The final 2026 ranking uses the `ext_et_all` points forecast and the `extra_trees_all_leaf50` missed-time injury classifier. On the locked 2022-2025 audit the adopted points procedure reached mean within-position-season Spearman 0.7557 (player-cluster 95% interval 0.7321 to 0.7751) against 0.7510 for the Phase 2 production model and 0.6472 for prior-season points. The injury classifier reached pooled 2022-2024 ROC AUC 0.7486 (95% interval 0.7306 to 0.7672) at 24.7% prevalence with calibration slope 1.019.

The default injury weight in the draft score is 0.20. The workbook is `outputs/fantasy_football_final_rankings_2026.xlsx`.

The paired gain of the adopted points procedure over Phase 2 production is 0.0047 with a player-cluster interval of -0.0001 to 0.0093, so the two forecasts are close to equivalent in rank accuracy; the adopted procedure has lower MAE (24.69 against 25.25), uses the 2025 roster reserve history, and passes the stricter null control described below.

## Draft scores

Two scores appear in the workbook.

- Position Draft Score, used on the QB, RB, WR, TE, and K pages: `100 * ((1 - w) * within-position percentile of projected points + w * (1 - missed-time injury probability))`.
- Overall Draft Score, used on the ALL and INJURED pages: points over replacement is projected points minus the projected points of the replacement-level player at the position (QB12, RB24, WR36, TE12, K12: 219.8, 157.0, 122.5, 124.9, and 117.2 points). Overall value is points over replacement divided by the board maximum with negatives set to zero, and the score is `100 * ((1 - w) * overall value + w * (healthy probability - position mean healthy probability + 0.5))`.

The position-mean centering in the overall score exists because the injury model is partly an exposure model: kickers average 13% predicted risk and running backs 30%. Without centering, a raw health blend placed four kickers at the top of the overall list. The replacement-level cutoffs follow the project's roster-cutoff constants and are a league-format assumption, not an estimate.

The injury weight `w = 0.20` is the largest weight in the predeclared grid whose development-fold rank accuracy against realized points stayed within 0.005 Spearman of the best weight. On the audit seasons the weight costs 0.004 Spearman (0.7538 to 0.7499). The weight does not reduce the top-k bust rate in either period, so it should be read as a transparent risk preference, not as a backtested improvement in downside outcomes.

## Injury label

A player-season is positive when the player is listed Out or Doubtful with a physical injury on any regular-season official report, or spends any regular-season roster week on an injury-compatible reserve list. Questionable-only and practice-only designations are negative. COVID, retired, did-not-report, left-squad, future, and suspension reserve codes are excluded. The label covers 2013-2024. On the audit seasons the selected learner reached ROC AUC 0.8676 against the Phase 3 any-designation label, for comparison.

## Points track

Discovery used validation seasons 2019-2021 with training from 2017 (plan amendment 1) and selected `ext_et_all`. The audit adoption rule then adopted `ext_et_all` because its 2022-2025 mean Spearman did not fall below the Phase 2 production value minus 0.005.

| candidate             |   components |   spearman |     mae |   ndcg_at_roster_cutoff |   top_k_recall |
|:----------------------|-------------:|-----------:|--------:|------------------------:|---------------:|
| phase2_production     |            1 |     0.7173 | 26.2524 |                  0.7685 |         0.5532 |
| ext_et_all            |            1 |     0.7196 | 25.9925 |                  0.7413 |         0.5069 |
| ext_hgb_all           |            1 |     0.7032 | 26.0564 |                  0.7500 |         0.5231 |
| ext_xgb_all           |            1 |     0.6791 | 27.2129 |                  0.7616 |         0.5255 |
| blend_ext_et_hgb      |            2 |     0.7233 | 25.6826 |                  0.7538 |         0.5370 |
| blend_ext_all3        |            3 |     0.7159 | 25.9825 |                  0.7494 |         0.5197 |
| blend_phase2_ext_et   |            2 |     0.7211 | 25.9871 |                  0.7670 |         0.5498 |
| blend_phase2_ext_all3 |            4 |     0.7203 | 25.9077 |                  0.7618 |         0.5336 |

Locked 2022-2025 audit:

| candidate                |   spearman |     mae |   ndcg_at_roster_cutoff |   top_k_recall |
|:-------------------------|-----------:|--------:|------------------------:|---------------:|
| phase2_production        |     0.7510 | 25.2496 |                  0.7895 |         0.5651 |
| ext_et_all               |     0.7557 | 24.6948 |                  0.7884 |         0.5521 |
| ext_hgb_all              |     0.7435 | 24.7806 |                  0.7889 |         0.5694 |
| ext_xgb_all              |     0.7293 | 26.1087 |                  0.7925 |         0.5616 |
| blend_ext_et_hgb         |     0.7580 | 24.4770 |                  0.7884 |         0.5634 |
| blend_ext_all3           |     0.7537 | 24.8481 |                  0.7928 |         0.5686 |
| blend_phase2_ext_et      |     0.7554 | 24.8700 |                  0.7943 |         0.5608 |
| blend_phase2_ext_all3    |     0.7573 | 24.8260 |                  0.7954 |         0.5747 |
| previous_points_baseline |     0.6472 | 27.1758 |                  0.7850 |         0.5547 |

| candidate         | model_position   |   spearman |     mae |
|:------------------|:-----------------|-----------:|--------:|
| ext_et_all        | QB               |     0.7179 | 36.8854 |
| ext_et_all        | RB               |     0.7430 | 29.3416 |
| ext_et_all        | TE               |     0.7934 | 16.7370 |
| ext_et_all        | WR               |     0.7684 | 22.3683 |
| phase2_production | QB               |     0.7092 | 37.8179 |
| phase2_production | RB               |     0.7485 | 29.2932 |
| phase2_production | TE               |     0.7867 | 17.4027 |
| phase2_production | WR               |     0.7595 | 23.1192 |

Paired gain of the adopted procedure over Phase 2 production: 0.0047 (-0.0001 to 0.0093). Gain over prior-season points: 0.1085 (0.0890 to 0.1279). Null control mean Spearman over 5 target permutations: 0.0210.

## Injury track

Development (2017-2021):

| learner                    |   roc_auc |   average_precision |   log_loss |   brier_score |
|:---------------------------|----------:|--------------------:|-----------:|--------------:|
| age_position_logistic      |    0.6025 |              0.2882 |     0.5205 |        0.1730 |
| elastic_all_c0.05          |    0.7280 |              0.3902 |     0.4859 |        0.1597 |
| extra_trees_all_leaf20     |    0.7256 |              0.4091 |     0.4793 |        0.1579 |
| extra_trees_all_leaf50     |    0.7303 |              0.4117 |     0.4771 |        0.1571 |
| hist_gradient_all_leaves15 |    0.7174 |              0.4022 |     0.4878 |        0.1601 |
| prior_games_logistic       |    0.6803 |              0.3625 |     0.5063 |        0.1650 |

Locked 2022-2024 audit:

| learner                |   roc_auc |   average_precision |   log_loss |   brier_score |   calibration_intercept |   calibration_slope |   prevalence |
|:-----------------------|----------:|--------------------:|-----------:|--------------:|------------------------:|--------------------:|-------------:|
| extra_trees_all_leaf50 |    0.7486 |              0.4441 |     0.4835 |        0.1602 |                 -0.0069 |              1.0194 |       0.2467 |
| age_position_logistic  |    0.5858 |              0.3013 |     0.5421 |        0.1816 |                 -0.3037 |              0.7273 |       0.2467 |
| prior_games_logistic   |    0.6770 |              0.3800 |     0.5225 |        0.1718 |                 -0.0815 |              0.9704 |       0.2467 |

| model_position   |      rows |   positive_rate |   roc_auc |
|:-----------------|----------:|----------------:|----------:|
| K                |  174.0000 |          0.1724 |    0.5995 |
| QB               |  416.0000 |          0.1803 |    0.7794 |
| RB               |  800.0000 |          0.2625 |    0.7230 |
| TE               |  689.0000 |          0.2540 |    0.7115 |
| WR               | 1338.0000 |          0.2638 |    0.7754 |

| subgroup              |   rows |   positive_rate |   roc_auc |   prior_games_auc |
|:----------------------|-------:|----------------:|----------:|------------------:|
| all audit candidates  |   3417 |          0.2467 |    0.7486 |            0.6719 |
| prior-year games > 0  |   1982 |          0.3259 |    0.6746 |            0.6214 |
| prior-year games >= 4 |   1594 |          0.3645 |    0.6458 |            0.5720 |
| prior-year games >= 8 |   1231 |          0.3997 |    0.6248 |            0.5183 |

Paired AUC gain over the age-position baseline: 0.1628 (0.1369 to 0.1859). Success rule passed: True. Null control mean AUC over 5 label permutations: 0.5288.

## Combination

| stage       |   injury_weight |   spearman |   ndcg_at_roster_cutoff |   top_k_recall |   bust_rate_top_k |
|:------------|----------------:|-----------:|------------------------:|---------------:|------------------:|
| development |          0.0000 |     0.7196 |                  0.7413 |         0.5069 |            0.2130 |
| development |          0.0500 |     0.7191 |                  0.7412 |         0.5069 |            0.2130 |
| development |          0.1000 |     0.7182 |                  0.7308 |         0.4873 |            0.2222 |
| development |          0.1500 |     0.7168 |                  0.7271 |         0.4850 |            0.2188 |
| development |          0.2000 |     0.7151 |                  0.7245 |         0.4931 |            0.2176 |
| development |          0.2500 |     0.7122 |                  0.7187 |         0.4769 |            0.2303 |
| development |          0.3000 |     0.7080 |                  0.6941 |         0.4468 |            0.2535 |
| development |          0.4000 |     0.6916 |                  0.6587 |         0.4120 |            0.2824 |
| development |          0.5000 |     0.6413 |                  0.6037 |         0.3646 |            0.3449 |
| audit       |          0.0000 |     0.7538 |                  0.7910 |         0.5417 |            0.1667 |
| audit       |          0.0500 |     0.7535 |                  0.7904 |         0.5417 |            0.1667 |
| audit       |          0.1000 |     0.7527 |                  0.7912 |         0.5382 |            0.1678 |
| audit       |          0.1500 |     0.7517 |                  0.7832 |         0.5336 |            0.1678 |
| audit       |          0.2000 |     0.7499 |                  0.7669 |         0.5255 |            0.1690 |
| audit       |          0.2500 |     0.7474 |                  0.7610 |         0.5266 |            0.1725 |
| audit       |          0.3000 |     0.7441 |                  0.7517 |         0.5197 |            0.1759 |
| audit       |          0.4000 |     0.7287 |                  0.7035 |         0.4745 |            0.2384 |
| audit       |          0.5000 |     0.6724 |                  0.6175 |         0.3981 |            0.3241 |

## Feature ranking

Points track top 20 by consensus rank (mean of the permutation, univariate, filter-panel, and FeatureRanker ranks):

|   consensus_rank | feature                                | family               |   permutation_rank |   univariate_rank |   panel_rank |   featureranker_rank |
|-----------------:|:---------------------------------------|:---------------------|-------------------:|------------------:|-------------:|---------------------:|
|                1 | lag1_offense_snaps                     | lag1_participation   |                2.0 |               1.0 |          1.0 |                  1.0 |
|                2 | lag1_games                             | lag1_participation   |               10.0 |               2.0 |          2.0 |                  2.0 |
|                3 | lag1_receptions                        | lag1_production      |                5.0 |               3.0 |          5.0 |                  7.0 |
|                4 | lag1_receiving_yards_after_catch       | lag1_production      |               18.3 |               4.0 |          3.0 |                  3.0 |
|                5 | lag1_targets                           | lag1_production      |               15.3 |               5.0 |          5.5 |                  6.0 |
|                6 | lag1_receiving_yards                   | lag1_production      |               17.3 |               8.0 |          5.5 |                  5.0 |
|                7 | lag1_receiving_10                      | lag1_production      |               27.3 |               7.0 |         13.0 |                 11.0 |
|                8 | lag1_receiving_first_downs             | lag1_production      |               27.3 |               6.0 |         11.0 |                 14.0 |
|                9 | lag1_receiving_tds                     | lag1_production      |               13.0 |              15.0 |         28.0 |                 19.0 |
|               10 | lag1_rushing_10                        | lag1_production      |               10.3 |              27.0 |         36.0 |                  9.0 |
|               11 | lag2_offense_snaps                     | lag2_4_history       |               16.3 |              28.0 |         22.5 |                 17.0 |
|               12 | lag1_late6_receiving_yards_after_catch | lag1_trajectory      |               32.7 |              12.0 |         18.0 |                 25.0 |
|               13 | missing_draft_number                   | metadata_missingness |                5.7 |              30.0 |         45.0 |                 15.0 |
|               14 | lag2_receiving_yards_after_catch       | lag2_4_history       |               11.0 |              42.0 |         43.0 |                 21.0 |
|               15 | draft_number                           | age_experience_draft |                6.7 |              87.0 |         14.0 |                 12.0 |
|               16 | lag1_late6_receptions                  | lag1_trajectory      |               70.0 |              11.0 |         25.5 |                 22.0 |
|               17 | lag1_receiving_air_yards               | lag1_production      |               40.0 |              34.0 |         27.0 |                 31.0 |
|               18 | lag1_receiving_16                      | lag1_production      |               80.0 |               9.0 |         23.5 |                 26.0 |
|               19 | lag1_receiving_yards_per_target        | lag1_rates           |               17.3 |              17.0 |         38.0 |                 72.0 |
|               20 | lag2_receptions                        | lag2_4_history       |               20.0 |              46.0 |         42.0 |                 39.0 |

Injury track top 20 by consensus rank:

|   consensus_rank | feature                                | family               |   permutation_rank |   univariate_rank |   panel_rank |   featureranker_rank |
|-----------------:|:---------------------------------------|:---------------------|-------------------:|------------------:|-------------:|---------------------:|
|                1 | lag1_receiving_yards_after_catch       | lag1_production      |               12.0 |                 1 |          3.5 |                  1.0 |
|                2 | lag1_games                             | lag1_participation   |                3.0 |                 9 |          8.0 |                  2.0 |
|                3 | lag1_catch_rate                        | lag1_rates           |                6.0 |                12 |          4.0 |                  3.0 |
|                4 | lag1_offense_snaps                     | lag1_participation   |               18.0 |                 7 |         13.0 |                  9.0 |
|                5 | lag1_receiving_yards                   | lag1_production      |               34.0 |                 4 |          7.0 |                  7.0 |
|                6 | lag1_receptions                        | lag1_production      |               13.0 |                 3 |         15.0 |                 25.0 |
|                7 | lag1_receiving_yards_per_target        | lag1_rates           |               21.0 |                10 |         30.0 |                 10.0 |
|                8 | lag1_receiving_first_down_rate         | lag1_rates           |               16.0 |                18 |         22.0 |                 19.0 |
|                9 | missing_draft_round                    | metadata_missingness |               10.0 |                25 |         35.5 |                 14.0 |
|               10 | was_drafted                            | age_experience_draft |                2.0 |                26 |         44.5 |                 15.0 |
|               11 | lag1_receiving_16                      | lag1_production      |               40.0 |                11 |         11.0 |                 27.0 |
|               12 | missing_draft_number                   | metadata_missingness |                7.0 |                24 |         39.0 |                 21.0 |
|               13 | lag1_receiving_20_rate                 | lag1_rates           |               32.0 |                21 |         29.0 |                 23.0 |
|               14 | lag2_offense_snaps                     | lag2_4_history       |                9.0 |                33 |         53.5 |                 28.0 |
|               15 | draft_round                            | age_experience_draft |               15.0 |                55 |         25.0 |                 47.0 |
|               16 | lag1_late6_receiving_yards_after_catch | lag1_trajectory      |               62.0 |                16 |         40.0 |                 31.0 |
|               17 | lag1_late6_receiving_yards             | lag1_trajectory      |               22.0 |                17 |         46.0 |                 68.0 |
|               18 | missing_age                            | metadata_missingness |                5.0 |               115 |         35.5 |                  8.0 |
|               19 | lag1_receiving_20                      | lag1_production      |               26.0 |                19 |         88.5 |                 61.0 |
|               20 | missing_combine_bench                  | metadata_missingness |               14.0 |                49 |         93.5 |                 38.0 |

FeatureRanker 3.0.4 (amendment 3) ran on the same discovery rows with random forest, XGBoost, mutual information, ANOVA F-test, and L1 rankers and probe-weighted reciprocal-rank voting. Points: 5605 rows, 375 nonconstant features, probe skill by method {'rf': 0.491, 'xg': 0.407, 'mi': 0.455, 'f_test': 0.457, 'l1': 0.513}, top-20 overlap between automatic and equal weights 19 of 20. Injury: 9077 rows, 505 features, probe skill {'rf': 0.025, 'xg': 0.021, 'mi': 0.021, 'f_test': 0.017, 'l1': 0.021}, overlap 19 of 20. The injury probes are close to chance because FeatureRanker's internal probes use random folds and small feature subsets; the chronological permutation and ablation evidence carries more weight for that track. Per-method ranks and probe tables are in `artifacts/*_featureranker_ranking.csv` and `artifacts/*_featureranker_summary.json`.

Family ablation (points):

| family                 |   columns |   spearman_without |   spearman_full |   delta |
|:-----------------------|----------:|-------------------:|----------------:|--------:|
| age_experience_draft   |         6 |             0.7100 |          0.7196 |  0.0095 |
| lag2_4_history         |       288 |             0.7118 |          0.7196 |  0.0077 |
| lag1_participation     |         3 |             0.7166 |          0.7196 |  0.0030 |
| reserve_history_roster |         4 |             0.7188 |          0.7196 |  0.0008 |
| lag1_trajectory        |        66 |             0.7204 |          0.7196 | -0.0009 |
| body_combine           |         8 |             0.7206 |          0.7196 | -0.0010 |
| lag1_production        |        73 |             0.7208 |          0.7196 | -0.0012 |
| injury_report_history  |        20 |             0.7210 |          0.7196 | -0.0014 |
| metadata_missingness   |        15 |             0.7212 |          0.7196 | -0.0016 |
| history_availability   |         4 |             0.7216 |          0.7196 | -0.0020 |
| position               |         5 |             0.7218 |          0.7196 | -0.0022 |
| lag1_rates             |        20 |             0.7218 |          0.7196 | -0.0022 |

Family ablation (injury):

| family                 |   columns |   roc_auc_without |   roc_auc_full |   delta |
|:-----------------------|----------:|------------------:|---------------:|--------:|
| lag1_participation     |         3 |            0.7281 |         0.7303 |  0.0022 |
| age_experience_draft   |         6 |            0.7283 |         0.7303 |  0.0020 |
| position               |         5 |            0.7293 |         0.7303 |  0.0009 |
| lag2_4_history         |       288 |            0.7295 |         0.7303 |  0.0008 |
| injury_report_history  |        20 |            0.7301 |         0.7303 |  0.0001 |
| history_availability   |         4 |            0.7306 |         0.7303 | -0.0003 |
| lag1_trajectory        |        66 |            0.7306 |         0.7303 | -0.0003 |
| body_combine           |         8 |            0.7308 |         0.7303 | -0.0005 |
| reserve_history_roster |         4 |            0.7308 |         0.7303 | -0.0005 |
| lag1_production        |        73 |            0.7309 |         0.7303 | -0.0006 |
| metadata_missingness   |        15 |            0.7320 |         0.7303 | -0.0018 |
| lag1_rates             |        20 |            0.7322 |         0.7303 | -0.0019 |

## Plan amendments and negative-control diagnostics

Amendment 1. The first full run (retained under `runs/20260822T190114656387Z-final-ranking-v1`) trained the extended points learners from 2013 and failed the predeclared null gate: mean within-position-season Spearman 0.0512 over five target permutations against a 0.05 gate. The cause is an era shift in the cohort: mean target points per season fall from about 57 (2013-2016) to about 40 (2017 onward) when the weekly roster files widen their coverage. A model fitted to permuted targets can still learn era-proxy features and recover some within-season structure. `artifacts/points_null_era_diagnostic.json` shows that for validation seasons 2019-2021 the null mean is 0.108 (sd 0.036, six seeds) with training from 2013 and 0.027 (sd 0.034) with training from 2017, while real `ext_et_all` skill is 0.714 against 0.720. The extended learners therefore train from 2017 in discovery, audit, and production; discovery folds become 2019-2021. Under the amended window the null mean is 0.0210 (seeds -0.083 to 0.094), inside the gate. Phase 2 production, which trains from 2006 on the stable feature tier, was not re-examined under this stricter control; its own null scored predictions against the permuted label and tests only the implementation path.

Amendment 2. The installed FeatureRanker package was version 1.2.2, whose internal hyperparameter search did not finish within ten minutes on a 600-row test. An in-house filter panel (mutual information and random-forest impurity importance on the same discovery rows) was added as an evidence line.

Amendment 3. FeatureRanker was upgraded to 3.0.4 and rerun on the completed run without refitting the models. The consensus rank is now the mean of four ranks: permutation, univariate, filter panel, and FeatureRanker.

Injury null control. Labels permuted within season and position give a mean pooled audit AUC of 0.529 and a mean within-season AUC of 0.543, both above 0.5. `artifacts/injury_null_position_diagnostic.json` shows that the mean within-season-and-position AUC of the same null models is 0.478 (four seeds), so the pooled shift comes from position prevalence preserved by the permutation (kickers 17%, running backs 26%), not from a label path. The real model's within-season-and-position AUC on the development seasons is 0.683 against a pooled 0.727; the within-position figure is the fairer summary of how well the model separates peers.

## Feature-ranking interpretation

Participation and receiving usage dominate both tracks. For points, `lag1_offense_snaps` has by far the largest held-out permutation importance, and removing the age, experience, and draft family costs the most Spearman (0.0095), followed by the two-to-four-season history (0.0077). For the injury label, removing prior-year participation costs the most AUC (0.0022), and the strongest injury-history input, `injury_lag2_injury_reported`, ranks 44th by consensus with a univariate AUC of 0.567. The roster reserve history added in this phase ranks 208th or lower for points and is not in the injury top 40. Injury history is therefore a secondary signal in this candidate pool: most of what the model learns is that players who played and were targeted last season are the players who can appear as Out or on reserve this season. Within established players (at least eight prior-year games) the audit AUC is 0.625 against 0.518 for prior-year games alone, which is the clearest evidence that the model carries information beyond exposure.

## Limits

- Audits are chronological and procedure-locked but not analyst-blinded; earlier phases inspected the same seasons.
- Historical August roster membership is reconstructed from the prior season, not observed.
- The injury label records observed absence events; players who play through injuries or leave the league are negatives.
- The official injury feed ends in 2024, so report-based history is at least two seasons old; roster reserve history fills the one-season gap.
- The draft scores are preference indexes. Expected value also depends on lineup rules, the replacement-level assumption, and league scoring.
- The injury weight does not lower the backtested bust rate; it trades a small amount of rank accuracy for a transparent preference toward players with lower predicted absence risk relative to peers.
- Points P10 and P90 are empirical residual percentiles from 2022-2025, not calibrated predictive intervals for 2026.
