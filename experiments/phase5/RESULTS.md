# Phase 5 methods and results (negative result)

## Answer

The final 2026 ranking uses the `phase4_production` points forecast and the `extra_trees_all_leaf50` missed-time injury classifier. On the locked 2022-2025 audit the adopted points procedure reached mean within-position-season Spearman 0.7557 (player-cluster 95% interval 0.7321 to 0.7751) against 0.7557 for the `phase4_production` reference and 0.6472 for prior-season points. The injury classifier reached pooled 2022-2024 ROC AUC 0.7492 (95% interval 0.7313 to 0.7678) at 24.7% prevalence with calibration slope 1.021.

The default injury weight in the draft score is 0.20. No workbook was kept: the adoption rule retained the Phase 4 reference on both tracks, so `outputs/fantasy_football_final_rankings_2026.xlsx` remains the deliverable.

## Injury label

A player-season is positive when the player is listed Out or Doubtful with a physical injury on any regular-season official report, or spends any regular-season roster week on an injury-compatible reserve list. Questionable-only and practice-only designations are negative. COVID, retired, did-not-report, left-squad, future, and suspension reserve codes are excluded. The label covers 2013-2024. On the audit seasons the selected learner reached ROC AUC 0.8679 against the Phase 3 any-designation label, for comparison.

## Points track

Discovery (2019-2021, training from the amended window) selected `{points['selected']}`; the audit adoption rule adopted `phase4_production`.

| candidate             |   components |   spearman |     mae |   ndcg_at_roster_cutoff |   top_k_recall |
|:----------------------|-------------:|-----------:|--------:|------------------------:|---------------:|
| phase4_production     |            1 |     0.7196 | 25.9925 |                  0.7413 |         0.5069 |
| ext_et_all            |            1 |     0.7197 | 25.8428 |                  0.7422 |         0.4919 |
| ext_hgb_all           |            1 |     0.6993 | 26.0729 |                  0.7475 |         0.5347 |
| ext_xgb_all           |            1 |     0.6778 | 27.1871 |                  0.7466 |         0.5220 |
| blend_ext_et_hgb      |            2 |     0.7207 | 25.6134 |                  0.7485 |         0.5278 |
| blend_ext_all3        |            3 |     0.7150 | 25.9318 |                  0.7569 |         0.5382 |
| blend_phase4_ext_et   |            2 |     0.7199 | 25.8946 |                  0.7411 |         0.5023 |
| blend_phase4_ext_all3 |            4 |     0.7185 | 25.8475 |                  0.7522 |         0.5347 |

Locked 2022-2025 audit:

| candidate                |   spearman |     mae |   ndcg_at_roster_cutoff |   top_k_recall |
|:-------------------------|-----------:|--------:|------------------------:|---------------:|
| phase4_production        |     0.7557 | 24.6948 |                  0.7884 |         0.5521 |
| ext_et_all               |     0.7548 | 24.6813 |                  0.7901 |         0.5729 |
| ext_hgb_all              |     0.7427 | 24.8941 |                  0.7714 |         0.5495 |
| ext_xgb_all              |     0.7281 | 26.0560 |                  0.8031 |         0.5686 |
| blend_ext_et_hgb         |     0.7574 | 24.5144 |                  0.7836 |         0.5460 |
| blend_ext_all3           |     0.7526 | 24.8505 |                  0.7935 |         0.5651 |
| blend_phase4_ext_et      |     0.7555 | 24.6721 |                  0.7880 |         0.5599 |
| blend_phase4_ext_all3    |     0.7566 | 24.7407 |                  0.7890 |         0.5660 |
| previous_points_baseline |     0.6472 | 27.1758 |                  0.7850 |         0.5547 |

| candidate         | model_position   |   spearman |     mae |
|:------------------|:-----------------|-----------:|--------:|
| phase4_production | QB               |     0.7179 | 36.8854 |
| phase4_production | RB               |     0.7430 | 29.3416 |
| phase4_production | TE               |     0.7934 | 16.7370 |
| phase4_production | WR               |     0.7684 | 22.3683 |

Paired gain of the adopted procedure over the `phase4_production` reference: 0.0000 (0.0000 to 0.0000). Gain over prior-season points: 0.1085 (0.0890 to 0.1279). Null control mean Spearman over 5 target permutations: 0.0242.

## Injury track

Development (2017-2021):

| learner                    |   roc_auc |   average_precision |   log_loss |   brier_score |
|:---------------------------|----------:|--------------------:|-----------:|--------------:|
| age_position_logistic      |    0.6025 |              0.2882 |     0.5205 |        0.1730 |
| elastic_all_c0.05          |    0.7280 |              0.3911 |     0.4858 |        0.1597 |
| extra_trees_all_leaf20     |    0.7249 |              0.4090 |     0.4797 |        0.1581 |
| extra_trees_all_leaf50     |    0.7303 |              0.4123 |     0.4770 |        0.1571 |
| hist_gradient_all_leaves15 |    0.7179 |              0.3991 |     0.4882 |        0.1602 |
| prior_games_logistic       |    0.6803 |              0.3625 |     0.5063 |        0.1650 |

Locked 2022-2024 audit:

| learner                |   roc_auc |   average_precision |   log_loss |   brier_score |   calibration_intercept |   calibration_slope |   prevalence |
|:-----------------------|----------:|--------------------:|-----------:|--------------:|------------------------:|--------------------:|-------------:|
| extra_trees_all_leaf50 |    0.7492 |              0.4446 |     0.4833 |        0.1601 |                 -0.0078 |              1.0208 |       0.2467 |
| age_position_logistic  |    0.5858 |              0.3013 |     0.5421 |        0.1816 |                 -0.3037 |              0.7273 |       0.2467 |
| prior_games_logistic   |    0.6770 |              0.3800 |     0.5225 |        0.1718 |                 -0.0815 |              0.9704 |       0.2467 |

| model_position   |      rows |   positive_rate |   roc_auc |
|:-----------------|----------:|----------------:|----------:|
| K                |  174.0000 |          0.1724 |    0.6042 |
| QB               |  416.0000 |          0.1803 |    0.7767 |
| RB               |  800.0000 |          0.2625 |    0.7250 |
| TE               |  689.0000 |          0.2540 |    0.7115 |
| WR               | 1338.0000 |          0.2638 |    0.7772 |

| subgroup              |   rows |   positive_rate |   roc_auc |   prior_games_auc |
|:----------------------|-------:|----------------:|----------:|------------------:|
| all audit candidates  |   3417 |          0.2467 |    0.7492 |            0.6719 |
| prior-year games > 0  |   1982 |          0.3259 |    0.6756 |            0.6214 |
| prior-year games >= 4 |   1594 |          0.3645 |    0.6471 |            0.5720 |
| prior-year games >= 8 |   1231 |          0.3997 |    0.6271 |            0.5183 |

Paired AUC gain over the age-position baseline: 0.1634 (0.1362 to 0.1860). Success rule passed: True. Null control mean AUC over 5 label permutations: 0.5286.

## Combination

| stage       |   injury_weight |   spearman |   ndcg_at_roster_cutoff |   top_k_recall |   bust_rate_top_k |
|:------------|----------------:|-----------:|------------------------:|---------------:|------------------:|
| development |          0.0000 |     0.7196 |                  0.7413 |         0.5069 |            0.2130 |
| development |          0.0500 |     0.7191 |                  0.7402 |         0.5046 |            0.2153 |
| development |          0.1000 |     0.7183 |                  0.7316 |         0.4873 |            0.2257 |
| development |          0.1500 |     0.7168 |                  0.7263 |         0.4850 |            0.2188 |
| development |          0.2000 |     0.7152 |                  0.7231 |         0.4826 |            0.2245 |
| development |          0.2500 |     0.7125 |                  0.7109 |         0.4745 |            0.2303 |
| development |          0.3000 |     0.7084 |                  0.6931 |         0.4549 |            0.2558 |
| development |          0.4000 |     0.6926 |                  0.6492 |         0.4039 |            0.2963 |
| development |          0.5000 |     0.6432 |                  0.5890 |         0.3426 |            0.3600 |
| audit       |          0.0000 |     0.7538 |                  0.7910 |         0.5417 |            0.1667 |
| audit       |          0.0500 |     0.7536 |                  0.7907 |         0.5417 |            0.1667 |
| audit       |          0.1000 |     0.7528 |                  0.7895 |         0.5382 |            0.1678 |
| audit       |          0.1500 |     0.7519 |                  0.7819 |         0.5336 |            0.1655 |
| audit       |          0.2000 |     0.7502 |                  0.7651 |         0.5197 |            0.1725 |
| audit       |          0.2500 |     0.7476 |                  0.7562 |         0.5174 |            0.1759 |
| audit       |          0.3000 |     0.7439 |                  0.7483 |         0.5174 |            0.1759 |
| audit       |          0.4000 |     0.7281 |                  0.7084 |         0.4653 |            0.2303 |
| audit       |          0.5000 |     0.6728 |                  0.6097 |         0.3808 |            0.3194 |

## Feature ranking

Points track top 20 by consensus rank:

|   consensus_rank | feature                                | family               |   permutation_importance |   univariate_spearman |
|-----------------:|:---------------------------------------|:---------------------|-------------------------:|----------------------:|
|                1 | lag1_offense_snaps                     | lag1_participation   |                   0.0783 |                0.6043 |
|                2 | lag1_games                             | lag1_participation   |                   0.0115 |                0.5944 |
|                3 | lag1_receiving_yards_after_catch       | lag1_production      |                   0.0053 |                0.5332 |
|                4 | lag1_receptions                        | lag1_production      |                   0.0147 |                0.5340 |
|                5 | lag1_share_receptions_prior_team       | lag1_usage_share     |                   0.0059 |                0.5327 |
|                6 | lag1_receiving_yards                   | lag1_production      |                   0.0040 |                0.5224 |
|                7 | lag1_targets                           | lag1_production      |                   0.0041 |                0.5322 |
|                8 | lag1_offense_snaps_per_game            | lag1_per_game        |                   0.0013 |                0.5647 |
|                9 | lag1_receiving_yards_per_game          | lag1_per_game        |                   0.0056 |                0.5059 |
|               10 | lag1_share_targets_prior_team          | lag1_usage_share     |                   0.0026 |                0.5318 |
|               11 | lag1_rushing_10                        | lag1_production      |                   0.0108 |                0.3966 |
|               12 | lag1_receiving_tds                     | lag1_production      |                   0.0068 |                0.4921 |
|               13 | missing_draft_number                   | metadata_missingness |                   0.0332 |               -0.3941 |
|               14 | lag2_offense_snaps                     | lag2_4_history       |                   0.0058 |                0.3964 |
|               15 | lag1_receiving_10                      | lag1_production      |                   0.0010 |                0.5230 |
|               16 | lag2_receiving_yards_after_catch       | lag2_4_history       |                   0.0097 |                0.3539 |
|               17 | lag1_share_carries_prior_team          | lag1_usage_share     |                   0.0027 |                0.4176 |
|               18 | draft_number                           | age_experience_draft |                   0.0470 |                0.2427 |
|               19 | lag1_receptions_per_game               | lag1_per_game        |                   0.0008 |                0.5172 |
|               20 | lag1_late6_receiving_yards_after_catch | lag1_trajectory      |                   0.0015 |                0.5031 |

Injury track top 20 by consensus rank:

|   consensus_rank | feature                                        | family               |   permutation_importance |   discriminative_auc |
|-----------------:|:-----------------------------------------------|:---------------------|-------------------------:|---------------------:|
|                1 | lag1_receiving_yards_after_catch               | lag1_production      |                   0.0008 |               0.6508 |
|                2 | lag1_games                                     | lag1_participation   |                   0.0036 |               0.6385 |
|                3 | lag1_catch_rate                                | lag1_rates           |                   0.0008 |               0.6328 |
|                4 | lag1_share_receptions_prior_team               | lag1_usage_share     |                   0.0002 |               0.6486 |
|                5 | lag1_offense_snaps                             | lag1_participation   |                   0.0007 |               0.6410 |
|                6 | lag1_receiving_yards                           | lag1_production      |                   0.0002 |               0.6462 |
|                7 | lag1_share_targets_prior_team                  | lag1_usage_share     |                   0.0003 |               0.6491 |
|                8 | lag1_receptions                                | lag1_production      |                   0.0005 |               0.6491 |
|                9 | missing_draft_number                           | metadata_missingness |                   0.0016 |               0.5963 |
|               10 | lag1_receiving_yards_after_catch_per_reception | lag1_rates           |                   0.0001 |               0.6393 |
|               11 | lag1_receiving_yards_per_target                | lag1_rates           |                   0.0002 |               0.6367 |
|               12 | lag1_targets_per_game                          | lag1_per_game        |                   0.0001 |               0.6430 |
|               13 | was_drafted                                    | age_experience_draft |                   0.0069 |               0.5963 |
|               14 | missing_draft_round                            | metadata_missingness |                   0.0020 |               0.5963 |
|               15 | lag1_receiving_first_down_rate                 | lag1_rates           |                   0.0003 |               0.6251 |
|               16 | lag2_offense_snaps                             | lag2_4_history       |                   0.0014 |               0.5900 |
|               17 | lag1_receiving_16                              | lag1_production      |                   0.0001 |               0.6333 |
|               18 | lag1_offense_snaps_per_game                    | lag1_per_game        |                   0.0000 |               0.6292 |
|               19 | draft_round                                    | age_experience_draft |                   0.0009 |               0.5726 |
|               20 | lag1_receiving_20                              | lag1_production      |                   0.0005 |               0.6223 |

Family ablation (points):

| family                 |   columns |   spearman_without |   spearman_full |   delta |
|:-----------------------|----------:|-------------------:|----------------:|--------:|
| age_experience_draft   |         6 |             0.7095 |          0.7197 |  0.0102 |
| lag2_4_history         |       288 |             0.7136 |          0.7197 |  0.0061 |
| lag1_participation     |         3 |             0.7189 |          0.7197 |  0.0008 |
| reserve_history_roster |         4 |             0.7192 |          0.7197 |  0.0004 |
| position               |         5 |             0.7197 |          0.7197 | -0.0001 |
| history_availability   |         4 |             0.7204 |          0.7197 | -0.0008 |
| lag1_trajectory        |        66 |             0.7205 |          0.7197 | -0.0008 |
| lag1_usage_share       |         4 |             0.7206 |          0.7197 | -0.0009 |
| metadata_missingness   |        15 |             0.7207 |          0.7197 | -0.0010 |
| body_combine           |         8 |             0.7211 |          0.7197 | -0.0014 |
| lag1_per_game          |         8 |             0.7212 |          0.7197 | -0.0015 |
| lag1_production        |        73 |             0.7214 |          0.7197 | -0.0018 |
| lag1_rates             |        20 |             0.7217 |          0.7197 | -0.0020 |
| injury_report_history  |        20 |             0.7223 |          0.7197 | -0.0027 |

Family ablation (injury):

| family                 |   columns |   roc_auc_without |   roc_auc_full |   delta |
|:-----------------------|----------:|------------------:|---------------:|--------:|
| lag1_participation     |         3 |            0.7284 |         0.7303 |  0.0018 |
| age_experience_draft   |         6 |            0.7285 |         0.7303 |  0.0018 |
| lag2_4_history         |       288 |            0.7291 |         0.7303 |  0.0012 |
| lag1_trajectory        |        66 |            0.7293 |         0.7303 |  0.0010 |
| history_availability   |         4 |            0.7295 |         0.7303 |  0.0008 |
| position               |         5 |            0.7298 |         0.7303 |  0.0005 |
| lag1_per_game          |         8 |            0.7301 |         0.7303 |  0.0002 |
| lag1_usage_share       |         4 |            0.7303 |         0.7303 |  0.0000 |
| body_combine           |         8 |            0.7304 |         0.7303 | -0.0001 |
| reserve_history_roster |         4 |            0.7307 |         0.7303 | -0.0004 |
| lag1_production        |        73 |            0.7309 |         0.7303 | -0.0006 |
| injury_report_history  |        20 |            0.7311 |         0.7303 | -0.0008 |
| lag1_rates             |        20 |            0.7315 |         0.7303 | -0.0012 |
| metadata_missingness   |        15 |            0.7317 |         0.7303 | -0.0014 |

## Limits

- Audits are chronological and procedure-locked but not analyst-blinded; earlier phases inspected the same seasons.
- Historical August roster membership is reconstructed from the prior season, not observed.
- The injury label records observed absence events; players who play through injuries or leave the league are negatives.
- The official injury feed ends in 2024, so report-based history is at least two seasons old; roster reserve history fills the one-season gap.
- The draft score is a preference index. Expected value also depends on lineup rules, replacement level, and league scoring.
