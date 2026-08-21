# FeatureRanker analysis: fantasy points and injury reports

This analysis ranks the leakage-controlled preseason features separately for:

- next-season ESPN full-PPR points for QB, RB, WR, and TE
- any physical regular-season injury or practice-report designation for QB, RB, WR, TE, and K

The rankings use discovery seasons 2013 through 2021. Audit seasons are not used for feature ranking.

## Results

Prior-year games is the strongest exact feature for both targets. The points line then emphasizes prior scoring production. The injury line emphasizes participation and receiving workload, followed by draft and age context.

Only four exact features overlap between the two top tens:

- `lag1_games`
- `lag1_receptions`
- `lag1_receiving_yards`
- `lag1_receiving_10`

Eight exact features overlap in the top 20. Across all 226 features shared by the two matrices, the consensus ranks have Spearman correlation 0.570. The rankings are therefore moderately similar at the feature-family level, but not interchangeable.

The similarity is expected for three reasons. First, both outcomes depend on whether a preseason candidate remains active and receives meaningful playing time. Second, usage creates injury-report exposure: players with more games, snaps, receptions, and targets have more opportunities to receive a physical designation. Third, the injury label is report presence rather than injury severity or games missed, so it partly measures league participation and observation opportunity.

This does not mean points production causes injury. It means availability, role, and workload are common upstream predictors of both recorded production and recorded injury-report presence.

## Top features

Points top five:

1. `lag1_games`
2. `lag1_receptions`
3. `lag1_passing_tds`
4. `lag1_receiving_yards`
5. `lag1_receiving_first_downs`

Injury-report top five:

1. `lag1_games`
2. `lag1_receiving_yards_after_catch`
3. `lag1_receiving_yards`
4. `lag1_receptions`
5. `lag1_offense_snaps`

The strongest prior injury-history feature is `injury_lag2_injury_report_weeks` at consensus rank 44. Injury history is useful, but secondary to participation and workload in this candidate pool.

## Files

- [`feature_ranker_report.html`](feature_ranker_report.html): self-contained technical report
- [`featureranker_points_injury.ipynb`](featureranker_points_injury.ipynb): executed analysis notebook
- [`artifacts/points_feature_ranking.csv`](artifacts/points_feature_ranking.csv): complete points ranking
- [`artifacts/injury_feature_ranking.csv`](artifacts/injury_feature_ranking.csv): complete injury ranking
- [`artifacts/points_top_features.png`](artifacts/points_top_features.png): points top-feature plot at 300 dpi
- [`artifacts/injury_top_features.png`](artifacts/injury_top_features.png): injury top-feature plot at 300 dpi
- [`artifacts/analysis_summary.json`](artifacts/analysis_summary.json): machine-readable summary and sensitivity checks
- [`artifacts/data_quality.json`](artifacts/data_quality.json): matrix and target validation results
- `artifacts/*_ranking_result.joblib`: serialized FeatureRanker results

## Method

FeatureRanker 3.0.4 combines Random Forest, XGBoost, mutual information, ANOVA F-test, and L1 rankings. The primary result uses cross-validated probe-skill weights with reciprocal-rank voting. Automatic and equal weights produce identical top-20 sets for both targets. Alternate voting geometry changes more of the injury ordering, so exact engineered columns should be interpreted as correlated feature families.

The points analysis contains 8,613 player-seasons and 226 nonconstant features. The injury analysis contains 9,077 player-seasons and 493 nonconstant features, with 36.8% positive labels.

The notebook reconstructs the matrices from the project feature pipeline. The saved rankings, plots, diagnostics, and serialized FeatureRanker results are self-contained in this directory.

## Interpretation limits

- Consensus scores are voting scores, not effect sizes or causal estimates.
- Both cohorts retain nonparticipants, so league persistence is part of the signal.
- The injury outcome measures report presence, not medical severity or biological fragility.
- FeatureRanker's internal probes use random folds and are diagnostics, not replacements for the project's chronological player-aware audits.
- Do not prune either production model to a small top-N list without a new chronological evaluation.
