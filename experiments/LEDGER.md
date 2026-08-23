# Experiment ledger

One record of every phase, what it changed, what it measured, and what was discarded. Phases 1 to 3 are summarized from records that were removed during the August 22, 2026 consolidation; their full files remain in git history before commit `62b1e64`'s successors. Phase 4 is the final deliverable. Phase 5 is a negative result.

## Shared contract

- Source: public nflverse Parquet releases frozen 2026-08-10 03:11 UTC (weekly player stats 1999-2025, weekly rosters 2002-2026, players, combine, draft picks, snap counts, depth charts, official injury reports 2009-2024). `data/raw/manifest.json` records URLs, sizes, and SHA-256 digests.
- Target: full regular-season ESPN full-PPR points, including Week 18 since 2021, from `config/scoring_espn_full_ppr_2026.json`. Reconstructed points match the nflverse comparison column on 99.33% of player-games (mean absolute difference 0.015); that column is never an input.
- Forecast origin: 2026-08-09 roster snapshot (2,930 records, 958 fantasy-position players, 951 available for a draft).
- Historical candidates for target season `t`: players on a `t - 1` regular-season roster plus target-year draft or combine entrants. Nonparticipants keep zero points. No historical target-year roster file decides membership.
- Primary points metric: mean within-position-season Spearman. Primary injury metric: pooled ROC AUC with a player-cluster bootstrap interval.
- Every model selection uses earlier development folds only; audits run once on later seasons. Plans are hashed before fitting. Audits are chronological but not analyst-blinded.

## Phase 1 (removed): data build and compact baseline

Built the download, scoring, and processed tables that every later phase uses (`player_games` 475,626 rows, `player_seasons` 66,907, 11,372 players). Selected a five-input Extra Trees offense model (draft number, receiving yards, rushing yards, passing touchdowns, age) with 2024-2025 retrospective Spearman 0.757. Its cohort used a target-year Week 1 roster proxy, which conditions on later participation and does not match the August 9 origin, so the result is not comparable with later phases and the models were retired.

## Phase 2 (production reference, record kept): August-origin points forecast

Two preliminary studies used post-cut Week 1 roster proxies without verified timestamps (248-column Extra Trees, stress-period Spearman 0.766, and a grouped-label null that did not center at zero, mean 0.118 over 11 permutations). They were diagnostics and were removed.

The production study (`experiments/phase2/`, plan `august9_plan_v1.json`) fixed the August 9 origin and the candidate rule above. Discovery on 2016-2021 compared Extra Trees, histogram gradient boosting, XGBoost, and ridge over feature counts 16 to all; the selection rule kept the best mean Spearman within a 0.005 tolerance with fewer columns preferred. Winner: `off_stable_core_et_all`, 228 lagged raw columns, Extra Trees with 180 trees, depth 14, leaf 5. Locked 2022-2025 audit: Spearman 0.751 (0.726 to 0.770) against 0.647 for prior-season points; QB 0.709, RB 0.748, WR 0.759, TE 0.787. Kicker model `k_stable_core_et_32`: 0.625 against 0.600, interval for the gain included zero, labeled low confidence. Null controls scored predictions against permuted labels: 499 fixed-prediction permutations mean 0.0004, 11 end-to-end permutations mean 0.0009.

Kept because Phase 4 uses its audit and 2026 predictions as the comparison reference and as the kicker forecast: the three run directories, `artifacts/` at the repository root, and the plan and manifests. The runner, promotion, and size-guard code were removed; the feature, cohort, and modeling modules remain.

## Phase 3 (superseded, data kept): any-designation injury model

Label: any physical designation on an official regular-season injury or practice report, including Questionable. Calibrated Extra Trees on 500 inputs; 2022-2024 audit AUC 0.868 (0.854 to 0.880), average precision 0.763 at 34.9% prevalence. Prior-year games alone reached 0.772, and the label's prevalence fell from about 45% (2013-2016) to 33% when Probable left the reports, so the model mostly measured exposure under a drifting label. A dated public injury screen (2026-08-14) matched 109 current listings.

Kept: the injury-report files (`experiments/phase3/data/raw`), the screen source and output (`data/current_injuries`, `artifacts/*.csv`, `current_injury_manifest_v1.json`), and the label and feature module `phase3_injury.py`. The model, run, workbook, and result files were removed.

## Phase 4 (final): missed-time label, re-audited points, feature rankings

Plan: `experiments/phase4/plan_v1.json` (with three recorded amendments). Run: `runs/20260822T193048718307Z-final-ranking-v2`. Results: `experiments/phase4/RESULTS.md`.

Inputs: the 500 Phase 3 columns plus roster reserve weeks at offsets one to four and Out or Doubtful report weeks at offsets two to five (512 columns). Label: Out or Doubtful with a physical injury, or an injury-reserve roster week, in the regular season; defined 2013-2024; prevalence 22% to 31% from 2016.

Points track: candidates were the Phase 2 reference, Extra Trees, histogram gradient boosting, and XGBoost on the extended inputs, and equal-weight blends. Discovery 2019-2021 with training from 2017 selected `ext_et_all` (0.7196 against 0.7173 for the reference). Audit 2022-2025: 0.7557 (0.7321 to 0.7751) against 0.7510; paired gain 0.0047 (-0.0001 to 0.0093); MAE 24.69 against 25.25. Adopted by the predeclared rule.

Injury track: grid of age-position logistic, prior-games logistic, elastic net, two Extra Trees, and histogram gradient boosting; base model through `y - 2`, sigmoid calibration on `y - 1`. Development 2017-2021 selected `extra_trees_all_leaf50` (AUC 0.730). Audit 2022-2024: 0.7486 (0.7306 to 0.7672), average precision 0.444, calibration slope 1.019, against 0.586 for age and position and 0.677 for prior-year games; within-season-and-position AUC 0.683. Among players with at least eight prior-year games: 0.625 against 0.518 for prior games alone.

Combination: injury weight 0.20 by the predeclared rule (largest weight within 0.005 Spearman of the best on development folds); it costs 0.004 Spearman on the audit and does not lower the top-k bust rate. The ALL page ranks by points over replacement (QB12, RB24, WR36, TE12, K12) scaled to the board maximum, blended with health relative to the position mean; position pages use the within-position percentile.

Feature rankings: consensus of held-out permutation importance (three learners, 2019-2021 folds for points; the selected learner for injury), discovery-only univariate association, a mutual-information and random-forest filter panel, and FeatureRanker 3.0.4 probe-weighted voting. Participation and receiving usage lead both tracks; injury history first appears at consensus rank 44. Family ablation: removing age, experience, and draft costs 0.0095 Spearman; removing prior-year participation costs 0.0022 AUC.

Amendments, all triggered by evidence recorded in the plan:

1. Training from 2013 failed the points null gate (0.0512 against 0.05). Mean target points drop from about 57 to 40 in 2017 when roster coverage widens, and permuted-target fits learn era proxies (null 0.108 from 2013 against 0.027 from 2017, real skill unchanged). Training now starts in 2017. The pre-amendment run was removed; its numbers are in the plan.
2. FeatureRanker 1.2.2 was too slow; a filter panel was added.
3. FeatureRanker 3.0.4 was installed and rerun; the consensus uses four ranks.

Diagnostic: the injury null is right-shifted when pooled (0.529) because permutation preserves position prevalence; within season and position it is 0.478.

## Phase 5 (negative result): usage shares and per-game rates

Plan: `experiments/phase5/plan_v1.json`. Run: `runs/*usage-features-v1`. Motivation from published fantasy and injury models: prior-season opportunity shares, per-game production, age, and prior injury are the main reported predictors. Twelve lag-1 columns were added: targets, carries, receptions, attempts, receiving yards, rushing yards, passing yards, and offense snaps per game, and the player's share of the final prior-season team's targets, carries, receptions, and attempts.

Points: discovery 0.7197 for the new Extra Trees against 0.7196 for the Phase 4 reference; audit 0.7548 against 0.7557; the best blend 0.7574 was inside tolerance and the rule kept the reference. Injury: see `experiments/phase5/RESULTS.md`. No adoption. The Phase 4 workbook remains final.

## Environment notes

- Hash chains recorded before the consolidation assume LF line endings. `.gitattributes` now normalizes text to LF; on a checkout with `core.autocrlf=true` earlier digests will not match the working files.
- scikit-learn 1.9 deprecated `penalty=`; the injury learners now pass `l1_ratio` and `C=inf` directly, which does not change the fitted models.
