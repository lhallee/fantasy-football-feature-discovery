# Phase 2

Phase 2 is the production forecast generation. Its fixed origin is August 9, 2026. It replaces the target-year Week 1 roster proxy with a historical candidate universe reconstructed from prior-season rosters and target-year draft or combine entrants.

The selected offense model reached chronological 2022-2025 mean within-position-season Spearman ρ = 0.751, compared with 0.647 for prior-season points. It passed every predeclared promotion gate, including position-specific minimums and two shuffled-target controls. It did not reach the exploratory 0.90 target.

## Start here

- [August 9 production record](production/README.md)
- [Production results](production/RESULTS.md)
- [Production experiment ledger](production/EXPERIMENT_LEDGER.md)
- [Locked pre-fit plan](production/august9_plan_v1.json)
- [Pre-fit code and data lock](production/august9_prefit_manifest_v1.json)
- [Production release manifest](production/experiment_manifest_v1.json)
- [Draft workbook guide](DRAFT_BOARD.md)
- [Scoring-scheme correlations](SCORING_CORRELATIONS.md)

The root [`artifacts/`](../../artifacts/) directory, generated [`players_2026.py`](../../fantasy_football/players_2026.py), and [draft workbook](../../outputs/fantasy_football_draft_board_2026.xlsx) now use this production generation.

## Production runs

| Stage | Purpose | Run directory |
|---|---|---|
| Discovery | Choose model family, recipe, source tier, and feature count using 2016-2021 only | [`20260810T174927...`](runs/20260810T174927548207Z-august9-discovery-3b07b79602c5-b96dfefdf4d91c71/) |
| Audit | Evaluate the locked procedures on 2022-2025 and run negative controls | [`20260810T175209...`](runs/20260810T175209276165Z-august9-audit-3b07b79602c5-d6a01b2f6040ba35/) |
| Fit | Refit through 2025 and forecast the frozen 2026 roster | [`20260810T175655...`](runs/20260810T175655682503Z-august9-fit-3b07b79602c5-134b25c8b34a916a/) |

Every run has a manifest, declared input hashes, and hashes for every output. Audit cannot proceed if its selection, plan, code, or pre-fit lock differs from discovery. Final fitting cannot proceed unless all audit gates pass. Promotion re-verifies the fit before replacing production files.

## Earlier Phase 2 studies

Phase 2 first explored a post-final-cuts question using historical Week 1 roster files. Those source files did not have verified publication timestamps. The first cohort also included CUT and RET rows in both the scored cohort and roster-room aggregates. A status-filtered rerun corrected that cohort issue but still answered a post-cut source-proxy question, not the user's August 9 question. Its grouped-label null also failed its zero-centering sentinel.

These studies are preserved, not erased:

- [`plan_v1.json`](plan_v1.json) and [`experiment_manifest_v1.json`](experiment_manifest_v1.json): superseded first source-proxy study
- [`plan_v2.json`](plan_v2.json) and [`experiment_manifest_v2.json`](experiment_manifest_v2.json): corrected post-cut proxy stress test
- [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md), [`RESULTS_LEDGER.md`](RESULTS_LEDGER.md), and [`RESULTS_V2.md`](RESULTS_V2.md): historical narrative and limitations
- [`null_diagnostic_v2.json`](null_diagnostic_v2.json): diagnostic follow-up to the failed grouped-label sentinel

They remain useful for failure analysis. They do not validate or replace the August 9 production forecast.

## Claim boundary

The 2022-2025 production audit is chronological and procedure-locked, but its outcomes were seen during earlier project work. It is not analyst-blinded. Public historical data still do not reconstruct exact August 9 injury, transaction, depth-chart, or roster states. The 2026 season is the sealed prospective test.
