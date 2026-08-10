# August 9 Phase 2 experiment ledger

This ledger records the production replacement in chronological order. Machine-readable plans, manifests, row-level predictions, and hashes are the authoritative evidence when prose and artifacts differ.

## Entry 0: question correction

The earlier Phase 2 experiments answered a post-final-cuts proxy question. The user required Phase 2 to replace production for the existing August 9 forecast. Historical Week 1 roster membership would expose information unavailable at that origin, so the production experiment needed a new candidate-universe contract.

Decision: use prior-season roster membership plus target-year draft or combine entry. Retain players with no target-season participation as explicit zero outcomes. Exclude every target-year roster-room, team, status, depth, and injury feature.

## Entry 1: pre-fit lock

At 2026-08-10 17:35:42 UTC, `august9_plan_v1.json` recorded the candidate universe, allowed and excluded features, chronological folds, model menu, feature-count menu, selection rule, null controls, and promotion gates.

At 2026-08-10 17:47:36 UTC, `august9_prefit_manifest_v1.json` locked:

- plan SHA-256 `3b07b79602c5a5e352d008f1cdd1b185011aa76b08140ae2313c940bf98babc0`
- eight implementation and focused-test hashes
- 95 raw files totaling 56,786,715 bytes
- five processed files totaling 15,790,421 bytes
- the scoring configuration and its SHA-256
- 13 passing focused tests

Known before fitting: the 2022-2025 outcomes had been viewed in earlier phases, and exact historical August 9 source timestamps were unavailable.

## Entry 2: discovery

The discovery run evaluated the fixed menu on 2016-2021 chronological folds. It completed in 143.4 seconds with numerical and estimator pools capped at four threads.

Offense result: `off_stable_core_et_all` led at ρ = 0.6988 with 228 columns. The 128-column version reached 0.6814. Because its loss exceeded 0.005, it did not qualify for the compactness tie-breaker.

Kicker result: `k_stable_core_et_32` led at ρ = 0.6094. Prior-season points reached 0.5571 and was eligible to win.

Excitement checkpoint: the richer raw history produced a strong, stable rank signal without roster-room shortcuts. Scientific restraint checkpoint: the offense result required the full safe core, so the experiment did not discover a minimal combination.

## Entry 3: locked audit

The audit run loaded the hashed discovery selection and rejected any plan or code mismatch. It evaluated 2022-2025 once, then ran the predeclared controls. Runtime was 267.1 seconds.

Offense reached ρ = 0.7510, versus 0.6472 for prior-season points. MAE improved from 27.18 to 25.25. Position-specific ρ ranged from 0.7092 for QB to 0.7867 for TE. All four validation seasons fell between 0.7413 and 0.7595.

Kicker reached ρ = 0.6255, versus 0.5995 for prior-season points. Its point error and NDCG were worse, so the improvement is not uniform across objectives.

The fixed-prediction null centered at ρ = 0.0004 for offense. The end-to-end null centered at ρ = 0.0009. Both passed their locked thresholds. The previous positive grouped-label-null failure did not recur because this protocol scores against independently permuted validation labels.

Decision: every promotion gate passed. The 0.90 research target did not.

## Entry 4: final fit and promotion

The final fit completed in 38.3 seconds. It trained the selected procedures on eligible history through 2025 and emitted 958 unique 2026 forecasts with no missing eligible prediction.

Independent checks replayed both saved estimators against the saved feature vectors and matched predictions within absolute tolerance `1e-12`. Roster IDs matched the frozen August 9 catalog exactly. Feature lineage contained no forbidden target, status, depth, room, team-context, or fantasy-composite input.

Promotion archived the former Phase 1 production files, copied the gated Phase 2 models and predictions into root `artifacts/`, and regenerated `fantasy_football/players_2026.py`.

## Entry 5: draft deliverables

The draft payload was regenerated from the promoted catalog. The Python Excel fallback produced six verified sheets: Overall, QB, RB, WR, TE, and K. Every sheet is sorted only by predicted points, with deterministic name and player-ID tie breaks.

The workbook includes editable draft status, fantasy team, round, pick, and notes columns; frozen panes; filters; validation lists; conditional formatting; point intervals; confidence flags; and identifiers. It is published at `outputs/fantasy_football_draft_board_2026.xlsx`.

## Entry 6: final interpretation

Phase 2 replaces Phase 1 because its information-origin contract, candidate universe, chronology, artifact isolation, and negative controls are more rigorous. It is not promoted because its headline number is larger by itself.

The active claim is ρ = 0.751 on a chronological but non-blinded retrospective. The 2026 season is the first prospective test. Exact historical August 9 injury, transaction, depth-chart, and camp-roster information remains the most important missing data class.
