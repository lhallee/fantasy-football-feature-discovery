# August 9 Phase 2 results

Phase 2 passed every promotion gate and replaced Phase 1 in production. It did not reach the exploratory 0.90 Spearman target.

## Selected procedures

| Cohort | Procedure | Columns | Discovery ρ |
|---|---|---:|---:|
| Offense | `off_stable_core_et_all` | 228 | 0.6988 |
| Kicker | `k_stable_core_et_32` | 32 | 0.6094 |

The offense result is not compact. The 128-column candidate reached discovery ρ = 0.6814 and failed the predeclared 0.005 tolerance. The evidence therefore favors retaining the complete leakage-controlled core vector.

## Locked 2022-2025 audit

| Cohort | Model ρ | Baseline ρ | Gain | Model MAE | Baseline MAE |
|---|---:|---:|---:|---:|---:|
| Offense | 0.7510 | 0.6472 | 0.1038 | 25.25 | 27.18 |
| Kicker | 0.6255 | 0.5995 | 0.0260 | 37.88 | 35.96 |

The offense bootstrap interval was 0.7263 to 0.7703. The paired gain interval was 0.0843 to 0.1244. Every offense position exceeded the locked 0.50 minimum.

The kicker rank gain interval was -0.0255 to 0.0766. Kicker MAE and NDCG were worse than the baseline, so the catalog keeps a low-confidence kicker label.

## Null controls

| Control | Offense mean ρ | Kicker mean ρ | Locked gate |
|---|---:|---:|---:|
| 499 fixed-prediction label permutations | 0.0004 | 0.0008 | absolute mean ≤ 0.015 |
| 11 end-to-end target permutations | 0.0009 | 0.0287 | offense ≤ 0.05; kicker ≤ 0.08 |

Both controls passed. They are implementation sentinels, not a formal discovery-wide permutation test.

## Production forecast

The final fit used eligible history through 2025 and generated 958 forecasts. The draft board contains 951 available players: 119 QB, 210 RB, 384 WR, 195 TE, and 43 K. Seven CUT or RET rows remain in the catalog but are excluded from the default board.

The 2026 outcomes are not yet observed. Retrospective evidence cannot be presented as prospective accuracy.
