# Phase 2 V2 Post-Fit Experiment Ledger

This file continues the immutable pre-fit [Phase 2 ledger](EXPERIMENT_LEDGER_PHASE2.md). The pre-fit file remains unchanged so `prefit_manifest_v2.json` can continue to verify the exact protocol that ran.

## Entry 7: corrected discovery run

**Recorded:** 2026-08-10, after the locked discovery run.

The corrected discovery run is:

`experiments/phase2/runs/20260810T135032148895Z-v2-discovery-507ac4f8eb7d-60bed99cda8e1b66`

The runner verified the seven-scope Phase 1 freeze, plan SHA-256 `507ac4f8...241916`, pre-fit manifest SHA-256 `eb2833f1...41b245`, and six Phase 2 code hashes before fitting. It retained 11,615 of 12,893 source rows through 2021. The fixed status rule removed 1,278 rows before feature and roster-room construction, including 1,176 `CUT` rows, seven `RET` rows, and two rows with missing status.

### Discovery result

The best eligible point estimate was the 305-column stable Extra Trees model at Spearman 0.72258. The locked tolerance rule selected `stable_core_room_et_all_points` at 0.72226 because its 248 columns were within 0.005 of the leader and used fewer inputs. This is the smallest qualifying declared procedure, not a globally minimal feature combination.

The selected input vector contains:

- 196 elementary counts, history indicators, and raw rates from one to four completed seasons earlier;
- 29 cutoff metadata, combine, and missingness fields;
- four position indicators; and
- 19 same-team, same-position room measurements built from cutoff membership and lag-one raw production.

All 248 columns were retained in every fold. The highest mean fold-local association scores were for lag-one receptions, receiving first downs, receiving yards, receiving explosive-play counts, room-relative receptions and receiving yards, lag-one games, receiving touchdowns, draft number, and lag-one carries. These are descriptive rankings inside a large correlated set, not isolated causal effects.

The stable Extra Trees feature-count curve was:

| Columns | Discovery Spearman | MAE |
|---:|---:|---:|
| 8 | 0.51676 | 49.25 |
| 16 | 0.68321 | 40.28 |
| 32 | 0.69726 | 38.88 |
| 64 | 0.70243 | 38.24 |
| 128 | 0.70991 | 37.49 |
| 305 | 0.72258 | 37.03 |

The previous-season-points baseline scored 0.60634. The selected 248-column model scored 0.72226 with MAE 36.99. Removing draft number, draft round, drafted status, and their missingness fields after selection reduced discovery Spearman to 0.67551.

### Null-control finding

The predeclared end-to-end grouped-label permutation scored 0.10956 rather than centering near zero. Its six yearly values were all positive. Ten additional post-selection discovery-only repeats were also positive, giving an 11-run mean of 0.11753 and sample standard deviation 0.04280. This means the raw shuffled-target zero-null sentinel failed.

The failure did not expose a direct real-label path:

- Original versus shuffled targets had macro Spearman 0.00202.
- One thousand within-group validation-label permutations against fixed predictions centered at -0.00039.
- Six independent centered Rademacher-label fits had mean -0.02652 with both signs.
- An antithetic centered-noise pair scored 0.00443 and -0.00271.
- Three within-group standardized-target permutations had mean -0.09413.

These checks favor an Extra Trees, target-skew, and feature-geometry effect. The selected all-column procedure still reorders columns by absolute shuffled-label association, and the fixed forest can turn structured support and sparse tails into a nonzero ranking even after row labels are destroyed. The observed-minus-null-mean difference is 0.60473, but it is only a descriptive contrast. It is not a bias-corrected Spearman or a confirmatory permutation result. The additional controls were executed after procedure selection and are recorded transparently in `experiments/phase2/null_diagnostic_v2.json`.

## Entry 8: locked retrospective stress test

**Recorded:** 2026-08-10, after the discovery selection and implementation hashes were fixed.

The locked stress run is:

`experiments/phase2/runs/20260810T140512585689Z-v2-retrospective-507ac4f8eb7d-3bcc8a436dc769ae`

The runner rejected implementation drift by matching the discovery selection's six code hashes and pre-fit manifest hash. It evaluated only the selected procedure and its nonselecting diagnostics. It retained 14,669 of 16,726 source rows through 2025, removing 2,057 rows before room aggregation. The removed rows included 1,927 `CUT`, 34 `RET`, 52 `UFA`, 29 `NWT`, nine `RSR`, one undocumented `E01`, one `TRD`, and two missing-status rows.

### Stress-period result

The selected model scored Spearman **0.76583** on 3,054 player-seasons from 2022 through 2025. Prior-season points scored 0.64594. The model's MAE was 33.79 points, NDCG at the roster cutoff was 0.79777, and top-k recall was 0.57726.

| Target season | Rows | Spearman | MAE |
|---:|---:|---:|---:|
| 2022 | 786 | 0.76253 | 33.07 |
| 2023 | 757 | 0.74407 | 34.32 |
| 2024 | 756 | 0.77754 | 34.95 |
| 2025 | 755 | 0.77917 | 32.83 |

| Position | Spearman | MAE |
|---|---:|---:|
| QB | 0.73449 | 46.29 |
| RB | 0.73491 | 39.73 |
| WR | 0.78624 | 32.29 |
| TE | 0.80766 | 22.24 |

The descriptive player-cluster 95% interval is 0.73942 to 0.78555. The four-season resampling interval is 0.75244 to 0.77835. Neither interval includes model-selection, source-timing, future-regime, or prospective-season uncertainty.

The joint post-selection no-draft ablation scored 0.72176, a reduction of 0.04407. This is consistent with useful incremental signal in the six draft-related fields, but correlated inputs prevent isolated attribution. The remaining rawer performance, physical, and room inputs still produced useful ranking. The locked grouped-label null scored 0.09039 on the stress years and retains the discovery null caveat.

### Decision

The experiment did not reach 0.90 and fails the declared success rule:

- point estimate 0.76583, not above 0.90;
- player-cluster lower bound 0.73942, not above 0.85;
- no position exceeds 0.85;
- no retrospective season exceeds 0.88; and
- the raw shuffled-target sentinel does not center at zero.

The result is a useful source-season-controlled Week 1 roster-proxy stress test, not a timestamp-proven or prospective estimate. Historical publication timestamps remain unverified, and the 2022-2025 outcomes were exposed by earlier work. No Phase 2 model has been promoted into the 2026 player catalog.

These results give more support to prioritizing timestamped preseason injury, roster, transaction, and role data than to another unconstrained search over the current feature set. The look-ahead diagnostics show that realized availability and workload are highly informative, but those variables are forbidden at forecast time. A defensible next study needs those snapshots collected at the same origin across seasons, followed by a fully sealed 2026 evaluation. Future feature-count work should use a new predeclared protocol and repeat the entire selection rule inside any confirmatory permutation analysis.
