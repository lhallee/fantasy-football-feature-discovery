# Phase 2 leakage-controlled ranking experiment

Phase 2 tests whether richer elementary NFL inputs can raise next-season within-position Spearman correlation above 0.90 without changing the cohort or using information unavailable at the forecast cutoff.

The v1 forecast origin is after final roster cuts and before Week 1. Historical membership comes from Week 1 roster proxies, which makes roster-room features valid only at that origin. The old files lack reliable publication timestamps, so this is a source-season-safe proxy analysis, not a timestamp-proven as-of study. This result must not be used to validate or replace the August 9 camp forecast. An August-origin model requires timestamp-aligned historical August roster snapshots.

The first locked result did not reach that threshold, but a later cohort audit invalidated its post-cut interpretation. V1 scored 0.7661 on an outcome-exposed 2022-2025 Week 1 source-row proxy that included cut and retired rows. It is preserved as a superseded audit trail. `plan_v2.json` declares the corrected status-filtered rerun before fitting, and `prefit_manifest_v2.json` locks its implementation and tests. The status rule applies to frozen upstream-deduplicated player rows; the ledger records the one mixed-status raw group that this conservative proxy excludes.

## Reproduce

Run from the project root:

```powershell
python -m fantasy_football.phase2_runner_v2 `
  --plan experiments/phase2/plan_v2.json `
  --stage discovery
```

The discovery command prints a unique run directory. Pass its `selection.json` to the retrospective stage:

```powershell
python -m fantasy_football.phase2_runner_v2 `
  --plan experiments/phase2/plan_v2.json `
  --stage retrospective `
  --selection experiments/phase2/runs/<discovery-run>/selection.json
```

Each run verifies the Phase 1 freeze before and after fitting, verifies the v2 pre-fit lock, hashes declared inputs and outputs, caps native numerical pools at four threads, and writes only inside its unique run directory. The retrospective stage also rejects any code or pre-fit-manifest hash that differs from discovery.

## Files

- `phase1_freeze_manifest.json`: protected Phase 1 tree and critical-file hashes.
- `plan_v1.json`: superseded fixed feature, model, split, and selection menu.
- `plan_v2.json`: pre-fit cohort and audit-control amendment over the unchanged v1 candidate menu.
- `prefit_manifest_v2.json`: exact pre-fit hashes for the amended plan, code, tests, ledger, and this guide.
- `experiment_manifest_v1.json`: superseded v1 plan and code hashes, run directories, and result summary.
- `runs/*/run_manifest.json`: exact source-input hashes and runtime environment for each run.
- `runs/*/output_hashes.json`: exact hashes for every generated run artifact.
- `runs/*/candidate_summary.csv`: aggregate metrics for every evaluated candidate.
- `runs/*/fold_metrics.csv`: season-level metrics.
- `runs/*/predictions.parquet`: row-level out-of-time predictions.
- `runs/*/selected_features.csv`: fold-local feature scores and selections.
- `runs/*/feature_lineage.json`: source, offset, and recipe for every input.
- `runs/*/uncertainty.json`: player-cluster and season-block intervals.

The complete research narrative is in [EXPERIMENT_LEDGER_PHASE2.md](../../EXPERIMENT_LEDGER_PHASE2.md).

## Interpretation

The 2022-2025 period is not a blinded holdout because Phase 1 already exposed those outcomes. The only sealed test is the future 2026 season, evaluated from an equivalent timestamped post-cut snapshot. Realized target-season games and snaps exceed 0.90 in an explicitly forbidden look-ahead diagnostic; they are not model inputs. They show that realized exposure is strongly co-ranked with fantasy points, not that a preseason model can predict that exposure or attain 0.90.
