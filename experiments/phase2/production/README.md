# August 9 Phase 2 production protocol

This is the active production experiment. It answers one fixed question: what can be predicted for the August 9, 2026 fantasy-player universe from elementary public data that can be reconstructed without historical target-season roster membership?

## Locked contract

- Forecast origin: 2026-08-09.
- Historical candidates: prior-season roster players plus target-year draft or combine entrants.
- Nonparticipants: retained with zero target points.
- Discovery: 2016-2021 chronological folds.
- Audit: 2022-2025 chronological folds.
- Primary metric: mean within-position-season Spearman ρ.
- Offense baseline: prior-season points, evaluation only.
- Kicker baseline: eligible to win selection.
- Promotion: requires the predeclared accuracy, baseline, position, null, coverage, and lineage gates.

The plan was hashed before fitting. The pre-fit manifest binds the plan, scientific inputs, implementation, and focused tests.

## Reproduce

Run from the repository root:

```powershell
python -m fantasy_football.phase2_august_runner `
  --plan experiments/phase2/production/august9_plan_v1.json `
  --stage discovery
```

Use the printed discovery directory in the audit command:

```powershell
python -m fantasy_football.phase2_august_runner `
  --plan experiments/phase2/production/august9_plan_v1.json `
  --stage audit `
  --selection experiments/phase2/runs/<discovery-run>/selection.json
```

Final fitting requires both locked outputs:

```powershell
python -m fantasy_football.phase2_august_runner `
  --plan experiments/phase2/production/august9_plan_v1.json `
  --stage fit `
  --selection experiments/phase2/runs/<discovery-run>/selection.json `
  --audit experiments/phase2/runs/<audit-run>
```

Promotion is intentionally separate:

```powershell
python -m fantasy_football.phase2_august_runner `
  --plan experiments/phase2/production/august9_plan_v1.json `
  --stage promote `
  --fit-run experiments/phase2/runs/<fit-run>
```

Promotion verifies the plan, code, fit manifest, output hashes, and audit gates before replacing root artifacts. Existing Phase 1 production files are copied to `experiments/phase1/production_archive/` before replacement.

## Evidence

- [Locked plan](august9_plan_v1.json)
- [Pre-fit manifest](august9_prefit_manifest_v1.json)
- [Release manifest](experiment_manifest_v1.json)
- [Results](RESULTS.md)
- [Experiment ledger](EXPERIMENT_LEDGER.md)
- [Discovery run](../runs/20260810T174927548207Z-august9-discovery-3b07b79602c5-b96dfefdf4d91c71/)
- [Audit run](../runs/20260810T175209276165Z-august9-audit-3b07b79602c5-d6a01b2f6040ba35/)
- [Final-fit run](../runs/20260810T175655682503Z-august9-fit-3b07b79602c5-134b25c8b34a916a/)

## Limitation

The audit is chronological and procedure-locked, but not analyst-blinded. Exact historical August 9 roster, transaction, injury, and depth-chart states remain unavailable. The candidate-universe rule avoids using target-season roster membership; it cannot reconstruct every player who was briefly available in an undocumented historical camp.
