# Experiments

The repository has two research phases. Phase 2 is production. Phase 1 and the earlier Phase 2 source-proxy studies remain available as immutable audit history.

| Phase | Purpose | Current status | Start here |
|---|---|---|---|
| [Phase 1](phase1/README.md) | Build the source data, scoring target, first compact model, and catalog | Archived baseline | [Phase 1 ledger](phase1/EXPERIMENT_LEDGER.md) |
| [Phase 2](phase2/README.md) | Replace production with an August 9 candidate universe, richer raw lags, chronological selection, and negative controls | Production; ρ = 0.751, below the 0.90 research target | [Production record](phase2/production/README.md) |

Repository-level `artifacts/` and `fantasy_football/players_2026.py` contain the gated Phase 2 production generation. The former Phase 1 production files are preserved in [`phase1/production_archive/`](phase1/production_archive/).

The earlier post-cut source-proxy experiments remain under [`phase2/runs/`](phase2/runs/) with their original manifests. Their roster-origin contract is not compatible with the August 9 production question, so they are historical diagnostics rather than active forecast evidence.

[`organization_manifest.json`](organization_manifest.json) records historical path and hash migrations. The [organization audit](ORGANIZATION_AUDIT.md) documents the directory contract.
