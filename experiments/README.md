# Experiments

This directory separates the production forecast from later feature-discovery research.

| Phase | Purpose | Status | Start here |
|---|---|---|---|
| [Phase 1](phase1/README.md) | Build the 2026 full-PPR player catalog and compact next-season forecast | Complete production baseline | [Experiment ledger](phase1/EXPERIMENT_LEDGER.md) |
| [Phase 2](phase2/README.md) | Test richer leakage-controlled feature sets and the 0.90 Spearman target | Complete retrospective stress test; target not reached | [Pre-fit ledger](phase2/EXPERIMENT_LEDGER.md) and [results ledger](phase2/RESULTS_LEDGER.md) |

The Phase 1 catalog, processed data, and production model artifacts remain in their repository-level package, `data/`, and `artifacts/` locations because the application loads them directly. Experiment narratives, protocols, manifests, diagnostics, and research-only outputs live here.

[`organization_manifest.json`](organization_manifest.json) records the path migration from the published pre-organization commit. Historical hashes remain identifiable after the move.

The [organization audit](ORGANIZATION_AUDIT.md) records the move contract and release checks.
