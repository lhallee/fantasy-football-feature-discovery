# Phase 1: production forecast

Phase 1 assembled the source data, defined the ESPN-style full-PPR target, compared model families with temporal validation, selected a compact offense model, and generated the complete 2026 player catalog.

## Research record

- [Experiment and excitement ledger](EXPERIMENT_LEDGER.md)
- [Completion audit](COMPLETION_AUDIT.md)
- [Methods and results](../../docs/methods_and_results.md)
- [Data sources](../../docs/data_sources.md)
- [Fantasy football and scoring](../../docs/fantasy_football.md)

## Provenance

`phase1_freeze_manifest_v1.json` is the original freeze record created before Phase 2. `phase1_freeze_manifest.json` is the current layout-aware record created after the documentation reorganization. The current record points back to the original freeze ID; the scientific data, trained artifacts, and generated player catalog are unchanged.

Production outputs remain at the repository root:

- [`artifacts/`](../../artifacts/)
- [`data/processed/`](../../data/processed/)
- [`fantasy_football/players_2026.py`](../../fantasy_football/players_2026.py)
