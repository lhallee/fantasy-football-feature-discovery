# Experiment organization audit

## Outcome

The repository now has explicit [`phase1/`](phase1/) and [`phase2/`](phase2/) research records. No experiment ledger or completion audit remains at the repository root. Production data, models, and the generated player catalog remain in their application-facing locations.

## Relocations

| Historical path | Current path |
|---|---|
| `EXPERIMENT_LEDGER.md` | `experiments/phase1/EXPERIMENT_LEDGER.md` |
| `COMPLETION_AUDIT.md` | `experiments/phase1/COMPLETION_AUDIT.md` |
| `EXPERIMENT_LEDGER_PHASE2.md` | `experiments/phase2/EXPERIMENT_LEDGER.md` |
| `EXPERIMENT_LEDGER_PHASE2_RESULTS.md` | `experiments/phase2/RESULTS_LEDGER.md` |
| `experiments/phase2/phase1_freeze_manifest.json` | `experiments/phase1/phase1_freeze_manifest_v1.json` |

Relative links were repaired after the moves. [`organization_manifest.json`](organization_manifest.json) binds every historical path and SHA-256 to its current path and SHA-256, using commit `f128c46d2cf503e2ddba7c8cb4aad06d2b32d972` as the published pre-organization baseline.

The Markdown compatibility pass also replaced unsupported LaTeX-style math delimiters with portable UTF-8 notation. Statistical symbols now render directly as ρ, τ, and R² without requiring a math extension.

## Provenance

The original Phase 1 freeze record remains byte-identical at [`phase1_freeze_manifest_v1.json`](phase1/phase1_freeze_manifest_v1.json). The current [`phase1_freeze_manifest.json`](phase1/phase1_freeze_manifest.json) records a new layout-aware freeze ID and points back to the original ID. Phase 1 data, trained artifacts, scoring configuration, and `players_2026.py` retain their original critical-file hashes.

Historical Phase 2 plans, pre-fit manifests, run manifests, selections, predictions, and output-hash files were not rewritten. The path resolver validates their historical hashes through the explicit organization record. Its maintenance scope is limited to path lookup, current freeze discovery, navigation, and corresponding tests.

## Verification

- 81 of 81 tests passed with external pytest plugin autoload disabled.
- Ruff passed, and the six changed Python files passed the formatter check.
- All 38 Python files compiled.
- The primary CLI, Phase 2 runner CLI, and draft-board CLI passed smoke tests.
- Both the current Phase 1 freeze and its reference to the original freeze verified.
- All organization records resolved and matched their current hashes.
- Every local Markdown link resolved.
- The size-cap scope is synchronized at 96,956,855 bytes, below the 900 MiB limit.
