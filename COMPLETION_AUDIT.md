# Completion audit

## Outcome

The frozen August 9, 2026 project satisfies every requested deliverable within the declared scope. It compiles public player data, reconstructs full-PPR scores from raw events, compares lightweight models with chronological validation, discovers a compact offense feature set, exports every current roster player and backup, and remains well below 1 GB.

The strongest defensible offense result is a locally irreducible five-atom Extra Trees model. It is not a proven global minimum. Kicker model rank evidence is negative, so explicit K and PK queries default to the stronger prior-points baseline.

## Requirement evidence

| Requested outcome | Evidence | Status |
|---|---|---|
| Find public, current sources | [Data sources](docs/data_sources.md) documents 94 nflverse assets, coverage, endpoints, update cadence, licensing, and a 2026-08-10 source cutoff. [manifest.json](data/raw/manifest.json) stores URL, retrieval time, verification time, size, and SHA-256 for every file. | Complete |
| Compile broad player and year coverage | Processed data contain 475,626 player-game rows from 1999-2025, 66,907 player-seasons, 56,505 roster proxy rows, and 18,630 modeling rows. The current snapshot contains 2,930 roster players, including 958 fantasy-eligible players. | Complete |
| Stay below 1 GB | `data/` plus `artifacts/` use 82,802,977 bytes. The generated catalog adds 3,468,732 bytes, for 86,271,709 bytes total, or 8.63% of decimal 1 GB. An automated 900 MiB guard and test enforce the limit. | Complete |
| Explain fantasy football and scoring concisely | [Fantasy football and scoring](docs/fantasy_football.md) explains gameplay, full-PPR popularity, every scoring coefficient, included positions, and interpretation limits. The executable scoring profile is [scoring_espn_full_ppr_2026.json](config/scoring_espn_full_ppr_2026.json). | Complete |
| Train on prior years to predict later years | Development uses rolling 2020-2023 folds. Retrospective evaluation trains before 2024 and 2025. The final forecast refits through 2025 for the 2026 snapshot. | Complete |
| Compare models and hyperparameters | The full search evaluates 30 settings across ridge, elastic net, random forest, Extra Trees, histogram gradient boosting, XGBoost, RBF SVR, and small MLP families. Search artifacts preserve fold results and parameters. | Complete |
| Run easily on the laptop | The observed full CPU search took 743.4 seconds, used at most four estimator and native numerical threads, and stayed near or below 500 MB RAM. CUDA is not required. A reduced smoke search is available. | Complete |
| Find a small discriminative combination | The final offense atoms are draft number, prior receiving yards, prior rushing yards, prior passing touchdowns, and age. They expand to 24 fitted columns. Development Spearman is 0.73865, within 0.00460 of the full 32-atom model, with a 1.07% MAE increase. Every one-atom deletion from this final set failed the declared threshold. | Complete with bounded claim |
| Explain exactly how the combination was found | [Methods and results](docs/methods_and_results.md) and the [ledger](EXPERIMENT_LEDGER.md) record temporal splits, three-signal atom ranking, prefix search, greedy deletion, thresholds, model selection, sensitivity tests, and negative results. CSV artifacts preserve every evaluated prefix and deletion. | Complete |
| Produce all-player Python data | [players_2026.py](fantasy_football/players_2026.py) contains all 2,930 source roster IDs. All 958 eligible players have predictions and exact fitted vectors; the other 1,972 retain metadata. `get_player` and `query_players` support position, team, status, name, college, confidence, depth, rank, points, availability, and distribution filters. | Complete |
| Include backups | The catalog contains 791 fantasy-eligible players ranked below first on their team-position group. Source, processed, and generated GSIS ID sets match exactly. | Complete |
| Keep a detailed excitement ledger | [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) records research, decisions, failed checks, fixes, runtimes, metrics, caveats, and the most informative findings. | Complete |
| Follow coding standards | The repository-wide pass covered all 18 eligible first-party Python files, added complete numerical shape traces to array code, normalized imports and formatting, preserved behavior, and excluded only the generated catalog. | Complete |
| Follow writing standards | User-facing prose is answer-first, evidence-led, qualified, and linked to primary or direct sources where possible. The language audit found no actionable prose issues; two repeated-opening warnings refer only to command lines inside code blocks. | Complete |

## Model evidence

On chronological 2024-2025 retrospective evaluation, the five-atom offense model reaches within-position Spearman 0.7575, MAE 29.46, RMSE 46.01, NDCG 0.7940, and top-k recall 0.5747. Prior-year points reach Spearman 0.6688, MAE 29.58, RMSE 56.10, NDCG 0.7642, and top-k recall 0.5382. A 500-sample player-cluster bootstrap gives a 95% interval of 0.7249-0.7822 for model Spearman and 0.0602-0.1157 for the paired Spearman improvement.

The full 32-atom model reaches a higher retrospective Spearman of 0.7694 and lower MAE of 28.83, so compactness has a measurable cost. Running-back MAE and overall position-rank MAE do not improve. The kicker model reaches Spearman 0.466 versus 0.512 for prior-year points and does not support a better-rank claim.

## Coding-standards inventory

| Classification | Files |
|---|---|
| Structural implementation and standards pass | `dataset.py`, `export.py`, `features.py`, `metrics.py`, `modeling.py`, `models.py`, `scoring.py` |
| Mechanical standards pass | `__init__.py`, `__main__.py`, `cli.py`, `constants.py`, `download.py`, `provenance.py`, and all five test modules |
| Excluded | `players_2026.py`, because it is generated by `export_current_players` and verified through source-to-catalog parity tests |
| Blocked | None |

The numerical scope uses symbolic shape comments at function inputs, DataFrame and Series construction, masking and slicing, model calls, reductions, mutations, concatenation, and returns. The pass also fixed import grouping, formatter consistency, package discovery, thread limits, smoke-output isolation, and refresh provenance.

## Final verification

The final clean checks are:

```text
25 tests passed
Ruff: all checks passed
compileall: passed
CLI help: passed
wheel build and isolated install: passed
saved estimator reproduction: passed
source hash verification: passed
current roster ID parity: passed
exact fitted-vector parity: passed
data-size guard: passed
```

The test command disables unrelated global pytest-plugin autoload because the host environment contains a broken third-party `torchtyping` plugin. The project itself does not use Torch.

## Claim limits

- The compact set is locally irreducible on one deterministic greedy path, not globally minimal over all (2^{32}) subsets.
- Draft number is a scalar source field but embeds NFL scouting judgment.
- Historical Week 1 rosters are proxies, not proven preseason snapshots.
- The 2024-2025 evaluation was inspected during development and is retrospective, not sealed.
- August rosters, injuries, roles, and team context can change before Week 1.
- The production source lacks reliable individual defensive-touchdown attribution for a two-way fantasy player.
- Full-season point rank is not league-specific draft value; lineup rules and replacement level still matter.
