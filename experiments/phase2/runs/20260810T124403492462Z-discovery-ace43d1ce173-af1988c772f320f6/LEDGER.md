# Phase 2 run: discovery

Plan: `phase2_leakage_safe_raw_rank_v1`.
Elapsed time: 177.2 seconds.
Selected procedure: `stable_core_room_et_all_points`.

| Candidate | Spearman | MAE | Columns | Kind |
|---|---:|---:|---:|---|
| `stable_all_et_all_points` | 0.7157 | 32.35 | 305 | model |
| `stable_core_room_et_all_points` | 0.7138 | 32.32 | 248 | model |
| `recent_all_et_all_points` | 0.7102 | 32.01 | 373 | model |
| `stable_core_team_et_all_points` | 0.7085 | 33.18 | 250 | model |
| `stable_core_et_all_points` | 0.7077 | 33.36 | 229 | model |
| `stable_core_trajectory_et_all_points` | 0.7060 | 33.47 | 265 | model |
| `stable_all_et_128_points` | 0.7019 | 33.07 | 128 | model |
| `stable_all_et_64_signed_log` | 0.7008 | 46.43 | 64 | model |
| `stable_all_et_64_position_points` | 0.6993 | 32.55 | 64 | model |
| `stable_all_hurdle_64_rank` | 0.6977 | 47.25 | 64 | model |

All folds train only on earlier target seasons. This run does not promote a model into Phase 1 artifacts or the player catalog.
