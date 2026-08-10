# Phase 2 run: discovery

Plan: `phase2_postcut_status_filtered_raw_rank_v2`.
Elapsed time: 195.3 seconds.
Selected procedure: `stable_core_room_et_all_points`.

| Candidate | Spearman | MAE | Columns | Kind |
|---|---:|---:|---:|---|
| `stable_all_et_all_points` | 0.7226 | 37.03 | 305 | model |
| `stable_core_room_et_all_points` | 0.7223 | 36.99 | 248 | model |
| `stable_core_team_et_all_points` | 0.7152 | 37.42 | 250 | model |
| `recent_all_et_all_points` | 0.7140 | 37.06 | 373 | model |
| `stable_core_trajectory_et_all_points` | 0.7120 | 37.63 | 265 | model |
| `stable_core_et_all_points` | 0.7117 | 37.61 | 229 | model |
| `stable_all_et_128_points` | 0.7099 | 37.49 | 128 | model |
| `stable_all_et_64_position_points` | 0.7088 | 37.12 | 64 | model |
| `stable_all_et_64_signed_log` | 0.7047 | 36.12 | 64 | model |
| `stable_all_et_64_points` | 0.7024 | 38.24 | 64 | model |

All folds train only on earlier target seasons. This run does not promote a model into Phase 1 artifacts or the player catalog.
