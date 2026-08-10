# Phase 2 run: retrospective

Plan: `phase2_postcut_status_filtered_raw_rank_v2`.
Elapsed time: 46.7 seconds.
Selected procedure: `stable_core_room_et_all_points`.

| Candidate | Spearman | MAE | Columns | Kind |
|---|---:|---:|---:|---|
| `stable_core_room_et_all_points` | 0.7658 | 33.79 | 248 | model |
| `previous_points_baseline` | 0.6459 | 36.02 | 1 | benchmark |

All folds train only on earlier target seasons. This run does not promote a model into Phase 1 artifacts or the player catalog.
