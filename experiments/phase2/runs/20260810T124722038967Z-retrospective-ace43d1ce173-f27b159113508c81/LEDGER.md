# Phase 2 run: retrospective

Plan: `phase2_leakage_safe_raw_rank_v1`.
Elapsed time: 22.2 seconds.
Selected procedure: `stable_core_room_et_all_points`.

| Candidate | Spearman | MAE | Columns | Kind |
|---|---:|---:|---:|---|
| `stable_core_room_et_all_points` | 0.7661 | 28.15 | 248 | model |
| `previous_points_baseline` | 0.6585 | 29.51 | 1 | benchmark |

All folds train only on earlier target seasons. This run does not promote a model into Phase 1 artifacts or the player catalog.
