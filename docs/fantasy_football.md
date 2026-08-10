# Fantasy football and the scoring target

Fantasy football converts real NFL player statistics into points. A manager drafts a roster, starts players at fixed positions, and competes on weekly totals. Redraft leagues usually reset each season. This project predicts each player's full NFL regular-season total and rank within position; it does not simulate weekly lineups, head-to-head schedules, or playoffs.

## Why full PPR

The target is ESPN-style full points per reception. RotoWire found full PPR in 47.6% of 36,599 imported 2025 leagues, compared with 38.4% half PPR and 13.4% non-PPR. That is a large convenience sample of RotoWire users, not a census of every fantasy league. [RotoWire league analysis](https://www.rotowire.com/football/article/we-analyzed-35000-fantasy-football-leagues-heres-how-america-actually-plays-119607), [ESPN scoring formats](https://support.espn.com/hc/en-us/articles/360003914032-Scoring-Formats)

The executable profile is [scoring_espn_full_ppr_2026.json](../config/scoring_espn_full_ppr_2026.json). The build records its SHA-256 digest, and the model manifest carries the same profile name and digest. nflverse's supplied fantasy-point columns are used only for a reconstruction check, never as model inputs.

## ESPN full-PPR coefficients

| Event | Points |
|---|---:|
| Passing yard | 0.04 |
| Passing touchdown | 4 |
| Interception thrown | -2 |
| Rushing yard | 0.10 |
| Rushing touchdown | 6 |
| Reception | 1 |
| Receiving yard | 0.10 |
| Receiving touchdown | 6 |
| Passing, rushing, or receiving two-point conversion | 2 |
| Fumble lost | -2 |
| Return, fumble-recovery, or individual defensive touchdown | 6 |
| PAT made | 1 |
| Field goal, 0-39 yards | 3 |
| Field goal, 40-49 yards | 4 |
| Field goal, 50-59 yards | 5 |
| Field goal, 60 or more yards | 6 |
| Field goal missed | -1 |

These coefficients follow ESPN's July 2026 public table. Yardage is fractional, so 17 receiving yards contribute 1.7 points. Kicker targets include any reception points plus kicking points. [ESPN scoring formats](https://support.espn.com/hc/en-us/articles/360003914032-Scoring-Formats)

For reception coefficient `r`:

`fantasy points = non-PPR fantasy points + r × R`

Here, `R` is receptions and `r` is 0, 0.5, or 1 for non-PPR, half PPR, or full PPR. The player-game table stores all three offense scoring variants. The season target sums every source row with `season_type == "REG"`, including Week 18 since 2021.

## Defensive-touchdown limitation

ESPN can award an offensive-position player six points for a defensive touchdown. This matters for a two-way player such as Travis Hunter. The scorer accepts a curated `individual_defensive_tds` field, but the production nflverse weekly-stat input does not provide a trustworthy individual field for this use. Historical `def_tds` values have early-era attribution problems and no fantasy-position values from 2012-2025. The builder therefore does not add that raw column automatically. A 2026 defensive touchdown by an offensive-eligible player would be absent unless a reliable individual event source is added. [ESPN two-way-player explanation](https://www.espn.com/fantasy/football/story?id=45525668)

## Included positions

The forecast covers QB, RB, FB and HB as RB, WR, TE, K, and PK as K. Team defense and special teams is excluded because it is a team entity. Individual defensive player leagues are also outside the target. The generated catalog still retains every position in the nflverse roster snapshot.

RotoWire found a kicker slot in 88.7% and a defense slot in 83.2% of its sample. A future D/ST model would need a separate team-level target based on sacks, turnovers, touchdowns, points allowed, and yards allowed. [RotoWire league analysis](https://www.rotowire.com/football/article/we-analyzed-35000-fantasy-football-leagues-heres-how-america-actually-plays-119607), [ESPN D/ST scoring](https://support.espn.com/hc/en-us/articles/115003847231-Defense-and-Special-Teams-D-ST-Scoring)

## Reading a forecast

Use `position_rank` to compare players at the same position. `overall_point_rank` is not a draft order because replacement value differs by position. The approximate 80% interval uses the 80th percentile of absolute 2022-2025 residuals within position. It is a heuristic width estimated and assessed on the same retrospective period, not a coverage guarantee for 2026 injuries, cuts, or role changes.

The Phase 2 kicker model had higher retrospective Spearman correlation than prior-year points, but worse MAE and NDCG. Its bootstrap interval for the rank gain included zero. An explicit K or PK query therefore retains the conservative prior-points default:

```python
query_players(position="K", sort_by="previous_season_points")
```

Pass `sort_by="prediction"` to inspect the lower-confidence kicker model order. The draft workbook follows the user's requested predicted-score-only ordering for every position, including K.
