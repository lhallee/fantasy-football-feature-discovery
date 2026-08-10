# Scoring-format correlations

The comparison holds every ESPN 2026 coefficient fixed except reception points. It evaluates non-PPR (0 points per reception), half PPR (0.5), and this project's full-PPR target (1.0). The pooled sample contains 16,586 regular-season QB, RB, WR, TE, and K player-seasons from 1999 through 2025. Kicker points are included in every format.

Full PPR and half PPR are nearly rank-equivalent in this sample: pooled Spearman correlation is 0.9978 and Pearson correlation is 0.9933. Full PPR versus non-PPR remains strongly associated but differs more in point magnitude, with Spearman 0.9890 and Pearson 0.9676. Reception scoring therefore changes total points more than it changes the global ordering of player-seasons.

These are pooled historical associations, not correlations between independent projection systems. Position and era remain mixed in the global sample, and a league-specific draft board can still change through lineup rules, replacement value, and positional scarcity.

Artifacts:

- [`scoring_scheme_correlations.png`](artifacts/scoring_scheme_correlations.png)
- [`scoring_scheme_correlations.csv`](artifacts/scoring_scheme_correlations.csv)

Reproduce them from the project root:

```powershell
python -m fantasy_football.phase2_scoring_comparison --root .
```
