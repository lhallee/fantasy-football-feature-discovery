# 2026 draft-board data

The draft-board builder reads the generated `PLAYERS` tuple directly and returns six ordered pages: Overall, QB, RB, WR, TE, and K. Every page is sorted only by predicted season points, from highest to lowest. Name and player ID are used only to make exact score ties deterministic.

The default board includes backups and other fantasy-eligible players with predictions. It excludes only records marked CUT or RET. Pass `available_only=False` to retain those records for archival review.

```python
from fantasy_football.phase2_draft_board import draft_board_pages

pages = draft_board_pages()
best_available_running_back = pages["RB"][0]
```

Each flat record includes editable draft fields (`drafted`, `fantasy_team`, `draft_round`, and `notes`) alongside the prediction, approximate 80% interval, ranks, team, roster status, depth information, confidence, and player identifiers. Kicker pages follow the same predicted-score sort contract as every other position.

The checked-in [`draft_board_2026.json`](artifacts/draft_board_2026.json) is the workbook-neutral page payload:

```powershell
python -m fantasy_football.phase2_draft_board
```

The JSON contract is ready for an Excel renderer with one worksheet per page. The approved spreadsheet runtime was unavailable during this release, so no `.xlsx` file is represented as verified.
