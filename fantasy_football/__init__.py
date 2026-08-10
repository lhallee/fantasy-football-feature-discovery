"""Fantasy-football data, modeling, and projection tools."""

from .scoring import ScoringProfile, load_scoring_profile, score_player_games


__all__ = ["ScoringProfile", "load_scoring_profile", "score_player_games"]
__version__ = "0.1.0"
