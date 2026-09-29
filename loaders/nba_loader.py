"""NBA data via nba_api (free, no key required)."""
import pandas as pd
from datetime import datetime, timedelta


def load_nba_season(season: str = "2024-25") -> pd.DataFrame:
    """
    Load NBA season game log.
    Columns: date, home_team, away_team, home_score, away_score,
             home_h1_score, away_h1_score, ft_line (if available)
    """
    try:
        from nba_api.stats.endpoints import scoreboards
        from nba_api.stats.endpoints import leaguegamefinder

        finder = leaguegamefinder.LeagueGameFinder(
            season=season,
            season_type_all_star="Regular Season",
            league_id_nullable="00",
        )
        games = finder.get_data_frames()[0]

        rows = []
        for _, g in games.iterrows():
            rows.append({
                "date": g.get("GAME_DATE", ""),
                "home_team": g.get("HOME_TEAM_NAME", ""),
                "away_team": g.get("VISITOR_TEAM_NAME", ""),
                "home_score": int(g.get("PTS", 0)),
                "away_score": int(g.get("PTS_VISITOR", 0)),
                "home_h1_score": int(g.get("HOME_TEAM_HALF_SCORE", 0)) if "HOME_TEAM_HALF_SCORE" in g.index else None,
                "away_h1_score": int(g.get("VISITOR_TEAM_HALF_SCORE", 0)) if "VISITOR_TEAM_HALF_SCORE" in g.index else None,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"NBA API error: {e}")
        return pd.DataFrame()


def load_nba_live_scores() -> pd.DataFrame:
    """Load today's NBA scoreboard."""
    try:
        from nba_api.stats.endpoints import scoreboards
        sb = scoreboards.ScoreBoard()
        games = sb.get_data_frames()[0]
        return games
    except Exception as e:
        print(f"NBA live error: {e}")
        return pd.DataFrame()
