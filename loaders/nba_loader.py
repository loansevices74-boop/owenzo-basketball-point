import pandas as pd

def load_nba_season(season="2024-25"):
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        finder = leaguegamefinder.LeagueGameFinder(season=season, season_type_all_star="Regular Season", league_id_nullable="00")
        games = finder.get_data_frames()[0]
        rows = []
        for _, g in games.iterrows():
            rows.append({
                "date": g.get("GAME_DATE", ""),
                "home_team": g.get("HOME_TEAM_NAME", ""),
                "away_team": g.get("VISITOR_TEAM_NAME", ""),
                "home_score": int(g.get("PTS", 0)),
                "away_score": int(g.get("PTS_VISITOR", 0)),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"NBA API error: {e}")
        return pd.DataFrame()

def load_nba_live_scores():
    return pd.DataFrame()
