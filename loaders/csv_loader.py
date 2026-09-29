"""EuroLeague live endpoint (free, no key)."""
import requests
import pandas as pd

EUROLEAGUE_API = "https://live.euroleague.net/api"


def get_euroleague_games(season: str = "2024-25", competition: str = "EL") -> pd.DataFrame:
    """
    Load EuroLeague/EuroCup games.
    competition: 'EL' = EuroLeague, 'EC' = EuroCup
    """
    url = f"{EUROLEAGUE_API}/tsv?systemcode={competition}&seasoncode={season}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        rows = []
        for g in data.get("Games", []):
            rows.append({
                "date": g.get("Date", ""),
                "home_team": g.get("Home", {}).get("Name", ""),
                "away_team": g.get("Away", {}).get("Name", ""),
                "home_score": g.get("Home", {}).get("Score", 0),
                "away_score": g.get("Away", {}).get("Score", 0),
                "home_h1_score": g.get("Home", {}).get("Score1", 0),
                "away_h1_score": g.get("Away", {}).get("Score1", 0),
                "status": g.get("Status", ""),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"EuroLeague API error: {e}")
        return pd.DataFrame()


def get_euroleague_standings(season: str = "2024-25", competition: str = "EL") -> pd.DataFrame:
    url = f"{EUROLEAGUE_API}/tsv?systemcode={competition}&seasoncode={season}&p=standings"
    try:
        resp = requests.get(url, timeout=10)
        return pd.DataFrame(resp.json().get("Standings", []))
    except Exception:
        return pd.DataFrame()
