import requests
import pandas as pd

EUROLEAGUE_API = "https://live.euroleague.net/api"

def get_euroleague_games(season="2024-25", competition="EL"):
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
                "status": g.get("Status", ""),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"EuroLeague API error: {e}")
        return pd.DataFrame()
