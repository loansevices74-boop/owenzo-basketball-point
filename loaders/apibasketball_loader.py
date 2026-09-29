"""API-Basketball (api-sports.io) — 100 free requests/day."""
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://v1.basketball.api-sports.io"
API_KEY = os.getenv("API_BASKETBALL_KEY", "")


def _headers():
    return {"x-apisports-key": API_KEY}


def get_leagues(country: str = None) -> pd.DataFrame:
    """List available leagues."""
    params = {}
    if country:
        params["country"] = country
    resp = requests.get(f"{BASE_URL}/leagues", headers=_headers(), params=params)
    data = resp.json().get("response", [])
    return pd.DataFrame(data)


def get_games(league_id: int, season: int = 2024, date: str = None) -> pd.DataFrame:
    """
    Get games for a league/season.
    Returns: date, home_team, away_team, home_score, away_score, status
    """
    params = {"league": league_id, "season": season}
    if date:
        params["date"] = date
    resp = requests.get(f"{BASE_URL}/games", headers=_headers(), params=params)
    data = resp.json().get("response", [])

    rows = []
    for g in data:
        rows.append({
            "date": g.get("date", ""),
            "home_team": g.get("teams", {}).get("home", {}).get("name", ""),
            "away_team": g.get("teams", {}).get("away", {}).get("name", ""),
            "home_score": g.get("scores", {}).get("home", {}).get("total", 0) or 0,
            "away_score": g.get("scores", {}).get("away", {}).get("total", 0) or 0,
            "home_h1_score": g.get("scores", {}).get("home", {}).get("quarter_1", 0) or 0,
            "away_h1_score": g.get("scores", {}).get("away", {}).get("quarter_1", 0) or 0,
            "status": g.get("status", {}).get("short", ""),
        })
    return pd.DataFrame(rows)


def get_odds(game_id: int) -> dict:
    """Get betting odds for a game (Blessings: real closing lines)."""
    resp = requests.get(f"{BASE_URL}/odds", headers=_headers(), params={"game": game_id})
    return resp.json().get("response", [])
