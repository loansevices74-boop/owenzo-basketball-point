import os
import requests
import pandas as pd

BASE_URL = "https://v1.basketball.api-sports.io"

def _get_key():
    try:
        import streamlit as st
        return st.secrets.get("API_BASKETBALL_KEY", "")
    except:
        return os.getenv("API_BASKETBALL_KEY", "")

def _headers():
    return {"x-apisports-key": _get_key()}

def get_leagues(country=None):
    params = {}
    if country:
        params["country"] = country
    resp = requests.get(f"{BASE_URL}/leagues", headers=_headers(), params=params)
    return pd.DataFrame(resp.json().get("response", []))

def get_games(league_id, season=2024, date=None):
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
            "status": g.get("status", {}).get("short", ""),
        })
    return pd.DataFrame(rows)
