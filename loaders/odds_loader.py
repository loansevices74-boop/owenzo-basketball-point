import os
import requests
import pandas as pd

def get_odds_api_sport(sport_key="basketball_nba", regions="us", markets="totals"):
    try:
        import streamlit as st
        api_key = st.secrets.get("ODDS_API_KEY", "")
    except:
        api_key = os.getenv("ODDS_API_KEY", "")
    
    if not api_key:
        return pd.DataFrame()
    
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"apiKey": api_key, "regions": regions, "markets": markets, "oddsFormat": "decimal"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        rows = []
        for event in data:
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        rows.append({
                            "event": event.get("description", ""),
                            "home_team": event.get("home_team", ""),
                            "away_team": event.get("away_team", ""),
                            "commence_time": event.get("commence_time", ""),
                            "bookmaker": bookmaker.get("key", ""),
                            "market": market.get("key", ""),
                            "outcome_name": outcome.get("name", ""),
                            "odds": outcome.get("price", 0),
                        })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Odds API error: {e}")
        return pd.DataFrame()
