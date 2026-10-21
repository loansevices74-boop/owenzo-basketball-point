import os
import requests
from openai import OpenAI

# ==========================================
# 1. DATA FETCHING & SMART FILTERING (REAL DATA ONLY)
# ==========================================
def get_odds_data(sport_key: str, api_key: str):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions=us,eu&markets=h2h,totals"
    res = requests.get(url)
    
    # Explicit error handling for real API limits
    if res.status_code == 429:
        return {"error": "429_TOO_MANY_REQUESTS"}
    elif res.status_code == 401:
        return {"error": "401_INVALID_API_KEY"}
    elif res.status_code == 200:
        return res.json()
    
    return []

def get_top_matches_for_league(sport_key: str, api_key: str, max_matches=2):
    result = get_odds_data(sport_key, api_key)
    
    # If we hit the API limit, bubble it up immediately
    if isinstance(result, dict) and "error" in result:
        return result
        
    if not result:
        return []
    
    scored_matches = []
    for match in result:
        bookmakers = match.get('bookmakers', [])
        if bookmakers and bookmakers[0].get('markets'):
            outcomes = bookmakers[0]['markets'][0].get('outcomes', [])
            if len(outcomes) >= 2:
                odds_diff = abs(outcomes[0]['price'] - outcomes[1]['price'])
                match['variance_score'] = odds_diff
                scored_matches.append(match)
                
    scored_matches.sort(key=lambda x: x['variance_score'])
    return scored_matches[:max_matches]

# ==========================================
# 2. LIGHTWEIGHT AI AGENT (Real Data Only)
# ==========================================
def generate_global_daily_report(DOMESTIC_LEAGUES: dict, odds_api_key: str, openrouter_key: str) -> str:
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    )

    country_matches_map = {}
    active_leagues_count = 0
    api_limit_hit = False

    # Fetch and filter odds
    for country, leagues in DOMESTIC_LEAGUES.items():
        country_matches = []
        for league in leagues:
            top_matches = get_top_matches_for_league(league, odds_api_key, max_matches=1)
            
            # Check if API limit was hit
            if isinstance(top_matches, dict) and top_matches.get("error") == "429_TOO_MANY_REQUESTS":
                api_limit_hit = True
                break
                
            if top_matches:
                active_leagues_count += 1
                for m in top_matches:
                    bookmakers = m.get('bookmakers', [])
                    odds_text = "N/A"
                    if bookmakers and bookmakers[0].get('markets'):
                        outcomes = bookmakers[0]['markets'][0].get('outcomes', [])
                        odds_text = ", ".join([f"{o['name']}: {o['price']}" for o in outcomes[:2]])
                    country_matches.append(f"- {m['home_team']} vs {m['away_team']} | Odds: {odds_text}")
        
        if api_limit_hit:
            break
            
        if country_matches:
            country_matches_map[country] = "\n".join(country_matches)

    # STRICT REAL DATA CHECK: No fake data fallback
    if api_limit_hit:
        return "⚠️ **API LIMIT REACHED:** You have used your 500 free monthly requests on The Odds API. Please wait until next month, or upgrade to a paid plan at the-odds-api.com to continue fetching real data."
    
    if not country_matches_map:
        return "ℹ️ **NO REAL MATCHES FOUND:** There are currently no active games scheduled for these leagues, or the API returned no data. Check back when games are scheduled."

    # Process AI analysis
    ai_analysis = ""
    system_prompt = """You are an elite professional sports handicapper. 
    Analyze the provided REAL basketball matches and odds. 
    For each match, provide ONE specific, high-confidence betting pick (Moneyline, Spread, or Total) 
    and write 1 sentence of 'Sharp Reasoning' based on team styles, pace, or odds value. 
    Format as a clean list."""

    countries = list(country_matches_map.keys())
    for i in range(0, len(countries), 5):
        batch_countries = countries[i:i+5]
        batch_text = ""
        for c in batch_countries:
            batch_text += f"\n### {c}\n{country_matches_map[c]}\n"

        response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct:free", # Active, reliable free model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze these REAL matches:\n{batch_text}"}
            ],
            temperature=0.3
        )
        ai_analysis += response.choices[0].message.content + "\n\n"

    # Format Final Product
    premium_report = f"""
# 🏀 **GLOBAL DAILY AI BETTING INTELLIGENCE REPORT** 
**Total Leagues Scanned:** {len(DOMESTIC_LEAGUES)}  
**Active Leagues with Real Picks:** {active_leagues_count}  
**Generated By:** Owenzo Autonomous AI Handicapper  
**Date:** Today  

---
### 📊 **TOP AI-GENERATED VALUE PICKS BY REGION**

{ai_analysis}

---
### 🔒 **UNLOCK THE FULL DEEP-DIVE ANALYSIS & LIVE TRACKING**
The free report above contains the top picks per active league. 
Our **Premium Daily Slip** includes:
✅ Full 10-Leg High-Confidence Accumulator  
✅ Live Line-Movement Alerts & Kelly Criterion Staking  
✅ Direct Email Alerts  

👉 **[GET LIFETIME ACCESS FOR $19.99](https://buy.stripe.com/YOUR_STRIPE_LINK_HERE)** 👈
"""
    return premium_report