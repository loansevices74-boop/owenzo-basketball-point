import os
import requests
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
from langchain_openai import ChatOpenAI

# ==========================================
# OPEN-SOURCE LLM CONFIGURATION
# ==========================================
def get_open_source_llm():
    return ChatOpenAI(
        model="mistralai/mistral-7b-instruct:free", 
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY", "your_free_key_here"),
        temperature=0.3
    )

# ==========================================
# 1. DATA FETCHING & SMART FILTERING
# ==========================================
def get_odds_data(sport_key: str, api_key: str):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions=us,eu&markets=h2h,totals"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else []

def calculate_bounds(avg_home_score: float, avg_away_score: float):
    expected_total = (avg_home_score + avg_away_score)
    return {
        "expected_fulltime": round(expected_total, 1),
        "ft_over": round(expected_total + 5.5, 1),
        "ft_under": round(expected_total - 5.5, 1)
    }

def get_top_matches_for_league(sport_key: str, api_key: str, max_matches=2):
    """Fetches odds and uses math to find the most volatile/valuable matches."""
    matches = get_odds_data(sport_key, api_key)
    if not matches:
        return []
    
    # Sort by a simple variance metric (difference in odds) to find the most interesting games
    # In a real app, you'd use Kelly Criterion, but this is light and efficient.
    scored_matches = []
    for match in matches:
        bookmakers = match.get('bookmakers', [])
        if bookmakers and bookmakers[0].get('markets'):
            outcomes = bookmakers[0]['markets'][0].get('outcomes', [])
            if len(outcomes) >= 2:
                odds_diff = abs(outcomes[0]['price'] - outcomes[1]['price'])
                match['variance_score'] = odds_diff
                scored_matches.append(match)
                
    # Sort by highest variance (closest odds = highest variance/interest) and take top N
    scored_matches.sort(key=lambda x: x['variance_score'])
    return scored_matches[:max_matches]

# ==========================================
# 2. AUTONOMOUS AI AGENTS (Batch Processing)
# ==========================================
def generate_global_daily_report(DOMESTIC_LEAGUES: dict, odds_api_key: str) -> str:
    """Generates a massive, multi-league daily report efficiently."""
    
    llm = get_open_source_llm()
    scrape_tool = ScrapeWebsiteTool()

    # AGENT 1: The Sharp Researcher
    researcher = Agent(
        role="Elite Sports Data Researcher",
        goal="Find critical real-world context (injuries, rest days, recent form) for basketball teams.",
        backstory="You use web search to find the hidden factors that bookmakers miss.",
        tools=[scrape_tool],
        llm=llm,
        verbose=False
    )

    # AGENT 2: The AI Handicapper
    handicapper = Agent(
        role="Professional AI Sports Handicapper",
        goal="Generate highly confident, data-backed betting predictions with clear reasoning.",
        backstory="You combine raw odds data with real-world research to find +EV (positive expected value) bets.",
        llm=llm,
        verbose=False
    )

    # Group leagues by country to batch AI tasks and save API calls
    country_tasks = []
    active_leagues_count = 0

    for country, leagues in DOMESTIC_LEAGUES.items():
        country_matches = []
        for league in leagues:
            top_matches = get_top_matches_for_league(league, odds_api_key, max_matches=1)
            if top_matches:
                active_leagues_count += 1
                for m in top_matches:
                    country_matches.append(f"- {m['home_team']} vs {m['away_team']} (League: {league})")
        
        if country_matches:
            match_list = "\n".join(country_matches)
            country_tasks.append({
                "country": country,
                "matches": match_list
            })

    if not country_tasks:
        return "⚠️ No active matches found across any leagues today. Check back tomorrow!"

    # Create ONE task per active country to prevent OpenRouter rate limits
    tasks = []
    for ct in country_tasks:
        tasks.append(Task(
            description=f"""
            Analyze the following basketball matches for {ct['country']}:
            {ct['matches']}
            
            For EACH match listed above:
            1. Briefly search for recent news/injuries for the teams.
            2. Provide ONE specific, high-confidence betting pick (Moneyline, Spread, or Total).
            3. Write 1 sentence of 'Sharp Reasoning' for the pick.
            Format as a clean list.
            """,
            expected_output="A structured list of betting picks for the provided matches.",
            agent=handicapper
        ))

    # Execute Crew
    crew = Crew(agents=[researcher, handicapper], tasks=tasks, process=Process.sequential)
    result = crew.kickoff()

    # Format Final Product
    premium_report = f"""
# 🏀 **GLOBAL DAILY AI BETTING INTELLIGENCE REPORT** 🏀
**Total Leagues Scanned:** {len(DOMESTIC_LEAGUES)}  
**Active Leagues with Picks:** {active_leagues_count}  
**Generated By:** Owenzo Autonomous AI Handicapper  
**Date:** Today  

---
### 📊 **TOP AI-GENERATED VALUE PICKS BY REGION**

{result.raw}

---
### 🔒 **UNLOCK THE FULL DEEP-DIVE ANALYSIS & LIVE TRACKING**
The free report above contains the top 1 pick per active league. 
Our **Premium Daily Slip** includes:
✅ Full 10-Leg High-Confidence Accumulator  
✅ Live Line-Movement Alerts & Kelly Criterion Staking  
✅ Direct Telegram/Discord Alerts  

👉 **[GET LIFETIME ACCESS FOR $19.99](https://buy.stripe.com/YOUR_STRIPE_LINK_HERE)** 👈
"""
    return premium_report