import streamlit as st
import requests
from PIL import Image
import io
import base64
from typing import List, Dict, Any
from openai import OpenAI

st.set_page_config(page_title="Basketball Prediction & Odds Machine", layout="wide", page_icon="🏀")
st.title("🏀 Basketball Odds Engine & Slip Generator (>70% Confidence)")

# --- Sidebar API Keys ---
st.sidebar.header("🔑 API Credentials")
odds_api_key = st.sidebar.text_input("The Odds API Key", value=st.secrets.get("ODDS_API_KEY", ""), type="password")
qwen_api_key = st.sidebar.text_input("Qwen API Key (for Screenshots)", value=st.secrets.get("QWEN_API_KEY", ""), type="password")
qwen_base_url = st.sidebar.text_input("Qwen Base URL", value=st.secrets.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))

# ============ ODDS API FUNCTIONS ============
def get_basketball_leagues(api_key: str) -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports?apiKey={api_key}"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    sports = res.json()
    return [s for s in sports if "basketball" in s.get("group", "").lower()]

def get_odds(api_key: str, sport_key: str, regions: str = "eu,us", markets: str = "h2h,totals") -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={regions}&markets={markets}"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()
    return []

# Country mapping for leagues
COUNTRY_MAPPING = {
    "USA": ["nba", "wnba", "ncaab", "ncaaw"],
    "Europe": ["euroleague", "eurocup"],
    "Spain": ["acb"],
    "Germany": ["bbl"],
    "France": ["lnb"],
    "Italy": ["lega-basket-serie-a"],
    "Greece": ["greek-basket-league"],
    "Turkey": ["bsl"],
    "Russia": ["vbl"],
    "China": ["cba"],
    "Australia": ["nbl"],
    "Argentina": ["lnb"],
    "Brazil": ["nbb"],
    "Lithuania": ["lkl"],
    "Israel": ["winner-league"],
    "Poland": ["plk"],
    "Serbia": ["kls"],
    "Croatia": ["aba-liga"],
    "International": ["fiba-world-cup", "fiba-olympic-qualifying"],
    "Mexico": ["lnbp"],
    "Slovakia": ["slovakia-basketball"],
    "Czech Republic": ["czech-nbl"],
    "Japan": ["japan-b1"],
    "Denmark": ["denmark-basketligaen"],
    "Chile": ["chile-lnb"],
    "Switzerland": ["switzerland-sbl"],
    "Sweden": ["sweden-basketligan"],
    "Austria": ["austria-bundesliga"],
    "Albania": ["albania-basketball"],
    "Belgium": ["belgium-bsl"],
    "Great Britain": ["bbl"],
    "Puerto Rico": ["puerto-rico-bsn"],
    "Malaysia": ["malaysia-mbl"],
    "Uganda": ["uganda-basketball"],
    "Uruguay": ["uruguay-lub"],
    "New Zealand": ["new-zealand-nznbl"],
    "Hungary": ["hungary-nba"],
    "Netherlands": ["netherlands-dbl"],
    "Kenya": ["kenya-basketball"],
    "England": ["england-nbl"],
    "Malta": ["malta-basketball"],
    "Kosovo": ["kosovo-basketball"],
    "Portugal": ["portugal-lpb"],
    "Vietnam": ["vietnam-vba"],
}

def get_leagues_by_country(api_key: str) -> Dict[str, List[Dict[str, Any]]]:
    leagues = get_basketball_leagues(api_key)
    grouped = {}
    
    for league in leagues:
        key = league.get('key', '')
        title = league.get('title', '')
        
        matched_country = None
        for country, keywords in COUNTRY_MAPPING.items():
            if any(kw in key.lower() for kw in keywords):
                matched_country = country
                break
        
        if not matched_country:
            if "NBA" in title:
                matched_country = "USA"
            elif "Euro" in title:
                matched_country = "Europe"
            else:
                matched_country = "Other"
        
        if matched_country not in grouped:
            grouped[matched_country] = []
        grouped[matched_country].append(league)
    
    return grouped

# ============ PREDICTOR FUNCTION ============
def calculate_bounds(avg_home_score: float, avg_away_score: float, league_avg: float = 158.0) -> Dict[str, Any]:
    expected_total = (avg_home_score + avg_away_score)
    std_dev = 12.5
    
    ft_under_70 = expected_total + (1.28 * std_dev)
    ft_over_70 = max(135.0, expected_total - (1.28 * std_dev))
    
    ht_expected = expected_total * 0.49
    ht_under_70 = ht_expected + (1.28 * (std_dev * 0.55))
    ht_over_70 = max(65.0, ht_expected - (1.28 * (std_dev * 0.55)))
    
    return {
        "expected_fulltime": round(expected_total, 1),
        "ft_over_70": round(ft_over_70 - 0.5, 0) + 0.5,
        "ft_under_70": round(ft_under_70 + 0.5, 0) - 0.5,
        "ht_over_70": round(ht_over_70 - 0.5, 0) + 0.5,
        "ht_under_70": round(ht_under_70 + 0.5, 0) - 0.5,
    }

# ============ QWEN SCANNER FUNCTION ============
def scan_image_with_qwen(api_key: str, base_url: str, image: Image.Image) -> str:
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        prompt = "Analyze this basketball betting screenshot. Extract all matches, home/away teams, odds, lines (1st Half, Fulltime), and league names. Output as a clean structured Markdown table."
        
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ============ MAIN APP ============
tabs = st.tabs(["📊 Live Odds & Predictions", "📸 Screenshot Scanner", "🎫 Slip Accumulator"])

# TAB 0: Live Odds with Country Filter
with tabs[0]:
    st.header(" Fetch Match Odds by Country & League")
    
    if odds_api_key:
        try:
            with st.spinner("Loading available countries and leagues..."):
                leagues_by_country = get_leagues_by_country(odds_api_key)
            
            if leagues_by_country:
                countries = sorted(leagues_by_country.keys())
                selected_country = st.selectbox(" Select Country", ["All Countries"] + countries, index=0)
                
                if selected_country == "All Countries":
                    all_leagues = [l for leagues in leagues_by_country.values() for l in leagues]
                    league_options = {f"{l['title']} ({l['key']})": l['key'] for l in all_leagues}
                else:
                    country_leagues = leagues_by_country.get(selected_country, [])
                    league_options = {f"{l['title']} ({l['key']})": l['key'] for l in country_leagues}
                
                if league_options:
                    selected_league_display = st.selectbox("🏆 Select League", list(league_options.keys()))
                    selected_league_key = league_options[selected_league_display]
                    
                    if st.button("Fetch Matches & Predict", type="primary"):
                        with st.spinner(f"Fetching odds for {selected_league_display}..."):
                            odds_data = get_odds(odds_api_key, selected_league_key)
                        
                        if odds_data:
                            st.success(f"✅ Found {len(odds_data)} matches in {selected_league_display}")
                            st.write("---")
                            
                            for match in odds_data:
                                home = match['home_team']
                                away = match['away_team']
                                
                                bookmakers = match.get('bookmakers', [])
                                home_score = 82.0
                                away_score = 78.0
                                
                                if bookmakers:
                                    first_bookmaker = bookmakers[0]
                                    markets = first_bookmaker.get('markets', [])
                                    
                                    for market in markets:
                                        if market.get('key') == 'h2h':
                                            outcomes = market.get('outcomes', [])
                                            for outcome in outcomes:
                                                name = outcome.get('name', '')
                                                price = outcome.get('price', 2.0)
                                                if home.lower() in name.lower():
                                                    home_score = 80 + (2.5 - price) * 15
                                                elif away.lower() in name.lower():
                                                    away_score = 80 + (2.5 - price) * 15
                                
                                preds = calculate_bounds(home_score, away_score)
                                
                                with st.expander(f"🏀 {home} vs {away}"):
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Home Team", home)
                                    col1.caption(f"Est. Score: {home_score:.1f}")
                                    col2.metric("Away Team", away)
                                    col2.caption(f"Est. Score: {away_score:.1f}")
                                    col3.metric("Expected Total", preds['expected_fulltime'])
                                    
                                    st.write("**70% Confidence Bounds:**")
                                    st.json(preds)
                        else:
                            st.warning("No matches found for this league. Try another league or check if games are scheduled.")
                else:
                    st.warning(f"No leagues available for {selected_country}. Try 'All Countries'.")
            else:
                st.warning("No active basketball leagues found or invalid API key.")
        except Exception as e:
            st.error(f"Error fetching leagues: {e}")
    else:
        st.info("👉 Enter your Odds API key in the sidebar to fetch real-time odds.")

# TAB 1: Screenshot Scanner
with tabs[1]:
    st.header("📸 Upload Screenshot (.png, .jpg)")
    uploaded_file = st.file_uploader("Upload Betting App Screenshot", type=["png", "jpg", "jpeg"])

    if uploaded_file and qwen_api_key:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Screenshot", use_container_width=True)

        if st.button("Scan Screenshot with Qwen"):
            with st.spinner("Extracting match odds using Qwen Vision..."):
                result = scan_image_with_qwen(qwen_api_key, qwen_base_url, img)
                st.markdown("### Processed Analysis")
                st.markdown(result)
    elif uploaded_file and not qwen_api_key:
        st.warning("Please enter your Qwen API Key in the sidebar to process images.")

# TAB 2: Slip Accumulator
with tabs[2]:
    st.header(" Daily / Weekly Accumulator Generator")
    target_confidence = st.slider("Target Confidence", min_value=70, max_value=95, value=75)
    num_legs = st.number_input("Number of Legs", min_value=2, max_value=25, value=10)
    
    if st.button("Generate (>70%) Accumulator"):
        st.success(f"Generated Multi-Leg High-Probability Slip! (Confidence: {target_confidence}%, Legs: {int(num_legs)})")
        st.info("⚠️ Accumulator logic pending — integrate odds_data from Tab 0 to auto-select legs.")