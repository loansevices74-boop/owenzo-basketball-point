import streamlit as st
import requests
from PIL import Image
import io
import base64
from typing import List, Dict, Any
from openai import OpenAI

st.set_page_config(page_title="Domestic Basketball Leagues Odds", layout="wide", page_icon="")
st.title("🏀 Domestic Country Leagues - Odds & Predictions")

# --- Sidebar API Keys ---
st.sidebar.header("🔑 API Credentials")
odds_api_key = st.sidebar.text_input("The Odds API Key", value=st.secrets.get("ODDS_API_KEY", ""), type="password")
qwen_api_key = st.sidebar.text_input("Qwen API Key (for Screenshots)", value=st.secrets.get("QWEN_API_KEY", ""), type="password")
qwen_base_url = st.sidebar.text_input("Qwen Base URL", value=st.secrets.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))

# ============ DOMESTIC LEAGUES MAPPING ============
DOMESTIC_LEAGUES = {
    "🇺🇸 USA": ["basketball_nba", "basketball_wnba", "basketball_ncaab"],
    "🇪 Spain": ["basketball_spain_acb"],
    "🇩🇪 Germany": ["basketball_germany_bbl"],
    "🇷 France": ["basketball_france_lnb"],
    "🇮🇹 Italy": ["basketball_italy_serie_a"],
    "🇬🇷 Greece": ["basketball_greek_basket_league"],
    "🇹🇷 Turkey": ["basketball_turkey_bsl"],
    "🇷 Russia": ["basketball_russia_vbl"],
    "🇨🇳 China": ["basketball_china_cba"],
    "🇺 Australia": ["basketball_australia_nbl"],
    "🇦 Argentina": ["basketball_argentina_lnb"],
    "🇧🇷 Brazil": ["basketball_brazil_nbb"],
    "🇱🇹 Lithuania": ["basketball_lithuania_lkl"],
    "🇮🇱 Israel": ["basketball_israel_winner_league"],
    "🇵🇱 Poland": ["basketball_poland_plk"],
    "🇷🇸 Serbia": ["basketball_serbia_kls"],
    "🇭🇷 Croatia": ["basketball_croatia_aba_liga"],
    "🇲🇽 Mexico": ["basketball_mexico_lnbp"],
    "🇰 Slovakia": ["basketball_slovakia_extraliga"],
    "🇨 Czech Republic": ["basketball_czech_nbl"],
    "🇯🇵 Japan": ["basketball_japan_b1"],
    "🇰 Denmark": ["basketball_denmark_basketligaen"],
    "🇨 Chile": ["basketball_chile_lnb"],
    "🇨🇭 Switzerland": ["basketball_switzerland_sbl"],
    "🇸🇪 Sweden": ["basketball_sweden_basketligan"],
    "🇦🇹 Austria": ["basketball_austria_bundesliga"],
    "🇦🇱 Albania": ["basketball_albania_superliga"],
    "🇪 Belgium": ["basketball_belgium_bsl"],
    "🇬🇧 Great Britain": ["basketball_bbl"],
    "🇵🇷 Puerto Rico": ["basketball_puerto_rico_bsn"],
    "🇾 Malaysia": ["basketball_malaysia_mbl"],
    "🇺🇬 Uganda": ["basketball_uganda_nbl"],
    "🇺🇾 Uruguay": ["basketball_uruguay_lub"],
    "🇳🇿 New Zealand": ["basketball_new_zealand_nznbl"],
    "🇭🇺 Hungary": ["basketball_hungary_nba"],
    "🇳🇱 Netherlands": ["basketball_netherlands_dbl"],
    "🇰🇪 Kenya": ["basketball_kenya_premier_league"],
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England": ["basketball_england_nbl"],
    "🇲🇹 Malta": ["basketball_malta_nbl"],
    "🇽🇰 Kosovo": ["basketball_kosovo_superliga"],
    "🇵 Portugal": ["basketball_portugal_lpb"],
    "🇻🇳 Vietnam": ["basketball_vietnam_vba"],
}

# ============ API FUNCTIONS ============
def get_odds(api_key: str, sport_key: str, regions: str = "eu,us", markets: str = "h2h,totals") -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions={regions}&markets={markets}"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()
    return []

def calculate_bounds(avg_home_score: float, avg_away_score: float) -> Dict[str, Any]:
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

# ============ SESSION STATE INITIALIZATION ============
if 'fetched_matches' not in st.session_state:
    st.session_state.fetched_matches = []
if 'selected_league_name' not in st.session_state:
    st.session_state.selected_league_name = ""

# ============ MAIN APP ============
tabs = st.tabs(["🌍 Domestic Leagues", "📸 Screenshot Scanner", " Slip Generator"])

# TAB 0: Domestic Leagues by Country
with tabs[0]:
    st.header("🌍 Select Country & League")
    
    if odds_api_key:
        countries = list(DOMESTIC_LEAGUES.keys())
        selected_country = st.selectbox("🌍 Select Country", countries)
        
        available_leagues = DOMESTIC_LEAGUES.get(selected_country, [])
        
        if available_leagues:
            league_options = {league.replace("basketball_", "").replace("_", " ").title(): league for league in available_leagues}
            selected_league_display = st.selectbox("🏆 Select League", list(league_options.keys()))
            selected_league_key = league_options[selected_league_display]
            
            if st.button("Fetch Matches & Predict", type="primary"):
                with st.spinner(f"Fetching odds for {selected_league_display}..."):
                    odds_data = get_odds(odds_api_key, selected_league_key)
                
                if odds_data:
                    st.success(f"✅ Found {len(odds_data)} matches in {selected_league_display}")
                    st.write("---")
                    
                    # Store matches in session state for Tab 2
                    st.session_state.fetched_matches = []
                    st.session_state.selected_league_name = selected_league_display
                    
                    for match in odds_data:
                        home = match['home_team']
                        away = match['away_team']
                        
                        bookmakers = match.get('bookmakers', [])
                        home_score = 82.0
                        away_score = 78.0
                        best_home_odds = 1.5
                        best_away_odds = 1.5
                        
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
                                            best_home_odds = price
                                        elif away.lower() in name.lower():
                                            away_score = 80 + (2.5 - price) * 15
                                            best_away_odds = price
                        
                        preds = calculate_bounds(home_score, away_score)
                        
                        # Determine the best pick
                        if home_score > away_score:
                            pick = f"{home} ML"
                            pick_odds = best_home_odds
                            confidence = "HIGH" if home_score - away_score > 5 else "MEDIUM"
                        else:
                            pick = f"{away} ML"
                            pick_odds = best_away_odds
                            confidence = "HIGH" if away_score - home_score > 5 else "MEDIUM"
                        
                        # Store match data for slip generator
                        st.session_state.fetched_matches.append({
                            "match": f"{home} vs {away}",
                            "pick": pick,
                            "odds": pick_odds,
                            "confidence": confidence,
                            "expected_total": preds['expected_total'],
                            "ft_over": preds['ft_over_70'],
                            "ft_under": preds['ft_under_70']
                        })
                        
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
                    st.warning("No matches found for this league. Games may not be scheduled yet.")
        else:
            st.warning(f"No domestic leagues available for {selected_country}.")
    else:
        st.info("👉 Enter your Odds API key in the sidebar to fetch real-time odds.")

# TAB 1: Screenshot Scanner
with tabs[1]:
    st.header("📸 Upload Betting Screenshot")
    uploaded_file = st.file_uploader("Upload (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

    if uploaded_file and qwen_api_key:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Screenshot", use_container_width=True)

        if st.button("Scan with Qwen Vision"):
            with st.spinner("Analyzing screenshot..."):
                result = scan_image_with_qwen(qwen_api_key, qwen_base_url, img)
                st.markdown("### Extracted Data")
                st.markdown(result)
    elif uploaded_file and not qwen_api_key:
        st.warning("Please enter your Qwen API Key in the sidebar.")

# TAB 2: REAL Slip Generator
with tabs[2]:
    st.header("🎫 Accumulator Slip Generator")
    
    # Check if we have matches from Tab 0
    if not st.session_state.fetched_matches:
        st.warning("⚠️ No matches loaded yet! Go to **Tab 0 (Domestic Leagues)**, select a league, and click 'Fetch Matches & Predict' first.")
    else:
        st.success(f"✅ {len(st.session_state.fetched_matches)} matches loaded from {st.session_state.selected_league_name}")
        
        target_confidence = st.slider("Target Confidence (%)", min_value=70, max_value=95, value=75)
        num_legs = st.number_input("Number of Legs", min_value=2, max_value=25, value=5)
        
        # Filter matches by confidence
        high_confidence_matches = [m for m in st.session_state.fetched_matches if m['confidence'] == 'HIGH']
        
        if st.button("Generate Slip", type="primary"):
            # Select top picks
            selected_picks = high_confidence_matches[:num_legs]
            
            # If not enough HIGH confidence, add MEDIUM
            if len(selected_picks) < num_legs:
                medium_confidence = [m for m in st.session_state.fetched_matches if m['confidence'] == 'MEDIUM']
                for m in medium_confidence:
                    if m not in selected_picks:
                        selected_picks.append(m)
                    if len(selected_picks) >= num_legs:
                        break
            
            # Calculate combined odds
            combined_odds = 1.0
            for pick in selected_picks:
                combined_odds *= pick['odds']
            
            # Display the slip
            st.markdown("---")
            st.subheader(f"🎫 Your {num_legs}-Leg Accumulator Slip")
            st.markdown(f"**League:** {st.session_state.selected_league_name}")
            st.markdown(f"**Combined Odds:** {combined_odds:.2f}")
            st.markdown(f"**Potential Return (on $10 stake):** ${(combined_odds * 10):.2f}")
            st.markdown("---")
            
            # Create a table of picks
            slip_data = []
            for i, pick in enumerate(selected_picks, 1):
                slip_data.append({
                    "Leg #": i,
                    "Match": pick['match'],
                    "Pick": pick['pick'],
                    "Odds": f"{pick['odds']:.2f}",
                    "Confidence": pick['confidence']
                })
            
            import pandas as pd
            slip_df = pd.DataFrame(slip_data)
            st.table(slip_df)
            
            st.markdown("---")
            st.info("💡 **Tip:** Higher combined odds = higher risk. Consider reducing legs for safer bets.")