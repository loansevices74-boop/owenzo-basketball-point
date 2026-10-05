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
    "🇷 Serbia": ["basketball_serbia_kls"],
    "🇭🇷 Croatia": ["basketball_croatia_aba_liga"],
    "🇲🇽 Mexico": ["basketball_mexico_lnbp"],
    "🇸 Slovakia": ["basketball_slovakia_extraliga"],
    "🇨🇿 Czech Republic": ["basketball_czech_nbl"],
    "🇯🇵 Japan": ["basketball_japan_b1"],
    "🇩🇰 Denmark": ["basketball_denmark_basketligaen"],
    "🇨🇱 Chile": ["basketball_chile_lnb"],
    "🇨🇭 Switzerland": ["basketball_switzerland_sbl"],
    "🇸 Sweden": ["basketball_sweden_basketligan"],
    "🇦🇹 Austria": ["basketball_austria_bundesliga"],
    "🇦🇱 Albania": ["basketball_albania_superliga"],
    "🇪 Belgium": ["basketball_belgium_bsl"],
    "🇬🇧 Great Britain": ["basketball_bbl"],
    "🇷 Puerto Rico": ["basketball_puerto_rico_bsn"],
    "🇲🇾 Malaysia": ["basketball_malaysia_mbl"],
    "🇺🇬 Uganda": ["basketball_uganda_nbl"],
    "🇺🇾 Uruguay": ["basketball_uruguay_lub"],
    "🇳🇿 New Zealand": ["basketball_new_zealand_nznbl"],
    "🇭 Hungary": ["basketball_hungary_nba"],
    "🇳🇱 Netherlands": ["basketball_netherlands_dbl"],
    "🇪 Kenya": ["basketball_kenya_premier_league"],
    "󠁧󠁢󠁥󠁮󠁧󠁿 England": ["basketball_england_nbl"],
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

# ============ SESSION STATE ============
if 'fetched_matches' not in st.session_state:
    st.session_state.fetched_matches = []
if 'selected_match_ids' not in st.session_state:
    st.session_state.selected_match_ids = set()

# ============ MAIN APP ============
tabs = st.tabs(["🌍 Domestic Leagues", "📸 Screenshot Scanner", "🎫 Slip Generator"])

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
                    
                    # Store fetched matches in session state
                    st.session_state.fetched_matches = []
                    st.session_state.selected_match_ids = set()
                    
                    for idx, match in enumerate(odds_data):
                        home = match['home_team']
                        away = match['away_team']
                        match_id = f"{home}_vs_{away}_{idx}"
                        
                        bookmakers = match.get('bookmakers', [])
                        home_score = 82.0
                        away_score = 78.0
                        home_odds = 1.85
                        away_odds = 1.95
                        
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
                                            home_odds = price
                                        elif away.lower() in name.lower():
                                            away_score = 80 + (2.5 - price) * 15
                                            away_odds = price
                        
                        preds = calculate_bounds(home_score, away_score)
                        
                        # Store match data
                        match_data = {
                            "id": match_id,
                            "match": f"{home} vs {away}",
                            "home": home,
                            "away": away,
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                            "expected_total": preds['expected_fulltime'],
                            "ft_over_70": preds['ft_over_70'],
                            "ft_under_70": preds['ft_under_70'],
                            "confidence": 75
                        }
                        st.session_state.fetched_matches.append(match_data)
                    
                    st.rerun()
                else:
                    st.warning("No matches found for this league. Games may not be scheduled yet.")
        else:
            st.warning(f"No domestic leagues available for {selected_country}.")
    else:
        st.info("👉 Enter your Odds API key in the sidebar to fetch real-time odds.")
    
    # Display matches with checkboxes AFTER fetching
    if st.session_state.fetched_matches:
        st.write("---")
        st.subheader("📋 Select Matches for Your Slip")
        
        for match_data in st.session_state.fetched_matches:
            match_id = match_data['id']
            
            # Checkbox with unique key
            is_selected = st.checkbox(
                f"Select: {match_data['match']}",
                key=f"checkbox_{match_id}",
                value=match_id in st.session_state.selected_match_ids
            )
            
            # Update session state based on checkbox
            if is_selected:
                st.session_state.selected_match_ids.add(match_id)
            else:
                st.session_state.selected_match_ids.discard(match_id)
            
            # Show match details in expander
            with st.expander(f" {match_data['match']} - View Details"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Home Team", match_data['home'])
                col1.caption(f"Odds: {match_data['home_odds']:.2f}")
                col2.metric("Away Team", match_data['away'])
                col2.caption(f"Odds: {match_data['away_odds']:.2f}")
                col3.metric("Expected Total", match_data['expected_total'])
                st.write(f"**FT Over 70%:** {match_data['ft_over_70']} | **FT Under 70%:** {match_data['ft_under_70']}")
        
        st.info(f"✅ {len(st.session_state.selected_match_ids)} matches selected for slip")

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

# TAB 2: Slip Generator with Actual Matches
with tabs[2]:
    st.header("🎫 Accumulator Slip Generator")
    
    # Get selected matches from session state
    selected_matches = [m for m in st.session_state.fetched_matches if m['id'] in st.session_state.selected_match_ids]
    
    if selected_matches:
        st.success(f"✅ {len(selected_matches)} matches selected from Tab 0")
        
        target_confidence = st.slider("Target Confidence (%)", min_value=70, max_value=95, value=75)
        
        st.write("---")
        st.subheader("📋 Your Accumulator Slip")
        
        combined_odds = 1.0
        for idx, match in enumerate(selected_matches, 1):
            st.write(f"**{idx}. {match['match']}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Home Odds", f"{match['home_odds']:.2f}")
            col2.metric("Away Odds", f"{match['away_odds']:.2f}")
            col3.metric("Expected Total", match['expected_total'])
            st.write(f"FT Over: {match['ft_over_70']} | FT Under: {match['ft_under_70']}")
            st.write("---")
            
            # Use average odds for accumulator calculation
            avg_odds = (match['home_odds'] + match['away_odds']) / 2
            combined_odds *= avg_odds
        
        st.subheader("📊 Accumulator Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Legs", len(selected_matches))
        col2.metric("Combined Odds", f"{combined_odds:.2f}x")
        col3.metric("Target Confidence", f"{target_confidence}%")
        
        if len(selected_matches) >= 2:
            st.success(f"✅ {len(selected_matches)}-leg accumulator with {combined_odds:.2f}x combined odds!")
            st.info(f"💡 If you bet $10, potential return: ${10 * combined_odds:.2f}")
        else:
            st.warning("Need at least 2 matches for an accumulator. Select more matches in Tab 0.")
    else:
        st.warning("⚠️ No matches selected yet. Go to Tab 0, fetch matches, and use the checkboxes to select matches for your slip.")
        
        target_confidence = st.slider("Target Confidence (%)", min_value=70, max_value=95, value=75)
        num_legs = st.number_input("Number of Legs", min_value=2, max_value=25, value=5)
        
        if st.button("Generate Slip"):
            st.info("Please select matches from Tab 0 first!")