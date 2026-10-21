import streamlit as st
import os
from agent_engine import generate_premium_ai_report, DOMESTIC_LEAGUES

# Note: We import DOMESTIC_LEAGUES from agent_engine, but for clarity, 
# ensure it's defined in agent_engine.py or imported correctly. 
# For this snippet, I'll define it here to match your exact request.

DOMESTIC_LEAGUES = {
    "🇺🇸 USA": ["basketball_nba", "basketball_wnba", "basketball_ncaab"],
    "🇪🇸 Spain": ["basketball_spain_acb"],
    "🇩🇪 Germany": ["basketball_germany_bbl"],
    "🇫🇷 France": ["basketball_france_lnb"],
    "🇮🇹 Italy": ["basketball_italy_serie_a"],
    "🇬🇷 Greece": ["basketball_greek_basket_league"],
    "🇹🇷 Turkey": ["basketball_turkey_bsl"],
    "🇷🇺 Russia": ["basketball_russia_vbl"],
    "🇨🇳 China": ["basketball_china_cba"],
    "🇦🇺 Australia": ["basketball_australia_nbl"],
    "🇦🇷 Argentina": ["basketball_argentina_lnb"],
    "🇧🇷 Brazil": ["basketball_brazil_nbb"],
    "🇱🇹 Lithuania": ["basketball_lithuania_lkl"],
    "🇮🇱 Israel": ["basketball_israel_winner_league"],
    "🇵🇱 Poland": ["basketball_poland_plk"],
    "🇷🇸 Serbia": ["basketball_serbia_kls"],
    "🇭🇷 Croatia": ["basketball_croatia_aba_liga"],
    "🇲🇽 Mexico": ["basketball_mexico_lnbp"],
    "🇸🇰 Slovakia": ["basketball_slovakia_extraliga"],
    "🇨🇿 Czech Republic": ["basketball_czech_nbl"],
    "🇯🇵 Japan": ["basketball_japan_b1"],
    "🇩🇰 Denmark": ["basketball_denmark_basketligaen"],
    "🇨🇱 Chile": ["basketball_chile_lnb"],
    "🇨🇭 Switzerland": ["basketball_switzerland_sbl"],
    "🇸🇪 Sweden": ["basketball_sweden_basketligan"],
    "🇦🇹 Austria": ["basketball_austria_bundesliga"],
    "🇦🇱 Albania": ["basketball_albania_superliga"],
    "🇧🇪 Belgium": ["basketball_belgium_bsl"],
    "🇬🇧 Great Britain": ["basketball_bbl"],
    "🇵🇷 Puerto Rico": ["basketball_puerto_rico_bsn"],
    "🇲🇾 Malaysia": ["basketball_malaysia_mbl"],
    "🇺🇬 Uganda": ["basketball_uganda_nbl"],
    "🇺🇾 Uruguay": ["basketball_uruguay_lub"],
    "🇳🇿 New Zealand": ["basketball_new_zealand_nznbl"],
    "🇭🇺 Hungary": ["basketball_hungary_nba"],
    "🇳🇱 Netherlands": ["basketball_netherlands_dbl"],
    "🇰🇪 Kenya": ["basketball_kenya_premier_league"],
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England": ["basketball_england_nbl"],
    "🇲🇹 Malta": ["basketball_malta_nbl"],
    "🇽🇰 Kosovo": ["basketball_kosovo_superliga"],
    "🇵🇹 Portugal": ["basketball_portugal_lpb"],
    "🇻🇳 Vietnam": ["basketball_vietnam_vba"],
}

st.set_page_config(page_title="Owenzo AI Betting Empire", layout="wide", page_icon="🏀")

# --- SIDEBAR: ADMIN & API SETTINGS ---
st.sidebar.header("⚙️ System Configuration")
st.sidebar.markdown("*(Uses 100% Free Open-Source LLMs via OpenRouter)*")

odds_api_key = st.sidebar.text_input("The Odds API Key", type="password", value=os.getenv("ODDS_API_KEY", ""))
openrouter_key = st.sidebar.text_input("OpenRouter API Key (Optional for free tier)", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 E-Commerce Settings")
stripe_link = st.sidebar.text_input("Premium Product Stripe Link", value="https://buy.stripe.com/YOUR_LINK_HERE")

# --- MAIN DASHBOARD ---
st.title("🏀 Owenzo AI Autonomous Betting Factory")
st.markdown("Generate high-value, AI-researched betting reports to sell to your audience. **100% Automated. $0 Inference Cost.**")

tab1, tab2 = st.tabs(["🏭 Generate Digital Product", "💼 Monetization & Analytics"])

with tab1:
    st.header("Autonomous Report Generator")
    st.info("The AI will fetch live odds, scrape real-time injury/news data, and generate a premium, sellable markdown report.")
    
    col1, col2 = st.columns(2)
    with col1:
        countries = list(DOMESTIC_LEAGUES.keys())
        selected_country = st.selectbox("🌍 Select Country", countries)
    
    with col2:
        available_leagues = DOMESTIC_LEAGUES.get(selected_country, [])
        league_options = {league.replace("basketball_", "").replace("_", " ").title(): league for league in available_leagues}
        selected_league_display = st.selectbox("🏆 Select League", list(league_options.keys()))
        selected_league_key = league_options[selected_league_display]

    if st.button("🚀 Generate Premium AI Report", type="primary", use_container_width=True):
        if not odds_api_key:
            st.error("⚠️ Please enter The Odds API Key in the sidebar.")
        else:
            with st.spinner("🤖 AI Agents are fetching odds, scraping injury reports, and analyzing sharp money..."):
                try:
                    # Set env var for the engine to use
                    if openrouter_key:
                        os.environ["OPENROUTER_API_KEY"] = openrouter_key
                    
                    report = generate_premium_ai_report(
                        country=selected_country,
                        league_name=selected_league_display,
                        sport_key=selected_league_key,
                        odds_api_key=odds_api_key
                    )
                    st.session_state['latest_report'] = report
                    st.success("✅ Premium Digital Product Generated Successfully!")
                except Exception as e:
                    st.error(f"❌ Agent Error: {str(e)}")

    # Display the generated product
    if 'latest_report' in st.session_state:
        st.markdown("---")
        st.subheader("📄 Preview of Generated Digital Product")
        
        # E-COMMERCE GATING SIMULATION
        report_text = st.session_state['latest_report']
        parts = report_text.split("### 🔒 **UNLOCK THE FULL 10-LEG ACCUMULATOR SLIP**")
        
        # Show the free teaser
        st.markdown(parts[0])
        
        # Blur/Gate the premium section
        st.markdown("""
        <style>
        .blurred-box {
            filter: blur(5px);
            user-select: none;
            pointer-events: none;
            background: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin-top: 10px;
        }
        .cta-box {
            text-align: center;
            margin-top: -40px;
            padding: 20px;
            background: rgba(255,255,255,0.9);
            border-radius: 10px;
            border: 2px dashed #ff4b4b;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="blurred-box">' + parts[1] + '</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cta-box">
            <h3>🔒 This is how your customers will see it</h3>
            <p>They get the free teaser, but must pay to unlock the full AI analysis.</p>
            <a href="{stripe_link}" target="_blank">
                <button style="background-color: #ff4b4b; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer;">
                    💳 Simulate Customer Buying Full Report ($19.99)
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("💼 E-Commerce Monetization Strategy")
    st.markdown("""
    ### How this makes money automatically:
    1. **The Product:** The AI generates a unique, data-backed report every day (no manual work).
    2. **The Distribution:** You post the "Free Teaser" (the top half of the report) on Twitter, Reddit, or a free Telegram/Discord channel.
    3. **The Gate:** The link to the "Full 10-Leg Slip" goes to a Stripe Payment Page or Gumroad.
    4. **The Fulfillment:** When they pay, Stripe automatically emails them the full markdown report (or grants them access to a private page).
    
    ### Next Steps to Go Live:
    - [ ] Get a free API key from [The Odds API](https://the-odds-api.com/)
    - [ ] Get a free API key from [OpenRouter](https://openrouter.ai/) (to use free open-source models like Mistral 7B).
    - [ ] Create a $19.99 product on [Gumroad](https://gumroad.com/) or [Stripe Payment Links](https://dashboard.stripe.com/payment-links).
    - [ ] Deploy this Streamlit app to [Streamlit Community Cloud](https://share.streamlit.io/) (100% Free Hosting).
    """)