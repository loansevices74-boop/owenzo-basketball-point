import streamlit as st
import os
from agent_engine import generate_global_daily_report

# ============ DOMESTIC LEAGUES MAPPING ============
DOMESTIC_LEAGUES = {
    "🇸 USA": ["basketball_nba", "basketball_wnba", "basketball_ncaab"],
    "🇪🇸 Spain": ["basketball_spain_acb"],
    "🇩🇪 Germany": ["basketball_germany_bbl"],
    "🇫🇷 France": ["basketball_france_lnb"],
    "🇮🇹 Italy": ["basketball_italy_serie_a"],
    "🇬🇷 Greece": ["basketball_greek_basket_league"],
    "🇷 Turkey": ["basketball_turkey_bsl"],
    "🇷 Russia": ["basketball_russia_vbl"],
    "🇨🇳 China": ["basketball_china_cba"],
    "🇦🇺 Australia": ["basketball_australia_nbl"],
    "🇦🇷 Argentina": ["basketball_argentina_lnb"],
    "🇧🇷 Brazil": ["basketball_brazil_nbb"],
    "🇱 Lithuania": ["basketball_lithuania_lkl"],
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
    "🇸 Sweden": ["basketball_sweden_basketligan"],
    "🇦🇹 Austria": ["basketball_austria_bundesliga"],
    "🇦🇱 Albania": ["basketball_albania_superliga"],
    "🇧🇪 Belgium": ["basketball_belgium_bsl"],
    "🇬🇧 Great Britain": ["basketball_bbl"],
    "🇷 Puerto Rico": ["basketball_puerto_rico_bsn"],
    "🇲 Malaysia": ["basketball_malaysia_mbl"],
    "🇺🇬 Uganda": ["basketball_uganda_nbl"],
    "🇺🇾 Uruguay": ["basketball_uruguay_lub"],
    "🇳 New Zealand": ["basketball_new_zealand_nznbl"],
    "🇭🇺 Hungary": ["basketball_hungary_nba"],
    "🇳🇱 Netherlands": ["basketball_netherlands_dbl"],
    "🇰🇪 Kenya": ["basketball_kenya_premier_league"],
    "🏴󠁢󠁥󠁧󠁿 England": ["basketball_england_nbl"],
    "🇲🇹 Malta": ["basketball_malta_nbl"],
    "🇽 Kosovo": ["basketball_kosovo_superliga"],
    "🇵🇹 Portugal": ["basketball_portugal_lpb"],
    "🇳 Vietnam": ["basketball_vietnam_vba"],
}

st.set_page_config(page_title="Owenzo AI Global Betting Factory", layout="wide", page_icon="🏀")

# --- SIDEBAR ---
st.sidebar.header("️ System Configuration")
st.sidebar.markdown("*(Uses 100% Free Open-Source LLMs via OpenRouter)*")

odds_api_key = st.sidebar.text_input("The Odds API Key", type="password", value=os.getenv("ODDS_API_KEY", ""))
openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 E-Commerce Settings")
stripe_link = st.sidebar.text_input("Premium Product Stripe Link", value="https://buy.stripe.com/YOUR_LINK_HERE")

# --- MAIN DASHBOARD ---
st.title(" Owenzo AI Global Betting Factory")
st.markdown("Autonomously scans **all 40+ domestic leagues**, filters for high-value variance, and generates a global daily report. **100% Automated. $0 Inference Cost.**")

tab1, tab2 = st.tabs(["🏭 Generate Global Daily Report", " Monetization Strategy"])

with tab1:
    st.header("Global Autonomous Report Generator")
    st.info("The AI will scan all leagues, use math to filter the best matches, and use CrewAI to generate picks. This takes 2-4 minutes.")
    
    if st.button("🚀 Scan All Leagues & Generate Report", type="primary", use_container_width=True):
        if not odds_api_key:
            st.error("⚠️ Please enter The Odds API Key in the sidebar.")
        else:
            if openrouter_key:
                os.environ["OPENROUTER_API_KEY"] = openrouter_key
                
            with st.status("🤖 AI Agents are scanning the globe...", expanded=True) as status:
                st.write("📡 Fetching odds from 40+ leagues...")
                try:
                    report = generate_global_daily_report(DOMESTIC_LEAGUES, odds_api_key)
                    st.session_state['latest_global_report'] = report
                    status.update(label="✅ Global Report Generated Successfully!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Error generating report", state="error", expanded=True)
                    st.error(f"Agent Error: {str(e)}")

    # Display the generated product
    if 'latest_global_report' in st.session_state:
        st.markdown("---")
        st.subheader("📄 Preview of Generated Global Digital Product")
        
        report_text = st.session_state['latest_global_report']
        parts = report_text.split("### 🔒 **UNLOCK THE FULL DEEP-DIVE ANALYSIS & LIVE TRACKING**")
        
        # Show the free teaser
        st.markdown(parts[0])
        
        # Blur/Gate the premium section
        st.markdown("""
        <style>
        .blurred-box {
            filter: blur(6px);
            user-select: none;
            pointer-events: none;
            background: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin-top: 10px;
            opacity: 0.5;
        }
        .cta-box {
            text-align: center;
            margin-top: -40px;
            padding: 20px;
            background: rgba(255,255,255,0.95);
            border-radius: 10px;
            border: 2px dashed #ff4b4b;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="blurred-box">' + parts[1] + '</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cta-box">
            <h3>🔒 Premium Content Locked</h3>
            <p>Free users see the top picks. Paid users get the deep-dive analysis, staking guides, and live alerts.</p>
            <a href="{stripe_link}" target="_blank">
                <button style="background-color: #ff4b4b; color: white; padding: 12px 24px; border: none; border-radius: 5px; font-size: 16px; font-weight: bold; cursor: pointer;">
                    💳 Unlock Full Global Report ($19.99)
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("💼 E-Commerce Monetization Strategy")
    st.markdown("""
    ### How this makes money automatically:
    1. **The Product:** Every morning, this app scans 40+ leagues and generates a unique, data-backed global report.
    2. **The Distribution:** You copy the "Free Teaser" (the top half) and post it on Twitter/X, Reddit (r/sportsbook), or a free Discord channel.
    3. **The Gate:** The blurred section shows them exactly what they are missing. The button links to your Stripe/Gumroad checkout.
    4. **The Fulfillment:** When they pay, you automatically email them the full un-blurred Markdown text, or grant them access to a private "Premium" page on this same Streamlit app.
    
    ### Next Steps to Go Live:
    - [ ] Get a free API key from [The Odds API](https://the-odds-api.com/)
    - [ ] Get a free API key from [OpenRouter](https://openrouter.ai/) (to use free open-source models like Mistral 7B).
    - [ ] Create a $19.99 product on [Gumroad](https://gumroad.com/) or [Stripe Payment Links](https://dashboard.stripe.com/payment-links).
    - [ ] Deploy this Streamlit app to [Streamlit Community Cloud](https://share.streamlit.io/) (100% Free Hosting).
    """)