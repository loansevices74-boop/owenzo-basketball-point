"""
Owenzõ Basketball Points — Streamlit UI
OCR + 70% Blessings Predictions Engine
Accepts CSV, PNG, JPG files
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

from src.config import LEAGUE_REGISTRY, LeagueProfile
from src.leagues import list_all_countries, list_all_leagues
from src.model import fit_model, predict, market_probabilities
from src.lines import evaluate_value
from src.staking import fractional_kelly, bankroll_simulator
from src.backtest import walk_forward_backtest
from src.utils import total_probabilities
from loaders.csv_loader import load_csv
from src.ocr_parser import process_image_to_dataframe

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Owenzõ Basketball Points",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    .stButton>button {
        background-color: #ADFF2F;
        color: #000000;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton>button:hover { background-color: #9ACD32; }
    h1, h2, h3 { color: #262730; }
    .blessings-banner {
        background: linear-gradient(135deg, #ADFF2F 0%, #7FFF00 50%, #32CD32 100%);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(173, 255, 47, 0.3);
    }
    .blessings-banner h2 { color: #000 !important; margin: 0; font-size: 2em; }
    .blessings-banner p { color: #000; margin: 10px 0 0 0; font-size: 1.1em; }
    .gold-pick { background-color: #FFF9C4 !important; }
    .silver-pick { background-color: #F5F5F5 !important; }
    .bronze-pick { background-color: #FFE0B2 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F2F6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ADFF2F;
        color: #000000;
    }
    .upload-box {
        border: 3px dashed #ADFF2F;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        background-color: #F9F9F9;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────
if "fitted_models" not in st.session_state:
    st.session_state.fitted_models = {}
if "vip_authenticated" not in st.session_state:
    st.session_state.vip_authenticated = False
if "ocr_games" not in st.session_state:
    st.session_state.ocr_games = []

# ─ Title ────────────────────────────────────────────────────────────────
st.title("🏀 Owenzõ Basketball Points")
st.caption("OCR-powered 70% Blessings Predictions Engine")

# ── Tabs ────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯 Blessings Predictor",
    "📊 Line Explorer",
    " Value & Combos",
    "📈 Backtest",
    "💰 Bankroll",
    "🌍 League Board",
    " VIP",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: BLESSINGS PREDICTOR (OCR + CSV)
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("🎯 70% Blessings Predictor")
    
    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Country / Region", ["All"] + list_all_countries())
    with col2:
        if country == "All":
            leagues = list_all_leagues()
        else:
            leagues = [k for k, v in LEAGUE_REGISTRY.items() if v.country == country]
        league_name = st.selectbox("League", leagues)

    st.markdown("---")
    
    # ── File Upload Section ──────────────────────────────────────────────
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.subheader("📂 Upload Game Data")
    st.write("📄 **CSV** — Direct data import")
    st.write("️ **PNG / JPG** — Screenshot OCR extraction")
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Choose a file (CSV, PNG, JPG, JPEG)",
        type=["csv", "png", "jpg", "jpeg"],
        help="Upload CSV for direct data or PNG/JPG screenshot for OCR extraction"
    )

    game_log = None

    if uploaded is not None:
        file_type = uploaded.name.split(".")[-1].lower()
        
        # ── CSV Processing ───────────────────────────────────────────────
        if file_type == "csv":
            try:
                game_log = load_csv(uploaded)
                st.success(f"✅ Loaded {len(game_log)} games from CSV")
            except Exception as e:
                st.error(f"CSV Error: {e}")
        
        # ── IMAGE OCR Processing ─────────────────────────────────────────
        elif file_type in ["png", "jpg", "jpeg"]:
            st.info("🔍 Processing image with OCR... This may take 10-30 seconds.")
            
            progress_bar = st.progress(0)
            progress_bar.progress(30, text="Reading image...")
            
            try:
                # Reset file pointer
                uploaded.seek(0)
                
                # Show the uploaded image
                st.image(uploaded, caption="Uploaded Screenshot", use_container_width=True)
                
                progress_bar.progress(60, text="Extracting text with OCR...")
                
                # Reset again for OCR
                uploaded.seek(0)
                ocr_df = process_image_to_dataframe(uploaded)
                
                progress_bar.progress(90, text="Parsing game data...")
                
                if len(ocr_df) > 0:
                    st.success(f"✅ OCR extracted {len(ocr_df)} game(s) from image")
                    st.dataframe(ocr_df, use_container_width=True)
                    
                    # Let user verify/edit extracted data
                    st.subheader("✏️ Verify Extracted Data")
                    st.caption("Edit any incorrect team names or scores before prediction")
                    
                    edited_df = st.data_editor(
                        ocr_df,
                        num_rows="dynamic",
                        use_container_width=True
                    )
                    
                    if st.button("Confirm & Predict", type="primary"):
                        game_log = edited_df
                        progress_bar.progress(100, text="Ready!")
                else:
                    st.warning("️ OCR could not extract game data from this image.")
                    st.info("Try a clearer screenshot with team names and scores visible.")
                    
                    # Fallback: Manual entry
                    st.subheader("✏️ Manual Entry (Fallback)")
                    with st.form("manual_entry"):
                        num = st.number_input("Number of games", 1, 10, 1)
                        games_data = []
                        for i in range(num):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                h = st.text_input(f"Home Team {i+1}", key=f"mh_{i}")
                                hs = st.number_input(f"Home Score {i+1}", 0, 200, 0, key=f"mhs_{i}")
                            with col2:
                                a = st.text_input(f"Away Team {i+1}", key=f"ma_{i}")
                                as_ = st.number_input(f"Away Score {i+1}", 0, 200, 0, key=f"mas_{i}")
                            with col3:
                                odds = st.number_input(f"Odds {i+1}", 1.01, 50.0, 1.90, step=0.05, key=f"mo_{i}")
                            games_data.append({
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "home_team": h,
                                "away_team": a,
                                "home_score": hs,
                                "away_score": as_,
                                "odds": odds,
                                "home_h1_score": 0,
                                "away_h1_score": 0,
                                "ft_line": None,
                            })
                        submit = st.form_submit_button("Load Games")
                        if submit:
                            valid = [g for g in games_data if g["home_team"] and g["away_team"]]
                            if valid:
                                game_log = pd.DataFrame(valid)
                                st.success(f"✅ Loaded {len(valid)} games")
                
            except Exception as e:
                st.error(f"OCR Error: {e}")
                st.info("Please try a clearer image or use CSV upload.")

    # ── RUN PREDICTIONS ──────────────────────────────────────────────────
    if game_log is not None and len(game_log) > 0:
        profile = LEAGUE_REGISTRY.get(league_name, LeagueProfile(league_name, "Custom", "csv"))
        
        with st.spinner(" Running prediction model..."):
            fitted = fit_model(game_log, profile)
            st.session_state.fitted_models[league_name] = fitted
            
            # Generate predictions for all matchups
            predictions = []
            teams = fitted["ratings"]["team"].tolist() if "ratings" in fitted else []
            
            # Create all possible matchups from the data
            unique_teams = list(set(game_log["home_team"].tolist() + game_log["away_team"].tolist()))
            
            for home in unique_teams:
                for away in unique_teams:
                    if home != away and home in teams and away in teams:
                        try:
                            pred = predict(home, away, fitted)
                            probs = market_probabilities(
                                pred,
                                ft_line=pred["ft_total"],
                                ht_line=pred["ht_total"]
                            )
                            
                            ft_over = probs["ft"]["over"]
                            ft_under = probs["ft"]["under"]
                            ht_over = probs["ht"]["over"]
                            ht_under = probs["ht"]["under"]
                            home_win = probs["combo"]["p_home_win"]
                            away_win = probs["combo"]["p_away_win"]
                            
                            # Find best pick
                            all_probs = {
                                "FT Over": ft_over,
                                "FT Under": ft_under,
                                "HT Over": ht_over,
                                "HT Under": ht_under,
                                "Home Win": home_win,
                                "Away Win": away_win,
                            }
                            best_pick = max(all_probs, key=all_probs.get)
                            best_prob = all_probs[best_pick]
                            
                            # Rating
                            if best_prob >= 0.80:
                                rating = "⭐⭐⭐ GOLD"
                            elif best_prob >= 0.75:
                                rating = "⭐⭐ SILVER"
                            elif best_prob >= 0.70:
                                rating = "⭐ BRONZE"
                            else:
                                rating = "— Standard"
                            
                            predictions.append({
                                "Matchup": f"{home} vs {away}",
                                "FT Total (μ)": f"{pred['ft_total']:.1f}",
                                "HT Total (μ)": f"{pred['ht_total']:.1f}",
                                "Best Pick": best_pick,
                                "Confidence": best_prob,
                                "Confidence %": f"{best_prob:.1%}",
                                "Rating": rating,
                                "FT Over %": f"{ft_over:.1%}",
                                "FT Under %": f"{ft_under:.1%}",
                                "HT Over %": f"{ht_over:.1%}",
                                "HT Under %": f"{ht_under:.1%}",
                                "Home Win %": f"{home_win:.1%}",
                                "Away Win %": f"{away_win:.1%}",
                            })
                        except Exception as e:
                            pass
            
            if predictions:
                pred_df = pd.DataFrame(predictions)
                
                # ── 200% BLESSINGS BANNER ─────────────────────────────────
                blessings_df = pred_df[pred_df["Confidence"] >= 0.70].copy()
                blessings_df = blessings_df.sort_values(by="Confidence", ascending=False)
                
                st.markdown(f'''
                <div class="blessings-banner">
                    <h2>🎁 200% BLESSINGS PICKS</h2>
                    <p>{len(blessings_df)} high-confidence picks (70%+) from {len(predictions)} total matchups</p>
                </div>
                ''', unsafe_allow_html=True)
                
                if len(blessings_df) > 0:
                    # Display blessings picks
                    display_cols = ["Matchup", "Best Pick", "Confidence %", "Rating", "FT Total (μ)", "HT Total (μ)", "FT Over %", "FT Under %"]
                    st.dataframe(
                        blessings_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Matchups", len(predictions))
                    with col2:
                        st.metric("Blessings Picks (70%+)", len(blessings_df))
                    with col3:
                        gold_count = len(blessings_df[blessings_df["Rating"].str.contains("GOLD")])
                        st.metric("GOLD Picks (80%+)", gold_count)
                    with col4:
                        avg_conf = blessings_df["Confidence"].mean()
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                    
                    # Download button
                    st.subheader("📥 Export Blessings Picks")
                    csv_export = blessings_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Blessings CSV",
                        data=csv_export,
                        file_name=f"owenzo_blessings_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        type="primary",
                    )
                    
                    # Visual breakdown
                    st.subheader("📊 Pick Distribution")
                    rating_counts = blessings_df["Rating"].value_counts()
                    fig = px.bar(
                        x=rating_counts.index,
                        y=rating_counts.values,
                        labels={"x": "Rating", "y": "Count"},
                        color=rating_counts.index,
                        color_discrete_map={
                            "⭐⭐⭐ GOLD": "#FFD700",
                            "⭐⭐ SILVER": "#C0C0C0",
                            "⭐ BRONZE": "#CD7F32",
                        },
                    )
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("⚠️ No picks above 70% confidence found.")
                    st.info("Showing top 5 closest picks:")
                    top5 = pred_df.nlargest(5, "Confidence")
                    st.dataframe(top5[display_cols], use_container_width=True)
                
                # Full predictions table (collapsible)
                with st.expander(" View All Predictions (including below 70%)"):
                    st.dataframe(pred_df[display_cols], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: LINE EXPLORER
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📊 Line Explorer")
    if st.session_state.fitted_models:
        league_name = st.selectbox("Select League", list(st.session_state.fitted_models.keys()))
        fitted = st.session_state.fitted_models[league_name]
        teams = fitted["ratings"]["team"].tolist() if "ratings" in fitted else []
        
        if len(teams) >= 2:
            col1, col2 = st.columns(2)
            with col1: home_team = st.selectbox("Home Team", teams, key="le_home")
            with col2: away_team = st.selectbox("Away Team", [t for t in teams if t != home_team], key="le_away")
            
            pred = predict(home_team, away_team, fitted)
            ft_lines = np.arange(pred["ft_total"] - 20, pred["ft_total"] + 20, 1.0)
            ft_probs_list = [total_probabilities(l, pred["ft_total"], pred["sigma_ft"]) for l in ft_lines]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ft_lines, y=[p["over"] for p in ft_probs_list], name="Over", line=dict(color="#ADFF2F")))
            fig.add_trace(go.Scatter(x=ft_lines, y=[p["under"] for p in ft_probs_list], name="Under", line=dict(color="#FF6B6B")))
            fig.update_layout(title="FT Total Probabilities by Line", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload data in the Predictor tab first.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: VALUE & COMBOS
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("💎 Value & Combos")
    if st.session_state.fitted_models:
        league_name = st.selectbox("League", list(st.session_state.fitted_models.keys()), key="vc_league")
        fitted = st.session_state.fitted_models[league_name]
        teams = fitted["ratings"]["team"].tolist() if "ratings" in fitted else []
        
        if len(teams) >= 2:
            col1, col2 = st.columns(2)
            with col1: home_team = st.selectbox("Home Team", teams, key="vc_home")
            with col2: away_team = st.selectbox("Away Team", [t for t in teams if t != home_team], key="vc_away")
            
            pred = predict(home_team, away_team, fitted)
            probs = market_probabilities(pred)
            
            st.subheader("Enter Odds (Decimal)")
            col1, col2, col3 = st.columns(3)
            with col1: ft_over_odds = st.number_input("FT Over Odds", min_value=1.01, value=1.90, step=0.05)
            with col2: home_win_odds = st.number_input("Home Win Odds", min_value=1.01, value=1.85, step=0.05)
            with col3: away_win_odds = st.number_input("Away Win Odds", min_value=1.01, value=2.00, step=0.05)
            
            ft_over_eval = evaluate_value(probs["ft"]["over"], ft_over_odds)
            home_eval = evaluate_value(probs["combo"]["p_home_win"], home_win_odds)
            away_eval = evaluate_value(probs["combo"]["p_away_win"], away_win_odds)
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("FT Over Edge", f"{ft_over_eval['edge']:.4f}", delta="VALUE" if ft_over_eval["is_value"] else "NO")
            with col2: st.metric("Home Win Edge", f"{home_eval['edge']:.4f}", delta="VALUE" if home_eval["is_value"] else "NO")
            with col3: st.metric("Away Win Edge", f"{away_eval['edge']:.4f}", delta="VALUE" if away_eval["is_value"] else "NO")
    else:
        st.info("Upload data in the Predictor tab first.")

# ═════════════════════════════════════════════════════════════════════════
# TAB 4: BACKTEST
# ═════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header(" Backtest")
    st.info("Upload historical CSV data in the Predictor tab to enable backtesting.")

# ═════════════════════════════════════════════════════════════════════════
# TAB 5: BANKROLL
# ═════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("💰 Bankroll Simulator")
    starting_bankroll = st.number_input("Starting Bankroll ($)", 100, 100000, 1000, step=100)
    kelly_fraction = st.slider("Kelly Fraction", 0.1, 0.5, 0.25, step=0.05)
    st.write("Simulate bankroll growth based on Blessings picks.")

# ═════════════════════════════════════════════════════════════════════════
# TAB 6: LEAGUE BOARD
# ═════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("🌍 League Board — 42 Categories")
    rows = []
    for name, profile in LEAGUE_REGISTRY.items():
        rows.append({"League": name, "Country": profile.country, "Feed": profile.feed})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════
# TAB 7: VIP
# ═════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("👑 VIP Access")
    if not st.session_state.vip_authenticated:
        col1, col2 = st.columns(2)
        with col1: username = st.text_input("Username")
        with col2: password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "owenzo" and password == "basketball2026":
                st.session_state.vip_authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.success("✅ VIP Access Granted")
        if st.button("Logout"):
            st.session_state.vip_authenticated = False
            st.rerun()

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🏀 Owenzõ Basketball Points v1.0 | MIT License | Responsible gambling: owenzo contactowenzo@gmail.com")
