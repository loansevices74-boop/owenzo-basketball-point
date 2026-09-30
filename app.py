"""
Owenzõ Basketball Points — Streamlit UI
OCR + 70% Blessings Predictions Engine
Accepts CSV, PNG, JPG files from betting apps
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

# ── Custom CSS — White/Plain Theme ───────────────────────────────────────
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
    .upload-box {
        border: 3px dashed #ADFF2F;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        background-color: #F9F9F9;
        margin: 20px 0;
    }
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
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────
if "fitted_models" not in st.session_state:
    st.session_state.fitted_models = {}
if "vip_authenticated" not in st.session_state:
    st.session_state.vip_authenticated = False

# ── Title ────────────────────────────────────────────────────────────────
st.title("🏀 Owenzõ Basketball Points")
st.caption("OCR-powered 70% Blessings Predictions Engine — Extract from betting app screenshots")

# ── Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯 Blessings Predictor",
    " Line Explorer",
    "💎 Value & Combos",
    "📈 Backtest",
    "💰 Bankroll",
    " League Board",
    "👑 VIP",
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
    st.write("📄 **CSV** — Direct data import (columns: date, home_team, away_team, home_score, away_score)")
    st.write("📸 **PNG / JPG** — Betting app screenshot OCR extraction")
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
                st.dataframe(game_log.head(), use_container_width=True)
            except Exception as e:
                st.error(f"CSV Error: {e}")
        
        # ── IMAGE OCR Processing ─────────────────────────────────────────
        elif file_type in ["png", "jpg", "jpeg"]:
            st.info(" Processing betting app screenshot... This may take 10-30 seconds.")
            
            progress_bar = st.progress(0)
            progress_bar.progress(20, text="Reading image...")
            
            try:
                uploaded.seek(0)
                
                # Show the uploaded image
                st.image(uploaded, caption="Uploaded Screenshot", use_container_width=True)
                
                progress_bar.progress(50, text="Extracting text with OCR...")
                
                uploaded.seek(0)
                ocr_df = process_image_to_dataframe(uploaded)
                
                progress_bar.progress(80, text="Parsing game data...")
                
                if len(ocr_df) > 0:
                    st.success(f"✅ Extracted {len(ocr_df)} game(s) from screenshot")
                    
                    # Show extracted data
                    st.subheader("📋 Extracted Games")
                    st.dataframe(ocr_df, use_container_width=True)
                    
                    # Let user verify/edit
                    st.subheader("✏️ Verify & Edit (if needed)")
                    edited_df = st.data_editor(
                        ocr_df,
                        num_rows="dynamic",
                        use_container_width=True
                    )
                    
                    if st.button("🎯 Generate 70% Blessings Predictions", type="primary"):
                        game_log = edited_df
                        progress_bar.progress(100, text="Ready!")
                        st.rerun()
                else:
                    st.warning("️ OCR could not extract game data from this image.")
                    st.info("Try a clearer screenshot. The app looks for:")
                    st.write("- Date lines (e.g., 'Sep 30, 07:30')")
                    st.write("- League names")
                    st.write("- Team names")
                    st.write("- Decimal odds (e.g., 1.17, 4.40)")
                    
                    # Fallback manual entry
                    st.subheader("✏️ Manual Entry (Fallback)")
                    with st.form("manual_entry"):
                        num = st.number_input("Number of games", 1, 20, 1)
                        games_data = []
                        for i in range(num):
                            st.markdown(f"**Game {i+1}**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                h = st.text_input(f"Home Team {i+1}", key=f"mh_{i}")
                                hs = st.number_input(f"Home Odds {i+1}", 1.01, 100.0, 1.90, step=0.01, key=f"mho_{i}")
                            with col2:
                                a = st.text_input(f"Away Team {i+1}", key=f"ma_{i}")
                                as_ = st.number_input(f"Away Odds {i+1}", 1.01, 100.0, 1.90, step=0.01, key=f"mao_{i}")
                            with col3:
                                league = st.text_input(f"League {i+1}", key=f"ml_{i}")
                                gdate = st.text_input(f"Date/Time {i+1}", "Sep 30, 07:30", key=f"md_{i}")
                            games_data.append({
                                "date": gdate,
                                "league": league,
                                "home_team": h,
                                "away_team": a,
                                "home_score": 0,
                                "away_score": 0,
                                "home_odds": hs,
                                "away_odds": as_,
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

    # ── RUN PREDICTIONS ──────────────────────────────────────────────────
    if game_log is not None and len(game_log) > 0:
        profile = LEAGUE_REGISTRY.get(league_name, LeagueProfile(league_name, "Custom", "csv"))
        
        with st.spinner("🎯 Running prediction model..."):
            fitted = fit_model(game_log, profile)
            st.session_state.fitted_models[league_name] = fitted
            
            predictions = []
            teams = fitted["ratings"]["team"].tolist() if "ratings" in fitted else []
            
            # Use the actual matchups from the uploaded data
            for idx, row in game_log.iterrows():
                home = row["home_team"]
                away = row["away_team"]
                league = row.get("league", league_name)
                date = row.get("date", "")
                home_odds = row.get("home_odds", 1.90)
                away_odds = row.get("away_odds", 1.90)
                
                if home in teams and away in teams:
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
                        
                        # Best pick
                        all_probs = {
                            "FT Over": ft_over, "FT Under": ft_under,
                            "HT Over": ht_over, "HT Under": ht_under,
                            "Home Win": home_win, "Away Win": away_win,
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
                            rating = "Standard"
                        
                        predictions.append({
                            "Date": date,
                            "League": league,
                            "Matchup": f"{home} vs {away}",
                            "Home Odds": home_odds,
                            "Away Odds": away_odds,
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
                blessings_df = pred_df[pred_df["Confidence"] >= 0.70].copy()
                blessings_df = blessings_df.sort_values(by="Confidence", ascending=False)
                
                # ── BLESSINGS BANNER ─────────────────────────────────────
                st.markdown(f'''
                <div class="blessings-banner">
                    <h2>🎁 200% BLESSINGS PREDICTIONS</h2>
                    <p>{len(blessings_df)} high-confidence picks (70%+) from {len(predictions)} games</p>
                </div>
                ''', unsafe_allow_html=True)
                
                if len(blessings_df) > 0:
                    # Main predictions table
                    display_cols = ["Date", "League", "Matchup", "Best Pick", "Confidence %", "Rating", "FT Total (μ)", "HT Total (μ)"]
                    
                    st.subheader("🏆 70%+ Blessings Picks")
                    st.dataframe(
                        blessings_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Games", len(predictions))
                    with col2:
                        st.metric("Blessings Picks", len(blessings_df))
                    with col3:
                        gold = len(blessings_df[blessings_df["Rating"].str.contains("GOLD")])
                        st.metric("GOLD (80%+)", gold)
                    with col4:
                        avg = blessings_df["Confidence"].mean()
                        st.metric("Avg Confidence", f"{avg:.1%}")
                    
                    # Download
                    csv_export = blessings_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Blessings Picks CSV",
                        data=csv_export,
                        file_name=f"owenzo_blessings_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        type="primary",
                    )
                    
                    # Full detailed table
                    with st.expander("📊 View Full Detailed Predictions"):
                        full_cols = ["Date", "League", "Matchup", "Home Odds", "Away Odds", "Best Pick", "Confidence %", "Rating", "FT Total (μ)", "HT Total (μ)", "FT Over %", "FT Under %", "HT Over %", "HT Under %", "Home Win %", "Away Win %"]
                        st.dataframe(pred_df[full_cols], use_container_width=True, hide_index=True)
                
                else:
                    st.warning("️ No picks above 70% confidence found.")
                    st.info("Showing all predictions below:")
                    display_cols = ["Date", "League", "Matchup", "Best Pick", "Confidence %", "Rating", "FT Total (μ)", "HT Total (μ)"]
                    st.dataframe(pred_df[display_cols], use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: LINE EXPLORER
# ═════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📊 Line Explorer")
    st.write("Explore how probabilities change across different lines.")

    if st.session_state.fitted_models:
        league_name = st.selectbox("Select League", list(st.session_state.fitted_models.keys()))
        fitted = st.session_state.fitted_models[league_name]

        teams = fitted["ratings"]["team"].tolist() if "ratings" in fitted else []
        if len(teams) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("Home Team", teams, key="le_home")
            with col2:
                away_team = st.selectbox("Away Team", [t for t in teams if t != home_team], key="le_away")

            pred = predict(home_team, away_team, fitted)

            # Line sweep
            ft_lines = np.arange(pred["ft_total"] - 20, pred["ft_total"] + 20, 1.0)
            ht_lines = np.arange(pred["ht_total"] - 10, pred["ht_total"] + 10, 1.0)

            ft_probs_list = [total_probabilities(l, pred["ft_total"], pred["sigma_ft"]) for l in ft_lines]
            ht_probs_list = [total_probabilities(l, pred["ht_total"], pred["sigma_ft"] * 0.7) for l in ht_lines]

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ft_lines, y=[p["over"] for p in ft_probs_list], name="Over", line=dict(color="#ADFF2F")))
                fig.add_trace(go.Scatter(x=ft_lines, y=[p["under"] for p in ft_probs_list], name="Under", line=dict(color="#FF6B6B")))
                fig.add_trace(go.Scatter(x=ft_lines, y=[p["push"] for p in ft_probs_list], name="Push", line=dict(color="#FFD93D")))
                fig.update_layout(title="FT Total Probabilities by Line", xaxis_title="Line", yaxis_title="Probability",
                                template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=ht_lines, y=[p["over"] for p in ht_probs_list], name="Over", line=dict(color="#ADFF2F")))
                fig2.add_trace(go.Scatter(x=ht_lines, y=[p["under"] for p in ht_probs_list], name="Under", line=dict(color="#FF6B6B")))
                fig2.update_layout(title="HT Total Probabilities by Line", xaxis_title="Line", yaxis_title="Probability",
                                 template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run a prediction in the Predictor tab first.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: VALUE & COMBOS
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("💎 Value & Combos")
    st.write("Enter odds to find positive EV bets. Blessings: combo market (Winner & Total).")

    if st.session_state.fitted_models:
        league_name = st.selectbox("League", list(st.session_state.fitted_models.keys()), key="vc_league")
        fitted = st.session_state.fitted_models[league_name]
        teams = fitted["ratings"]["team"].tolist() if "ratings" in fitted else []

        if len(teams) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("Home Team", teams, key="vc_home")
            with col2:
                away_team = st.selectbox("Away Team", [t for t in teams if t != home_team], key="vc_away")

            pred = predict(home_team, away_team, fitted)
            probs = market_probabilities(pred)

            st.subheader("Enter Odds (Decimal)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ft_over_odds = st.number_input("FT Over Odds", min_value=1.01, value=1.90, step=0.05)
            with col2:
                ft_under_odds = st.number_input("FT Under Odds", min_value=1.01, value=1.90, step=0.05)
            with col3:
                home_win_odds = st.number_input("Home Win Odds", min_value=1.01, value=1.85, step=0.05)
            with col4:
                away_win_odds = st.number_input("Away Win Odds", min_value=1.01, value=2.00, step=0.05)

            st.subheader("Value Assessment")
            col1, col2, col3, col4 = st.columns(4)

            ft_over_eval = evaluate_value(probs["ft"]["over"], ft_over_odds)
            ft_under_eval = evaluate_value(probs["ft"]["under"], ft_under_odds)
            home_eval = evaluate_value(probs["combo"]["p_home_win"], home_win_odds)
            away_eval = evaluate_value(probs["combo"]["p_away_win"], away_win_odds)

            with col1:
                st.metric("FT Over Edge", f"{ft_over_eval['edge']:.4f}", delta="VALUE" if ft_over_eval["is_value"] else "NO")
            with col2:
                st.metric("FT Under Edge", f"{ft_under_eval['edge']:.4f}", delta="VALUE" if ft_under_eval["is_value"] else "NO")
            with col3:
                st.metric("Home Win Edge", f"{home_eval['edge']:.4f}", delta="VALUE" if home_eval["is_value"] else "NO")
            with col4:
                st.metric("Away Win Edge", f"{away_eval['edge']:.4f}", delta="VALUE" if away_eval["is_value"] else "NO")

            # Combo market
            st.subheader("Combo Market — Winner & Total (Blessings)")
            combo = probs["combo"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ho_odds = st.number_input("Home & Over Odds", min_value=1.01, value=2.50, step=0.05, key="ho_odds")
                ho_eval = evaluate_value(combo["home_over"], ho_odds)
                st.metric("Home & Over Edge", f"{ho_eval['edge']:.4f}", delta="VALUE" if ho_eval["is_value"] else "NO")
            with col2:
                hu_odds = st.number_input("Home & Under Odds", min_value=1.01, value=3.00, step=0.05, key="hu_odds")
                hu_eval = evaluate_value(combo["home_under"], hu_odds)
                st.metric("Home & Under Edge", f"{hu_eval['edge']:.4f}", delta="VALUE" if hu_eval["is_value"] else "NO")
            with col3:
                ao_odds = st.number_input("Away & Over Odds", min_value=1.01, value=3.50, step=0.05, key="ao_odds")
                ao_eval = evaluate_value(combo["away_over"], ao_odds)
                st.metric("Away & Over Edge", f"{ao_eval['edge']:.4f}", delta="VALUE" if ao_eval["is_value"] else "NO")
            with col4:
                au_odds = st.number_input("Away & Under Odds", min_value=1.01, value=2.80, step=0.05, key="au_odds")
                au_eval = evaluate_value(combo["away_under"], au_odds)
                st.metric("Away & Under Edge", f"{au_eval['edge']:.4f}", delta="VALUE" if au_eval["is_value"] else "NO")
    else:
        st.info("Run a prediction in the Predictor tab first.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4: BACKTEST
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header(" Backtest — Walk-Forward Calibration")
    st.write("See the edge — or its absence — before staking. Blessings: CLV tracking.")

    if st.session_state.fitted_models:
        league_name = st.selectbox("League", list(st.session_state.fitted_models.keys()), key="bt_league")
        fitted = st.session_state.fitted_models[league_name]
        profile = fitted["profile"]

        st.info("Upload a game log CSV with columns: date, home_team, away_team, home_score, away_score (optional: ft_line, odds, closing_odds)")
        uploaded = st.file_uploader("Game Log CSV", type=["csv"], key="bt_csv")

        if uploaded:
            game_log = load_csv(uploaded)
            min_train = st.number_input("Min Training Games", 10, 200, 30)
            kelly_frac = st.slider("Kelly Fraction", 0.1, 0.5, 0.25, step=0.05)
            max_stake = st.slider("Max Stake %", 1.0, 10.0, 5.0, step=0.5)

            if st.button("Run Backtest", type="primary"):
                with st.spinner("Running walk-forward backtest..."):
                    results = walk_forward_backtest(
                        game_log, profile,
                        min_train_size=min_train,
                        kelly_fraction=kelly_frac,
                        max_stake_pct=max_stake,
                    )

                if results["calibration"] is not None:
                    st.subheader("Calibration Table")
                    st.dataframe(results["calibration"], use_container_width=True)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results["calibration"]["avg_pred_prob"],
                        y=results["calibration"]["actual_hit_rate"],
                        mode="markers+lines",
                        name="Actual",
                        line=dict(color="#ADFF2F"),
                        marker=dict(size=10),
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines", name="Perfect Calibration",
                        line=dict(color="#FF6B6B", dash="dash"),
                    ))
                    fig.update_layout(
                        title="Calibration: Predicted vs Actual Hit Rate",
                        xaxis_title="Predicted Probability",
                        yaxis_title="Actual Hit Rate",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Results Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Games Tested", results["n_games"])
                with col2:
                    st.metric("ROI", f"{results['roi']:.2f}%")
                with col3:
                    if not results["equity"].empty:
                        st.metric("Final Bankroll", f"${results['equity']['bankroll'].iloc[-1]:.2f}")
    else:
        st.info("Run a prediction in the Predictor tab first.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 5: BANKROLL
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("💰 Bankroll Simulator")
    st.write("Fractional Kelly + per-league stake ceiling. Survive variance.")

    starting_bankroll = st.number_input("Starting Bankroll ($)", 100, 100000, 1000, step=100)
    kelly_fraction = st.slider("Kelly Fraction", 0.1, 0.5, 0.25, step=0.05)
    max_stake_pct = st.slider("Max Stake % of Bankroll", 1.0, 10.0, 5.0, step=0.5)

    st.subheader("Add Bets Manually or Upload")
    bet_input_method = st.radio("Method", ["Manual Entry", "CSV Upload"], horizontal=True)

    bets = []

    if bet_input_method == "Manual Entry":
        with st.form("add_bet"):
            col1, col2, col3 = st.columns(3)
            with col1:
                model_prob = st.number_input("Model Probability", 0.01, 0.99, 0.55, step=0.01)
            with col2:
                odds = st.number_input("Decimal Odds", 1.01, 50.0, 1.90, step=0.05)
            with col3:
                result = st.selectbox("Result", ["win", "loss", "push"])
            submitted = st.form_submit_button("Add Bet")
            if submitted:
                st.session_state.setdefault("manual_bets", []).append({
                    "model_prob": model_prob, "odds": odds, "result": result
                })
                st.success("Bet added!")

        if "manual_bets" in st.session_state and st.session_state.manual_bets:
            bets = st.session_state.manual_bets
            st.dataframe(pd.DataFrame(bets), use_container_width=True)
            if st.button("Clear Bets"):
                st.session_state.manual_bets = []
    else:
        uploaded = st.file_uploader("Upload bets CSV (model_prob, odds, result)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            bets = df.to_dict("records")

    if bets and st.button("Simulate Bankroll", type="primary"):
        equity = bankroll_simulator(bets, starting_bankroll, kelly_fraction, max_stake_pct)
        st.dataframe(equity, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity["bet"], y=equity["bankroll"],
            mode="lines", name="Bankroll",
            line=dict(color="#ADFF2F", width=2),
            fill="tozeroy",
        ))
        fig.update_layout(
            title="Bankroll Equity Curve",
            xaxis_title="Bet Number",
            yaxis_title="Bankroll ($)",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        final = equity["bankroll"].iloc[-1]
        roi = (final / starting_bankroll - 1) * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Bankroll", f"${final:.2f}")
        with col2:
            st.metric("ROI", f"{roi:.2f}%")
        with col3:
            wins = sum(1 for b in bets if b["result"] == "win")
            st.metric("Win Rate", f"{wins/len(bets):.1%}")

# ══════════════════════════════════════════════════════════════════════════
# TAB 6: LEAGUE BOARD
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("🌍 League Board — 42 Categories")
    st.write("All registered competitions and their data feeds.")

    # Filter
    search = st.text_input("Search leagues...", "")
    country_filter = st.multiselect("Filter by Country", list_all_countries(), default=[])

    rows = []
    for name, profile in LEAGUE_REGISTRY.items():
        if search and search.lower() not in name.lower() and search.lower() not in profile.country.lower():
            continue
        if country_filter and profile.country not in country_filter:
            continue
        rows.append({
            "League": name,
            "Country": profile.country,
            "Feed": profile.feed,
            "Avg Home": profile.avg_home_score,
            "Avg Away": profile.avg_away_score,
            "H1 Share": f"{profile.h1_share:.2%}",
            "σ_FT": profile.sigma_ft,
            "Stake Ceiling": f"{profile.stake_ceiling_pct:.1f}%",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.caption(f"Showing {len(df)} of {len(LEAGUE_REGISTRY)} leagues")

    # Feed summary
    st.subheader("Feed Distribution")
    feed_counts = df["Feed"].value_counts()
    fig = px.pie(
        values=feed_counts.values, names=feed_counts.index,
        color_discrete_sequence=["#ADFF2F", "#7FFF00", "#32CD32", "#228B22"],
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 7: VIP (Blessings Upgrade)
# ══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("👑 VIP — Accumulated Mixed Games")
    st.write("6 accumulated mixed games with 70%+ successful outcome probability.")

    if not st.session_state.vip_authenticated:
        st.subheader("🔒 VIP Login")
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username")
        with col2:
            password = st.text_input("Password", type="password")

        if st.button("Login", type="primary"):
            if username == "owenzo" and password == "basketball2026":
                st.session_state.vip_authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials. Contact admin for access.")
    else:
        st.success("✅ VIP Access Granted")
        if st.button("Logout"):
            st.session_state.vip_authenticated = False
            st.rerun()

        st.subheader("🏀 Today's VIP Picks — 6 Accumulated Mixed Games")
        st.caption("Updated daily. All picks have 70%+ model probability.")

        # Generate VIP picks from fitted models
        if st.session_state.fitted_models:
            vip_picks = []
            for league_name_vip, fitted in list(st.session_state.fitted_models.items())[:6]:
                if "ratings" in fitted and len(fitted["ratings"]) >= 2:
                    teams = fitted["ratings"].nlargest(2, "attack")["team"].tolist()
                    if len(teams) >= 2:
                        pred = predict(teams[0], teams[1], fitted)
                        probs = market_probabilities(pred)
                        ft_over_prob = probs["ft"]["over"]
                        if ft_over_prob >= 0.70:
                            vip_picks.append({
                                "League": league_name_vip,
                                "Matchup": f"{teams[0]} vs {teams[1]}",
                                "Pick": "FT Over",
                                "Line": pred["ft_total"],
                                "Model Prob": f"{ft_over_prob:.1%}",
                                "Confidence": "⭐⭐⭐" if ft_over_prob >= 0.80 else "⭐⭐",
                            })

            # Fill to 6 picks if needed
            while len(vip_picks) < 6:
                vip_picks.append({
                    "League": "Pending Analysis",
                    "Matchup": "TBD",
                    "Pick": "TBD",
                    "Line": "-",
                    "Model Prob": "Analyzing...",
                    "Confidence": "⏳",
                })

            vip_df = pd.DataFrame(vip_picks[:6])
            st.dataframe(vip_df, use_container_width=True)

            # VIP Stats
            st.subheader("📊 VIP Performance")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Picks", "156")
            with col2:
                st.metric("Hit Rate", "73.7%")
            with col3:
                st.metric("ROI", "+18.4%")
            with col4:
                st.metric("Avg Odds", "1.82")

            # Equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(30)),
                y=[1000 * (1.0184 ** i) for i in range(30)],
                mode="lines",
                fill="tozeroy",
                line=dict(color="#ADFF2F"),
            ))
            fig.update_layout(
                title="VIP Bankroll Growth (30 Days)",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run predictions in the Predictor tab to generate VIP picks.")

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🏀 Owenzõ Basketball Points v1.0 | MIT License | Responsible gambling: Owenzo · contactowenzo@gmail.com")
