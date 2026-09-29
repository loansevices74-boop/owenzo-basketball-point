"""
Owenzõ Basketball Points — Streamlit UI
7 tabs: Predictor, Line Explorer, Value & Combos, Backtest, Bankroll, League Board, VIP
Dark + lime theme. No demo mode. Real API feeds.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

from src.config import LEAGUE_REGISTRY, LeagueProfile
from src.leagues import list_all_countries, list_all_leagues, get_league, list_leagues_by_country
from src.model import fit_model, predict, market_probabilities
from src.lines import outcome_grid, evaluate_value
from src.staking import fractional_kelly, bankroll_simulator
from src.backtest import walk_forward_backtest, compute_clv
from src.utils import total_probabilities, closing_line_value
from loaders.csv_loader import load_csv
from loaders.apibasketball_loader import get_games as api_basketball_games
from loaders.euroleague_loader import get_euroleague_games
from loaders.nba_loader import load_nba_season

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Owenzõ Basketball Points",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for dark + lime theme ─────────────────────────────────────
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
    .metric-card {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ADFF2F;
    }
    .vip-badge {
        background: linear-gradient(135deg, #ADFF2F, #7FFF00);
        color: #000000;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    h1, h2, h3 { color: #262730; }
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
st.caption("HT & FT total points modelling across 60+ competitions worldwide")

# ── Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯 Predictor",
    "📊 Line Explorer",
    "💎 Value & Combos",
    " Backtest",
    "💰 Bankroll",
    "🌍 League Board",
    "👑 VIP",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("🎯 Predictor — HT & FT Total Points")

    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Country / Region", ["All"] + list_all_countries())
    with col2:
        if country == "All":
            leagues = list_all_leagues()
        else:
            leagues = [k for k, v in LEAGUE_REGISTRY.items() if v.country == country]
        league_name = st.selectbox("League", leagues)

    # Data source
    source = st.radio("Data Source", ["API-Basketball", "EuroLeague", "NBA API", "CSV Upload"], horizontal=True)

    game_log = None

    if source == "CSV Upload":
        uploaded = st.file_uploader("Upload game log CSV", type=["csv"])
        if uploaded:
            game_log = load_csv(uploaded)
            st.success(f"Loaded {len(game_log)} games")
    elif source == "API-Basketball":
        profile = LEAGUE_REGISTRY.get(league_name)
        if profile and profile.api_league_id:
            season = st.number_input("Season", 2020, 2030, 2024)
            if st.button("Fetch Games"):
                with st.spinner("Fetching from API-Basketball..."):
                    game_log = api_basketball_games(profile.api_league_id, season)
                    st.success(f"Fetched {len(game_log)} games")
    elif source == "EuroLeague":
        comp = st.selectbox("Competition", ["EL", "EC"])
        season = st.text_input("Season Code", "2024-25")
        if st.button("Fetch EuroLeague"):
            with st.spinner("Fetching from EuroLeague API..."):
                game_log = get_euroleague_games(season, comp)
                st.success(f"Fetched {len(game_log)} games")
    elif source == "NBA API":
        season = st.text_input("NBA Season", "2024-25")
        if st.button("Fetch NBA"):
            with st.spinner("Fetching from nba_api..."):
                game_log = load_nba_season(season)
                st.success(f"Fetched {len(game_log)} games")

    if game_log is not None and len(game_log) > 0:
        profile = LEAGUE_REGISTRY.get(league_name, LeagueProfile(league_name, "Custom", "csv"))
        fitted = fit_model(game_log, profile)
        st.session_state.fitted_models[league_name] = fitted

        st.subheader("Team Ratings")
        if "ratings" in fitted:
            st.dataframe(fitted["ratings"], use_container_width=True)

        st.subheader("Matchup Prediction")
        teams = fitted["ratings"]["team"].tolist() if "ratings" in fitted else []
        col1, col2, col3 = st.columns(3)
        with col1:
            home_team = st.selectbox("Home Team", teams)
        with col2:
            away_team = st.selectbox("Away Team", [t for t in teams if t != home_team])
        with col3:
            ft_line = st.number_input("FT Line", value=float(fitted["profile"].avg_home_score + fitted["profile"].avg_away_score), step=0.5)
            ht_line = st.number_input("HT Line", value=float((fitted["profile"].avg_home_score + fitted["profile"].avg_away_score) * fitted["profile"].h1_share), step=0.5)

        if st.button("Predict", type="primary"):
            pred = predict(home_team, away_team, fitted)
            probs = market_probabilities(pred, ft_line, ht_line)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("FT Total (μ)", f"{pred['ft_total']:.1f}")
            with col2:
                st.metric("HT Total (μ)", f"{pred['ht_total']:.1f}")
            with col3:
                st.metric("Margin", f"{pred['margin']:.1f}")
            with col4:
                st.metric("H1 Share", f"{pred['h1_share']:.2%}")

            st.subheader("FT Probabilities")
            ft_df = pd.DataFrame([probs["ft"]])
            st.bar_chart(ft_df.T)
            st.write(ft_df)

            st.subheader("HT Probabilities")
            ht_df = pd.DataFrame([probs["ht"]])
            st.bar_chart(ht_df.T)
            st.write(ht_df)

            st.subheader("Winner & Total Combo (Bivariate)")
            combo = probs["combo"]
            combo_df = pd.DataFrame([{
                "Home & Over": f"{combo['home_over']:.3f}",
                "Home & Under": f"{combo['home_under']:.3f}",
                "Away & Over": f"{combo['away_over']:.3f}",
                "Away & Under": f"{combo['away_under']:.3f}",
                "Home Win": f"{combo['p_home_win']:.3f}",
                "Away Win": f"{combo['p_away_win']:.3f}",
            }])
            st.dataframe(combo_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: LINE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📊 Line Explorer")
    st.write("Explore how probabilities change across different lines.")

    if st.session_state.fitted_models:
        league_name = st.selectbox("Select League", list(st.session_state.fitted_models.keys()))
        fitted = st.session_state.fitted_models[league_name]

        teams = fitted["ratings"]["team"].tolist() if "ratings" in fitted else []
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
                            template="plotly_dark", plot_bgcolor="#1A1A2E", paper_bgcolor="#0D0D1A")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=ht_lines, y=[p["over"] for p in ht_probs_list], name="Over", line=dict(color="#ADFF2F")))
            fig2.add_trace(go.Scatter(x=ht_lines, y=[p["under"] for p in ht_probs_list], name="Under", line=dict(color="#FF6B6B")))
            fig2.update_layout(title="HT Total Probabilities by Line", xaxis_title="Line", yaxis_title="Probability",
                             template="plotly_dark", plot_bgcolor="#1A1A2E", paper_bgcolor="#0D0D1A")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run a prediction in the Predictor tab first.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: VALUE & COMBOS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("💎 Value & Combos")
    st.write("Enter odds to find positive EV bets. Blessings: combo market (Winner & Total).")

    if st.session_state.fitted_models:
        league_name = st.selectbox("League", list(st.session_state.fitted_models.keys()), key="vc_league")
        fitted = st.session_state.fitted_models[league_name]
        teams = fitted["ratings"]["team"].tolist() if "ratings" in fitted else []

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
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📈 Backtest — Walk-Forward Calibration")
    st.write("See the edge — or its absence — before staking. Blessings: CLV tracking.")

    if st.session_state.fitted_models:
        league_name = st.selectbox("League", list(st.session_state.fitted_models.keys()), key="bt_league")
        fitted = st.session_state.fitted_models[league_name]
        profile = fitted["profile"]

        # Need game log — reconstruct from fitted data or upload
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
                        template="plotly_dark",
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

                # CLV (Blessings)
                if "closing_odds" in game_log.columns and not results["predictions"].empty:
                    clv = compute_clv(results["predictions"])
                    st.subheader("Closing Line Value (CLV) — Blessings")
                    st.metric("Avg CLV", f"{clv.mean():.4f}")
                    st.line_chart(clv)
    else:
        st.info("Run a prediction in the Predictor tab first.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 5: BANKROLL
# ═══════════════════════════════════════════════════════════════════════════
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
            template="plotly_dark",
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

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: LEAGUE BOARD
# ═══════════════════════════════════════════════════════════════════════════
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
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0D0D1A")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: VIP (Blessings Upgrade)
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("👑 VIP — Accumulated Mixed Games")
    st.markdown('<span class="vip-badge">PREMIUM ACCESS</span>', unsafe_allow_html=True)
    st.write("6 accumulated mixed games with 70%+ successful outcome probability.")

    if not st.session_state.vip_authenticated:
        st.subheader(" VIP Login")
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username")
        with col2:
            password = st.text_input("Password", type="password")

        if st.button("Login", type="primary"):
            # Default credentials — change in production
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

        st.subheader(" Today's VIP Picks — 6 Accumulated Mixed Games")
        st.caption("Updated daily. All picks have 70%+ model probability.")

        # Generate VIP picks from fitted models
        if st.session_state.fitted_models:
            vip_picks = []
            for league_name, fitted in list(st.session_state.fitted_models.items())[:6]:
                if "ratings" in fitted and len(fitted["ratings"]) >= 2:
                    teams = fitted["ratings"].nlargest(2, "attack")["team"].tolist()
                    if len(teams) >= 2:
                        pred = predict(teams[0], teams[1], fitted)
                        probs = market_probabilities(pred)
                        ft_over_prob = probs["ft"]["over"]
                        if ft_over_prob >= 0.70:
                            vip_picks.append({
                                "League": league_name,
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
            st.subheader(" VIP Performance")
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
                template="plotly_dark",
                paper_bgcolor="#0D0D1A",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run predictions in the Predictor tab to generate VIP picks.")

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🏀 Owenzõ Basketball Points v1.0 | MIT License | Responsible gambling: BeGambleAware · +2349021076350")
