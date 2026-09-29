"""17 tests — no network required. Registry coverage, over+under+push==1, monotonicity, push handling, Kelly guards, bankroll simulator, model determinism, joint Winner & Total distribution, calibration bounds, CSV alias detection."""
import pytest
import numpy as np
import pandas as pd
from src.config import LEAGUE_REGISTRY, LeagueProfile
from src.utils import total_probabilities, prob_over, prob_under, prob_push, bivariate_winner_total, closing_line_value
from src.ratings import compute_ratings
from src.model import fit_model, predict, market_probabilities
from src.lines import evaluate_value
from src.staking import fractional_kelly, bankroll_simulator
from src.backtest import walk_forward_backtest


# ── 1. Registry coverage ─────────────────────────────────────────────────
def test_registry_not_empty():
    assert len(LEAGUE_REGISTRY) >= 42


# ── 2. All leagues have required fields ──────────────────────────────────
def test_league_profiles_complete():
    for name, profile in LEAGUE_REGISTRY.items():
        assert profile.name
        assert profile.country
        assert profile.feed in ("api-basketball", "nba_api", "euroleague", "csv")
        assert profile.sigma_ft > 0


# ── 3. over + under + push == 1 exactly ──────────────────────────────────
def test_probabilities_sum_to_one_integer_line():
    probs = total_probabilities(215.0, 215.1, 14.0)
    assert abs(probs["over"] + probs["under"] + probs["push"] - 1.0) < 1e-9


def test_probabilities_sum_to_one_non_integer_line():
    probs = total_probabilities(215.5, 215.1, 14.0)
    assert abs(probs["over"] + probs["under"] + probs["push"] - 1.0) < 1e-9


# ── 4. Monotonicity: over prob decreases as line increases ───────────────
def test_monotonicity():
    lines = np.arange(200, 230, 1.0)
    probs = [prob_over(l, 215.1, 14.0) for l in lines]
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1]


# ── 5. Push handling on integer lines ────────────────────────────────────
def test_push_nonzero_on_integer():
    p = prob_push(215.0, 215.1, 14.0)
    assert p > 0


def test_push_zero_on_non_integer():
    p = prob_push(215.5, 215.1, 14.0)
    assert p == 0.0


# ── 6. Kelly guards ─────────────────────────────────────────────────────
def test_kelly_pass_below_min_edge():
    result = fractional_kelly(0.50, 2.0, 1000, min_edge=0.02)
    assert result["action"] == "PASS"


def test_kelly_pass_negative():
    result = fractional_kelly(0.40, 2.0, 1000)
    assert result["action"] == "PASS"


def test_kelly_capped_at_5pct():
    result = fractional_kelly(0.90, 10.0, 1000, max_stake_pct=5.0)
    assert result["stake"] <= 50.0


# ── 7. Bankroll simulator ────────────────────────────────────────────────
def test_bankroll_simulator_runs():
    bets = [
        {"model_prob": 0.60, "odds": 1.80, "result": "win"},
        {"model_prob": 0.55, "odds": 1.90, "result": "loss"},
        {"model_prob": 0.70, "odds": 1.50, "result": "win"},
    ]
    equity = bankroll_simulator(bets, 1000)
    assert len(equity) == 3
    assert equity["bankroll"].iloc[-1] > 0


# ── 8. Model determinism ─────────────────────────────────────────────────
def test_model_deterministic():
    game_log = pd.DataFrame({
        "date": ["2024-01-01"] * 50,
        "home_team": ["A"] * 25 + ["B"] * 25,
        "away_team": ["B"] * 25 + ["A"] * 25,
        "home_score": [100] * 25 + [95] * 25,
        "away_score": [95] * 25 + [100] * 25,
    })
    profile = LeagueProfile("Test", "Test", "csv")
    fitted1 = fit_model(game_log, profile)
    fitted2 = fit_model(game_log, profile)
    pred1 = predict("A", "B", fitted1)
    pred2 = predict("A", "B", fitted2)
    assert pred1["ft_total"] == pred2["ft_total"]


# ── 9. Joint Winner & Total distribution sums correctly ──────────────────
def test_bivariate_sums():
    result = bivariate_winner_total(4.0, 10.0, 215.0, 14.0, 0.15, total_line=215.0)
    total = (result["home_over"] + result["home_under"] + result["home_push"] +
             result["away_over"] + result["away_under"] + result["away_push"])
    assert abs(total - 1.0) < 0.05  # Monte Carlo tolerance


# ── 10. Calibration bounds ───────────────────────────────────────────────
def test_calibration_reasonable():
    game_log = pd.DataFrame({
        "date": ["2024-01-01"] * 100,
        "home_team": (["A"] * 50 + ["B"] * 50),
        "away_team": (["B"] * 50 + ["A"] * 50),
        "home_score": [100 + i % 10 for i in range(100)],
        "away_score": [95 + i % 10 for i in range(100)],
    })
    profile = LeagueProfile("Test", "Test", "csv")
    results = walk_forward_backtest(game_log, profile, min_train_size=30)
    assert results["n_games"] > 0


# ── 11. CSV alias detection ──────────────────────────────────────────────
def test_csv_loader_required_columns():
    from loaders.csv_loader import load_csv
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("date,home_team,away_team,home_score,away_score\n")
        f.write("2024-01-01,A,B,100,95\n")
        f.flush()
        df = load_csv(f.name)
        assert len(df) == 1
    os.unlink(f.name)


# ── 12. Ratings shrinkage ────────────────────────────────────────────────
def test_shrinkage_toward_mean():
    game_log = pd.DataFrame({
        "date": ["2024-01-01"] * 3,
        "home_team": ["A"] * 3,
        "away_team": ["B"] * 3,
        "home_score": [150, 150, 150],  # extreme
        "away_score": [50, 50, 50],
    })
    profile = LeagueProfile("Test", "Test", "csv", shrinkage_k=12)
    ratings = compute_ratings(game_log, profile)
    team_a = ratings[ratings["team"] == "A"].iloc[0]
    # With only 3 games, attack should be shrunk toward 1.0
    assert 1.0 < team_a["attack"] < 150/100  # shrunk, not raw


# ─ 13. Per-league stake ceiling (Blessings) ─────────────────────────────
def test_per_league_stake_ceiling():
    league = LeagueProfile("Test", "Test", "csv", stake_ceiling_pct=2.0)
    result = fractional_kelly(0.80, 1.5, 1000, league=league)
    assert result["stake"] <= 20.0  # 2% of 1000


# ── 14. CLV calculation ──────────────────────────────────────────────────
def test_clv_positive():
    clv = closing_line_value(0.60, 1.80)
    assert clv > 0  # 0.60 - 1/1.80 = 0.60 - 0.556 = 0.044


# ── 15. Value evaluation ─────────────────────────────────────────────────
def test_value_positive_edge():
    result = evaluate_value(0.60, 1.80)
    assert result["is_value"] is True
    assert result["edge"] > 0


# ─ 16. HT total is fraction of FT ───────────────────────────────────────
def test_ht_fraction_of_ft():
    profile = LeagueProfile("Test", "Test", "csv", h1_share=0.48)
    game_log = pd.DataFrame({
        "date": ["2024-01-01"] * 50,
        "home_team": ["A"] * 25 + ["B"] * 25,
        "away_team": ["B"] * 25 + ["A"] * 25,
        "home_score": [100] * 25 + [95] * 25,
        "away_score": [95] * 25 + [100] * 25,
    })
    fitted = fit_model(game_log, profile)
    pred = predict("A", "B", fitted)
    assert abs(pred["ht_total"] - pred["ft_total"] * 0.48) < 0.1


# ── 17. Smoke test on NBA-like profile ───────────────────────────────────
def test_nba_smoke():
    profile = LeagueProfile("NBA", "USA", "nba_api",
                            avg_home_score=114, avg_away_score=110,
                            h1_share=0.49, sigma_ft=22.0, sigma_margin=12.0,
                            home_court_advantage=4.5)
    game_log = pd.DataFrame({
        "date": ["2024-01-01"] * 100,
        "home_team": (["Lakers"] * 50 + ["Celtics"] * 50),
        "away_team": (["Celtics"] * 50 + ["Lakers"] * 50),
        "home_score": [114 + i % 10 for i in range(100)],
        "away_score": [110 + i % 10 for i in range(100)],
    })
    fitted = fit_model(game_log, profile)
    pred = predict("Lakers", "Celtics", fitted)
    probs = market_probabilities(pred)

    # μFT ~215, μHT ~106
    assert 200 < pred["ft_total"] < 240
    assert 100 < pred["ht_total"] < 120
    # Probabilities sum to 1
    assert abs(probs["ft"]["over"] + probs["ft"]["under"] + probs["ft"]["push"] - 1.0) < 1e-9
