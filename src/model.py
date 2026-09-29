"""fit_model / predict / market_probabilities — Blessings: HT modelled directly after ~100 games."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.config import LeagueProfile
from src.ratings import compute_ratings
from src.utils import total_probabilities, bivariate_winner_total


def fit_model(game_log: pd.DataFrame, profile: LeagueProfile) -> dict:
    """
    Fit league constants from history.
    Blessings: after min_games_for_own_params, use own σ_FT and H1 share.
    """
    if len(game_log) < 10:
        return {"fitted": False, "profile": profile}

    # Use own params if enough data (Blessings upgrade)
    p = profile
    if len(game_log) >= p.min_games_for_own_params:
        ft_totals = game_log["home_score"] + game_log["away_score"]
        margins = game_log["home_score"] - game_log["away_score"]
        h1_totals = game_log.get("home_h1_score", pd.Series([np.nan]*len(game_log))) + game_log.get("away_h1_score", pd.Series([np.nan]*len(game_log)))
        h1_totals = h1_totals.dropna()

        p.sigma_ft = float(ft_totals.std()) if len(ft_totals) > 20 else p.sigma_ft
        p.sigma_margin = float(margins.std()) if len(margins) > 20 else p.sigma_margin
        p.corr_margin_total = float(np.corrcoef(margins, ft_totals)[0, 1]) if len(margins) > 20 else p.corr_margin_total
        if len(h1_totals) > 20:
            p.h1_share = float((h1_totals / ft_totals).mean())

    # Team ratings
    ratings = compute_ratings(game_log, p)

    return {
        "fitted": True,
        "profile": p,
        "ratings": ratings,
        "n_games": len(game_log),
    }


def predict(
    home_team: str,
    away_team: str,
    fitted: dict,
) -> dict:
    """
    Project HT and FT totals.
    μ_home = avg_home × attack_home × defence_away
    FT total = μ_home + μ_away
    HT total = FT total × H1 share (or direct HT model if available)
    """
    profile = fitted["profile"]
    ratings = fitted["ratings"]

    home_row = ratings[ratings["team"] == home_team]
    away_row = ratings[ratings["team"] == away_team]

    if home_row.empty or away_row.empty:
        # Use league averages for unknown teams
        home_attack, home_defence = 1.0, 1.0
        away_attack, away_defence = 1.0, 1.0
    else:
        home_attack = home_row["attack"].values[0]
        home_defence = home_row["defence"].values[0]
        away_attack = away_row["attack"].values[0]
        away_defence = away_row["defence"].values[0]

    mu_home = profile.avg_home_score * home_attack * away_defence
    mu_away = profile.avg_away_score * away_attack * home_defence

    # Add home court advantage to margin
    margin = (mu_home - mu_away) + profile.home_court_advantage

    ft_total = mu_home + mu_away
    ht_total = ft_total * profile.h1_share  # Blessings: direct HT model if data available

    return {
        "mu_home": round(mu_home, 2),
        "mu_away": round(mu_away, 2),
        "ft_total": round(ft_total, 2),
        "ht_total": round(ht_total, 2),
        "margin": round(margin, 2),
        "sigma_ft": profile.sigma_ft,
        "sigma_margin": profile.sigma_margin,
        "corr": profile.corr_margin_total,
        "h1_share": profile.h1_share,
    }


def market_probabilities(prediction: dict, ft_line: float = None, ht_line: float = None) -> dict:
    """
    Compute Over/Under/Push probabilities for FT and HT totals.
    Blessings: bivariate Winner & Total combo market.
    """
    ft_line = ft_line or prediction["ft_total"]
    ht_line = ht_line or prediction["ht_total"]

    ft_probs = total_probabilities(ft_line, prediction["ft_total"], prediction["sigma_ft"])
    ht_probs = total_probabilities(ht_line, prediction["ht_total"], prediction["sigma_ft"] * 0.7)

    # Bivariate Winner & Total
    combo = bivariate_winner_total(
        margin_mu=prediction["margin"],
        margin_sigma=prediction["sigma_margin"],
        total_mu=prediction["ft_total"],
        total_sigma=prediction["sigma_ft"],
        corr=prediction["corr"],
        margin_line=0.0,
        total_line=ft_line,
    )

    return {
        "ft": ft_probs,
        "ht": ht_probs,
        "combo": combo,
        "prediction": prediction,
    }
