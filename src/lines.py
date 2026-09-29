"""Outcome grid, market eval, Winner & Total — Blessings: combo market attack."""
import pandas as pd
from src.model import predict, market_probabilities


def outcome_grid(fitted: dict, teams: list[tuple[str, str]], ft_lines: list[float] = None) -> pd.DataFrame:
    """
    Generate outcome grid for multiple matchups.
    Returns DataFrame with team pairs, projections, and probabilities at various lines.
    """
    rows = []
    for home, away in teams:
        pred = predict(home, away, fitted)
        probs = market_probabilities(pred)
        rows.append({
            "Home": home,
            "Away": away,
            "μ_Home": pred["mu_home"],
            "μ_Away": pred["mu_away"],
            "FT Total": pred["ft_total"],
            "HT Total": pred["ht_total"],
            "Margin": pred["margin"],
            "FT Over": probs["ft"]["over"],
            "FT Under": probs["ft"]["under"],
            "FT Push": probs["ft"]["push"],
            "HT Over": probs["ht"]["over"],
            "HT Under": probs["ht"]["under"],
            "HT Push": probs["ht"]["push"],
            "Home Win": probs["combo"]["p_home_win"],
            "Away Win": probs["combo"]["p_away_win"],
            "Home & Over": probs["combo"]["home_over"],
            "Home & Under": probs["combo"]["home_under"],
            "Away & Over": probs["combo"]["away_over"],
            "Away & Under": probs["combo"]["away_under"],
        })
    return pd.DataFrame(rows)


def evaluate_value(model_prob: float, decimal_odds: float) -> dict:
    """
    Evaluate whether a bet has positive expected value.
    Edge = model_prob - implied_prob
    """
    implied = 1.0 / decimal_odds
    edge = model_prob - implied
    ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
    return {
        "model_prob": round(model_prob, 4),
        "implied_prob": round(implied, 4),
        "edge": round(edge, 4),
        "ev_per_unit": round(ev, 4),
        "is_value": edge > 0,
    }
