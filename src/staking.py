"""Fractional Kelly + bankroll simulator — Blessings: per-league stake ceiling."""
import numpy as np
import pandas as pd
from src.config import LeagueProfile


def fractional_kelly(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 5.0,
    min_edge: float = 0.02,
    league: LeagueProfile = None,
) -> dict:
    """
    Calculate stake using fractional Kelly criterion.
    Blessings: per-league stake ceiling, skip below min_edge, zero-edge = pass.
    """
    implied = 1.0 / decimal_odds
    edge = model_prob - implied

    # Zero-edge selections read as pass, not as a tip
    if edge < min_edge:
        return {"stake": 0.0, "action": "PASS", "reason": f"Edge {edge:.4f} below minimum {min_edge}"}

    # Full Kelly fraction: (bp - q) / b where b = odds - 1, p = model_prob, q = 1 - p
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    full_kelly = (b * p - q) / b

    if full_kelly <= 0:
        return {"stake": 0.0, "action": "PASS", "reason": "Negative Kelly"}

    # Fractional Kelly
    stake_fraction = full_kelly * kelly_fraction

    # Hard cap at max_stake_pct of bankroll
    stake_fraction = min(stake_fraction, max_stake_pct / 100.0)

    # Blessings: per-league stake ceiling
    if league is not None:
        league_cap = league.stake_ceiling_pct / 100.0
        stake_fraction = min(stake_fraction, league_cap)

    stake = stake_fraction * bankroll
    return {
        "stake": round(stake, 2),
        "stake_fraction": round(stake_fraction, 4),
        "full_kelly": round(full_kelly, 4),
        "action": "BET",
        "edge": round(edge, 4),
    }


def bankroll_simulator(
    bets: list[dict],
    starting_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 5.0,
) -> pd.DataFrame:
    """
    Walk-forward bankroll simulation.
    bets: list of {model_prob, odds, result: 'win'|'loss'|'push'}
    """
    bankroll = starting_bankroll
    history = []

    for i, bet in enumerate(bets):
        stake_info = fractional_kelly(
            bet["model_prob"], bet["odds"], bankroll,
            kelly_fraction=kelly_fraction, max_stake_pct=max_stake_pct
        )
        stake = stake_info["stake"]

        if stake_info["action"] == "PASS":
            pnl = 0.0
        elif bet["result"] == "win":
            pnl = stake * (bet["odds"] - 1)
        elif bet["result"] == "loss":
            pnl = -stake
        else:  # push
            pnl = 0.0

        bankroll += pnl
        history.append({
            "bet": i + 1,
            "stake": stake,
            "odds": bet["odds"],
            "model_prob": bet["model_prob"],
            "result": bet["result"],
            "pnl": round(pnl, 2),
            "bankroll": round(bankroll, 2),
        })

    return pd.DataFrame(history)
