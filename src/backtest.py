"""Walk-forward, calibration, ROI — Blessings: CLV tracking."""
import numpy as np
import pandas as pd
from src.model import fit_model, predict, market_probabilities
from src.staking import fractional_kelly, bankroll_simulator
from src.utils import closing_line_value


def walk_forward_backtest(
    game_log: pd.DataFrame,
    profile,
    min_train_size: int = 30,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 5.0,
) -> dict:
    """
    Walk-forward backtest: train on rolling window, predict next game.
    Returns calibration table, equity curve, ROI.
    """
    results = []
    n = len(game_log)

    for i in range(min_train_size, n):
        train = game_log.iloc[:i]
        test_game = game_log.iloc[i]

        fitted = fit_model(train, profile)
        if not fitted["fitted"]:
            continue

        pred = predict(test_game["home_team"], test_game["away_team"], fitted)
        probs = market_probabilities(pred)

        ft_line = test_game.get("ft_line", pred["ft_total"])
        actual_total = test_game["home_score"] + test_game["away_score"]
        actual_margin = test_game["home_score"] - test_game["away_score"]

        ft_over_actual = actual_total > ft_line
        ft_over_prob = probs["ft"]["over"]

        results.append({
            "game_idx": i,
            "ft_total_pred": pred["ft_total"],
            "ft_line": ft_line,
            "ft_over_prob": ft_over_prob,
            "ft_over_actual": ft_over_actual,
            "actual_total": actual_total,
            "margin_pred": pred["margin"],
            "margin_actual": actual_margin,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return {"calibration": None, "roi": 0.0, "n_bets": 0}

    # Calibration: bin predicted probabilities and check actual hit rate
    df["prob_bin"] = pd.cut(df["ft_over_prob"], bins=10, labels=False)
    calibration = df.groupby("prob_bin").agg(
        avg_pred_prob=("ft_over_prob", "mean"),
        actual_hit_rate=("ft_over_actual", "mean"),
        count=("ft_over_actual", "size"),
    ).reset_index()

    # ROI simulation (if odds available)
    if "odds" in game_log.columns:
        bets = []
        for _, row in df.iterrows():
            idx = row["game_idx"]
            if idx < len(game_log) and "odds" in game_log.columns:
                odds = game_log.iloc[idx].get("odds", None)
                if odds and odds > 1:
                    bets.append({
                        "model_prob": row["ft_over_prob"],
                        "odds": odds,
                        "result": "win" if row["ft_over_actual"] else "loss",
                    })
        if bets:
            equity = bankroll_simulator(bets, kelly_fraction=kelly_fraction, max_stake_pct=max_stake_pct)
            roi = (equity["bankroll"].iloc[-1] / equity["bankroll"].iloc[0] - 1) * 100
        else:
            equity = pd.DataFrame()
            roi = 0.0
    else:
        equity = pd.DataFrame()
        roi = 0.0

    return {
        "calibration": calibration,
        "equity": equity,
        "roi": round(roi, 2),
        "n_games": len(df),
        "predictions": df,
    }


def compute_clv(predictions_df: pd.DataFrame, closing_odds_col: str = "closing_odds") -> pd.Series:
    """Blessings: Track closing-line value."""
    if closing_odds_col not in predictions_df.columns:
        return pd.Series(dtype=float)
    return predictions_df.apply(
        lambda row: closing_line_value(row["ft_over_prob"], row[closing_odds_col]),
        axis=1,
    )
