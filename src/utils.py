"""Normal + bivariate maths, push handling (Blessings: correct discreteness)."""
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal


# ── Univariate total probabilities with push handling ────────────────────

def prob_over(line: float, mu: float, sigma: float) -> float:
    """P(Total > line) with continuity correction for integer lines."""
    if line == int(line):
        # Integer line: push is possible
        return 1.0 - norm.cdf(line + 0.5, loc=mu, scale=sigma)
    return 1.0 - norm.cdf(line, loc=mu, scale=sigma)


def prob_under(line: float, mu: float, sigma: float) -> float:
    """P(Total < line) with continuity correction."""
    if line == int(line):
        return norm.cdf(line - 0.5, loc=mu, scale=sigma)
    return norm.cdf(line, loc=mu, scale=sigma)


def prob_push(line: float, mu: float, sigma: float) -> float:
    """P(Total == line) — only nonzero for integer lines."""
    if line == int(line):
        return norm.cdf(line + 0.5, loc=mu, scale=sigma) - norm.cdf(line - 0.5, loc=mu, scale=sigma)
    return 0.0


def total_probabilities(line: float, mu: float, sigma: float) -> dict:
    """Return {over, under, push} that sum to exactly 1.0."""
    over = prob_over(line, mu, sigma)
    under = prob_under(line, mu, sigma)
    push = prob_push(line, mu, sigma)
    # Normalise to guarantee sum == 1.0 (floating-point guard)
    total = over + under + push
    return {"over": over / total, "under": under / total, "push": push / total}


# ── Bivariate normal for Winner & Total combo market ─────────────────────

def bivariate_winner_total(
    margin_mu: float, margin_sigma: float,
    total_mu: float, total_sigma: float,
    corr: float,
    margin_line: float = 0.0,
    total_line: float = None,
) -> dict:
    """
    P(Home wins & Total over/under/push) from bivariate normal.
    Blessings: fair joint price for combo market.
    """
    if total_line is None:
        total_line = total_mu
    cov = corr * margin_sigma * total_sigma
    mean = [margin_mu, total_mu]
    cov_matrix = [[margin_sigma**2, cov], [cov, total_sigma**2]]

    # P(Home wins) = P(margin > 0)
    p_home_win = 1.0 - norm.cdf(0, loc=margin_mu, scale=margin_sigma)
    p_away_win = 1.0 - p_home_win

    # Total probs (univariate)
    total_probs = total_probabilities(total_line, total_mu, total_sigma)

    # Joint: P(Home win AND Total over) via bivariate CDF
    # P(margin > 0, total > line) = 1 - P(margin<=0) - P(total<=line) + P(margin<=0, total<=line)
    mvn = multivariate_normal(mean=mean, cov=cov_matrix)

    # Numerical integration via Monte Carlo for joint probabilities
    n_samples = 50000
    rng = np.random.default_rng(42)
    samples = rng.multivariate_normal(mean, cov_matrix, size=n_samples)
    margins = samples[:, 0]
    totals = samples[:, 1]

    p_home_over = np.mean((margins > 0) & (totals > total_line + 0.5)) if total_line == int(total_line) else np.mean((margins > 0) & (totals > total_line))
    p_home_under = np.mean((margins > 0) & (totals < total_line - 0.5)) if total_line == int(total_line) else np.mean((margins > 0) & (totals < total_line))
    p_home_push = np.mean((margins > 0) & (np.abs(totals - total_line) <= 0.5)) if total_line == int(total_line) else 0.0

    p_away_over = np.mean((margins <= 0) & (totals > total_line + 0.5)) if total_line == int(total_line) else np.mean((margins <= 0) & (totals > total_line))
    p_away_under = np.mean((margins <= 0) & (totals < total_line - 0.5)) if total_line == int(total_line) else np.mean((margins <= 0) & (totals < total_line))
    p_away_push = np.mean((margins <= 0) & (np.abs(totals - total_line) <= 0.5)) if total_line == int(total_line) else 0.0

    return {
        "home_over": p_home_over, "home_under": p_home_under, "home_push": p_home_push,
        "away_over": p_away_over, "away_under": p_away_under, "away_push": p_away_push,
        "p_home_win": p_home_win, "p_away_win": p_away_win,
        "total_over": total_probs["over"], "total_under": total_probs["under"], "total_push": total_probs["push"],
    }


# ── CLV calculation (Blessings) ──────────────────────────────────────────

def closing_line_value(model_prob: float, closing_odds: float) -> float:
    """CLV = model_prob - implied_prob_from_closing_odds."""
    implied = 1.0 / closing_odds
    return model_prob - implied
