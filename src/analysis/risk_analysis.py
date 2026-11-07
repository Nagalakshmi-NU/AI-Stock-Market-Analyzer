"""Volatility and risk calculations for stock portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_daily_volatility(daily_returns: pd.Series) -> float:
    """Return the standard deviation of daily returns."""
    return float(daily_returns.std(ddof=1)) if not daily_returns.empty else 0.0


def calculate_annualized_volatility(daily_returns: pd.Series, annualization_factor: float = 252.0) -> float:
    """Annualize daily volatility using a standard trading-day assumption."""
    daily_vol = calculate_daily_volatility(daily_returns)
    return float(daily_vol * np.sqrt(annualization_factor)) if daily_vol else 0.0


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown as a percentage."""
    if prices.empty:
        return 0.0
    cumulative = prices
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min()) * 100


def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.03, annualization_factor: float = 252.0) -> float:
    """Calculate Sharpe ratio using annualized risk-free rate assumption."""
    if daily_returns.empty:
        return 0.0
    avg_return = daily_returns.mean()
    vol = daily_returns.std(ddof=1)
    if vol == 0:
        return 0.0
    annualized_return = (1 + avg_return) ** annualization_factor - 1
    excess_return = annualized_return - risk_free_rate
    return float(excess_return / (vol * np.sqrt(annualization_factor)))


def calculate_risk_summary(df: pd.DataFrame, risk_free_rate: float = 0.03) -> dict[str, float | str]:
    """Compute a summary of volatility and risk measures for display."""
    daily_returns = df["close"].pct_change().fillna(0) if "close" in df.columns else pd.Series([], dtype=float)
    summary = {
        "average_daily_return": float(daily_returns.mean()) if not daily_returns.empty else 0.0,
        "daily_volatility": calculate_daily_volatility(daily_returns),
        "annualized_volatility": calculate_annualized_volatility(daily_returns),
        "max_drawdown": calculate_max_drawdown(df["close"]) if "close" in df.columns else 0.0,
        "sharpe_ratio": calculate_sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate),
        "assumption": "Risk-free rate assumed to be 3% per year and daily volatility annualized using 252 trading days.",
    }
    return summary
