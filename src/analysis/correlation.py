"""Correlation analysis utilities for multiple stock comparisons."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_daily_returns_for_multiple_stocks(dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Return each stock's daily return series aligned by date."""
    returns: dict[str, pd.Series] = {}
    for symbol, df in dataframes.items():
        if "close" not in df.columns:
            continue
        series = df[["date", "close"]].copy()
        series["daily_return"] = series["close"].pct_change().fillna(0)
        returns[symbol] = series.set_index("date")["daily_return"]
    return returns


def build_correlation_matrix(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple daily returns and compute the correlation matrix."""
    returns_by_symbol = compute_daily_returns_for_multiple_stocks(dataframes)
    if not returns_by_symbol:
        return pd.DataFrame()

    merged = pd.concat(returns_by_symbol.values(), axis=1)
    merged.columns = list(returns_by_symbol.keys())
    return merged.corr().fillna(0)


def compute_pairwise_correlation(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Compute the daily return correlation between two stocks."""
    if df_a.empty or df_b.empty:
        return 0.0
    price_a = df_a["close"].pct_change().fillna(0)
    price_b = df_b["close"].pct_change().fillna(0)
    overlap = pd.concat([price_a.rename("A"), price_b.rename("B")], axis=1).dropna()
    if overlap.empty:
        return 0.0
    return float(overlap["A"].corr(overlap["B"]))
