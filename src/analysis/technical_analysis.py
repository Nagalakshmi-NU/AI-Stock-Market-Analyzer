"""Technical analysis calculations for stock data."""

from __future__ import annotations

import pandas as pd


def calculate_moving_average(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute the simple moving average for the selected window."""
    if "close" not in df.columns:
        raise KeyError("Missing 'close' column for moving average calculation.")
    return df["close"].rolling(window=window, min_periods=1).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving average, daily return, and cumulative return columns."""
    result = df.copy()
    if result.empty:
        return result
    result["sma_20"] = calculate_moving_average(result, 20)
    result["sma_50"] = calculate_moving_average(result, 50)
    result["sma_200"] = calculate_moving_average(result, 200)
    result["daily_returns"] = result["close"].pct_change().fillna(0)
    result["cumulative_returns"] = (1 + result["daily_returns"]).cumprod() - 1
    result["daily_percentage_change"] = result["daily_returns"] * 100
    return result


def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    """Return daily returns for the closing price."""
    if "close" not in df.columns:
        raise KeyError("Missing 'close' column for daily return calculation.")
    return df["close"].pct_change().fillna(0)


def calculate_cumulative_returns(df: pd.DataFrame) -> pd.Series:
    """Return cumulative return values."""
    returns = calculate_daily_returns(df)
    return (1 + returns).cumprod() - 1
