"""Exploratory data analysis utilities for stock data."""

from __future__ import annotations

import pandas as pd
import numpy as np


def describe_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for price and volume columns."""
    numeric_columns = [column for column in ["open", "high", "low", "close", "adj_close", "volume"] if column in df.columns]
    return df[numeric_columns].describe().T


def calculate_price_statistics(df: pd.DataFrame) -> dict[str, float]:
    """Compute key price statistics."""
    stats = {
        "mean_close": float(df["close"].mean()) if "close" in df.columns else 0.0,
        "median_close": float(df["close"].median()) if "close" in df.columns else 0.0,
        "std_close": float(df["close"].std()) if "close" in df.columns else 0.0,
        "min_close": float(df["close"].min()) if "close" in df.columns else 0.0,
        "max_close": float(df["close"].max()) if "close" in df.columns else 0.0,
        "highest_price": float(df["high"].max()) if "high" in df.columns else 0.0,
        "lowest_price": float(df["low"].min()) if "low" in df.columns else 0.0,
    }
    return stats


def calculate_volume_statistics(df: pd.DataFrame) -> dict[str, float]:
    """Compute key volume statistics."""
    stats = {
        "avg_volume": float(df["volume"].mean()) if "volume" in df.columns else 0.0,
        "total_volume": float(df["volume"].sum()) if "volume" in df.columns else 0.0,
        "max_volume": float(df["volume"].max()) if "volume" in df.columns else 0.0,
        "min_volume": float(df["volume"].min()) if "volume" in df.columns else 0.0,
    }
    return stats


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily return and cumulative return series."""
    result = df.copy()
    if "close" in result.columns:
        result["daily_returns"] = result["close"].pct_change().fillna(0)
        result["cumulative_returns"] = (1 + result["daily_returns"]).cumprod() - 1
    else:
        result["daily_returns"] = 0
        result["cumulative_returns"] = 0
    return result


def calculate_mode(values: pd.Series) -> float | None:
    """Return mode for a numeric series when available."""
    if values.empty:
        return None
    mode = values.mode(dropna=True)
    return float(mode.iloc[0]) if not mode.empty else None


def quartile_summary(values: pd.Series) -> dict[str, float]:
    """Return quartile-based summary."""
    if values.empty:
        return {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0}
    return {
        "Q1": float(values.quantile(0.25)),
        "Q2": float(values.quantile(0.50)),
        "Q3": float(values.quantile(0.75)),
    }
