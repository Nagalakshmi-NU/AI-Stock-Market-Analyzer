"""Reusable data cleaning and preprocessing utilities for stock datasets."""

from __future__ import annotations

import pandas as pd
import numpy as np


def detect_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values per column."""
    if df is None or df.empty:
        return pd.Series(dtype=int)
    return df.isnull().sum()


def detect_duplicates(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows in the dataset."""
    if df is None:
        return 0
    return int(df.duplicated().sum())


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the stock dataset while preserving valid financial observations."""
    if df is None:
        raise ValueError("Input dataframe cannot be None.")

    cleaned = df.copy()
    if cleaned.empty:
        return cleaned

    cleaned["date"] = pd.to_datetime(cleaned.get("date", pd.Series(pd.NaT, index=cleaned.index)))
    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    cleaned = cleaned.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["volume"] = cleaned["volume"].fillna(0)
    numeric_columns = ["open", "high", "low", "close", "adj_close"]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace([np.inf, -np.inf], np.nan)
            cleaned[column] = cleaned[column].ffill().bfill()

    cleaned = cleaned.dropna(subset=["date", "close"]).reset_index(drop=True)
    cleaned["daily_return"] = cleaned["close"].pct_change().fillna(0)
    cleaned["daily_percentage_change"] = cleaned["close"].pct_change().fillna(0) * 100
    return cleaned


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-like columns to numeric types."""
    cleaned = df.copy()
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def validate_data_consistency(df: pd.DataFrame) -> bool:
    """Check that the data is logically consistent for stock analysis."""
    if df.empty:
        return False
    required = ["open", "high", "low", "close", "volume"]
    for column in required:
        if column in df.columns and df[column].isnull().all():
            return False
    if "high" in df.columns and "low" in df.columns:
        invalid_rows = df[(df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])].shape[0]
        if invalid_rows > 0:
            return False
    return True


def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate daily percentage returns from the closing price."""
    if "close" not in df.columns:
        raise KeyError("Missing 'close' column for daily return calculation.")
    return df["close"].pct_change().fillna(0)


def calculate_moving_average(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate the moving average for a given window size."""
    if "close" not in df.columns:
        raise KeyError("Missing 'close' column for moving average calculation.")
    return df["close"].rolling(window=window, min_periods=1).mean()
