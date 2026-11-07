"""Functions to fetch and validate stock market data using yfinance."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from config.config import PERIOD_OPTIONS, SUGGESTED_TICKERS


def validate_ticker(symbol: str) -> str:
    """Normalize the provided ticker symbol."""
    if not isinstance(symbol, str):
        raise ValueError("Ticker must be a string.")
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("Ticker cannot be empty.")
    return normalized


def fetch_stock_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical data for the selected symbol using yfinance.

    Note: yfinance is an unofficial Yahoo Finance interface, so data availability can vary.
    Some tickers may be unavailable or delayed depending on the exchange and market conditions.
    """
    normalized = validate_ticker(symbol)
    if period not in PERIOD_OPTIONS.values():
        normalized_period = PERIOD_OPTIONS.get(period, "1y")
    else:
        normalized_period = period

    try:
        stock = yf.Ticker(normalized)
        data = stock.history(period=normalized_period, auto_adjust=False)
    except Exception as exc:  # pragma: no cover - network and API variability
        raise RuntimeError(f"Unable to fetch stock data for {normalized}: {exc}") from exc

    if data.empty:
        raise ValueError(f"No historical data available for ticker '{normalized}'.")

    data = data.reset_index()
    data.columns = [str(column).strip().lower().replace(" ", "_") for column in data.columns]
    data = data.rename(columns={"adj_close": "adj_close"})
    if "date" not in data.columns:
        data["date"] = pd.to_datetime(data.index)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    return data


def fetch_multiple_stocks(symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch multiple stock histories and return a dictionary keyed by symbol."""
    results: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            history = fetch_stock_history(symbol, period)
            results[symbol.upper()] = history
        except Exception:
            continue
    return results


def get_stock_suggestions() -> list[str]:
    """Return the suggested ticker list."""
    return SUGGESTED_TICKERS


def ensure_valid_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist before analysis."""
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        for column in missing:
            df[column] = pd.NA
    return df
