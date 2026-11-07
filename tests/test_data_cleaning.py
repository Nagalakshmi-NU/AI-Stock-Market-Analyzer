"""Tests for data cleaning and financial calculations."""

import pandas as pd

from database.stock_database import get_stock_history, save_stock_data
from src.ai_insights.insight_generator import generate_ai_insights
from src.analysis.correlation import build_correlation_matrix
from src.analysis.risk_analysis import calculate_annualized_volatility, calculate_daily_volatility
from src.analysis.technical_analysis import calculate_moving_average
from src.preprocessing.data_cleaning import calculate_daily_returns, clean_stock_data, detect_duplicates, detect_missing_values


def build_sample_stock_data() -> pd.DataFrame:
    """Create a small sample dataset for unit tests."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.5, 102.0, 103.3, 104.8],
            "adj_close": [100.0, 101.5, 102.0, 103.3, 104.8],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )


def test_missing_values_detection() -> None:
    """Missing value detection should return a count for null entries."""
    df = build_sample_stock_data().copy()
    df.loc[0, "close"] = None
    missing = detect_missing_values(df)
    assert missing["close"] == 1


def test_duplicate_detection() -> None:
    """Duplicate rows should be counted correctly."""
    df = build_sample_stock_data().copy()
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert detect_duplicates(duplicated) == 1


def test_clean_stock_data_handles_invalid_rows() -> None:
    """The cleaning routine should remove invalid values and preserve valid rows."""
    df = build_sample_stock_data().copy()
    df.loc[2, "close"] = None
    result = clean_stock_data(df)
    assert not result.empty
    assert "daily_return" in result.columns


def test_daily_returns_calculation() -> None:
    """Daily returns should be computed from close price percent change."""
    df = build_sample_stock_data()
    daily_returns = calculate_daily_returns(df)
    assert daily_returns.iloc[1] > 0


def test_moving_average_calculation() -> None:
    """Moving average should smooth the close price series."""
    df = build_sample_stock_data()
    moving_average = calculate_moving_average(df, 3)
    assert len(moving_average) == len(df)
    assert moving_average.iloc[-1] > 0


def test_risk_calculations() -> None:
    """Volatility calculations should return non-negative results."""
    df = build_sample_stock_data()
    daily_returns = df["close"].pct_change().fillna(0)
    assert calculate_daily_volatility(daily_returns) >= 0
    assert calculate_annualized_volatility(daily_returns) >= 0


def test_database_insert_and_retrieve() -> None:
    """Stored stock records should be inserted and retrieved by ticker."""
    df = build_sample_stock_data()
    save_stock_data(df, "TEST")
    saved = get_stock_history("TEST")
    assert not saved.empty
    assert saved["ticker"].iloc[0].upper() == "TEST"


def test_correlation_matrix_computes_for_multiple_stocks() -> None:
    """Daily return correlation should be computed for multiple stock dataframes."""
    df_a = build_sample_stock_data().rename(columns={"close": "close"})
    df_b = build_sample_stock_data().copy()
    df_b["close"] = df_b["close"] * 1.05
    matrix = build_correlation_matrix({"A": df_a, "B": df_b})
    assert matrix.shape == (2, 2)
    assert matrix.loc["A", "B"] >= -1 and matrix.loc["A", "B"] <= 1


def test_ai_fallback_insights_are_generated_without_api_key() -> None:
    """AI insight generation should provide a fallback summary when no Groq key is configured."""
    df = build_sample_stock_data()
    metrics = {"annualized_volatility": 0.25, "average_daily_return": 0.01}
    summary, status = generate_ai_insights(df, metrics)
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "Fallback" in status or "Groq" in status or "fallback" in status.lower()
