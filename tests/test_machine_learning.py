"""Tests for the stock prediction model and evaluation metrics."""

from __future__ import annotations

import pandas as pd

from src.machine_learning.feature_engineering import prepare_ml_dataset
from src.machine_learning.predict import predict_next_day_close
from src.machine_learning.train_model import train_linear_regression_model


def create_test_stock_frame() -> pd.DataFrame:
    """Create a small ordered stock dataset for model training tests."""
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    closes = [100 + i * 0.7 + (i % 5) * 0.4 for i in range(40)]
    volumes = [1000 + i * 15 for i in range(40)]

    data = pd.DataFrame(
        {
            "date": dates,
            "open": [value - 1.5 for value in closes],
            "high": [value + 2.2 for value in closes],
            "low": [value - 2.0 for value in closes],
            "close": closes,
            "adj_close": closes,
            "volume": volumes,
        }
    )
    return data


def test_prepare_ml_dataset_creates_features() -> None:
    """The feature engineering should create lagged and rolling features."""
    df = create_test_stock_frame()
    X, y = prepare_ml_dataset(df)
    assert not X.empty
    assert not y.empty
    assert set(["prev_close_1", "prev_close_2", "prev_close_3", "rolling_mean_3", "rolling_mean_5", "rolling_mean_10", "volume_change", "close_change"]).issubset(X.columns)


def test_train_linear_regression_model_runs() -> None:
    """The model should train successfully and produce valid evaluation metrics."""
    df = create_test_stock_frame()
    result = train_linear_regression_model(df)
    assert result["model"] is not None
    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert result["r2"] is not None


def test_predict_next_day_close_returns_numeric_value() -> None:
    """The next-day prediction should return a numeric price using the trained model."""
    df = create_test_stock_frame()
    result = train_linear_regression_model(df)
    prediction = predict_next_day_close(result["model"], df)
    assert isinstance(prediction, float)
    assert prediction > 0


def test_chronological_split_has_no_data_leakage() -> None:
    """Training data must remain earlier in time than testing data with no shuffling."""
    df = create_test_stock_frame()
    result = train_linear_regression_model(df)
    train_index = result["X_train"].index
    test_index = result["X_test"].index

    assert train_index.max() < test_index.min()
    assert train_index.min() == 0
    assert test_index.min() > 0
