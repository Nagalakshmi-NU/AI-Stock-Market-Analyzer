"""Prediction helpers for the stock price forecasting model."""

from __future__ import annotations

import pandas as pd

from src.machine_learning.feature_engineering import create_prediction_features


def predict_next_day_close(model, df: pd.DataFrame) -> float:
    """Predict the next trading-day close price for the latest available data."""
    features = create_prediction_features(df).tail(1)
    if features.empty:
        raise ValueError("No stock data available for prediction.")

    feature_columns = [
        "prev_close_1",
        "prev_close_2",
        "prev_close_3",
        "rolling_mean_3",
        "rolling_mean_5",
        "rolling_mean_10",
        "volume_change",
        "close_change",
    ]
    row = features[feature_columns]
    prediction = model.predict(row)
    return float(prediction[0])
