"""Feature engineering utilities for stock price prediction."""

from __future__ import annotations

import pandas as pd


def create_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged and rolling price features used in the ML model."""
    result = df.copy()
    if result.empty:
        return result

    result["prev_close_1"] = result["close"].shift(1)
    result["prev_close_2"] = result["close"].shift(2)
    result["prev_close_3"] = result["close"].shift(3)
    result["rolling_mean_3"] = result["close"].rolling(window=3, min_periods=1).mean()
    result["rolling_mean_5"] = result["close"].rolling(window=5, min_periods=1).mean()
    result["rolling_mean_10"] = result["close"].rolling(window=10, min_periods=1).mean()
    result["volume_change"] = result["volume"].pct_change().fillna(0)
    result["close_change"] = result["close"].pct_change().fillna(0)
    result = result.dropna().reset_index(drop=True)
    return result


def prepare_ml_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare the independent and target variables for supervised learning."""
    features = create_prediction_features(df)
    if features.empty:
        raise ValueError("Not enough data to train the model.")

    target = features["close"].shift(-1).fillna(features["close"])
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
    X = features[feature_columns]
    y = target
    return X, y
