"""Train a baseline linear regression model for next-day close prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.machine_learning.feature_engineering import prepare_ml_dataset


def train_linear_regression_model(df: pd.DataFrame) -> dict[str, object]:
    """Train a linear regression model on chronological data using a split based on time."""
    if df is None or df.empty:
        raise ValueError("Dataframe is empty; cannot train the model.")

    X, y = prepare_ml_dataset(df)
    if len(X) < 10:
        raise ValueError("Not enough observations to train a robust model.")

    split_index = int(len(X) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    result = {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": pd.Series(predictions, index=y_test.index),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    return result
