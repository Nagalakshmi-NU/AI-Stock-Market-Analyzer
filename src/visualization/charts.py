"""Visualization helpers for the stock analysis dashboard."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def plot_price_history(df: pd.DataFrame) -> go.Figure:
    """Create a Plotly line chart for stock prices."""
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close Price", line=dict(color="#2dd4bf", width=2)))
    fig.update_layout(title="Historical Closing Price", template="plotly_dark", height=420, xaxis_title="Date", yaxis_title="Price")
    return fig


def plot_volume(df: pd.DataFrame) -> go.Figure:
    """Create a Plotly bar chart for trading volume."""
    if df.empty:
        return go.Figure()
    fig = go.Figure(data=[go.Bar(x=df["date"], y=df["volume"], marker_color="#60a5fa")])
    fig.update_layout(title="Trading Volume", template="plotly_dark", height=320, xaxis_title="Date", yaxis_title="Volume")
    return fig


def plot_moving_average(df: pd.DataFrame) -> go.Figure:
    """Create a chart with multiple moving averages."""
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    for column in ["sma_20", "sma_50", "sma_200"]:
        if column in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[column], mode="lines", name=column.upper()))
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close Price", line=dict(color="#f8fafc", width=2)))
    fig.update_layout(title="Moving Averages", template="plotly_dark", height=420, xaxis_title="Date", yaxis_title="Price")
    return fig


def plot_returns(df: pd.DataFrame) -> go.Figure:
    """Create a daily returns chart."""
    if df.empty:
        return go.Figure()
    fig = go.Figure(data=[go.Scatter(x=df["date"], y=df.get("daily_returns", 0), mode="lines", line=dict(color="#fbbf24", width=2))])
    fig.update_layout(title="Daily Returns", template="plotly_dark", height=320, xaxis_title="Date", yaxis_title="Return")
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create a heatmap for stock correlations using Plotly."""
    if corr_matrix.empty:
        return go.Figure()
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale="RdYlGn", zmid=0))
    fig.update_layout(title="Correlation Heatmap", template="plotly_dark", height=420, xaxis_title="Ticker", yaxis_title="Ticker")
    return fig


def plot_actual_vs_predicted(actual: pd.Series, predicted: pd.Series) -> go.Figure:
    """Create an actual versus predicted price chart."""
    if actual.empty or predicted.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines", name="Actual", line=dict(color="#22c55e")))
    fig.add_trace(go.Scatter(x=predicted.index, y=predicted.values, mode="lines", name="Predicted", line=dict(color="#f97316")))
    fig.update_layout(title="Actual vs Predicted Close Price", template="plotly_dark", height=380, xaxis_title="Index", yaxis_title="Price")
    return fig


def render_distribution_chart(series: pd.Series, title: str, color: str = "#38bdf8") -> go.Figure:
    """Create a distribution histogram for a series."""
    fig = px.histogram(series, title=title, color_discrete_sequence=[color], template="plotly_dark")
    fig.update_layout(height=350)
    return fig
