"""Metric cards and summary panel helpers for the dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_metric_cards(df: pd.DataFrame) -> None:
    """Display key overview metrics in Streamlit cards."""
    if df.empty:
        return

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    close_change = latest["close"] - previous["close"]
    pct_change = (close_change / previous["close"]) * 100 if previous["close"] else 0.0
    avg_volume = df["volume"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Latest Close", f"${latest['close']:.2f}")
    col2.metric("Daily Change", f"${close_change:.2f}", delta=f"{pct_change:.2f}%")
    col3.metric("High", f"${df['high'].max():.2f}")
    col4.metric("Low", f"${df['low'].min():.2f}")
    col5.metric("Avg Volume", f"{avg_volume:,.0f}")
