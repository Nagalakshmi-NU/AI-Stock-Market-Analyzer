"""Sidebar configuration for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from config.config import PERIOD_OPTIONS, SUGGESTED_TICKERS


def render_sidebar() -> tuple[str, str, list[str], str]:
    """Render the dashboard sidebar and return key user selections."""
    st.sidebar.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    st.sidebar.title("Stock Analyzer")
    st.sidebar.markdown("yfinance is an unofficial Yahoo Finance data interface. Availability may vary by ticker and market conditions.")

    ticker = st.sidebar.text_input("Stock ticker", value="AAPL")
    period = st.sidebar.selectbox("Historical period", list(PERIOD_OPTIONS.keys()), index=3)
    selected_symbols = st.sidebar.multiselect("Compare stocks", SUGGESTED_TICKERS, default=["AAPL", "MSFT"])
    st.sidebar.caption("Use official ticker symbols, including Indian stocks like INFY.NS or RELIANCE.NS.")

    return ticker, PERIOD_OPTIONS[period], selected_symbols, period
