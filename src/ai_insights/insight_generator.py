"""AI insight generation and fallback metric-based analysis."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from config.config import BASE_DIR, GROQ_API_KEY


def get_groq_client() -> Groq | None:
    """Create and return the Groq client using environment variables or Streamlit secrets."""
    api_key = None

    try:
        import streamlit as st

        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
        elif hasattr(st, "secrets") and "groq" in st.secrets:
            api_key = st.secrets["groq"].get("api_key")
    except Exception:
        api_key = None

    if not api_key:
        load_dotenv(BASE_DIR / ".env")
        api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY

    if not api_key:
        return None
    return Groq(api_key=str(api_key))


def build_fallback_summary(df: pd.DataFrame, metrics: dict[str, Any]) -> str:
    """Generate an analytical summary without relying on Groq."""
    if df.empty:
        return "No data is available to analyze yet."

    latest_close = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
    avg_volume = float(df["volume"].mean()) if "volume" in df.columns else 0.0
    volatility = float(metrics.get("annualized_volatility", 0.0))
    trend = "uptrend" if df["close"].iloc[-1] >= df["close"].iloc[0] else "downtrend"
    trend_text = f"The stock shows a {trend} over the selected period, with a latest close of {latest_close:,.2f}."
    volume_text = f"Average volume is {avg_volume:,.0f}, which helps contextualize liquidity and participation."
    risk_text = f"Annualized volatility is approximately {volatility:.2f} and should be interpreted as a measure of daily price dispersion, not as investment advice."
    return " ".join([trend_text, volume_text, risk_text])


def generate_ai_insights(df: pd.DataFrame, metrics: dict[str, Any]) -> tuple[str, str]:
    """Generate model-based insights with a fallback summary if no API key is available."""
    summary = build_fallback_summary(df, metrics)
    client = get_groq_client()
    if client is None:
        return summary, "Fallback analysis is active because no Groq key is configured."

    prompt = (
        "You are a careful financial analysis assistant. Use only the provided data and do not invent figures. "
        "Explain trends, volatility, volume, risk, and moving average context based on the metrics below. "
        "Keep the answer concise and educational. Do not provide investment advice.\n\n"
        f"Data summary: {metrics}\n"
        "Return exactly two paragraphs: first a trend and risk summary, second a moving-average and volume explanation."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250,
        )
        text = response.choices[0].message.content.strip()
        if not text:
            return summary, "Groq returned an empty response; fallback analysis is shown."
        return text, "Groq analysis generated successfully."
    except Exception:
        return summary, "Groq request failed or rate limits were hit; fallback analysis is shown."
