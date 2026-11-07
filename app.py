"""AI Stock Market Analyzer Dashboard main Streamlit application."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from components.dashboard import render_section_heading
from components.metrics import render_metric_cards
from components.sidebar import render_sidebar
from config.config import BASE_DIR, GROQ_API_KEY
from database.db_connection import initialize_database
from database.stock_database import get_recent_tickers, get_stock_history, save_stock_data
from src.ai_insights.insight_generator import generate_ai_insights
from src.analysis.correlation import build_correlation_matrix
from src.analysis.eda import calculate_price_statistics, calculate_volume_statistics, describe_stock_data
from src.analysis.risk_analysis import calculate_risk_summary
from src.analysis.technical_analysis import add_technical_indicators
from src.data_collection.stock_fetcher import fetch_multiple_stocks, fetch_stock_history, get_stock_suggestions
from src.machine_learning.feature_engineering import create_prediction_features
from src.machine_learning.predict import predict_next_day_close
from src.machine_learning.train_model import train_linear_regression_model
from src.preprocessing.data_cleaning import clean_stock_data, detect_duplicates, detect_missing_values, validate_data_consistency
from src.visualization.charts import plot_actual_vs_predicted, plot_correlation_heatmap, plot_moving_average, plot_price_history, plot_returns, plot_volume, render_distribution_chart

st.set_page_config(page_title="AI Stock Market Analyzer", page_icon="📈", layout="wide")

with open(BASE_DIR / "assets" / "custom.css", "r", encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

initialize_database()


@st.cache_data
def load_stock_data(symbol: str, period: str) -> pd.DataFrame:
    """Cache and return cleaned stock data for a selected symbol and period."""
    data = fetch_stock_history(symbol, period)
    cleaned = clean_stock_data(data)
    return cleaned


@st.cache_data
def load_multiple_stock_data(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Cache multiple stock data frames loaded for correlation analysis."""
    return fetch_multiple_stocks(symbols, period)


st.title("AI Stock Market Analyzer Dashboard")
st.caption("Beginner-friendly stock analytics with real market data, cleaning, technical indicators, risk metrics, machine learning and AI explanations.")

symbol_input, period_value, compare_symbols, period_label = render_sidebar()
selected_ticker = symbol_input.strip().upper() if symbol_input.strip() else "AAPL"

if selected_ticker not in get_stock_suggestions() and selected_ticker not in get_recent_tickers():
    st.sidebar.warning("Ticker is not in the quick suggestions. The app still attempts to fetch it if it exists.")

try:
    stock_data = load_stock_data(selected_ticker, period_value)
except Exception as exc:
    st.error(f"Unable to load data for {selected_ticker}. Please check the ticker or network connection. Error: {exc}")
    stock_data = pd.DataFrame()

if not stock_data.empty:
    missing_values = detect_missing_values(stock_data)
    duplicates = detect_duplicates(stock_data)
    consistency_ok = validate_data_consistency(stock_data)

    save_stock_data(stock_data, selected_ticker)
    stock_data = add_technical_indicators(stock_data)

    if "close" in stock_data.columns:
        latest_close = float(stock_data["close"].iloc[-1])
        previous_close = float(stock_data["close"].iloc[-2]) if len(stock_data) > 1 else latest_close
        daily_change = latest_close - previous_close
        pct_change = (daily_change / previous_close) * 100 if previous_close else 0.0
    else:
        latest_close = 0.0
        daily_change = 0.0
        pct_change = 0.0

    metrics = {
        "latest_close": latest_close,
        "daily_change": daily_change,
        "pct_change": pct_change,
        "highest_price": float(stock_data["high"].max()) if "high" in stock_data.columns else 0.0,
        "lowest_price": float(stock_data["low"].min()) if "low" in stock_data.columns else 0.0,
        "average_volume": float(stock_data["volume"].mean()) if "volume" in stock_data.columns else 0.0,
    }

    risk_metrics = calculate_risk_summary(stock_data)
    metrics.update({
        "annualized_volatility": risk_metrics["annualized_volatility"],
        "max_drawdown": risk_metrics["max_drawdown"],
        "sharpe_ratio": risk_metrics["sharpe_ratio"],
    })

    render_metric_cards(stock_data)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        render_section_heading("Historical Stock Analysis", "View price movement, volume, and indicators across the selected range.")
        st.plotly_chart(plot_price_history(stock_data), use_container_width=True)
        with st.expander("OHLC Information"):
            st.dataframe(stock_data[["date", "open", "high", "low", "close", "volume"]].tail(10).reset_index(drop=True), use_container_width=True)
        st.plotly_chart(plot_volume(stock_data), use_container_width=True)
        st.plotly_chart(plot_moving_average(stock_data), use_container_width=True)
    with col2:
        render_section_heading("Overview", "Selected stock and the latest market snapshot.")
        st.markdown(f"<div class='metric-card'><h4>Selected Stock</h4><p>{selected_ticker}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Latest Available Closing Price</h4><p>${latest_close:,.2f}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Daily Change</h4><p>${daily_change:,.2f}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Percentage Change</h4><p>{pct_change:.2f}%</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Highest Price</h4><p>${metrics['highest_price']:,.2f}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Lowest Price</h4><p>${metrics['lowest_price']:,.2f}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Average Volume</h4><p>{metrics['average_volume']:,.0f}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Annualized Volatility</h4><p>{metrics['annualized_volatility']:.4f}</p></div>", unsafe_allow_html=True)

    render_section_heading("Exploratory Data Analysis", "Descriptive statistics and distributions from the cleaned stock dataset.")
    stats = describe_stock_data(stock_data)
    st.dataframe(stats, use_container_width=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.plotly_chart(render_distribution_chart(stock_data["close"], "Closing Price Distribution", color="#34d399"), use_container_width=True)
    with col_b:
        st.plotly_chart(render_distribution_chart(stock_data["close"].pct_change().fillna(0), "Daily Returns Distribution", color="#fbbf24"), use_container_width=True)
    with col_c:
        st.plotly_chart(render_distribution_chart(stock_data["volume"], "Volume Distribution", color="#60a5fa"), use_container_width=True)

    render_section_heading("Technical Analysis", "Moving averages and return metrics for trend observation.")
    st.plotly_chart(plot_returns(stock_data), use_container_width=True)
    st.dataframe(stock_data[["date", "sma_20", "sma_50", "sma_200", "daily_returns", "cumulative_returns"]].tail(20), use_container_width=True)

    render_section_heading("Risk Analysis", "Volatility and risk indicators using daily returns and market drawdown.")
    st.write(risk_metrics)
    st.write("Assumption: annualized volatility uses a 252-day trading year and the risk-free rate is assumed to be 3% per year. This is educational and not financial advice.")

    render_section_heading("Correlation Analysis", "Choose multiple stocks to compare their return correlations.")
    comparison_data = load_multiple_stock_data(compare_symbols if compare_symbols else [selected_ticker], period_value)
    if comparison_data:
        corr_matrix = build_correlation_matrix(comparison_data)
        st.dataframe(corr_matrix, use_container_width=True)
        st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)
    else:
        st.info("Insufficient overlapping data to compute a meaningful correlation matrix.")

    render_section_heading("Machine Learning", "Linear regression model for next-trading-day close estimation.")
    ml_data = stock_data.copy()
    try:
        training_result = train_linear_regression_model(ml_data)
        model = training_result["model"]
        predicted_values = training_result["predictions"]
        actual_values = training_result["y_test"]
        next_day_prediction = predict_next_day_close(model, ml_data)
        st.write(f"Algorithm: Linear Regression")
        st.write(f"Training data size: {len(training_result['X_train'])} rows")
        st.write(f"Testing data size: {len(training_result['X_test'])} rows")
        st.write(f"MAE: {training_result['mae']:.2f}")
        st.write(f"RMSE: {training_result['rmse']:.2f}")
        st.write(f"R²: {training_result['r2']:.4f}")
        st.write(f"Estimated next trading-day closing price: ${next_day_prediction:,.2f}")
        st.plotly_chart(plot_actual_vs_predicted(actual_values, predicted_values), use_container_width=True)
        st.caption("This is a beginner-friendly educational model. It does not guarantee future stock prices and may not capture market regime changes.")
    except Exception as exc:
        st.warning(f"Machine learning model could not be trained with the current dataset: {exc}")

    render_section_heading("AI Insights", "AI-generated analyses based on actual metrics. The project falls back gracefully without a Groq key.")
    ai_summary, status = generate_ai_insights(stock_data, metrics)
    st.success(status)
    st.write(ai_summary)

    render_section_heading("About Project", "Project description, features, and disclaimer.")
    st.markdown(
        """
        This dashboard brings together stock data collection, data cleaning, database storage, exploratory analysis, technical indicators, risk evaluation, correlation mapping, machine learning prediction, and AI summarization into one educational application.

        Objectives:
        - Learn data collection with yfinance.
        - Clean and validate financial data.
        - Run EDA using Pandas, NumPy, Matplotlib and Seaborn.
        - Build beginner-friendly stock calculations.
        - Explore simple ML forecasting.
        - Produce AI-assisted summaries using Groq when available.

        Technology stack: Python, Pandas, NumPy, Matplotlib, Seaborn, yfinance, SQLite, SQLAlchemy, Scikit-learn, Streamlit, Plotly, HTML, CSS, Groq API.

        Disclaimer: This dashboard is for educational purposes only and is not investment advice.
        """
    )

    st.sidebar.success(f"Data loaded for {selected_ticker} using {period_label} period. Missing values: {int(missing_values.sum())}, duplicates: {duplicates}, consistency valid: {consistency_ok}.")
else:
    st.warning("No valid data is available for the selected stock symbol. Please try another ticker or check network connectivity.")

st.sidebar.markdown("---")
st.sidebar.markdown("Security reminder: keep `.env` and Streamlit secrets private and never hardcode API keys.")
