# AI Stock Market Analyzer Dashboard

## Project Overview

The AI Stock Market Analyzer Dashboard is a beginner-friendly data science project built with Python, Pandas, NumPy, Matplotlib, Seaborn, SQLAlchemy, yfinance, Scikit-learn, Streamlit and Groq. The goal is to provide an interactive stock analysis dashboard for searching a ticker, downloading historical market data, cleaning the dataset, analyzing price and volume patterns, calculating technical and risk metrics, building a small machine learning forecast, and generating AI-assisted insights based on actual metrics.

## Problem Statement

Stock market analysis often feels difficult for beginners because it combines financial concepts, data wrangling, statistics, visualization, and prediction. This project helps bridge that gap by simplifying the workflow and showing real calculations in a clear dashboard.

## Project Objectives

- Fetch real historical market data using yfinance.
- Clean and validate market data.
- Store data in SQLite with SQLAlchemy.
- Perform EDA and technical analysis.
- Evaluate risk and correlation.
- Train a beginner model for next-day close estimation.
- Generate AI-assisted insights when a Groq API key is available.
- Deploy the app through Streamlit Community Cloud.

## Features

- Search stock symbols for US and Indian markets.
- Historical period filters: 1M, 3M, 6M, 1Y, 2Y, 5Y.
- Data cleaning and duplicate handling.
- SQLite database integration.
- Descriptive statistics and distribution analysis.
- Technical indicators: SMA 20, 50, 200.
- Risk metrics: daily volatility, annualized volatility, drawdown, Sharpe ratio.
- Correlation analysis across multiple tickers.
- Linear regression prediction for next-day close price.
- Groq AI insights with fallback summary when credentials are missing.
- Streamlit dashboard with custom dark theme.

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- yfinance
- SQLite
- SQLAlchemy
- Scikit-learn
- Streamlit
- Plotly
- HTML
- CSS
- Groq API
- Git
- GitHub
- Streamlit Community Cloud

## Architecture

The project follows a modular architecture:

- `app.py` contains the dashboard entry point.
- `config/config.py` stores configuration and environment variables.
- `src/data_collection` fetches market data.
- `src/preprocessing` cleans and validates data.
- `database` stores data in SQLite.
- `src/analysis` performs EDA, technical analysis, and risk analysis.
- `src/visualization` creates charts.
- `src/machine_learning` trains and predicts.
- `src/ai_insights` generates AI summaries and fallback insights.
- `components` holds reusable UI elements.

## Folder Structure

```text
AI-Stock-Market-Analyzer/
├── app.py
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
├── config/
│   ├── __init__.py
│   └── config.py
├── data/
│   ├── raw/
│   └── processed/
├── database/
│   ├── __init__.py
│   ├── db_connection.py
│   └── stock_database.py
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   └── stock_fetcher.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_cleaning.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py
│   │   ├── technical_analysis.py
│   │   ├── risk_analysis.py
│   │   └── correlation.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── charts.py
│   ├── machine_learning/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── train_model.py
│   │   └── predict.py
│   └── ai_insights/
│       ├── __init__.py
│       └── insight_generator.py
├── components/
│   ├── __init__?.py
│   ├── sidebar.py
│   ├── metrics.py
│   └── dashboard.py
├── assets/
│   └── custom.css
├── tests/
│   └── test_data_cleaning.py
├── .streamlit/
│   └── config.toml
└── .env.example
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AI-Stock-Market-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Execution Instructions

```bash
streamlit run app.py
```

## API Configuration

Create a `.env` file using `.env.example`:

```bash
copy .env.example .env
```

Then update the file with your Groq API key:

```bash
GROQ_API_KEY=your_api_key_here
```

For Streamlit Community Cloud, add the key in the app’s secrets manager or environment settings.

## Database Integration

The application connects to SQLite using SQLAlchemy. Each time a stock is fetched and cleaned, it is stored in the `stock_history` table to avoid duplicate records and enable retrieval by ticker.

## Data Science Methodology

The project follows a clear data science workflow:

1. Collect raw market data.
2. Validate and clean the dataset.
3. Explore statistics and distributions.
4. Calculate technical indicators.
5. Analyze risk and correlation.
6. Train a simple machine-learning model.
7. Generate AI-based summaries from calculated metrics.

## Machine Learning Methodology

The model uses a simple linear regression approach for educational next-day close estimation. It uses historical price features such as previous close values and moving averages. The model is trained chronologically without shuffling to avoid data leakage and uses only information available at prediction time.

## Deployment Instructions

### Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Log in to Streamlit Community Cloud.
3. Create a new app and point it to the repository.
4. Set the app entry file as `app.py`.
5. Add environment variables or secrets for `GROQ_API_KEY`.
6. Deploy the app.

### Deployment Notes

- The app is cloud-friendly and uses environment variables for API keys.
- It does not rely on uncommitted local files.
- Database storage is local to the app instance, so the app should be treated as a demo deployment rather than a production-scale financial system.

## Screenshots

No screenshots are included in this repository. Please run the app locally to view the dashboard.

## Limitations

- yfinance is an unofficial Yahoo Finance interface and data availability can vary.
- AI-generated responses depend on Groq API availability and rate limits.
- The model is educational and not suitable for investment decisions.
- SQLite is used for lightweight local storage rather than a production-grade database.

## Future Enhancements

- More advanced technical indicators.
- Portfolio tracking.
- Forecasting using time series models.
- More robust caching and performance tuning.
- CSV export and report generation.

## Financial Disclaimer

This dashboard is intended for educational and analytical purposes only. It does not provide investment advice, trading recommendations, or guaranteed predictive results.
