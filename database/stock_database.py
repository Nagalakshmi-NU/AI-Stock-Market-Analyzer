"""SQLAlchemy models and database operations for stock history."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, UniqueConstraint, select

from database.db_connection import Base, SessionLocal


class StockHistory(Base):
    """Table to store stock historical records."""

    __tablename__ = "stock_history"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(50), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    adj_close = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "adj_close": self.adj_close,
            "volume": self.volume,
        }


def save_stock_data(df: pd.DataFrame, ticker: str) -> None:
    """Insert cleaned stock data into SQLite if it does not already exist."""
    if df is None or df.empty:
        return

    session = SessionLocal()
    try:
        for _, row in df.iterrows():
            statement = select(StockHistory).where(
                StockHistory.ticker == ticker.upper(),
                StockHistory.date == row["date"],
            )
            existing = session.execute(statement).scalar_one_or_none()
            if existing is None:
                record = StockHistory(
                    ticker=ticker.upper(),
                    date=row["date"],
                    open=float(row.get("open", 0) or 0),
                    high=float(row.get("high", 0) or 0),
                    low=float(row.get("low", 0) or 0),
                    close=float(row.get("close", 0) or 0),
                    adj_close=float(row.get("adj_close", 0) or 0),
                    volume=float(row.get("volume", 0) or 0),
                )
                session.add(record)
            else:
                existing.open = float(row.get("open", existing.open) or existing.open)
                existing.high = float(row.get("high", existing.high) or existing.high)
                existing.low = float(row.get("low", existing.low) or existing.low)
                existing.close = float(row.get("close", existing.close) or existing.close)
                existing.adj_close = float(row.get("adj_close", existing.adj_close) or existing.adj_close)
                existing.volume = float(row.get("volume", existing.volume) or existing.volume)
        session.commit()
    finally:
        session.close()


def get_stock_history(ticker: str) -> pd.DataFrame:
    """Retrieve historical records for a ticker from the database."""
    session = SessionLocal()
    try:
        records = session.query(StockHistory).filter(StockHistory.ticker == ticker.upper()).order_by(StockHistory.date).all()
        if not records:
            return pd.DataFrame()
        data = pd.DataFrame([record.to_dict() for record in records])
        return data
    finally:
        session.close()


def update_existing_records(df: pd.DataFrame, ticker: str) -> None:
    """Synchronize stored rows with latest values."""
    save_stock_data(df, ticker)


def get_recent_tickers() -> list[str]:
    """Return all unique tickers currently in the database."""
    session = SessionLocal()
    try:
        tickers = session.query(StockHistory.ticker).distinct().all()
        return [ticker[0] for ticker in tickers]
    finally:
        session.close()
