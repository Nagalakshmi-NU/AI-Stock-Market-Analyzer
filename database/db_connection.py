"""Database connection helpers for SQLite and SQLAlchemy."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config.config import DB_PATH

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db_session():
    """Return a database session for operations."""
    return SessionLocal()


def initialize_database() -> None:
    """Create all database tables if they do not already exist."""
    from database.stock_database import StockHistory

    Base.metadata.create_all(bind=engine)
    return None
