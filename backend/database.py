"""
database.py — SQLite database setup for CattleVision AI.

Uses SQLite (a single local file, no separate server process) rather than
MongoDB or a hosted SQL server, so the whole project still runs with one
`uvicorn main:app` command on a laptop with nothing extra installed or
running in the background.

The database file and uploaded images live in backend/data/, created
automatically on first run.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "cattlevision.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session, closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables if they don't already exist. Safe to call every startup."""
    import db_models  # noqa: F401 — ensures models are registered on Base before create_all
    Base.metadata.create_all(bind=engine)
