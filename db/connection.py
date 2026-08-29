"""
Database connection and session management.

Usage in any node:
    from db.connection import get_session, init_db

    init_db()   # call once at startup - creates tables if they don't exist

    with get_session() as session:
        session.add(some_model)
        session.commit()
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

from config.settings import settings

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        connect_args = {}
        if settings.DATABASE_URL.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(
            settings.DATABASE_URL,
            echo=False,
            connect_args=connect_args,
        )
    return _engine


def init_db() -> None:
    """Create all tables. Safe to call multiple times (CREATE IF NOT EXISTS)."""
    import db.models  # ensure all models are registered with SQLModel metadata
    SQLModel.metadata.create_all(_get_engine())


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a database session, committing on success and rolling back on error."""
    engine = _get_engine()
    with Session(engine, expire_on_commit=False) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
