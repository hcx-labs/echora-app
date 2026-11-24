from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.db.config import DatabaseConfig, require_database_url


class Base(DeclarativeBase):
    pass


_config = DatabaseConfig.load()
_database_url = require_database_url().replace("postgresql://", "postgresql+psycopg://", 1)
_engine = create_engine(
    _database_url,
    pool_pre_ping=True,
    pool_size=_config.db_pool_size,
    max_overflow=_config.db_max_overflow,
)
SessionLocal = sessionmaker(_engine, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    with SessionLocal.begin() as session:
        yield session
