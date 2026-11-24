from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.settings import get_settings


class Base(DeclarativeBase):
    pass


_settings = get_settings()
if not _settings.database_url:
    raise RuntimeError("DATABASE_URL is required")
_database_url = _settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
_engine = create_engine(_database_url, pool_pre_ping=True, pool_size=_settings.db_pool_size, max_overflow=_settings.db_max_overflow)
SessionLocal = sessionmaker(_engine, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    with SessionLocal.begin() as session:
        yield session
