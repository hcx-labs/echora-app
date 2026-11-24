"""Database configuration."""

from __future__ import annotations

from core.config import EchoraConfig, setting
from core.errors import ConfigurationError


class DatabaseConfig(EchoraConfig, env_prefix=""):
    """The PostgreSQL database every Echora process shares."""

    database_url: str = setting("", repr=False, exclude=True, description="PostgreSQL URL.")
    db_pool_size: int = setting(10, gt=0, description="SQLAlchemy pool size per process.")
    db_max_overflow: int = setting(20, ge=0, description="Connections beyond the pool size.")


def require_database_url() -> str:
    """The configured database URL; fails before a driver can fall back to its defaults."""
    url = DatabaseConfig.load().database_url
    if not url:
        raise ConfigurationError("DATABASE_URL is required")
    return url
