"""Exercise migration configuration without importing the application engine."""

import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from alembic.config import Config

import core.db  # noqa: F401 - the real package, before its modules are stubbed
from core.config import reset_config_cache
from core.errors import ConfigurationError


def run_offline_migrations(monkeypatch):
    from alembic import context

    configuration = Config()
    observed = {}
    monkeypatch.setattr(context, "config", configuration, raising=False)
    monkeypatch.setattr(context, "is_offline_mode", lambda: True)
    monkeypatch.setattr(context, "configure", lambda **kwargs: observed.update(kwargs))
    monkeypatch.setattr(context, "begin_transaction", nullcontext)
    monkeypatch.setattr(context, "run_migrations", lambda: None)
    db = ModuleType("core.db.session")
    db.Base = SimpleNamespace(metadata=object())
    monkeypatch.setitem(sys.modules, "core.db.session", db)
    monkeypatch.setitem(sys.modules, "core.db.models", ModuleType("core.db.models"))
    runpy.run_path(str(Path(__file__).parents[2] / "src" / "db" / "migrations" / "env.py"))
    return observed


def test_migration_accepts_case_insensitive_url_and_encoded_password(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("database_url", "postgresql://user:p%40ss%25word@example.invalid/echora")
    reset_config_cache()
    observed = run_offline_migrations(monkeypatch)
    assert observed["url"] == "postgresql+psycopg://user:p%40ss%25word@example.invalid/echora"


def test_migration_rejects_missing_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("database_url", raising=False)
    reset_config_cache()
    with pytest.raises(ConfigurationError, match="DATABASE_URL is required"):
        run_offline_migrations(monkeypatch)
