"""``core.db.connect``: the one way the packages open database connections."""

import re
from pathlib import Path

import pytest
from psycopg.rows import dict_row, tuple_row

import core.db.connection as connection
from core.config import reset_config_cache
from core.errors import ConfigurationError

REPO = Path(__file__).resolve().parents[3]


def test_connect_passes_the_configured_url_and_row_shape(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://echora@db.invalid/echora")
    reset_config_cache()
    calls = []
    monkeypatch.setattr(connection.psycopg, "connect", lambda *a, **kw: calls.append((a, kw)))
    connection.connect()
    connection.connect(dict_rows=True, autocommit=True)
    url = ("postgresql://echora@db.invalid/echora",)
    assert calls == [
        (url, {"row_factory": tuple_row, "autocommit": False}),
        (url, {"row_factory": dict_row, "autocommit": True}),
    ]


def test_connect_refuses_a_missing_url(monkeypatch):
    # psycopg would otherwise fall back to libpq's defaults (a local socket, the OS user).
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_config_cache()
    with pytest.raises(ConfigurationError, match="DATABASE_URL is required"):
        connection.connect()


def test_packages_open_connections_only_through_core_db():
    allowed = {"core/src/db/connection.py", "core/src/testing.py"}
    offenders = [
        str(path.relative_to(REPO))
        for package in ("core", "worker", "backend")
        for path in (REPO / package / "src").rglob("*.py")
        if "migrations" not in path.parts
        and str(path.relative_to(REPO)) not in allowed
        and re.search(r"psycopg\.connect\(", path.read_text())
    ]
    assert offenders == []
