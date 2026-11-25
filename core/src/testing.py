"""Pytest fixtures shared by the core, worker and backend test suites.

Each project's ``conftest.py`` imports what it needs: ``from core.testing import database``.
Database fixtures need ``TEST_DATABASE_URL`` (a database where the tests may create schemas) and
skip without it.
"""

from __future__ import annotations

import os
import runpy
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

import core.jobs.queue as jobs

MIGRATIONS = Path(__file__).resolve().parent / "db" / "migrations" / "versions"


def migration(name: str) -> dict[str, Any]:
    """The namespace of one migration file, e.g. ``migration("0034_jobs")["upgrade"]``."""
    return runpy.run_path(str(MIGRATIONS / f"{name}.py"))


def apply_migrations(connection: psycopg.Connection, *names: str) -> None:
    """Run the ``upgrade()`` of each named migration on ``connection``."""
    with patch("alembic.op.execute", side_effect=connection.execute):
        for name in names:
            migration(name)["upgrade"]()


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[], psycopg.Connection]]:
    """An isolated schema with the job queue, which ``core.jobs.queue`` uses for the test."""
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL tests")
    schema = "test_jobs_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    def connect() -> psycopg.Connection:
        return psycopg.connect(url, options=f"-c search_path={schema}", row_factory=dict_row)

    monkeypatch.setattr(jobs, "_db", connect)
    try:
        with connect() as db:
            apply_migrations(db, "0034_jobs", "0049_job_dismissal")
        yield connect
    finally:
        with psycopg.connect(url, autocommit=True) as db:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def stub_module(monkeypatch: pytest.MonkeyPatch, name: str, stub: Any) -> None:
    """Replace the module ``name`` for one test, also where it is reached as an attribute.

    ``import a.b.c as m`` reads ``a.b.c`` through the parent packages' attributes, so replacing
    only ``sys.modules["a.b.c"]`` misses a module that is already imported.
    """
    import importlib
    import sys

    parent, _, child = name.rpartition(".")
    monkeypatch.setitem(sys.modules, name, stub)
    if parent:
        monkeypatch.setattr(importlib.import_module(parent), child, stub, raising=False)
