"""Connections to the configured database.

Every process opens its psycopg connections here, never with ``psycopg.connect()`` directly.
Store modules (``core/<domain>/store.py``) take the connection as their first argument, so a
caller decides the transaction: ``with connect() as connection:`` commits on success and rolls
back on an exception, as psycopg does. A store reads through ``fetch_one``/``fetch_all``, which
return dict rows whatever row shape the caller's connection has.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row, tuple_row

from core.db.config import require_database_url


def connect(*, dict_rows: bool = False, autocommit: bool = False) -> psycopg.Connection:
    """A new connection: rows as tuples, or as dicts with ``dict_rows``."""
    return psycopg.connect(
        require_database_url(),
        row_factory=dict_row if dict_rows else tuple_row,
        autocommit=autocommit,
    )


def fetch_one(db: psycopg.Connection, query: str, params: Any = ()) -> dict[str, Any] | None:
    """The first row of ``query`` as a dict, or ``None``."""
    with db.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, params)
        return cursor.fetchone()


def fetch_all(db: psycopg.Connection, query: str, params: Any = ()) -> list[dict[str, Any]]:
    """Every row of ``query`` as a dict."""
    with db.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()
