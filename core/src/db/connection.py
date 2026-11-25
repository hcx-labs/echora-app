"""Connections to the configured database.

Every process opens its psycopg connections here, never with ``psycopg.connect()`` directly.
Store modules (``core/<domain>/store.py``) take the connection as their first argument, so a
caller decides the transaction: ``with connect() as connection:`` commits on success and rolls
back on an exception, as psycopg does.
"""

from __future__ import annotations

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
