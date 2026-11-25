"""Where a batch's songs come from: the database, and the user's songs in one library."""

from __future__ import annotations

import uuid

from core.db.connection import connect
from core.library import store


def source_rows(user_id, url: str, ids: list[str] | None = None) -> list[tuple[str, uuid.UUID]]:
    """``(external_id, track_id)`` of the user's songs in this library, optionally only ``ids``."""
    with connect() as connection:
        rows = store.song_sources(connection, user_id, url.rstrip("/"), ids)
    return [(row["external_id"], row["track_id"]) for row in rows]
