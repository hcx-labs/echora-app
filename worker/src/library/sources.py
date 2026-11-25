"""Where a batch's songs come from: the database, and the user's songs in one library."""

from __future__ import annotations

import uuid

from core.db.connection import connect


def source_rows(user_id, url: str, ids: list[str] | None = None) -> list[tuple[str, uuid.UUID]]:
    """``(external_id, track_id)`` of the user's songs in this library, optionally only ``ids``."""
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT DISTINCT ts.external_id, ts.track_id FROM track_sources ts
               JOIN libraries l ON l.id=ts.library_id
               JOIN user_track_links u ON u.library_id=ts.library_id AND u.track_id=ts.track_id
                 AND u.external_id=ts.external_id
               WHERE u.user_id=%s AND l.root_path=%s AND ts.source_type='subsonic'
                 AND (%s::text[] IS NULL OR ts.external_id=ANY(%s)) ORDER BY ts.external_id""",
            (user_id, url.rstrip("/"), ids, ids),
        )
        return cursor.fetchall()
