"""Queries on karaoke stems: each song's encoded vocals and instrumental (``karaoke_stems``).

One row per song: the current file, or the failure of its current recipe. Every function takes
an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection

from core.db.connection import fetch_all, fetch_one


def songs_without_stems(
    db: Connection, library_id: uuid.UUID, external_ids: list[str], recipe: str
) -> list[dict[str, Any]]:
    """The library's selected songs with lyrics that have no stems of ``recipe``.

    A song whose stems failed with this recipe is not planned again until the recipe changes,
    unless it has a stems resync request (as has a song asked to be made again). Rows are ``track_id``, ``external_id`` and ``title``.
    """
    return fetch_all(
        db,
        """SELECT DISTINCT ON (ts.track_id) ts.track_id, ts.external_id, t.title
           FROM track_sources ts JOIN tracks t ON t.id=ts.track_id
           WHERE ts.source_type='subsonic' AND ts.library_id=%s AND ts.external_id=ANY(%s)
             AND EXISTS (SELECT 1 FROM lyrics l WHERE l.track_id=ts.track_id
                         AND nullif(btrim(l.text), '') IS NOT NULL)
             AND (NOT EXISTS (SELECT 1 FROM karaoke_stems ks
                              WHERE ks.track_id=ts.track_id AND ks.recipe=%s)
                  OR EXISTS (SELECT 1 FROM track_resync_requests r
                             WHERE r.track_id=ts.track_id AND r.pipeline='karaoke_stems'))
           ORDER BY ts.track_id, ts.id""",
        (library_id, list(external_ids), recipe),
    )


def save_stems(
    db: Connection,
    track_id: uuid.UUID,
    recipe: str,
    *,
    path: str | None = None,
    file_bytes: int | None = None,
    duration_seconds: float | None = None,
    error: str | None = None,
) -> str | None:
    """Store a song's stems (``path``) or their failure (``error``).

    New stems replace whatever the song had; a failure never replaces complete stems (of an
    older recipe), so the song keeps them and is tried again by the next sync.
    Returns the replaced file's stored path, for the caller to delete after committing.
    """
    previous = fetch_one(
        db, "SELECT path FROM karaoke_stems WHERE track_id=%s FOR UPDATE", (track_id,)
    )
    db.execute(
        """INSERT INTO karaoke_stems
             (id, track_id, recipe, status, path, bytes, duration_seconds, error)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (track_id) DO UPDATE SET
             id=EXCLUDED.id, recipe=EXCLUDED.recipe, status=EXCLUDED.status, path=EXCLUDED.path,
             bytes=EXCLUDED.bytes, duration_seconds=EXCLUDED.duration_seconds,
             error=EXCLUDED.error, created_at=now()
           WHERE EXCLUDED.status='complete' OR karaoke_stems.status='failed'""",
        (
            uuid.uuid4(),
            track_id,
            recipe,
            "failed" if path is None else "complete",
            path,
            file_bytes,
            duration_seconds,
            error,
        ),
    )
    old = previous["path"] if previous and path is not None else None
    return old if old and old != path else None


def track_stems(db: Connection, user_id: uuid.UUID, track_id: uuid.UUID) -> dict[str, Any] | None:
    """The complete stems of a song in the user's library (``id``, ``duration_seconds``)."""
    return fetch_one(
        db,
        """SELECT ks.id, ks.duration_seconds FROM karaoke_stems ks
           WHERE ks.track_id=%s AND ks.status='complete' AND EXISTS (
             SELECT 1 FROM user_track_links utl WHERE utl.track_id=ks.track_id
             AND utl.user_id=%s)""",
        (track_id, user_id),
    )


def visible_stems_path(db: Connection, stems_id: uuid.UUID, user_id: uuid.UUID) -> str | None:
    """The stored file path of complete stems of a song in the user's library."""
    row = fetch_one(
        db,
        """SELECT ks.path FROM karaoke_stems ks
           WHERE ks.id=%s AND ks.status='complete' AND EXISTS (
             SELECT 1 FROM user_track_links utl WHERE utl.track_id=ks.track_id
             AND utl.user_id=%s)""",
        (stems_id, user_id),
    )
    return row["path"] if row else None


def stems_counts(db: Connection) -> dict[str, int]:
    """Complete and failed stems, and the complete ones' total size (``bytes``)."""
    return dict(
        fetch_one(
            db,
            """SELECT count(*) FILTER (WHERE status='complete') AS complete,
                      count(*) FILTER (WHERE status='failed') AS failed,
                      coalesce(sum(bytes) FILTER (WHERE status='complete'), 0) AS bytes
               FROM karaoke_stems""",
        )
        or {}
    )
