"""Per-song resync requests (``track_resync_requests``).

A request makes the next sync, partial or full, redo one pipeline of one song whatever its stored
revision; the stage that redoes it removes the request. Every function takes an open connection;
the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from core.db.connection import fetch_all

PIPELINES = ("transcription", "karaoke", "karaoke_stems", "translation", "themes")
BULK_PIPELINES = ("karaoke", "karaoke_stems", "translation", "themes")
"""The pipelines a full sync can redo for every song (a transcription needs a language)."""


def request(
    db: Connection,
    track_id: uuid.UUID,
    pipeline: str,
    options: dict[str, Any] | None = None,
    requested_by: uuid.UUID | None = None,
) -> None:
    """Ask the next sync to redo ``pipeline`` for the song (replaces an earlier request)."""
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline!r}")
    db.execute(
        """INSERT INTO track_resync_requests (track_id, pipeline, options, requested_by)
           VALUES (%s,%s,%s,%s) ON CONFLICT (track_id, pipeline) DO UPDATE
           SET options=EXCLUDED.options, requested_by=EXCLUDED.requested_by,
               requested_at=now()""",
        (track_id, pipeline, Jsonb(options or {}), requested_by),
    )


def request_library(
    db: Connection,
    library_id: uuid.UUID,
    external_ids: list[str],
    pipelines: Iterable[str],
    requested_by: uuid.UUID | None = None,
) -> None:
    """Ask the next sync to redo ``pipelines`` for every known song of the library's selection."""
    for pipeline in pipelines:
        if pipeline not in BULK_PIPELINES:
            raise ValueError(f"Pipeline {pipeline!r} cannot be resynced for every song")
        db.execute(
            """INSERT INTO track_resync_requests (track_id, pipeline, requested_by)
               SELECT DISTINCT ts.track_id, %s, %s FROM track_sources ts
               WHERE ts.source_type='subsonic' AND ts.library_id=%s AND ts.external_id=ANY(%s)
               ON CONFLICT (track_id, pipeline) DO UPDATE
               SET requested_by=EXCLUDED.requested_by, requested_at=now()""",
            (pipeline, requested_by, library_id, list(external_ids)),
        )


def cancel(db: Connection, track_id: uuid.UUID, pipeline: str) -> None:
    db.execute(
        "DELETE FROM track_resync_requests WHERE track_id=%s AND pipeline=%s",
        (track_id, pipeline),
    )


done = cancel
"""The stage redid the pipeline: the request is fulfilled."""


def for_track(db: Connection, track_id: uuid.UUID) -> dict[str, dict[str, Any]]:
    """The song's open requests by pipeline: ``options`` and ``requested_at``."""
    rows = fetch_all(
        db,
        """SELECT pipeline, options, requested_at FROM track_resync_requests
           WHERE track_id=%s""",
        (track_id,),
    )
    return {row["pipeline"]: row for row in rows}


def requested(
    db: Connection, pipeline: str, track_ids: Iterable[uuid.UUID]
) -> dict[uuid.UUID, dict[str, Any]]:
    """The given songs with an open request for ``pipeline``, with its options."""
    rows = fetch_all(
        db,
        """SELECT track_id, options FROM track_resync_requests
           WHERE pipeline=%s AND track_id=ANY(%s)""",
        (pipeline, list(track_ids)),
    )
    return {row["track_id"]: row["options"] or {} for row in rows}


def requested_sources(db: Connection, library_id: uuid.UUID, external_ids: list[str]) -> set[str]:
    """The library's selected songs (Navidrome ids) with any open request."""
    rows = fetch_all(
        db,
        """SELECT DISTINCT ts.external_id FROM track_sources ts
           JOIN track_resync_requests r ON r.track_id=ts.track_id
           WHERE ts.source_type='subsonic' AND ts.library_id=%s AND ts.external_id=ANY(%s)""",
        (library_id, list(external_ids)),
    )
    return {str(row["external_id"]) for row in rows}
