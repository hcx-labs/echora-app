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
"""Pipelines one sync can bring up to date for every song of an older revision (an option of
that sync, never stored as requests: a transcription needs a language)."""


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


# Songs of a pipeline whose stored output is of another revision than ``current`` (the %s).
_OUTDATED = {
    "karaoke": """EXISTS (SELECT 1 FROM karaoke_lyrics_variants kv
        WHERE kv.track_id=ts.track_id AND NOT kv.bounded AND kv.model<>'navidrome'
          AND kv.provenance->>'pipeline_revision' || ' ' || kv.model_revision <> %s)""",
    "karaoke_stems": """EXISTS (SELECT 1 FROM karaoke_stems ks
        WHERE ks.track_id=ts.track_id AND ks.recipe<>%s)""",
    "translation": """EXISTS (SELECT 1 FROM lyric_translations lt
        WHERE lt.track_id=ts.track_id AND lt.provenance='ai'
          AND lt.model || ' ' || lt.prompt_revision <> %s)""",
    "themes": """EXISTS (SELECT 1 FROM track_lyric_themes th
        WHERE th.track_id=ts.track_id AND th.prompt_revision::text <> %s)""",
}


def outdated_sources(
    db: Connection,
    library_id: uuid.UUID,
    external_ids: list[str],
    pipeline: str,
    current: str,
) -> set[str]:
    """The selection's songs (Navidrome ids) whose ``pipeline`` output is of an older revision.

    ``current`` is the pipeline's current revision as stored: karaoke ``"<pipeline revision>
    <model revision>"``, stems the recipe, translations ``"<model> <prompt sha256>"``, themes the
    prompt revision. Songs without output are not included (a sync makes those anyway).
    """
    if pipeline not in BULK_PIPELINES:
        raise ValueError(f"Pipeline {pipeline!r} cannot be brought up to date for every song")
    rows = fetch_all(
        db,
        f"""SELECT DISTINCT ts.external_id FROM track_sources ts
           WHERE ts.source_type='subsonic' AND ts.library_id=%s AND ts.external_id=ANY(%s)
             AND {_OUTDATED[pipeline]}""",
        (library_id, list(external_ids), current),
    )
    return {str(row["external_id"]) for row in rows}


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
