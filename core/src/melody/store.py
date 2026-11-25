"""Queries on melody contours (hum search) and the hum corpora built from them.

``melody_contours`` holds each song's pitch contour per source and contour run;
``hum_corpora`` and ``hum_corpus_tracks`` are a user's indexed song sets. Every function takes
an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

from core.db.connection import fetch_all, fetch_one


def contour_run(
    db: Connection,
    revision: str,
    config_hash: str,
    config: dict[str, Any],
    environment: dict[str, Any],
) -> uuid.UUID:
    """The melody contour run of this recipe, (re)started now."""
    row = fetch_one(
        db,
        """INSERT INTO analysis_runs (kind,model_name,model_revision,config_hash,config,environment,device,precision,status,started_at)
           VALUES ('melody_contour','melody_contour',%s,%s,%s,%s,'mixed','float32','running',now())
           ON CONFLICT (kind,model_name,model_revision,config_hash)
           DO UPDATE SET status='running',started_at=now(),finished_at=NULL RETURNING id""",
        (revision, config_hash, Jsonb(config), Jsonb(environment)),
    )
    return cast(dict, row)["id"]


def save_contour(
    db: Connection,
    track_id: uuid.UUID,
    run_id: uuid.UUID,
    source: str,
    hop_seconds: float,
    pitch: list[float],
    voiced: list[bool],
) -> None:
    """Store the run's contour of the song from one source."""
    db.execute(
        """INSERT INTO melody_contours (track_id,run_id,source,hop_seconds,pitch,voiced)
           VALUES (%s,%s,%s,%s,%s,%s)
           ON CONFLICT (track_id,run_id,source) DO UPDATE
           SET hop_seconds=EXCLUDED.hop_seconds,pitch=EXCLUDED.pitch,voiced=EXCLUDED.voiced,
               created_at=clock_timestamp()""",
        (track_id, run_id, source, hop_seconds, pitch, voiced),
    )


def contour_digest(db: Connection, revision: str, user_id: uuid.UUID) -> dict[str, Any]:
    """The ``count`` of the user's current contours and the newest one's time (``latest``)."""
    return cast(
        dict,
        fetch_one(
            db,
            """SELECT count(*) AS count,coalesce(max(mc.created_at)::text,'') AS latest
               FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
               WHERE ar.model_revision=%s AND ar.status IN ('complete','running')
                 AND EXISTS (SELECT 1 FROM user_track_links utl
                             WHERE utl.user_id=%s AND utl.track_id=mc.track_id)""",
            (revision, user_id),
        ),
    )


def user_contours(db: Connection, revision: str, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """The user's newest contour per song and source: ``track_id``, ``source``, ``pitch``, ``voiced``."""
    return fetch_all(
        db,
        """SELECT DISTINCT ON (mc.track_id,mc.source)
                  mc.track_id,mc.source,mc.pitch,mc.voiced
           FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
           WHERE ar.status IN ('complete','running') AND ar.model_revision=%s
             AND EXISTS (SELECT 1 FROM user_track_links utl
                         WHERE utl.user_id=%s AND utl.track_id=mc.track_id)
           ORDER BY mc.track_id,mc.source,ar.created_at DESC""",
        (revision, user_id),
    )


def matched_songs(db: Connection, user_id: uuid.UUID, track_ids: list) -> list[dict[str, Any]]:
    """The user's songs among ``track_ids`` with a source to play and its cover.

    Rows are ``id``, ``title``, ``artist``, ``album``, ``duration_seconds``, ``source_id`` and
    ``cover_art``.
    """
    return fetch_all(
        db,
        """SELECT t.id,t.title,t.artist,t.album,t.duration_seconds,max(utl.external_id) source_id,
                  max(ts.source_data->>'coverArt') cover_art
           FROM tracks t JOIN user_track_links utl ON utl.track_id=t.id AND utl.user_id=%s
           LEFT JOIN track_sources ts ON ts.library_id=utl.library_id AND ts.track_id=utl.track_id
                                      AND ts.external_id=utl.external_id
           WHERE t.id=ANY(%s) GROUP BY t.id""",
        (user_id, track_ids),
    )


def owns_corpus(db: Connection, corpus_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    """Whether the hum corpus is the user's."""
    return (
        fetch_one(db, "SELECT id FROM hum_corpora WHERE id=%s AND user_id=%s", (corpus_id, user_id))
        is not None
    )


def set_corpus_run(
    db: Connection, corpus_id: uuid.UUID, user_id: uuid.UUID, run_id: uuid.UUID
) -> None:
    """Point the user's hum corpus at a contour run."""
    db.execute(
        "UPDATE hum_corpora SET run_id=%s WHERE id=%s AND user_id=%s", (run_id, corpus_id, user_id)
    )


def has_contours(db: Connection, track_id: uuid.UUID, run_id: uuid.UUID) -> bool:
    """Whether the run has any contour of the song."""
    row = fetch_one(
        db,
        "SELECT count(*) AS n FROM melody_contours WHERE track_id=%s AND run_id=%s",
        (track_id, run_id),
    )
    return bool(cast(dict, row)["n"])


def add_corpus_track(db: Connection, corpus_id: uuid.UUID, track_id: uuid.UUID) -> None:
    """Put the song in the hum corpus."""
    db.execute(
        """INSERT INTO hum_corpus_tracks(corpus_id,track_id) VALUES (%s,%s)
           ON CONFLICT DO NOTHING""",
        (corpus_id, track_id),
    )
