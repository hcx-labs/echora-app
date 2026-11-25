"""Queries on voice classification: its runs, per-song voice activations and vocal activity.

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

from core.db.connection import fetch_all, fetch_one


def voice_run(
    db: Connection,
    model_name: str,
    config_hash: str,
    config: dict[str, Any],
    environment: dict[str, Any],
) -> uuid.UUID:
    """The voice classification run of these settings, started now when new."""
    row = fetch_one(
        db,
        """INSERT INTO analysis_runs
             (kind, model_name, model_revision, config_hash, config, environment, device, precision, status, started_at)
           VALUES ('voice_classification', %s, 'joint-v2', %s, %s, %s, 'cpu', 'float32', 'running', now())
           ON CONFLICT (kind, model_name, model_revision, config_hash)
           DO UPDATE SET id=analysis_runs.id RETURNING id""",
        (model_name, config_hash, Jsonb(config), Jsonb(environment)),
    )
    return cast(dict, row)["id"]


def songs_without_voice(
    db: Connection,
    library_id: uuid.UUID,
    external_ids: list[str] | None,
    embedding_type: str,
    run_id: uuid.UUID,
    limit: int | None,
) -> list[dict[str, Any]]:
    """The library's songs (optionally only some sources) the run has not classified.

    Rows are ``track_id``, ``external_id`` and ``title``, at most ``limit``.
    """
    restriction, parameters = (
        ("", []) if external_ids is None else (" AND ts.external_id=ANY(%s)", [external_ids])
    )
    return fetch_all(
        db,
        f"""SELECT DISTINCT ON (ts.track_id) ts.track_id, ts.external_id, t.title
           FROM track_sources ts JOIN tracks t ON t.id=ts.track_id
           WHERE ts.source_type='subsonic'
             AND ts.library_id=%s{restriction}
             AND NOT EXISTS (
               SELECT 1 FROM embeddings e
               JOIN track_vocal_activity activity ON activity.track_id=e.track_id AND activity.run_id=e.run_id
               WHERE e.track_id=ts.track_id AND e.embedding_type=%s AND e.run_id=%s
             )
           ORDER BY ts.track_id, ts.id
           LIMIT %s""",
        (library_id, *parameters, embedding_type, run_id, limit),
    )


def save_voice(
    db: Connection,
    track_id: uuid.UUID,
    run_id: uuid.UUID,
    embedding_type: str,
    vector: str,
    dimension: int,
    activity: dict[str, Any],
) -> None:
    """Replace the run's voice activation of the song and store its vocal activity."""
    db.execute(
        "DELETE FROM embeddings WHERE track_id=%s AND run_id=%s AND embedding_type=%s",
        (track_id, run_id, embedding_type),
    )
    db.execute(
        """INSERT INTO embeddings
             (track_id, run_id, embedding_type, dimension, aggregation, embedding)
           VALUES (%s, %s, %s, %s, 'mean_activation', %s::vector)""",
        (track_id, run_id, embedding_type, dimension, vector),
    )
    db.execute(
        """INSERT INTO track_vocal_activity (track_id, run_id, activity)
           VALUES (%s,%s,%s) ON CONFLICT (track_id, run_id)
           DO UPDATE SET activity=EXCLUDED.activity, created_at=now()""",
        (track_id, run_id, Jsonb(activity)),
    )
