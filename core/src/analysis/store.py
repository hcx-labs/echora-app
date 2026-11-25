"""Queries on analysis runs and their attempts.

A run (``analysis_runs``) is one model configuration; each domain's store creates its own kind.
An attempt (``analysis_attempts``) is one execution of a run over a batch of songs, with one
``analysis_attempt_tracks`` row per song. Every function takes an open connection; the caller
owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import cast

from psycopg import Connection

from core.db.connection import fetch_one


def complete_run(db: Connection, run_id: uuid.UUID) -> None:
    """Mark the run complete now."""
    db.execute(
        "UPDATE analysis_runs SET status='complete', finished_at=now() WHERE id=%s", (run_id,)
    )


def start_attempt(
    db: Connection, run_id: uuid.UUID, requested: int, job_id: str | None
) -> uuid.UUID:
    """Store a new attempt of the run over ``requested`` songs; returns its id."""
    row = fetch_one(
        db,
        "INSERT INTO analysis_attempts (run_id, requested, job_id) VALUES (%s,%s,%s) RETURNING id",
        (run_id, requested, job_id),
    )
    return cast(dict, row)["id"]


def record_attempt_track(
    db: Connection,
    attempt_id: uuid.UUID,
    external_id: str,
    track_id: uuid.UUID | None,
    error: str | None,
) -> None:
    """Record one song's outcome in the attempt: failed with ``error``, else complete."""
    db.execute(
        """INSERT INTO analysis_attempt_tracks (attempt_id, external_id, track_id, status, error)
           VALUES (%s,%s,%s,%s,%s)
           ON CONFLICT (attempt_id, external_id) DO UPDATE SET
             track_id=EXCLUDED.track_id, status=EXCLUDED.status, error=EXCLUDED.error""",
        (attempt_id, external_id, track_id, "failed" if error else "complete", error),
    )


def finish_attempt(db: Connection, attempt_id: uuid.UUID) -> None:
    """Count the attempt's outcomes and set its status: complete, partial or failed."""
    db.execute(
        """UPDATE analysis_attempts a SET
             succeeded=t.succeeded, failed=t.failed, finished_at=now(),
             status=CASE WHEN t.failed=0 AND t.succeeded=a.requested THEN 'complete'
                         WHEN t.succeeded>0 THEN 'partial' ELSE 'failed' END
           FROM (SELECT count(*) FILTER (WHERE status='complete') AS succeeded,
                        count(*) FILTER (WHERE status='failed') AS failed
                 FROM analysis_attempt_tracks WHERE attempt_id=%s) t
           WHERE a.id=%s""",
        (attempt_id, attempt_id),
    )
