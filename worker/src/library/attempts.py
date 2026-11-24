"""Durable execution status, separate from reusable representation definitions."""
from __future__ import annotations

import os


def start_attempt(connection, run_id, requested: int):
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO analysis_attempts (run_id, requested, job_id) VALUES (%s,%s,%s) RETURNING id",
            (run_id, requested, os.getenv('ECHORA_JOB_ID')),
        )
        return cursor.fetchone()[0]


def record_track(connection, attempt_id, external_id: str, track_id=None, error: str | None = None):
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO analysis_attempt_tracks (attempt_id, external_id, track_id, status, error)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (attempt_id, external_id) DO UPDATE SET
                 track_id=EXCLUDED.track_id, status=EXCLUDED.status, error=EXCLUDED.error""",
            (attempt_id, external_id, track_id, "failed" if error else "complete", error),
        )


def finish_attempt(connection, attempt_id):
    with connection.cursor() as cursor:
        cursor.execute(
            """UPDATE analysis_attempts a SET
                 succeeded=t.succeeded, failed=t.failed, finished_at=now(),
                 status=CASE WHEN t.failed=0 AND t.succeeded=a.requested THEN 'complete'
                             WHEN t.succeeded>0 THEN 'partial' ELSE 'failed' END
               FROM (SELECT count(*) FILTER (WHERE status='complete') AS succeeded,
                            count(*) FILTER (WHERE status='failed') AS failed
                     FROM analysis_attempt_tracks WHERE attempt_id=%s) t
               WHERE a.id=%s""", (attempt_id, attempt_id),
        )
