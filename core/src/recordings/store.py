"""Queries on recording search: song fingerprints, searches and calibration clips.

``recording_fingerprints`` holds each song's segment vectors per encoder representation;
``recording_searches`` a search job's uploaded clip and result; ``recording_calibration_samples``
the clips users saved for calibration. Every function takes an open connection; the caller owns
the transaction.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any, cast

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from core.db.connection import fetch_all, fetch_one


def indexed_song_count(db: Connection, representation_id: str, user_id: uuid.UUID) -> int:
    """How many of the user's songs have usable fingerprints of the representation."""
    row = fetch_one(
        db,
        """SELECT count(*) AS n FROM recording_fingerprints f
        WHERE f.representation_id=%s AND f.segment_count>0 AND EXISTS (
            SELECT 1 FROM user_track_links u WHERE u.track_id=f.track_id AND u.user_id=%s)
        """,
        (representation_id, user_id),
    )
    return cast(dict, row)["n"]


def catalog_coverage(
    db: Connection,
    representation_id: str,
    namespace: uuid.UUID,
    external_ids: list[str],
    user_id: uuid.UUID,
) -> dict[str, int]:
    """Of the user's sources in the library, how many are ``indexed`` or ``unsupported``.

    A source is unsupported when its song's fingerprints have no segments.
    """
    return cast(
        dict,
        fetch_one(
            db,
            """SELECT count(DISTINCT ts.external_id) FILTER (WHERE f.segment_count>0) AS indexed,
                   count(DISTINCT ts.external_id) FILTER (WHERE f.segment_count=0) AS unsupported
            FROM track_sources ts JOIN libraries l ON l.id=ts.library_id
            JOIN recording_fingerprints f ON f.track_id=ts.track_id AND f.representation_id=%s
            WHERE l.namespace=%s AND ts.source_type='subsonic' AND ts.external_id=ANY(%s)
              AND EXISTS (SELECT 1 FROM user_track_links u WHERE u.user_id=%s
                  AND u.library_id=ts.library_id AND u.track_id=ts.track_id)""",
            (representation_id, namespace, external_ids, user_id),
        ),
    )


def expire_searches(db: Connection) -> None:
    """Erase expired search audio, then expired searches and diagnostic captures."""
    db.execute(
        "UPDATE recording_searches SET audio=NULL WHERE audio IS NOT NULL AND audio_expires_at<=now()"
    )
    db.execute("DELETE FROM recording_searches WHERE expires_at<=now()")
    db.execute("DELETE FROM recording_diagnostic_captures WHERE expires_at<=now()")


def lock_admission(db: Connection) -> None:
    """Serialize search admission until the transaction ends."""
    db.execute("SELECT pg_advisory_xact_lock(hashtextextended('recording-search-admission',0))")


def active_searches(db: Connection, user_id: uuid.UUID) -> dict[str, int]:
    """Searches queued or running: in ``total``, and the user's (``owned``)."""
    return cast(
        dict,
        fetch_one(
            db,
            """SELECT count(*) AS total,
            count(*) FILTER (WHERE user_id=%s) AS owned FROM jobs
            WHERE kind='recording_search' AND status IN ('queued','running','waiting')""",
            (user_id,),
        ),
    )


def recent_searches(db: Connection, user_id: uuid.UUID) -> int:
    """The user's searches of the last minute."""
    row = fetch_one(
        db,
        """SELECT count(*) AS n FROM jobs WHERE kind='recording_search'
        AND user_id=%s AND created_at>now()-interval '1 minute'""",
        (user_id,),
    )
    return cast(dict, row)["n"]


def add_search(
    db: Connection,
    job_id: uuid.UUID,
    representation_id: str,
    policy_id: str,
    audio: bytes,
    audio_retention_seconds: int,
    result_retention_seconds: int,
) -> None:
    """Make the job a single attempt and store its clip until the retentions pass."""
    db.execute("UPDATE jobs SET max_attempts=1 WHERE id=%s", (job_id,))
    db.execute(
        """INSERT INTO recording_searches(job_id,representation_id,policy_id,audio,audio_expires_at,expires_at)
        VALUES (%s,%s,%s,%s,now()+%s*interval '1 second',now()+%s*interval '1 second')""",
        (
            job_id,
            representation_id,
            policy_id,
            audio,
            audio_retention_seconds,
            result_retention_seconds,
        ),
    )


def capture_diagnostic(
    db: Connection, audio: bytes, job_id: uuid.UUID, retention_seconds: int, user_id: uuid.UUID
) -> None:
    """Fill the user's armed diagnostic capture, if one is waiting, with this clip."""
    db.execute(
        """UPDATE recording_diagnostic_captures
        SET audio=%s,job_id=%s,captured_at=now(),expires_at=now()+%s*interval '1 second'
        WHERE user_id=%s AND captured_at IS NULL AND expires_at>now()""",
        (audio, job_id, retention_seconds, user_id),
    )


def has_fingerprints(db: Connection, track_id: uuid.UUID, representation_id: str) -> bool:
    """Whether the song has fingerprints of the representation."""
    return (
        fetch_one(
            db,
            "SELECT 1 FROM recording_fingerprints WHERE track_id=%s AND representation_id=%s",
            (track_id, representation_id),
        )
        is not None
    )


def track_audio_hash(db: Connection, track_id: uuid.UUID) -> str | None:
    """The SHA-256 of the song's source audio, if the song exists."""
    row = fetch_one(db, "SELECT audio_hash FROM tracks WHERE id=%s", (track_id,))
    return row["audio_hash"] if row else None


def save_fingerprints(
    db: Connection, track_id: uuid.UUID, representation_id: str, segments: int, data: bytes
) -> bool:
    """Store the song's fingerprints; ``False`` when they were stored already."""
    with db.cursor() as cursor:
        cursor.execute(
            """INSERT INTO recording_fingerprints
            (track_id,representation_id,segment_count,fingerprints) VALUES (%s,%s,%s,%s)
            ON CONFLICT (track_id,representation_id) DO NOTHING""",
            (track_id, representation_id, segments, data),
        )
        return cursor.rowcount == 1


def reference_rows(
    db: Connection, representation_id: str, user_id: uuid.UUID
) -> Iterator[dict[str, Any]]:
    """The user's usable fingerprints, a few rows at a time (a server-side cursor).

    Rows are ``track_id``, ``segment_count``, ``fingerprints`` and the recording ``group_id``.
    """
    with db.cursor(name="recording_references", row_factory=dict_row) as cursor:
        cursor.itersize = 8
        cursor.execute(
            """SELECT f.track_id,f.segment_count,f.fingerprints,g.group_id
            FROM recording_fingerprints f
            LEFT JOIN recording_group_members g ON g.track_id=f.track_id
            WHERE f.representation_id=%s AND f.segment_count>0 AND EXISTS (
                SELECT 1 FROM user_track_links u WHERE u.track_id=f.track_id AND u.user_id=%s)
            ORDER BY f.track_id""",
            (representation_id, user_id),
        )
        yield from cursor


def pending_search(db: Connection, job_id: uuid.UUID, user_id: uuid.UUID) -> dict | None:
    """The user's search of the job while its audio is kept."""
    return fetch_one(
        db,
        """SELECT q.* FROM recording_searches q JOIN jobs j ON j.id=q.job_id
        WHERE q.job_id=%s AND j.user_id=%s AND q.audio_expires_at>now()""",
        (job_id, user_id),
    )


def lock_live_claim(db: Connection, job_id: uuid.UUID, token: Any) -> bool:
    """Lock the job's row if this claim still runs it; whether it does."""
    return (
        fetch_one(
            db,
            """SELECT id FROM jobs WHERE id=%s AND token=%s
            AND status='running' AND NOT cancel_requested AND lease_until>now()
            FOR UPDATE""",
            (job_id, token),
        )
        is not None
    )


def save_search_result(db: Connection, job_id: uuid.UUID, result: dict[str, Any]) -> None:
    """Store the search's result."""
    db.execute("UPDATE recording_searches SET result=%s WHERE job_id=%s", (Jsonb(result), job_id))


def visible_songs(
    db: Connection, user_id: uuid.UUID, track_ids: list[uuid.UUID]
) -> list[dict[str, Any]]:
    """The songs of the user's library among ``track_ids``, with how to play each.

    Rows are ``id``, ``title``, ``artist``, ``album``, ``duration_seconds`` and, through a
    connection of the user that serves the song's library (the selected one first),
    ``source_id``, ``connection_id`` and ``cover_art``.
    """
    return fetch_all(
        db,
        """SELECT t.id,t.title,t.artist,t.album,t.duration_seconds,
        CASE WHEN u.connection_id IS NOT NULL THEN u.external_id END AS source_id,
        u.connection_id::text AS connection_id,
        CASE WHEN u.connection_id IS NOT NULL THEN ts.source_data->>'coverArt' END AS cover_art
        FROM tracks t JOIN LATERAL (
            SELECT link.external_id,link.library_id,nc.id AS connection_id
            FROM user_track_links link
            LEFT JOIN libraries library ON library.id=link.library_id
            LEFT JOIN navidrome_connections nc ON nc.owner_user_id=link.user_id
                AND rtrim(nc.url,'/')=rtrim(library.root_path,'/')
            LEFT JOIN user_preferences preference ON preference.user_id=link.user_id
            WHERE link.track_id=t.id AND link.user_id=%s
            ORDER BY (nc.id=preference.navidrome_connection_id) DESC NULLS LAST,
                     nc.id NULLS LAST,link.library_id,link.external_id LIMIT 1
        ) u ON true
        LEFT JOIN track_sources ts ON ts.track_id=t.id AND ts.library_id=u.library_id
            AND ts.external_id=u.external_id AND ts.source_type='subsonic'
        WHERE t.id=ANY(%s)""",
        (user_id, track_ids),
    )


def search_result(db: Connection, job_id: uuid.UUID, user_id: uuid.UUID) -> dict | None:
    """The user's search job: ``status``, ``error`` and its unexpired ``result``."""
    return fetch_one(
        db,
        """SELECT j.status,j.error,q.result FROM jobs j
        LEFT JOIN recording_searches q ON q.job_id=j.id AND q.expires_at>now()
        WHERE j.id=%s AND j.user_id=%s AND j.kind='recording_search'""",
        (job_id, user_id),
    )


# Calibration clips. Their writers hold the calibration transaction lock first.


def lock_calibration(db: Connection, key: str) -> None:
    """Hold the calibration lock until the transaction ends."""
    db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (key,))


def try_lock_calibration(db: Connection, key: str) -> bool:
    """Take the calibration lock until the transaction ends if no one holds it."""
    row = fetch_one(
        db, "SELECT pg_try_advisory_xact_lock(hashtextextended(%s,0)) AS locked", (key,)
    )
    return bool(row and row["locked"])


def in_library(db: Connection, user_id: uuid.UUID, track_id: uuid.UUID) -> bool:
    """Whether the song is in the user's library."""
    return (
        fetch_one(
            db,
            "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s",
            (user_id, track_id),
        )
        is not None
    )


def calibration_quota(db: Connection, user_id: uuid.UUID) -> dict[str, int]:
    """Stored clips: in ``total``, the user's (``owned``) and theirs of the last minute."""
    return cast(
        dict,
        fetch_one(
            db,
            """SELECT count(*) AS total,
            count(*) FILTER (WHERE user_id=%s) AS owned,
            count(*) FILTER (WHERE user_id=%s AND created_at>now()-interval '1 minute') AS recent
            FROM recording_calibration_samples""",
            (user_id, user_id),
        ),
    )


def add_calibration_sample(db: Connection, **values: Any) -> dict[str, Any]:
    """Store a clip's record, kept for ``retention_seconds``; returns the row.

    ``values`` are ``id``, ``user_id``, ``expected_track_id``, ``not_in_library``, ``notes``,
    ``consent_version``, ``audio_sha256``, ``byte_count``, ``duration_seconds`` and
    ``retention_seconds``.
    """
    return cast(
        dict,
        fetch_one(
            db,
            """INSERT INTO recording_calibration_samples
            (id,user_id,expected_track_id,not_in_library,notes,consent_version,
             audio_sha256,byte_count,duration_seconds,expires_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now()+%s*interval '1 second') RETURNING *""",
            (
                values["id"],
                values["user_id"],
                values["expected_track_id"],
                values["not_in_library"],
                values["notes"],
                values["consent_version"],
                values["audio_sha256"],
                values["byte_count"],
                values["duration_seconds"],
                values["retention_seconds"],
            ),
        ),
    )


def calibration_samples(db: Connection, user_id: uuid.UUID, limit: int) -> list[dict[str, Any]]:
    """The user's unexpired clips, newest first, with the expected song while it is theirs."""
    return fetch_all(
        db,
        """SELECT s.*,t.title AS expected_title,t.artist AS expected_artist
        FROM recording_calibration_samples s
        LEFT JOIN tracks t ON t.id=s.expected_track_id AND EXISTS (
            SELECT 1 FROM user_track_links u WHERE u.user_id=s.user_id AND u.track_id=t.id)
        WHERE s.user_id=%s AND s.expires_at>now() ORDER BY s.created_at DESC,s.id
        LIMIT %s""",
        (user_id, limit),
    )


def lock_calibration_sample(
    db: Connection, sample_id: uuid.UUID, user_id: uuid.UUID
) -> uuid.UUID | None:
    """Lock the user's clip record; its id, or ``None`` when it is not theirs."""
    row = fetch_one(
        db,
        "SELECT id FROM recording_calibration_samples WHERE id=%s AND user_id=%s FOR UPDATE",
        (sample_id, user_id),
    )
    return row["id"] if row else None


def delete_calibration_sample(db: Connection, sample_id: uuid.UUID) -> None:
    """Forget a clip's record."""
    db.execute("DELETE FROM recording_calibration_samples WHERE id=%s", (sample_id,))


def lock_expired_calibration_samples(db: Connection) -> list[uuid.UUID]:
    """Lock the expired clip records; their ids."""
    return [
        row["id"]
        for row in fetch_all(
            db, "SELECT id FROM recording_calibration_samples WHERE expires_at<=now() FOR UPDATE"
        )
    ]


def calibration_sample_ids(db: Connection) -> set[str]:
    """Every stored clip's id, as text."""
    return {str(row["id"]) for row in fetch_all(db, "SELECT id FROM recording_calibration_samples")}
