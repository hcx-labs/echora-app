"""Queries on curations and the user's concepts.

Curations are records (``core.db.models.Curation``): those functions take an open ``Session``.
The rest take an open psycopg connection. Either way the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db.connection import fetch_all, fetch_one
from core.db.models import Curation
from core.errors import NotFoundError

_CONCEPT_COLUMNS = """id, name, description, positive_prompts, negative_prompts,
                      positive_track_ids, negative_track_ids, enabled, created_at, updated_at"""


def concepts(db: Connection, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """The user's own concepts by name."""
    return fetch_all(
        db,
        f"SELECT {_CONCEPT_COLUMNS} FROM concepts WHERE user_id=%s ORDER BY lower(name)",
        (user_id,),
    )


def enabled_concepts(db: Connection, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """The user's enabled concepts: name, prompts and seed songs."""
    return fetch_all(
        db,
        """SELECT name, positive_prompts, negative_prompts, positive_track_ids, negative_track_ids
           FROM concepts WHERE user_id=%s AND enabled""",
        (user_id,),
    )


def create_concept(
    db: Connection,
    user_id: uuid.UUID,
    name: str,
    description: str,
    positive_prompts: list[str],
    negative_prompts: list[str],
    positive_track_ids: list[uuid.UUID],
    negative_track_ids: list[uuid.UUID],
) -> dict[str, Any]:
    """Store a new concept of the user; returns it."""
    return cast(
        dict[str, Any],
        fetch_one(
            db,
            f"""INSERT INTO concepts (user_id, name, description, positive_prompts, negative_prompts,
                                      positive_track_ids, negative_track_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING {_CONCEPT_COLUMNS}""",
            (
                user_id,
                name,
                description,
                positive_prompts,
                negative_prompts,
                positive_track_ids,
                negative_track_ids,
            ),
        ),
    )


def curations(db: Connection, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """The user's curations, newest change first, with their latest revision.

    Each has its ``recipe``, its seed and avoided songs (``positive_tracks``,
    ``negative_tracks``) and the revision's ``tracks`` in order, with evidence and covers.
    """
    return fetch_all(
        db,
        """SELECT c.*, latest.recipe,
                   coalesce((SELECT jsonb_agg(jsonb_build_object('id', pt.id, 'title', pt.title, 'artist', pt.artist, 'album', pt.album)) FROM tracks pt WHERE pt.id=ANY(c.positive_track_ids)), '[]') AS positive_tracks,
                   coalesce((SELECT jsonb_agg(jsonb_build_object('id', nt.id, 'title', nt.title, 'artist', nt.artist, 'album', nt.album)) FROM tracks nt WHERE nt.id=ANY(c.negative_track_ids)), '[]') AS negative_tracks,
                   coalesce(jsonb_agg(jsonb_build_object(
                     'id', t.id, 'title', t.title, 'artist', t.artist, 'album', t.album,
                     'duration_seconds', t.duration_seconds, 'source_id', crt.source_id,
                     'position', crt.position, 'score', crt.score,
                     'percentile', nullif(crt.evidence->>'percentile','')::real,
                     'retained', coalesce((crt.evidence->>'retained')::boolean, false),
                     'evidence', crt.evidence,
                     'cover_art', (
                       SELECT max(ts.source_data->>'coverArt')
                       FROM user_track_links utl
                       JOIN track_sources ts ON ts.library_id=utl.library_id
                                             AND ts.track_id=utl.track_id
                                             AND ts.external_id=crt.source_id
                       WHERE utl.user_id=c.user_id AND utl.track_id=crt.track_id
                     )
                   ) ORDER BY crt.position) FILTER (WHERE crt.track_id IS NOT NULL), '[]') AS tracks
               FROM curations c
               LEFT JOIN LATERAL (
                 SELECT id, recipe FROM curation_revisions
                 WHERE curation_id=c.id ORDER BY revision_number DESC LIMIT 1
               ) latest ON true
               LEFT JOIN curation_revision_tracks crt ON crt.revision_id=latest.id
               LEFT JOIN tracks t ON t.id=crt.track_id
               WHERE c.user_id=%s GROUP BY c.id, latest.recipe ORDER BY c.updated_at DESC""",
        (user_id,),
    )


def add_curation(
    session: Session, user_id: uuid.UUID, connection_id: uuid.UUID, columns: dict[str, Any]
) -> uuid.UUID:
    """Store a new curation; returns its id. A duplicate name fails when the row is written."""
    curation = Curation(user_id=user_id, navidrome_connection_id=connection_id, **columns)
    session.add(curation)
    session.flush()
    return curation.id


def owned_curation(session: Session, curation_id: uuid.UUID, user_id: uuid.UUID) -> Curation | None:
    """The user's curation."""
    return session.scalar(
        select(Curation).where(Curation.id == curation_id, Curation.user_id == user_id)
    )


def require_owned_curation(
    session: Session, curation_id: uuid.UUID, user_id: uuid.UUID
) -> Curation:
    """The user's curation, or ``NotFoundError("Curation not found")``."""
    curation = owned_curation(session, curation_id, user_id)
    if curation is None:
        raise NotFoundError("Curation not found")
    return curation


def remove_curation(session: Session, curation_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """Delete the user's curation (and, by cascade, its revisions), if it still exists."""
    curation = owned_curation(session, curation_id, user_id)
    if curation is not None:
        session.delete(curation)


# Refreshing and publishing a curation (worker.curation.jobs). The caller holds the curation's
# advisory lock on the connection, which must be dedicated to it: the lock is session-level.

_REFRESH_NEXT = (
    "next_refresh_at=CASE WHEN refresh_enabled THEN "
    "now()+make_interval(hours => refresh_interval_hours) END"
)


def lock_curation(db: Connection, curation_id: uuid.UUID) -> None:
    """Wait for and take the curation's session advisory lock."""
    db.execute(
        "SELECT pg_advisory_lock(hashtextextended(%s, 0))", ("curation:" + str(curation_id),)
    )


def try_lock_curation(db: Connection, curation_id: uuid.UUID) -> bool:
    """Take the curation's session advisory lock if no one holds it."""
    row = fetch_one(
        db,
        "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired",
        ("curation:" + str(curation_id),),
    )
    return bool(row and row["acquired"])


def unlock_curation(db: Connection, curation_id: uuid.UUID) -> None:
    """Release the curation's session advisory lock."""
    db.execute(
        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", ("curation:" + str(curation_id),)
    )


def curation_row(db: Connection, curation_id: uuid.UUID, user_id: uuid.UUID) -> dict | None:
    """Every column of the user's curation."""
    return fetch_one(
        db, "SELECT * FROM curations WHERE id=%s AND user_id=%s", (curation_id, user_id)
    )


def due_curation_ids(db: Connection, limit: int) -> list[uuid.UUID]:
    """Drafts and curations whose automatic refresh is due, the longest waiting first."""
    return [
        row["id"]
        for row in fetch_all(
            db,
            """SELECT id FROM curations
            WHERE status='draft' OR (refresh_enabled AND next_refresh_at <= now())
            ORDER BY next_refresh_at NULLS FIRST LIMIT %s""",
            (limit,),
        )
    ]


def due_curation(db: Connection, curation_id: uuid.UUID) -> dict | None:
    """Every column of the curation, if it is still a draft or due."""
    return fetch_one(
        db,
        """SELECT * FROM curations WHERE id=%s
        AND (status='draft' OR (refresh_enabled AND next_refresh_at <= now()))""",
        (curation_id,),
    )


def mark_queued(db: Connection, curation_id: uuid.UUID) -> None:
    """A refresh is queued: a draft becomes pending and the next automatic refresh moves on."""
    db.execute(
        f"""UPDATE curations SET status=CASE WHEN status='draft' THEN 'pending' ELSE status END,
        {_REFRESH_NEXT}, updated_at=now() WHERE id=%s""",
        (curation_id,),
    )


def unresolved_publication(db: Connection, curation_id: uuid.UUID) -> dict | None:
    """The curation's publication that has not completed, if any."""
    return fetch_one(
        db,
        "SELECT * FROM curation_publications WHERE curation_id=%s AND state <> 'complete'",
        (curation_id,),
    )


def latest_revision_track_ids(db: Connection, curation_id: uuid.UUID, limit: int) -> list:
    """The songs of the curation's newest revision, in playlist order."""
    return [
        row["track_id"]
        for row in fetch_all(
            db,
            """SELECT crt.track_id
        FROM curation_revision_tracks crt JOIN curation_revisions cr ON cr.id=crt.revision_id
        WHERE cr.curation_id=%s AND cr.revision_number=(SELECT max(revision_number)
            FROM curation_revisions WHERE curation_id=%s) ORDER BY crt.position LIMIT %s""",
            (curation_id, curation_id, limit),
        )
    ]


def track_names(db: Connection, track_ids: list) -> list[dict[str, Any]]:
    """``id`` (text), ``title`` and ``artist`` of each song."""
    return fetch_all(
        db, "SELECT id::text AS id, title, artist FROM tracks WHERE id=ANY(%s)", (track_ids,)
    )


def next_revision_number(db: Connection, curation_id: uuid.UUID) -> int:
    """The number the curation's next revision takes."""
    row = fetch_one(
        db,
        "SELECT coalesce(max(revision_number), 0)+1 AS value FROM curation_revisions WHERE curation_id=%s",
        (curation_id,),
    )
    return row["value"] if row else 1


def job_publication(db: Connection, curation_id: uuid.UUID, job_id: uuid.UUID) -> dict | None:
    """The job's publication, or else the curation's unresolved one."""
    return fetch_one(
        db,
        """SELECT * FROM curation_publications
        WHERE curation_id=%s AND (job_id=%s OR state <> 'complete')
        ORDER BY (job_id=%s) DESC LIMIT 1""",
        (curation_id, job_id, job_id),
    )


def add_publication(
    db: Connection,
    curation_id: uuid.UUID,
    job_id: uuid.UUID,
    plan: dict[str, Any],
    playlist_id: str | None,
) -> dict[str, Any]:
    """Store a prepared publication of the plan; returns its row."""
    return cast(
        dict[str, Any],
        fetch_one(
            db,
            """INSERT INTO curation_publications
            (id, curation_id, job_id, state, plan, playlist_id)
            VALUES (%s,%s,%s,'prepared',%s,%s) RETURNING *""",
            (uuid.uuid4(), curation_id, job_id, Jsonb(plan), playlist_id),
        ),
    )


def adopt_publication(db: Connection, publication_id: uuid.UUID, job_id: uuid.UUID) -> None:
    """Hand a publication to another job."""
    db.execute(
        "UPDATE curation_publications SET job_id=%s, updated_at=now() WHERE id=%s",
        (job_id, publication_id),
    )


def mark_publishing(db: Connection, publication_id: uuid.UUID, curation_id: uuid.UUID) -> None:
    """The publication goes to Navidrome next; the curation shows it as pending meanwhile."""
    db.execute(
        "UPDATE curation_publications SET state='publishing', updated_at=now() WHERE id=%s",
        (publication_id,),
    )
    db.execute(
        "UPDATE curations SET status='pending', last_error='Publication pending; refresh resumes the saved selection', updated_at=now() WHERE id=%s",
        (curation_id,),
    )


def record_playlist(
    db: Connection, publication_id: uuid.UUID, curation_id: uuid.UUID, playlist_id: str
) -> None:
    """Store the Navidrome playlist id on the publication and the curation."""
    db.execute(
        "UPDATE curation_publications SET playlist_id=%s, updated_at=now() WHERE id=%s",
        (playlist_id, publication_id),
    )
    db.execute(
        "UPDATE curations SET navidrome_playlist_id=%s WHERE id=%s", (playlist_id, curation_id)
    )


def add_revision(
    db: Connection,
    curation_id: uuid.UUID,
    revision_number: int,
    recipe: dict[str, Any],
    tracks: list[dict[str, Any]],
) -> None:
    """Store a revision and its songs in order.

    Each song has ``id``, ``score``, ``source_id`` and ``evidence``.
    """
    row = fetch_one(
        db,
        """INSERT INTO curation_revisions (curation_id, revision_number, recipe)
        VALUES (%s,%s,%s) RETURNING id""",
        (curation_id, revision_number, Jsonb(recipe)),
    )
    revision = cast(dict[str, Any], row)["id"]
    for position, track in enumerate(tracks):
        db.execute(
            """INSERT INTO curation_revision_tracks
            (revision_id, position, track_id, score, evidence, source_id)
            VALUES (%s,%s,%s,%s,%s,%s)""",
            (
                revision,
                position,
                track["id"],
                track["score"],
                Jsonb(track["evidence"]),
                track["source_id"],
            ),
        )


def complete_publication(db: Connection, publication_id: uuid.UUID, curation_id: uuid.UUID) -> None:
    """The curation is ready and refreshed now; its publication is complete."""
    db.execute(
        f"""UPDATE curations SET status='ready', last_error=NULL,
        last_refreshed_at=now(), {_REFRESH_NEXT},
        updated_at=now() WHERE id=%s""",
        (curation_id,),
    )
    db.execute(
        "UPDATE curation_publications SET state='complete', updated_at=now() WHERE id=%s",
        (publication_id,),
    )


def mark_failed(db: Connection, curation_id: uuid.UUID, error: str) -> None:
    """The refresh failed with ``error``."""
    db.execute(
        "UPDATE curations SET status='failed', last_error=%s, updated_at=now() WHERE id=%s",
        (error, curation_id),
    )
