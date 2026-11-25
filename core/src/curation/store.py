"""Queries on curations and the user's concepts.

Curations are records (``core.db.models.Curation``): those functions take an open ``Session``.
The rest take an open psycopg connection. Either way the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from psycopg import Connection
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
