"""Queries on stored embeddings: vectors and nearest neighbours (pgvector).

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection

from core.db.connection import fetch_all, fetch_one


def semantic_fusion_vector(db: Connection, track_id: uuid.UUID) -> str | None:
    """The song's current semantic fusion vector, in pgvector's text form."""
    row = fetch_one(
        db,
        """SELECT e.embedding FROM current_embeddings e
           WHERE e.track_id=%s AND e.embedding_type='semantic_fusion'""",
        (track_id,),
    )
    return row["embedding"] if row else None


def nearest_semantic_fusion(
    db: Connection, vector: str, exclude: uuid.UUID, limit: int
) -> list[dict[str, Any]]:
    """The songs closest to ``vector`` (cosine), with id, title, artist, album and similarity."""
    return fetch_all(
        db,
        """SELECT t.id, t.title, t.artist, t.album,
                  1 - (e.embedding <=> %s::vector) AS similarity
           FROM current_embeddings e
           JOIN tracks t ON t.id=e.track_id
           WHERE e.embedding_type='semantic_fusion' AND e.track_id != %s
           ORDER BY e.embedding <=> %s::vector LIMIT %s""",
        (vector, exclude, vector, limit),
    )
