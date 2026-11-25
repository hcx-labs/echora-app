"""Queries on curations and the user's concepts.

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from psycopg import Connection

from core.db.connection import fetch_all, fetch_one

_CONCEPT_COLUMNS = """id, name, description, positive_prompts, negative_prompts,
                      positive_track_ids, negative_track_ids, enabled, created_at, updated_at"""


def concepts(db: Connection, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """The user's own concepts by name."""
    return fetch_all(
        db,
        f"SELECT {_CONCEPT_COLUMNS} FROM concepts WHERE user_id=%s ORDER BY lower(name)",
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
