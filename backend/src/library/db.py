"""Database access shared by the library endpoints (raw SQL until the data-access branch)."""

from __future__ import annotations

import uuid

from core.errors import NotFoundError


def require_visible(cursor, user_id, track_id: uuid.UUID, detail: str) -> None:
    """Raise ``NotFoundError(detail)`` unless the track is in the user's library."""
    cursor.execute(
        "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s", (user_id, track_id)
    )
    if cursor.fetchone() is None:
        raise NotFoundError(detail)


def parse_vector(text: str):
    """A pgvector's text form as a NumPy array."""
    import numpy as np

    return np.fromstring(text.strip("[]"), sep=",")
