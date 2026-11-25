"""Saved Navidrome connections as the sync and media endpoints use them."""

from __future__ import annotations

import hashlib
from typing import Any

import core.jobs.queue as jobs
from core.errors import NotFoundError
from core.integrations.navidrome import NavidromeTrack
from core.library.connections import load_connection


def owned_connection(connection_id: str, user_id: Any) -> tuple[str, str, str]:
    """The user's connection credentials (404 when the connection is not theirs)."""
    credentials = load_connection(connection_id, user_id)
    if credentials is None:
        raise NotFoundError("Connection not found")
    return credentials


def track_summary(track: NavidromeTrack) -> dict[str, object]:
    """A Navidrome song as onboarding shows it."""
    return {
        "id": track.id,
        "title": track.title,
        "artist": track.artist,
        "album": track.album,
        "duration": track.duration,
        "genre": track.genre,
        "cover_art": track.raw.get("coverArt"),
    }


def enqueue_connection_job(
    kind: str, connection_id: str, user: dict[str, Any], payload: dict[str, object] | None = None
) -> dict[str, Any]:
    """Queue a job over one connection, one active per user and connection."""
    # Validate ownership without putting decrypted credentials into job payloads.
    owned_connection(connection_id, user["id"])
    return jobs.enqueue(
        kind,
        "analysis",
        user["id"],
        connection_id=connection_id,
        payload=payload or {},
        dedupe_key=f"{kind}:{user['id']}:{connection_id}",
    )


def enqueue_import(connection_id: str, user_id: Any, track_ids: list[str]) -> dict[str, Any]:
    """Queue analysis of chosen songs; the same selection is one job while it is active."""
    chosen = sorted(set(track_ids))
    selection = hashlib.sha256("\n".join(chosen).encode()).hexdigest()
    return jobs.enqueue(
        "import",
        "analysis",
        user_id,
        connection_id=connection_id,
        payload={"track_ids": chosen},
        dedupe_key=f"import:{user_id}:{connection_id}:{selection}",
    )
