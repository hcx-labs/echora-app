"""The user's most played library songs: Last.fm's periods, or Navidrome's all-time counts."""

from __future__ import annotations

import threading
import time
import uuid

from core.curation.listening import navidrome_play_counts
from core.db.connection import connect
from core.integrations.navidrome import NavidromeClient
from core.library import store
from core.library.connections import load_connection

# Navidrome play counts need a full catalog scan, so keep each user's ranking briefly.
NAVIDROME_TOP_TTL_SECONDS = 300
_navidrome_top_cache: dict[tuple[str, str], tuple[float, list[tuple[str, int]]]] = {}
_navidrome_top_lock = threading.Lock()


def navidrome_ranking(user_id: str, connection_id: str) -> list[tuple[str, int]] | None:
    """Navidrome song ids by play count (cached briefly), or None without the connection."""
    key = (user_id, connection_id)
    with _navidrome_top_lock:
        cached = _navidrome_top_cache.get(key)
    if cached and time.monotonic() - cached[0] < NAVIDROME_TOP_TTL_SECONDS:
        return cached[1]
    credentials = load_connection(connection_id, uuid.UUID(user_id))
    if credentials is None:
        return None
    with NavidromeClient(*credentials) as client:
        ranking = navidrome_play_counts([track.raw for track in client.all_tracks()])
    with _navidrome_top_lock:
        _navidrome_top_cache[key] = (time.monotonic(), ranking)
    return ranking


def library_track_rows(user_id: object) -> list[dict[str, object]]:
    """Every song of the user's library with its Navidrome source id and cover."""
    with connect() as connection:
        return store.library_songs(connection, user_id)
