"""``GET /library/top-tracks``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.library import top_tracks as ranking
from core.curation.listening import (
    TOP_TRACK_PERIODS,
    match_navidrome_play_counts,
    match_top_tracks,
    top_tracks,
)
from core.db.session import session_scope
from core.errors import IntegrationError, InvalidQueryError
from core.library.connections import credential_cipher
from core.users import store as users


class TopTrack(ResponseModel):
    """A library song and its play count."""

    id: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration_seconds: float | None = None
    source_id: str | None = None
    cover_art: str | None = None
    play_count: int


class TopTracks(ResponseModel):
    """Most played first; ``source`` is ``lastfm``, ``navidrome`` or none (unavailable)."""

    available: bool
    source: str | None
    period: str
    """The Last.fm period asked for, or ``overall`` for Navidrome's counts."""
    tracks: list[TopTrack]


class TopTracksEndpoint(Endpoint):
    """The user's most played library tracks. Echora records no plays itself.

    Last.fm supplies ranked listening periods. Without Last.fm, Navidrome's per-song play counts
    give an all-time ranking (``source`` says which).
    """

    method = "GET"
    path = "/top-tracks"
    response = TopTracks

    def handle(
        self, limit: int = 10, period: str = "1month", user: SessionUser = Depends(current_user)
    ) -> dict:
        limit = min(max(limit, 1), 50)
        if period not in TOP_TRACK_PERIODS:
            raise InvalidQueryError("Unknown listening period")
        with session_scope() as session:
            preference = users.preference(session, user["id"])
            username = preference.lastfm_username if preference else None
            encrypted_key = (
                bytes(preference.lastfm_api_key_encrypted)
                if preference and preference.lastfm_api_key_encrypted
                else None
            )
        if username and encrypted_key:
            try:
                # Ask for more than we show: some top tracks are not in this library.
                entries = top_tracks(
                    username,
                    credential_cipher().decrypt(encrypted_key).decode(),
                    period,
                    min(limit * 5, 200),
                )
            except Exception as error:
                raise IntegrationError(
                    f"Could not read Last.fm listening history: {error}"
                ) from error
            matched = match_top_tracks(ranking.library_track_rows(user["id"]), entries)
            return {
                "available": True,
                "source": "lastfm",
                "period": period,
                "tracks": [{**row, "play_count": count} for row, count in matched[:limit]],
            }
        connection_id = user.get("navidrome_connection_id")
        if not connection_id:
            return {"available": False, "source": None, "period": period, "tracks": []}
        try:
            ranked = ranking.navidrome_ranking(str(user["id"]), str(connection_id))
        except Exception as error:
            raise IntegrationError(f"Could not read Navidrome play counts: {error}") from error
        if ranked is None:
            return {"available": False, "source": None, "period": period, "tracks": []}
        matched = match_navidrome_play_counts(ranking.library_track_rows(user["id"]), ranked)
        return {
            "available": True,
            "source": "navidrome",
            "period": "overall",
            "tracks": [{**row, "play_count": count} for row, count in matched[:limit]],
        }
