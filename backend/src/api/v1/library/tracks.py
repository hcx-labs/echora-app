"""``GET /library/tracks``."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import Depends, Query

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.library import browse


class LibraryTrack(ResponseModel):
    """A song of the user's library, with its Navidrome source."""

    id: uuid.UUID
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    year: int | None = None
    duration_seconds: float | None = None
    genres: list[str] | None = None
    ingested_at: datetime | None = None
    lyrics_status: str | None = None
    date_added: datetime | None = None
    embedding_runs: int
    """Distinct current audio embedding runs."""
    source_id: str | None = None
    album_id: str | None = None
    cover_art: str | None = None


class TrackPage(ResponseModel):
    """A page of songs and the total matching the filters."""

    tracks: list[LibraryTrack]
    total: int
    limit: int
    offset: int


class LibraryTracks(Endpoint):
    """Browse the user's songs: search, artist and album, browse filters, sorting and paging."""

    method = "GET"
    path = "/tracks"
    response = TrackPage

    def handle(
        self,
        limit: int = 10,
        offset: int = 0,
        q: str = "",
        artist: str = "",
        album: str = "",
        sort_by: str = "name",
        sort_direction: str = "asc",
        track_id: uuid.UUID | None = None,
        vocals: list[str] | None = Query(default=None),
        language: list[str] | None = Query(default=None),
        lyrics: list[str] | None = Query(default=None),
        translation: list[str] | None = Query(default=None),
        genre: list[str] | None = Query(default=None),
        year_from: int | None = None,
        year_to: int | None = None,
        user: SessionUser = Depends(current_user),
    ) -> dict[str, object]:
        limit = min(max(limit, 1), 100)
        offset = max(offset, 0)
        with connect() as connection:
            tracks, total = browse.track_page(
                connection,
                user["id"],
                limit=limit,
                offset=offset,
                q=q,
                artist=artist,
                album=album,
                sort_by=sort_by,
                sort_direction=sort_direction,
                track_id=track_id,
                vocals=vocals,
                language=language,
                lyrics=lyrics,
                translation=translation,
                genre=genre,
                year_from=year_from,
                year_to=year_to,
            )
        return {"tracks": tracks, "total": total, "limit": limit, "offset": offset}
