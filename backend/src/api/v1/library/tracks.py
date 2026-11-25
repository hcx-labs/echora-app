"""``GET /library/tracks``."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import Depends, Query
from sqlalchemy import text

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.library.browse_filters import predicates
from core.db.session import session_scope
from core.errors import InvalidQueryError

SORT_COLUMNS = {
    "name": "lower(t.title)",
    "artist": "lower(t.artist)",
    "released": "t.year",
    "date_added": "source.date_added",
}


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
        sort_column = SORT_COLUMNS.get(sort_by)
        if sort_column is None:
            raise InvalidQueryError("Unknown track sort order")
        if sort_direction not in {"asc", "desc"}:
            raise InvalidQueryError("Unknown sort direction")
        direction = sort_direction.upper()
        nulls = " NULLS LAST" if sort_by in {"artist", "released", "date_added"} else ""
        ordering = f"{sort_column} {direction}{nulls}, lower(t.title), t.id"
        clauses: list[str] = [
            "EXISTS (SELECT 1 FROM user_track_links visible_links "
            "WHERE visible_links.track_id=t.id AND visible_links.user_id=%s)"
        ]
        parameters: list[object] = [user["id"]]
        if track_id is not None:
            clauses.append("t.id = %s")
            parameters.append(track_id)
        if q.strip():
            clauses.append(
                "(t.title ILIKE %s OR t.artist ILIKE %s OR t.album ILIKE %s OR array_to_string(t.genres, ' ') ILIKE %s)"
            )
            parameters.extend([f"%{q.strip()}%"] * 4)
        if artist:
            clauses.append("t.artist = %s")
            parameters.append(artist)
        if album:
            clauses.append("t.album = %s")
            parameters.append(album)
        filter_clauses, filter_parameters = predicates(
            vocals=vocals,
            language=language,
            lyrics=lyrics,
            translation=translation,
            genre=genre,
            year_from=year_from,
            year_to=year_to,
        )
        clauses.extend(filter_clauses)
        parameters.extend(filter_parameters)
        where = f"WHERE {' AND '.join(clauses)}"
        # The predicates use %s placeholders; SQLAlchemy's text() takes named ones.
        named_parameters = {f"p{index}": value for index, value in enumerate(parameters)}
        for name in named_parameters:
            where = where.replace("%s", f":{name}", 1)
        with session_scope() as session:
            total = session.execute(
                text(f"SELECT count(*) AS total FROM tracks t {where}"), named_parameters
            ).scalar_one()
            rows = (
                session.execute(
                    text(f"""
                SELECT t.id, t.title, t.artist, t.album, t.year, t.duration_seconds,
                       t.genres, t.ingested_at, l.availability_status AS lyrics_status,
                       source.date_added,
                       (SELECT count(DISTINCT e.run_id) FROM current_embeddings e
                        WHERE e.track_id=t.id AND e.embedding_type='audio-track') AS embedding_runs,
                       source.external_id AS source_id, source.album_id, source.cover_art
                FROM tracks t
                LEFT JOIN lyrics l ON l.track_id=t.id
                LEFT JOIN LATERAL (
                  SELECT ts.external_id, ts.source_data->>'albumId' AS album_id,
                         ts.source_data->>'coverArt' AS cover_art,
                         (ts.source_data->>'created')::timestamptz AS date_added
                  FROM track_sources ts
                  JOIN user_track_links source_links
                    ON source_links.user_id=:source_user_id AND source_links.library_id=ts.library_id
                   AND source_links.track_id=ts.track_id AND source_links.external_id=ts.external_id
                  WHERE ts.track_id=t.id AND ts.source_type='subsonic'
                  ORDER BY ts.id LIMIT 1
                ) source ON true
                {where}
                ORDER BY {ordering}
                LIMIT :limit OFFSET :offset
                """),
                    {
                        **named_parameters,
                        "source_user_id": user["id"],
                        "limit": limit,
                        "offset": offset,
                    },
                )
                .mappings()
                .all()
            )
            tracks = [dict(row) for row in rows]
        return {"tracks": tracks, "total": total, "limit": limit, "offset": offset}
