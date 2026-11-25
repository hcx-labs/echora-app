"""Browsing the user's library: the song page, its filters, facets and filter options.

``predicates`` turns the browse filters into bound SQL predicates on ``tracks t``: a song
matches any selected value of a group and every active group. The query functions take an open
connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection

from core.db.connection import fetch_all, fetch_one
from core.errors import InvalidQueryError

SORT_COLUMNS = {
    "name": "lower(t.title)",
    "artist": "lower(t.artist)",
    "released": "t.year",
    "date_added": "source.date_added",
}


def _values(value, *, split_commas=True):
    """Keep legacy comma-separated codes; genre names may themselves contain commas."""
    if not value:
        return []
    values = [value] if isinstance(value, str) else value
    result = list(
        dict.fromkeys(
            part
            for item in values
            for part in (item.split(",") if split_commas else [item])
            if part
        )
    )
    if len(result) > 128:
        raise InvalidQueryError("Too many filter values")
    return result


def predicates(
    *, vocals="", language="", lyrics="", translation="", genre="", year_from=None, year_to=None
):
    clauses, params = [], []

    def add(sql, *values):
        clauses.append(sql)
        params.extend(values)

    if year_from is not None and year_to is not None and year_from > year_to:
        raise InvalidQueryError("Release year range is reversed")
    genres = _values(genre, split_commas=False)
    if len(genres) == 1:
        add("t.genres @> ARRAY[%s]::text[]", genres[0])
    elif genres:
        add("t.genres && %s::text[]", genres)
    if year_from is not None:
        add("t.year >= %s", year_from)
    if year_to is not None:
        add("t.year <= %s", year_to)
    languages = _values(language)
    if languages:
        known = [item for item in languages if item != "unknown"]
        pieces = []
        if known:
            pieces.append(
                "EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND bl.language=ANY(%s))"
            )
            params.append(known)
        if "unknown" in languages:
            pieces.append(
                "NOT EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND NULLIF(bl.language,'') IS NOT NULL)"
            )
        clauses.append("(" + " OR ".join(pieces) + ")")
    lyric_values = _values(lyrics)
    if lyric_values:
        choices = {
            "available": "EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND NULLIF(btrim(bl.text),'') IS NOT NULL)",
            "missing": "NOT EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND NULLIF(btrim(bl.text),'') IS NOT NULL)",
            "ai": "EXISTS (SELECT 1 FROM lyrics bl WHERE bl.track_id=t.id AND bl.provenance->>'ai_generated'='true')",
        }
        if any(value not in choices for value in lyric_values):
            raise InvalidQueryError("Unknown lyrics filter")
        clauses.append("(" + " OR ".join(choices[value] for value in lyric_values) + ")")
    translations = _values(translation)
    if len(translations) == 1:
        add(
            "EXISTS (SELECT 1 FROM lyric_translations bt WHERE bt.track_id=t.id AND bt.status='ready' AND bt.target_language=%s)",
            translations[0],
        )
    elif translations:
        add(
            "EXISTS (SELECT 1 FROM lyric_translations bt WHERE bt.track_id=t.id AND bt.status='ready' AND bt.target_language=ANY(%s))",
            translations,
        )
    vocal_values = _values(vocals)
    if vocal_values:
        if any(
            value not in {"instrumental", "vocal", "female", "male", "unknown"}
            for value in vocal_values
        ):
            raise InvalidQueryError("Unknown vocals filter")
        base = "SELECT 1 FROM current_embeddings be WHERE be.track_id=t.id AND be.embedding_type='voice-gender' AND be.window_index IS NULL"
        # Same instrumental threshold as curation; gender is model evidence, not identity.
        score = "(be.embedding::real[])"
        conditions = {
            "instrumental": f"{score}[1] >= 0.5",
            "vocal": f"{score}[1] < 0.5",
            "female": f"{score}[1] < 0.5 AND {score}[2] >= {score}[3]",
            "male": f"{score}[1] < 0.5 AND {score}[3] > {score}[2]",
        }
        pieces = [
            "NOT EXISTS (" + base + ")"
            if value == "unknown"
            else "EXISTS (" + base + " AND " + conditions[value] + ")"
            for value in vocal_values
        ]
        clauses.append("(" + " OR ".join(pieces) + ")")
    return clauses, params


def track_page(
    db: Connection,
    user_id: uuid.UUID,
    *,
    limit: int,
    offset: int,
    q: str = "",
    artist: str = "",
    album: str = "",
    sort_by: str = "name",
    sort_direction: str = "asc",
    track_id: uuid.UUID | None = None,
    **filters: Any,
) -> tuple[list[dict[str, Any]], int]:
    """One page of the user's songs and how many match in all.

    The search narrows by text, artist, album or one song; ``filters`` are ``predicates``'s.
    """
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
    parameters: list[object] = [user_id]
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
    filter_clauses, filter_parameters = predicates(**filters)
    clauses.extend(filter_clauses)
    parameters.extend(filter_parameters)
    where = f"WHERE {' AND '.join(clauses)}"
    total = fetch_one(db, f"SELECT count(*) AS total FROM tracks t {where}", parameters)
    rows = fetch_all(
        db,
        f"""SELECT t.id, t.title, t.artist, t.album, t.year, t.duration_seconds,
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
                ON source_links.user_id=%s AND source_links.library_id=ts.library_id
               AND source_links.track_id=ts.track_id AND source_links.external_id=ts.external_id
              WHERE ts.track_id=t.id AND ts.source_type='subsonic'
              ORDER BY ts.id LIMIT 1
            ) source ON true
            {where}
            ORDER BY {ordering}
            LIMIT %s OFFSET %s""",
        [user_id, *parameters, limit, offset],
    )
    return rows, total["total"] if total else 0


def facets(
    db: Connection, user_id: uuid.UUID, artist_query: str, album_query: str, artist: str, limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Artists and albums of the user's library matching the queries.

    Rows are ``name`` and ``tracks`` (the song count); albums only of ``artist`` when given.
    """
    artists = fetch_all(
        db,
        """SELECT artist AS name, count(*) AS tracks FROM tracks t
           WHERE artist IS NOT NULL AND artist ILIKE %s
             AND EXISTS (SELECT 1 FROM user_track_links utl
                         WHERE utl.track_id=t.id AND utl.user_id=%s)
           GROUP BY artist ORDER BY artist LIMIT %s""",
        (f"%{artist_query.strip()}%", user_id, limit),
    )
    album_clauses = [
        "album IS NOT NULL",
        "album ILIKE %s",
        "EXISTS (SELECT 1 FROM user_track_links utl WHERE utl.track_id=t.id AND utl.user_id=%s)",
    ]
    album_parameters: list[object] = [f"%{album_query.strip()}%", user_id]
    if artist:
        album_clauses.append("artist = %s")
        album_parameters.append(artist)
    albums = fetch_all(
        db,
        f"SELECT album AS name, count(*) AS tracks FROM tracks t WHERE {' AND '.join(album_clauses)} GROUP BY album ORDER BY album LIMIT %s",
        [*album_parameters, limit],
    )
    return artists, albums


def filter_options(db: Connection, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """Lyrics languages, genres and ready translation languages of the user's songs.

    Rows are ``kind``, ``value`` and its song count (``tracks``).
    """
    return fetch_all(
        db,
        """WITH visible AS MATERIALIZED (
            SELECT DISTINCT track_id FROM user_track_links WHERE user_id=%s
        ), options AS (
            SELECT 'language' AS kind, coalesce(nullif(l.language,''),'unknown') AS value, v.track_id
            FROM visible v LEFT JOIN lyrics l ON l.track_id=v.track_id
            UNION ALL
            SELECT 'genre', g, v.track_id FROM visible v JOIN tracks t ON t.id=v.track_id
            CROSS JOIN LATERAL unnest(t.genres) g WHERE g<>''
            UNION ALL
            SELECT 'translation', lt.target_language, v.track_id FROM visible v
            JOIN lyric_translations lt ON lt.track_id=v.track_id WHERE lt.status='ready'
        ) SELECT kind,value,count(DISTINCT track_id) AS tracks FROM options
          GROUP BY kind,value ORDER BY kind,lower(value)""",
        (user_id,),
    )
