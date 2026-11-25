"""Queries on motion artwork: its settings, its loops and the cover of each song.

Loops (``motion_artworks``) are one per cover and recipe; ``track_motion_artwork`` maps each
song to its current cover. Every function takes an open connection; the caller owns the transaction. Recipes are passed
by their hash.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from core.db.connection import fetch_all, fetch_one

SETTINGS_COLUMNS = (
    "enabled",
    "comfyui_url",
    "resolution",
    "frames",
    "upscale",
    "camera_lock",
    "prompt_mode",
    "instructions",
    "fixed_prompt",
    "seed",
    "generate_during_sync",
    "generate_on_modal",
    "regenerate_outdated",
)


def settings(db: Connection) -> dict[str, Any] | None:
    """The stored settings (``SETTINGS_COLUMNS``), or ``None`` before they are first saved."""
    return fetch_one(
        db,
        f"SELECT {', '.join(SETTINGS_COLUMNS)} FROM motion_artwork_settings WHERE singleton=true",
    )


def save_settings(db: Connection, values: dict[str, Any]) -> None:
    """Store the settings; ``values`` has every one of ``SETTINGS_COLUMNS``."""
    columns = ", ".join(SETTINGS_COLUMNS)
    placeholders = ", ".join(f"%({name})s" for name in SETTINGS_COLUMNS)
    updates = ", ".join(f"{name}=EXCLUDED.{name}" for name in SETTINGS_COLUMNS)
    db.execute(
        f"""INSERT INTO motion_artwork_settings (singleton, {columns}, updated_at)
            VALUES (true, {placeholders}, now())
            ON CONFLICT (singleton) DO UPDATE SET {updates}, updated_at=now()""",
        {name: values[name] for name in SETTINGS_COLUMNS},
    )


def sources_missing_loops(
    db: Connection,
    library_id: uuid.UUID,
    external_ids: list[str],
    recipe_hash: str,
    regenerate_outdated: bool,
) -> set[str]:
    """Sources with a cover that has no complete loop, including covers not yet seen.

    With ``regenerate_outdated``, a loop of another recipe counts as missing too.
    """
    rows = fetch_all(
        db,
        """SELECT DISTINCT ts.external_id FROM track_sources ts
           LEFT JOIN track_motion_artwork tma ON tma.track_id = ts.track_id
           WHERE ts.library_id = %s AND ts.source_type = 'subsonic' AND ts.external_id = ANY(%s)
             AND ts.source_data->>'coverArt' IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM motion_artworks ma WHERE ma.cover_sha256 = tma.cover_sha256
                             AND (%s OR ma.recipe_hash = %s) AND ma.status = 'complete')""",
        (library_id, list(external_ids), not regenerate_outdated, recipe_hash),
    )
    return {row["external_id"] for row in rows}


def lock_out_syncs(db: Connection) -> bool:
    """Block new jobs until the transaction ends; whether a sync or import is active.

    The lock makes a check of this and a deletion that relies on it commit together.
    """
    db.execute("LOCK TABLE jobs IN SHARE ROW EXCLUSIVE MODE")
    return (
        fetch_one(
            db,
            """SELECT 1 FROM jobs WHERE status IN ('queued','running','waiting')
            AND (kind IN ('navidrome_sync','import')
                 OR (kind='analysis_batch' AND payload->>'operation' IN ('navidrome_sync','import')))
            LIMIT 1""",
        )
        is not None
    )


def delete_all_loops(db: Connection) -> list[str]:
    """Forget every loop and every song's cover; returns the loops' stored file paths."""
    paths = [
        row["path"]
        for row in fetch_all(db, "DELETE FROM motion_artworks RETURNING path")
        if row["path"]
    ]
    db.execute("DELETE FROM track_motion_artwork")
    return paths


def track_loop(
    db: Connection, user_id: uuid.UUID, track_id: uuid.UUID, recipe_hash: str
) -> dict[str, Any] | None:
    """The newest complete loop of a song in the user's library, preferring ``recipe_hash``."""
    return fetch_one(
        db,
        """SELECT ma.id, ma.width, ma.height, ma.frames, ma.fps, ma.prompt, ma.created_at
           FROM user_track_links utl
           JOIN track_motion_artwork tma ON tma.track_id = utl.track_id
           JOIN motion_artworks ma ON ma.cover_sha256 = tma.cover_sha256 AND ma.status = 'complete'
           WHERE utl.user_id = %s AND utl.track_id = %s
           ORDER BY (ma.recipe_hash = %s) DESC, ma.created_at DESC LIMIT 1""",
        (user_id, track_id, recipe_hash),
    )


def visible_loop_path(db: Connection, artwork_id: uuid.UUID, user_id: uuid.UUID) -> str | None:
    """The stored file path of a complete loop that a song of the user's library uses."""
    row = fetch_one(
        db,
        """SELECT ma.path FROM motion_artworks ma
           WHERE ma.id = %s AND ma.status = 'complete' AND EXISTS (
             SELECT 1 FROM track_motion_artwork tma
             JOIN user_track_links utl ON utl.track_id = tma.track_id
             WHERE tma.cover_sha256 = ma.cover_sha256 AND utl.user_id = %s)""",
        (artwork_id, user_id),
    )
    return row["path"] if row else None


def loop_counts(db: Connection) -> dict[str, int]:
    """Complete and failed loops, and the complete ones' total size (``bytes``)."""
    return dict(
        fetch_one(
            db,
            """SELECT count(*) FILTER (WHERE status='complete') AS complete,
                      count(*) FILTER (WHERE status='failed') AS failed,
                      coalesce(sum(bytes) FILTER (WHERE status='complete'), 0) AS bytes
               FROM motion_artworks""",
        )
        or {}
    )


def songs_with_loops(db: Connection, user_id: uuid.UUID) -> int:
    """How many of the user's songs have a complete loop."""
    row = fetch_one(
        db,
        """SELECT count(DISTINCT utl.track_id) AS tracks FROM user_track_links utl
           JOIN track_motion_artwork tma ON tma.track_id = utl.track_id
           JOIN motion_artworks ma ON ma.cover_sha256 = tma.cover_sha256
                                  AND ma.status='complete'
           WHERE utl.user_id = %s""",
        (user_id,),
    )
    return row["tracks"] if row else 0


def batch_songs(
    db: Connection, user_id: uuid.UUID, library_root: str, external_ids: list[str]
) -> list[dict[str, Any]]:
    """The user's selected songs that have a cover, with their newest non-empty lyrics.

    Rows are ``track_id``, ``external_id``, ``title``, ``album``, ``artist`` (the album artist
    when Navidrome has one), ``cover_art_id`` and ``lyrics``, by album and title.
    """
    return fetch_all(
        db,
        """SELECT t.id AS track_id, ts.external_id, t.title, t.album,
                  coalesce(ts.source_data->>'displayAlbumArtist', t.artist) AS artist,
                  ts.source_data->>'coverArt' AS cover_art_id, lyric.text AS lyrics
           FROM track_sources ts
           JOIN libraries lib ON lib.id = ts.library_id
           JOIN user_track_links u ON u.library_id = ts.library_id AND u.track_id = ts.track_id
             AND u.external_id = ts.external_id
           JOIN tracks t ON t.id = ts.track_id
           LEFT JOIN LATERAL (
             SELECT text FROM lyrics WHERE track_id=t.id AND nullif(btrim(text), '') IS NOT NULL
             ORDER BY created_at DESC LIMIT 1
           ) lyric ON true
           WHERE u.user_id = %s AND lib.root_path = %s AND ts.source_type = 'subsonic'
             AND ts.external_id = ANY(%s) AND ts.source_data->>'coverArt' IS NOT NULL
           ORDER BY t.album, t.title, t.id""",
        (user_id, library_root, list(external_ids)),
    )


def source_lyrics(
    db: Connection, library_root: str, external_ids: list[str]
) -> list[dict[str, Any]]:
    """The library's selected sources with their song's newest non-empty lyrics.

    Rows are ``external_id``, ``track_id`` and ``lyrics``.
    """
    return fetch_all(
        db,
        """SELECT ts.external_id, ts.track_id,
                  (SELECT text FROM lyrics WHERE track_id=ts.track_id
                     AND nullif(btrim(text), '') IS NOT NULL ORDER BY created_at DESC LIMIT 1) AS lyrics
           FROM track_sources ts JOIN libraries lib ON lib.id = ts.library_id
           WHERE lib.root_path = %s AND ts.source_type = 'subsonic' AND ts.external_id = ANY(%s)""",
        (library_root, list(external_ids)),
    )


def has_complete_loop(
    db: Connection, cover_sha256: str, recipe_hash: str, regenerate_outdated: bool
) -> bool:
    """Whether the cover has a complete loop: any, or with ``regenerate_outdated`` this recipe's."""
    return (
        fetch_one(
            db,
            """SELECT 1 FROM motion_artworks WHERE cover_sha256=%s AND status='complete'
                 AND (%s OR recipe_hash=%s)""",
            (cover_sha256, not regenerate_outdated, recipe_hash),
        )
        is not None
    )


def set_track_cover(
    db: Connection, track_id: uuid.UUID, cover_sha256: str, cover_art_id: str
) -> None:
    """Record which cover (by its asset key) the song has now."""
    db.execute(
        """INSERT INTO track_motion_artwork (track_id, cover_sha256, cover_art_id, checked_at)
           VALUES (%s,%s,%s,now()) ON CONFLICT (track_id) DO UPDATE SET
             cover_sha256=EXCLUDED.cover_sha256, cover_art_id=EXCLUDED.cover_art_id, checked_at=now()""",
        (track_id, cover_sha256, cover_art_id),
    )


def record_loop(
    db: Connection,
    *,
    cover_sha256: str,
    recipe_hash: str,
    recipe: dict[str, Any],
    status: str,
    prompt: str | None = None,
    error: str | None = None,
    path: str | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    fps: int | None = None,
    file_bytes: int | None = None,
) -> None:
    """Store the outcome of rendering a cover with a recipe, replacing an earlier one."""
    db.execute(
        """INSERT INTO motion_artworks
             (id, cover_sha256, recipe_hash, status, prompt, error, path, width, height, frames, fps, bytes, recipe)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (cover_sha256, recipe_hash) DO UPDATE SET
             id=EXCLUDED.id, status=EXCLUDED.status, prompt=EXCLUDED.prompt, error=EXCLUDED.error,
             path=EXCLUDED.path, width=EXCLUDED.width, height=EXCLUDED.height, frames=EXCLUDED.frames,
             fps=EXCLUDED.fps, bytes=EXCLUDED.bytes, recipe=EXCLUDED.recipe, created_at=now()""",
        (
            uuid.uuid4(),
            cover_sha256,
            recipe_hash,
            status,
            prompt,
            error,
            path,
            width,
            height,
            frames,
            fps,
            file_bytes,
            Jsonb(recipe),
        ),
    )


def delete_other_recipes(db: Connection, cover_sha256: str, recipe_hash: str) -> list[str]:
    """Forget the cover's loops of other recipes; returns their stored file paths."""
    return [
        row["path"]
        for row in fetch_all(
            db,
            """DELETE FROM motion_artworks WHERE cover_sha256=%s AND recipe_hash<>%s
               RETURNING path""",
            (cover_sha256, recipe_hash),
        )
        if row["path"]
    ]


def try_lock_cover(db: Connection, cover_sha256: str) -> bool:
    """Take the cover's session advisory lock if no one holds it (needs autocommit)."""
    row = fetch_one(
        db,
        "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS owned",
        (f"motion-artwork:{cover_sha256}",),
    )
    return bool(row and row["owned"])


def unlock_cover(db: Connection, cover_sha256: str) -> None:
    """Release the cover's session advisory lock."""
    db.execute(
        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (f"motion-artwork:{cover_sha256}",)
    )
