"""Queries on the Navidrome plugin integration and its keys.

The tables are ``navidrome_integrations`` (one row per user) and ``navidrome_api_keys`` (keys are
stored by token hash).

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from core.db.connection import fetch_all, fetch_one


def lock_owner(db: Connection, user_id: uuid.UUID) -> None:
    """Hold the user's row until the transaction ends.

    Enabling the integration, generating keys and changing its connection take this lock, so
    they run one at a time per user.
    """
    db.execute("SELECT id FROM users WHERE id=%s FOR UPDATE", (user_id,))


def integration(db: Connection, user_id: uuid.UUID) -> dict[str, Any] | None:
    """The user's integration row (``enabled``, ``connection_id``, ``profile``...)."""
    return fetch_one(db, "SELECT * FROM navidrome_integrations WHERE user_id=%s", (user_id,))


def save_integration(
    db: Connection,
    user_id: uuid.UUID,
    enabled: bool,
    connection_id: uuid.UUID | None,
    profile: dict[str, Any],
) -> None:
    """Store the user's integration settings."""
    db.execute(
        """INSERT INTO navidrome_integrations(user_id,enabled,connection_id,profile)
        VALUES (%s,%s,%s,%s) ON CONFLICT(user_id) DO UPDATE SET enabled=EXCLUDED.enabled,
        connection_id=EXCLUDED.connection_id,profile=EXCLUDED.profile,updated_at=now()""",
        (user_id, enabled, connection_id, Jsonb(profile)),
    )


def owned_connection_url(
    db: Connection, connection_id: uuid.UUID | None, user_id: uuid.UUID
) -> str | None:
    """The URL of the user's own Navidrome connection (``None`` if not theirs or no id)."""
    row = fetch_one(
        db,
        "SELECT url FROM navidrome_connections WHERE id=%s AND owner_user_id=%s",
        (connection_id, user_id),
    )
    return row["url"] if row else None


def keys(db: Connection, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """The user's keys, newest first, each with ``status`` (active, expired or revoked)."""
    return fetch_all(
        db,
        """SELECT id,label,prefix,connection_id,created_at,expires_at,revoked_at,
                  last_used_at,
        CASE WHEN revoked_at IS NOT NULL THEN 'revoked' WHEN expires_at<=now() THEN 'expired'
             ELSE 'active' END AS status
        FROM navidrome_api_keys WHERE user_id=%s ORDER BY created_at DESC""",
        (user_id,),
    )


def add_key(
    db: Connection,
    key_id: uuid.UUID,
    user_id: uuid.UUID,
    connection_id: uuid.UUID,
    label: str,
    prefix: str,
    token_hash: str,
    created_at: datetime,
    expires_at: datetime,
) -> None:
    """Store a new key by its token hash."""
    db.execute(
        """INSERT INTO navidrome_api_keys
        (id,user_id,connection_id,label,prefix,token_hash,created_at,expires_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (key_id, user_id, connection_id, label, prefix, token_hash, created_at, expires_at),
    )


def has_active_key(db: Connection, user_id: uuid.UUID, connection_id: uuid.UUID | None) -> bool:
    """Whether the user has an unrevoked, unexpired key for the connection."""
    return (
        fetch_one(
            db,
            """SELECT 1 FROM navidrome_api_keys WHERE user_id=%s AND connection_id=%s
            AND revoked_at IS NULL AND expires_at>now() LIMIT 1""",
            (user_id, connection_id),
        )
        is not None
    )


def revoke_key(db: Connection, key_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    """Revoke one of the user's keys (keeping an earlier revocation time); whether it exists."""
    return (
        fetch_one(
            db,
            """UPDATE navidrome_api_keys SET revoked_at=coalesce(revoked_at,now())
            WHERE id=%s AND user_id=%s RETURNING id""",
            (key_id, user_id),
        )
        is not None
    )


def revoke_all_keys(db: Connection, user_id: uuid.UUID) -> None:
    """Revoke every unrevoked key of the user."""
    db.execute(
        """UPDATE navidrome_api_keys SET revoked_at=now()
           WHERE user_id=%s AND revoked_at IS NULL""",
        (user_id,),
    )


def key_holder(db: Connection, token_hash: str) -> dict[str, Any] | None:
    """Who a valid key belongs to.

    The row has ``key_id``, ``user_id``, ``connection_id``, ``url`` and ``profile``. Only an unrevoked, unexpired key of an unblocked user, for the connection their enabled
    integration uses, is valid.
    """
    return fetch_one(
        db,
        """SELECT k.id AS key_id,k.user_id,k.connection_id,n.url,i.profile
        FROM navidrome_api_keys k
        JOIN navidrome_integrations i ON i.user_id=k.user_id AND i.enabled
            AND i.connection_id=k.connection_id
        JOIN users u ON u.id=k.user_id AND NOT u.is_blocked
        JOIN navidrome_connections n ON n.id=k.connection_id AND n.owner_user_id=k.user_id
        WHERE k.token_hash=%s AND k.revoked_at IS NULL AND k.expires_at>now()""",
        (token_hash,),
    )


def record_key_use(db: Connection, key_id: uuid.UUID) -> None:
    """Note that the key was just used."""
    db.execute("UPDATE navidrome_api_keys SET last_used_at=now() WHERE id=%s", (key_id,))


# The latest current embedding of each kind per visible song, as in the legacy plugin query.
_SONGS = """
    FROM user_source_memberships visible
    JOIN libraries lib ON lib.id=visible.library_id AND lib.namespace=%s
    JOIN track_sources ts ON ts.library_id=visible.library_id
        AND ts.external_id=visible.external_id AND ts.source_type='subsonic'
    JOIN tracks t ON t.id=ts.track_id
    LEFT JOIN recording_group_members member ON member.track_id=t.id
    LEFT JOIN LATERAL (
        SELECT e.id, e.embedding FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
        WHERE e.track_id=t.id AND e.embedding_type='audio-track' AND e.window_index IS NULL
            AND ar.model_name='muq_mulan' ORDER BY ar.created_at DESC LIMIT 1
    ) muq ON true
    LEFT JOIN LATERAL (
        SELECT e.id, e.embedding FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
        WHERE e.track_id=t.id AND e.embedding_type='audio-track' AND e.window_index IS NULL
            AND ar.model_name='mert' ORDER BY ar.created_at DESC LIMIT 1
    ) mert ON true
    LEFT JOIN LATERAL (
        SELECT e.id, e.embedding FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
        WHERE e.track_id=t.id AND e.embedding_type='lyrics' AND e.window_index IS NULL
            AND ar.model_name='bge_m3' ORDER BY ar.created_at DESC LIMIT 1
    ) lyric ON true
    WHERE visible.user_id=%s AND (muq.embedding IS NOT NULL OR mert.embedding IS NOT NULL
        OR lyric.embedding IS NOT NULL)"""

_COLUMNS = """t.id,t.title,t.artist,t.album,t.duration_seconds,ts.external_id AS source_id,
    coalesce(ts.source_data->>'artistId',ts.source_data#>>'{artists,0,id}') AS artist_id,
    coalesce(ts.source_data->>'artist',t.artist) AS source_artist,
    member.group_id::text AS recording_group_id"""

# Everything the corpus holds except the vectors, which the current embedding ids stand for.
# ROW()::text keeps NULLs in place, so different songs never digest alike.
_KEY_COLUMNS = """t.id, t.title, t.artist, t.album, t.duration_seconds, ts.external_id,
    ts.source_data->>'artistId', ts.source_data#>>'{artists,0,id}', ts.source_data->>'artist',
    member.group_id, muq.id, mert.id, lyric.id"""


def corpus_key(db: Connection, user_id: uuid.UUID, namespace: uuid.UUID) -> str:
    """A digest of the plugin corpus: it changes when a visible song or its embeddings do."""
    row = fetch_one(
        db,
        f"""SELECT md5(coalesce(string_agg(ROW({_KEY_COLUMNS})::text, ','
                           ORDER BY t.id, ts.external_id), '')) AS key
            {_SONGS}""",
        (namespace, user_id),
    )
    return row["key"] if row else ""


def corpus(db: Connection, user_id: uuid.UUID, namespace: uuid.UUID) -> list[dict[str, Any]]:
    """The user's songs in the library namespace that have an embedding.

    Each row has the newest vector of each kind in text form (``muq``, ``mert``, ``lyrics``).
    """
    return fetch_all(
        db,
        f"""SELECT {_COLUMNS},
               muq.embedding::text AS muq, mert.embedding::text AS mert,
               lyric.embedding::text AS lyrics
            {_SONGS} ORDER BY t.id,ts.external_id""",
        (namespace, user_id),
    )


def song_lyrics(
    db: Connection, user_id: uuid.UUID, namespace: uuid.UUID, source_id: str
) -> dict[str, Any] | None:
    """The lyrics and unbounded karaoke lines of one of the user's Navidrome songs."""
    return fetch_one(
        db,
        """SELECT t.id,l.text,l.language,l.provenance,k.lines AS karaoke_lines
        FROM user_source_memberships visible
        JOIN libraries lib ON lib.id=visible.library_id AND lib.namespace=%s
        JOIN track_sources ts ON ts.library_id=visible.library_id AND ts.external_id=visible.external_id
            AND ts.source_type='subsonic'
        JOIN tracks t ON t.id=ts.track_id
        LEFT JOIN lyrics l ON l.track_id=t.id
        LEFT JOIN karaoke_lyrics_variants k ON k.track_id=t.id AND k.bounded=false
        WHERE visible.user_id=%s AND visible.external_id=%s LIMIT 1""",
        (namespace, user_id, source_id),
    )
