"""Queries on a song's lyrics and what is derived from them (karaoke, the lyrics embedding).

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict, cast

from psycopg import Connection

from core.db.connection import fetch_one


class LyricsText(TypedDict):
    """A song's stored lyrics and their language."""

    text: str | None
    language: str | None


def lyrics_text(db: Connection, track_id: uuid.UUID) -> LyricsText | None:
    """The song's stored lyrics and language."""
    return cast(
        LyricsText | None,
        fetch_one(db, "SELECT text, language FROM lyrics WHERE track_id=%s", (track_id,)),
    )


def save_manual_lyrics(
    db: Connection, track_id: uuid.UUID, text: str, language: str | None
) -> None:
    """Store lyrics the user typed; line timing is dropped when the text changes."""
    db.execute(
        """INSERT INTO lyrics (track_id, source, text, language, provenance, availability_status)
           VALUES (%s,'manual',%s,%s,jsonb_build_object('manual',true,'edited_at',now()::text),'available')
           ON CONFLICT (track_id) DO UPDATE SET source='manual', text=EXCLUDED.text,
             language=EXCLUDED.language, provenance=(CASE WHEN lyrics.text IS DISTINCT FROM EXCLUDED.text THEN
                 coalesce(lyrics.provenance,'{}'::jsonb) - 'lines' - 'synced'
               ELSE coalesce(lyrics.provenance,'{}'::jsonb) END)
               || jsonb_build_object('manual',true,'edited_at',now()::text),
             availability_status='available', created_at=now()""",
        (track_id, text, language),
    )


def set_manual_status(db: Connection, track_id: uuid.UUID, status: str) -> None:
    """Clear the song's lyrics and record the user's status for them (e.g. instrumental)."""
    db.execute(
        """INSERT INTO lyrics (track_id, source, text, provenance, availability_status)
           VALUES (%s,'none',NULL,jsonb_build_object('manual_status',%s::text,'edited_at',now()::text),%s)
           ON CONFLICT (track_id) DO UPDATE SET source='none', text=NULL, language=NULL,
             provenance=(coalesce(lyrics.provenance,'{}'::jsonb)
               - 'manual' - 'lines' - 'synced' - 'forced_transcription' - 'transcription_language')
               || jsonb_build_object('manual_status',EXCLUDED.availability_status,'edited_at',now()::text),
             availability_status=EXCLUDED.availability_status, created_at=now()""",
        (track_id, status, status),
    )


def force_transcription(db: Connection, track_id: uuid.UUID, language: str) -> None:
    """Clear the song's lyrics so the next sync transcribes them in ``language``."""
    db.execute(
        """INSERT INTO lyrics (track_id, source, text, provenance, availability_status)
           VALUES (%s,'none',NULL,jsonb_build_object('transcription_language',%s::text,'forced_transcription',true),'missing')
           ON CONFLICT (track_id) DO UPDATE SET source='none', text=NULL, language=NULL,
             provenance=(coalesce(lyrics.provenance,'{}'::jsonb) - 'manual' - 'lines' - 'synced')
               || jsonb_build_object('transcription_language',%s::text,'forced_transcription',true),
             availability_status='missing', created_at=now()""",
        (track_id, language, language),
    )


def discard_derived(db: Connection, track_id: uuid.UUID) -> None:
    """Drop the song's karaoke variants and lyrics embedding, so the next sync redoes them."""
    db.execute("DELETE FROM karaoke_lyrics_variants WHERE track_id=%s", (track_id,))
    db.execute("DELETE FROM embeddings WHERE track_id=%s AND embedding_type='lyrics'", (track_id,))


def lyrics_facts(db: Connection, track_id: uuid.UUID) -> dict[str, Any] | None:
    """The song's lyrics with ``provenance`` and whether karaoke lines are stored."""
    return fetch_one(
        db,
        """SELECT text, language, provenance, karaoke_lines IS NOT NULL
                  AND jsonb_array_length(karaoke_lines) > 0 AS karaoke_lines
           FROM lyrics WHERE track_id=%s""",
        (track_id,),
    )


def current_themes(
    db: Connection, track_id: uuid.UUID, prompt_revision: str
) -> dict[str, Any] | None:
    """The song's theme description, only when written for its current lyrics and prompt."""
    return fetch_one(
        db,
        """SELECT th.description, th.themes FROM track_lyric_themes th
           JOIN lyrics ly ON ly.track_id=th.track_id
           WHERE th.track_id=%s AND th.status='ready' AND th.prompt_revision=%s
             AND th.source_checksum=encode(sha256(convert_to(ly.text, 'UTF8')), 'hex')""",
        (track_id, prompt_revision),
    )


def user_lyrics(db: Connection, user_id: uuid.UUID, track_id: uuid.UUID) -> dict[str, Any] | None:
    """The lyrics of a song in the user's library with its unbounded karaoke variant.

    ``None`` when the song is not in the library; the lyric columns are ``None`` when it has
    no lyrics row.
    """
    return fetch_one(
        db,
        """SELECT l.text, l.language, l.source, l.provenance, l.availability_status,
                  karaoke.lines AS karaoke_lines, karaoke.model AS karaoke_model,
                  karaoke.model_revision AS karaoke_model_revision,
                  karaoke.bounded AS karaoke_bounded
           FROM user_track_links link
           LEFT JOIN lyrics l ON l.track_id=link.track_id
           LEFT JOIN LATERAL (
             SELECT kv.* FROM karaoke_lyrics_variants kv
             WHERE kv.track_id=link.track_id AND kv.bounded=false LIMIT 1
           ) karaoke ON true
           WHERE link.user_id=%s AND link.track_id=%s LIMIT 1""",
        (user_id, track_id),
    )


def coverage(db: Connection) -> dict[str, int]:
    """How many songs of the instance have lyrics in each status, and a lyrics embedding."""
    return cast(
        dict[str, int],
        fetch_one(
            db,
            """SELECT count(*) AS total,
                      count(*) FILTER (WHERE l.availability_status='available') AS available,
                      count(*) FILTER (WHERE l.availability_status='missing') AS missing,
                      count(*) FILTER (WHERE l.availability_status='unavailable') AS unavailable,
                      count(*) FILTER (WHERE l.availability_status='instrumental') AS instrumental,
                      count(*) FILTER (WHERE EXISTS (
                        SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                        WHERE e.track_id=t.id AND e.embedding_type='lyrics' AND e.window_index IS NULL
                          AND ar.model_name='bge_m3'
                      )) AS embedded
               FROM tracks t LEFT JOIN lyrics l ON l.track_id=t.id""",
        ),
    )
