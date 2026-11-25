"""Queries on a song's lyrics and what is derived from them (karaoke, the lyrics embedding).

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

from core.db.connection import fetch_all, fetch_one


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


def synced_without_karaoke(db: Connection, pipeline_revision: str) -> int:
    """Songs with synced lyrics and no unbounded karaoke from ``pipeline_revision`` yet."""
    row = fetch_one(
        db,
        """SELECT count(*) AS pending FROM lyrics l
           WHERE l.text IS NOT NULL
             AND coalesce((l.provenance->>'synced')::boolean,false)
             AND NOT EXISTS (SELECT 1 FROM karaoke_lyrics_variants kv
                             WHERE kv.track_id=l.track_id AND kv.bounded=false
                               AND kv.provenance->>'pipeline_revision'=%s)""",
        (pipeline_revision,),
    )
    return int(row["pending"]) if row else 0


def ready_translations(db: Connection, track_id: uuid.UUID) -> list[dict[str, Any]]:
    """The song's ready translations by target language (every stored column)."""
    return fetch_all(
        db,
        """SELECT * FROM lyric_translations
        WHERE track_id=%s AND status='ready' ORDER BY target_language,provenance""",
        (track_id,),
    )


# Written by the lyrics stage of a sync (worker.lyrics.pipeline).


def lyrics_embedding_run(
    db: Connection,
    model_name: str,
    model_revision: str,
    config_hash: str,
    config: dict[str, Any],
    environment: dict[str, Any],
    device: str,
) -> uuid.UUID:
    """The lyrics embedding run of this model configuration, started now when new."""
    row = fetch_one(
        db,
        """INSERT INTO analysis_runs
             (kind, model_name, model_revision, config_hash, config, environment, device, precision, status, started_at)
           VALUES ('lyrics_embedding',%s,%s,%s,%s,%s,%s,'float32','running',now())
           ON CONFLICT (kind, model_name, model_revision, config_hash)
           DO UPDATE SET id=analysis_runs.id RETURNING id""",
        (model_name, model_revision, config_hash, Jsonb(config), Jsonb(environment), device),
    )
    return cast(dict, row)["id"]


def voice_activity(db: Connection, track_id: uuid.UUID) -> Any:
    """The song's newest vocal activity from a current voice embedding run, or ``None``."""
    row = fetch_one(
        db,
        """SELECT a.activity FROM track_vocal_activity a
           JOIN current_embeddings e ON e.track_id=a.track_id AND e.run_id=a.run_id
           WHERE a.track_id=%s AND e.embedding_type='voice-gender'
           ORDER BY a.created_at DESC LIMIT 1""",
        (track_id,),
    )
    return row["activity"] if row else None


def save_lyrics(
    db: Connection,
    track_id: uuid.UUID,
    source: str,
    text: str | None,
    language: str | None,
    provenance: dict[str, Any],
    status: str,
) -> uuid.UUID | None:
    """Store retrieved or transcribed lyrics; returns their id, or ``None`` when kept.

    A transcript or empty result never replaces stored non-empty lyrics.
    """
    row = fetch_one(
        db,
        """INSERT INTO lyrics (track_id, source, text, language, provenance, availability_status)
           VALUES (%s,%s,%s,%s,%s,%s)
           ON CONFLICT (track_id) DO UPDATE SET source=EXCLUDED.source, text=EXCLUDED.text,
             language=EXCLUDED.language, provenance=EXCLUDED.provenance,
             availability_status=EXCLUDED.availability_status, created_at=now()
           WHERE (EXCLUDED.source != 'transcribed' AND EXCLUDED.text IS NOT NULL)
              OR NULLIF(btrim(lyrics.text), '') IS NULL
           RETURNING id""",
        (track_id, source, text, language, Jsonb(provenance), status),
    )
    return row["id"] if row else None


def replace_lyrics_embeddings(
    db: Connection,
    track_id: uuid.UUID,
    run_id: uuid.UUID,
    whole: dict[str, Any],
    windows: list[tuple[Any, ...]],
) -> None:
    """Replace the run's lyrics embeddings of the song.

    ``whole`` has ``dimension``, ``embedding`` (a pgvector literal), ``inference_ms`` and
    ``peak_vram_bytes``; each window is ``(window_index, dimension, aggregation, embedding)``.
    """
    db.execute(
        "DELETE FROM embeddings WHERE track_id=%s AND run_id=%s AND embedding_type='lyrics'",
        (track_id, run_id),
    )
    db.execute(
        """INSERT INTO embeddings
             (track_id, run_id, embedding_type, dimension, aggregation, embedding, inference_ms, peak_vram_bytes)
           VALUES (%s,%s,'lyrics',%s,'normalized_mean',%s::vector,%s,%s)""",
        (
            track_id,
            run_id,
            whole["dimension"],
            whole["embedding"],
            whole["inference_ms"],
            whole["peak_vram_bytes"],
        ),
    )
    for window in windows:
        db.execute(
            """INSERT INTO embeddings
                 (track_id, run_id, embedding_type, window_index, dimension, aggregation, embedding)
               VALUES (%s,%s,'lyrics',%s,%s,%s,%s::vector)""",
            (track_id, run_id, *window),
        )


def source_titles(
    db: Connection, external_ids: list[str], library_id: uuid.UUID
) -> list[dict[str, Any]]:
    """One source per song among the library's sources: ``track_id``, ``external_id``, ``title``."""
    return fetch_all(
        db,
        """SELECT DISTINCT ON (ts.track_id) ts.track_id, ts.external_id, t.title
           FROM track_sources ts JOIN tracks t ON t.id=ts.track_id
           WHERE ts.source_type='subsonic' AND ts.external_id=ANY(%s) AND ts.library_id=%s
           ORDER BY ts.track_id, ts.id""",
        (external_ids, library_id),
    )


def stored_lyrics(db: Connection, track_id: uuid.UUID) -> dict[str, Any] | None:
    """The song's lyrics row: ``text``, ``provenance``, ``availability_status``, ``source``."""
    return fetch_one(
        db,
        "SELECT text, provenance, availability_status, source FROM lyrics WHERE track_id=%s",
        (track_id,),
    )


def kept_lyrics(db: Connection, track_id: uuid.UUID) -> dict[str, Any] | None:
    """The song's lyrics if they have text or are instrumental.

    Keys are ``text``, ``provenance`` and ``availability_status``.
    """
    return fetch_one(
        db,
        "SELECT text, provenance, availability_status FROM lyrics WHERE track_id=%s AND (NULLIF(btrim(text),'') IS NOT NULL OR availability_status='instrumental')",
        (track_id,),
    )


def track_artist(db: Connection, track_id: uuid.UUID) -> str | None:
    """The song's artist tag."""
    row = fetch_one(db, "SELECT artist FROM tracks WHERE id=%s", (track_id,))
    return row["artist"] if row else None


def refine_transcript(
    db: Connection,
    track_id: uuid.UUID,
    draft: str,
    text: str,
    language: str | None,
    provenance: dict[str, Any],
) -> int:
    """Replace the song's transcript if it is still ``draft``; how many rows changed."""
    with db.cursor() as cursor:
        cursor.execute(
            """UPDATE lyrics SET text=%s, language=%s, provenance=%s
               WHERE track_id=%s AND source='transcribed' AND text=%s""",
            (text, language, Jsonb(provenance), track_id, draft),
        )
        return cursor.rowcount


def batch_lyrics(
    db: Connection, user_id: uuid.UUID, namespace: uuid.UUID, external_ids: list[str]
) -> list[dict[str, Any]]:
    """The user's selected songs of the library that have lyrics, for translation.

    Rows are ``track_id``, ``text``, ``language`` and ``provenance``.
    """
    return fetch_all(
        db,
        """SELECT DISTINCT l.track_id, l.text, l.language, l.provenance
        FROM lyrics l JOIN user_track_links u ON u.track_id=l.track_id
        JOIN libraries lib ON lib.id=u.library_id
        WHERE u.user_id=%s AND lib.namespace=%s AND u.external_id=ANY(%s)
          AND NULLIF(btrim(l.text),'') IS NOT NULL""",
        (user_id, namespace, external_ids),
    )


def lyrics_source(db: Connection, track_id: uuid.UUID) -> dict[str, Any] | None:
    """The song's lyrics ``text`` and ``provenance``."""
    return fetch_one(db, "SELECT text, provenance FROM lyrics WHERE track_id=%s", (track_id,))


def songs_needing_themes(
    db: Connection,
    user_id: uuid.UUID,
    namespace: uuid.UUID,
    external_ids: list[str],
    prompt_revision: int,
    retry_failed_after_days: int,
) -> list[dict[str, Any]]:
    """The user's selected songs whose lyrics lack a current theme description.

    A description is current for the lyrics' checksum and the prompt revision; a failed one is
    retried after ``retry_failed_after_days``. Rows are ``track_id``, ``text`` and ``title``.
    """
    return fetch_all(
        db,
        """SELECT DISTINCT l.track_id, l.text, t.title
           FROM lyrics l JOIN tracks t ON t.id=l.track_id
           JOIN user_track_links u ON u.track_id=l.track_id
           JOIN libraries lib ON lib.id=u.library_id
           LEFT JOIN track_lyric_themes th ON th.track_id=l.track_id
           WHERE u.user_id=%s AND lib.namespace=%s AND u.external_id=ANY(%s)
             AND NULLIF(btrim(l.text),'') IS NOT NULL
             AND (th.track_id IS NULL
                  OR th.source_checksum <> encode(sha256(convert_to(l.text,'UTF8')),'hex')
                  OR th.prompt_revision <> %s
                  OR (th.status='failed'
                      AND th.updated_at < now() - make_interval(days => %s)))""",
        (user_id, namespace, external_ids, prompt_revision, retry_failed_after_days),
    )


def save_themes(
    db: Connection,
    track_id: uuid.UUID,
    checksum: str,
    prompt_revision: int,
    model: str,
    *,
    status: str,
    error_code: str | None,
    description: str | None,
    themes: list[str],
    embedding: str | None,
    embedding_model: str | None,
) -> None:
    """Store the song's theme description (or its failure) for the lyrics' checksum."""
    db.execute(
        """INSERT INTO track_lyric_themes (track_id, source_checksum, prompt_revision, model,
             status, error_code, description, themes, embedding, embedding_model, updated_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s,now())
           ON CONFLICT (track_id) DO UPDATE SET source_checksum=EXCLUDED.source_checksum,
             prompt_revision=EXCLUDED.prompt_revision, model=EXCLUDED.model,
             status=EXCLUDED.status, error_code=EXCLUDED.error_code,
             description=EXCLUDED.description, themes=EXCLUDED.themes,
             embedding=EXCLUDED.embedding, embedding_model=EXCLUDED.embedding_model,
             updated_at=now()""",
        (
            track_id,
            checksum,
            prompt_revision,
            model,
            status,
            error_code,
            description,
            Jsonb(themes),
            embedding,
            embedding_model,
        ),
    )


# Karaoke timing written by the karaoke stage (worker.kara.pipeline).


def karaoke_sources(
    db: Connection, external_ids: list[str], library_id: uuid.UUID
) -> list[dict[str, Any]]:
    """One source per song among the library's sources, with the song's lyrics.

    Rows are ``track_id``, ``external_id``, ``title``, ``text``, ``language``, ``lines`` (the
    provider's lines), ``word_timing`` and ``word_timing_digest`` (supplied word timing).
    """
    return fetch_all(
        db,
        """SELECT DISTINCT ON (ts.track_id) ts.track_id, ts.external_id, t.title, l.text, l.language,
                   l.provenance->'lines' AS lines, l.provenance->'word_timing' AS word_timing,
                   l.provenance->>'word_timing_digest' AS word_timing_digest
            FROM track_sources ts JOIN tracks t ON t.id=ts.track_id JOIN lyrics l ON l.track_id=ts.track_id
            WHERE ts.source_type='subsonic' AND ts.external_id=ANY(%s) AND ts.library_id=%s
            ORDER BY ts.track_id, ts.id""",
        (external_ids, library_id),
    )


def save_supplied_karaoke(
    db: Connection, track_id: uuid.UUID, lines: list[Any], provenance: dict[str, Any]
) -> None:
    """Store word timing that came with the lyrics as the song's unbounded karaoke."""
    db.execute(
        """INSERT INTO karaoke_lyrics_variants
             (track_id, bounded, lines, ass, lrc, model, model_revision, provenance,
              alignment_document, diagnostics, created_at)
           VALUES (%s,false,%s,NULL,NULL,'navidrome','supplied',%s,NULL,'{}',now())
           ON CONFLICT (track_id, bounded) DO UPDATE SET lines=EXCLUDED.lines,
             ass=NULL, lrc=NULL, model=EXCLUDED.model, model_revision=EXCLUDED.model_revision,
             provenance=EXCLUDED.provenance, alignment_document=NULL, diagnostics='{}',
             created_at=now()""",
        (track_id, Jsonb(lines), Jsonb(provenance)),
    )
    db.execute(
        """UPDATE lyrics SET karaoke_lines=%s, karaoke_ass=NULL, karaoke_lrc=NULL,
                  karaoke_model='navidrome', karaoke_model_revision='supplied',
                  karaoke_bounded=false, karaoke_created_at=now(),
                  provenance=provenance || %s
           WHERE track_id=%s""",
        (Jsonb(lines), Jsonb({"karaoke": provenance}), track_id),
    )


def save_aligned_karaoke(
    db: Connection,
    track_id: uuid.UUID,
    lines: list[Any],
    model_revision: str,
    provenance: dict[str, Any],
    alignment_document: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    """Store a FA-Kara alignment as the song's unbounded karaoke."""
    db.execute(
        """INSERT INTO karaoke_lyrics_variants
             (track_id, bounded, lines, ass, lrc, model, model_revision, provenance,
              alignment_document, diagnostics, created_at)
           VALUES (%s,%s,%s,NULL,NULL,'fa_kara',%s,%s,%s,%s,now())
           ON CONFLICT (track_id, bounded) DO UPDATE SET lines=EXCLUDED.lines,
             ass=EXCLUDED.ass, lrc=EXCLUDED.lrc, model=EXCLUDED.model,
             model_revision=EXCLUDED.model_revision, provenance=EXCLUDED.provenance,
             alignment_document=EXCLUDED.alignment_document,
             diagnostics=EXCLUDED.diagnostics, created_at=now()""",
        (
            track_id,
            False,
            Jsonb(lines),
            model_revision,
            Jsonb(provenance),
            Jsonb(alignment_document),
            Jsonb(diagnostics),
        ),
    )
    db.execute(
        """UPDATE lyrics SET karaoke_lines=%s, karaoke_ass=NULL, karaoke_lrc=NULL,
                  karaoke_model='fa_kara', karaoke_model_revision=%s, karaoke_bounded=%s,
                  karaoke_created_at=now(),
                  provenance=provenance || %s
           WHERE track_id=%s""",
        (Jsonb(lines), model_revision, False, Jsonb({"karaoke": provenance}), track_id),
    )
