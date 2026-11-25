"""Queries on stored embeddings: vectors and nearest neighbours (pgvector).

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

import numpy as np
from psycopg import Connection
from psycopg.types.json import Jsonb

from core.db.connection import fetch_all, fetch_one


def parse_vector(text: str) -> np.ndarray:
    """A pgvector's text form (``[0.1,0.2]``) as a NumPy array."""
    return np.fromstring(text.strip("[]"), sep=",")


def semantic_fusion_vector(db: Connection, track_id: uuid.UUID) -> str | None:
    """The song's current semantic fusion vector, in pgvector's text form."""
    row = fetch_one(
        db,
        """SELECT e.embedding FROM current_embeddings e
           WHERE e.track_id=%s AND e.embedding_type='semantic_fusion'""",
        (track_id,),
    )
    return row["embedding"] if row else None


def nearest_semantic_fusion(
    db: Connection, vector: str, exclude: uuid.UUID, limit: int
) -> list[dict[str, Any]]:
    """The songs closest to ``vector`` (cosine), with id, title, artist, album and similarity."""
    return fetch_all(
        db,
        """SELECT t.id, t.title, t.artist, t.album,
                  1 - (e.embedding <=> %s::vector) AS similarity
           FROM current_embeddings e
           JOIN tracks t ON t.id=e.track_id
           WHERE e.embedding_type='semantic_fusion' AND e.track_id != %s
           ORDER BY e.embedding <=> %s::vector LIMIT %s""",
        (vector, exclude, vector, limit),
    )


def artist_corpus(db: Connection, model: str) -> list[dict[str, Any]]:
    """Songs with an artist and their newest current audio embedding from ``model``.

    Rows are ``id``, ``title``, ``artist``, ``album``, ``embedding`` (text) and ``run_id``.
    """
    return fetch_all(
        db,
        """SELECT DISTINCT ON (e.track_id) t.id, t.title, t.artist, t.album,
                  e.embedding::text AS embedding, ar.id AS run_id
           FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id JOIN tracks t ON t.id=e.track_id
           WHERE e.embedding_type='audio-track' AND ar.model_name=%s AND t.artist IS NOT NULL
           ORDER BY e.track_id, ar.created_at DESC""",
        (model,),
    )


def map_corpus(
    db: Connection, user_id: uuid.UUID, embedding_type: str, model: str, limit: int
) -> list[dict[str, Any]]:
    """The user's songs with their newest whole-song embedding of one type and model.

    Rows are the song (``id``, ``title``, ``artist``, ``album``, ``duration_seconds``,
    ``genres``, ``source_id``, ``cover_art``) with ``embedding`` (text), ``run_id``,
    ``model_revision`` and ``run_created_at``, by song id.
    """
    return fetch_all(
        db,
        """
        WITH selected_embeddings AS (
          SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding,
                 ar.id AS run_id, ar.model_revision, ar.created_at AS run_created_at
          FROM current_embeddings e JOIN analysis_runs ar ON ar.id = e.run_id
          WHERE e.embedding_type = %s AND ar.model_name = %s AND e.window_index IS NULL
          ORDER BY e.track_id, ar.created_at DESC
        )
        SELECT t.id, t.title, t.artist, t.album, t.duration_seconds, t.genres,
               selected_embeddings.embedding, selected_embeddings.run_id,
               selected_embeddings.model_revision, selected_embeddings.run_created_at,
               max(ts.external_id) AS source_id, max(ts.source_data->>'coverArt') AS cover_art
        FROM selected_embeddings JOIN tracks t ON t.id = selected_embeddings.track_id
        LEFT JOIN track_sources ts ON ts.track_id = t.id
          AND EXISTS (SELECT 1 FROM user_track_links source_links
                      WHERE source_links.user_id=%s AND source_links.library_id=ts.library_id
                        AND source_links.track_id=ts.track_id AND source_links.external_id=ts.external_id)
        WHERE EXISTS (SELECT 1 FROM user_track_links visible_links
                      WHERE visible_links.track_id=t.id AND visible_links.user_id=%s)
        GROUP BY t.id, selected_embeddings.embedding, selected_embeddings.run_id,
                 selected_embeddings.model_revision, selected_embeddings.run_created_at
        ORDER BY t.id LIMIT %s
        """,
        (embedding_type, model, user_id, user_id, limit),
    )


def blend_map_corpus(db: Connection, user_id: uuid.UUID, limit: int) -> list[dict[str, Any]]:
    """The user's songs that have both a MuQ-MuLan and a MERT audio embedding.

    Rows are as in ``map_corpus``, with ``semantic_embedding`` and ``acoustic_embedding`` in
    place of ``embedding``, and run ids and revisions joined with ``:``.
    """
    return fetch_all(
        db,
        """
        WITH semantic AS (
          SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding,
                 ar.id AS run_id, ar.model_revision, ar.created_at
          FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
          WHERE e.embedding_type='audio-track' AND ar.model_name='muq_mulan'
          ORDER BY e.track_id, ar.created_at DESC
        ), acoustic AS (
          SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding,
                 ar.id AS run_id, ar.model_revision, ar.created_at
          FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
          WHERE e.embedding_type='audio-track' AND ar.model_name='mert'
          ORDER BY e.track_id, ar.created_at DESC
        )
        SELECT t.id, t.title, t.artist, t.album, t.duration_seconds, t.genres,
               semantic.embedding AS semantic_embedding, acoustic.embedding AS acoustic_embedding,
               semantic.run_id::text || ':' || acoustic.run_id::text AS run_id,
               semantic.model_revision || ':' || acoustic.model_revision AS model_revision,
               greatest(semantic.created_at, acoustic.created_at) AS run_created_at,
               max(ts.external_id) AS source_id, max(ts.source_data->>'coverArt') AS cover_art
        FROM semantic JOIN acoustic USING (track_id) JOIN tracks t ON t.id=semantic.track_id
        LEFT JOIN track_sources ts ON ts.track_id=t.id
          AND EXISTS (SELECT 1 FROM user_track_links source_links
                      WHERE source_links.user_id=%s AND source_links.library_id=ts.library_id
                        AND source_links.track_id=ts.track_id AND source_links.external_id=ts.external_id)
        WHERE EXISTS (SELECT 1 FROM user_track_links visible_links
                      WHERE visible_links.track_id=t.id AND visible_links.user_id=%s)
        GROUP BY t.id, semantic.embedding, acoustic.embedding, semantic.run_id, acoustic.run_id,
                 semantic.model_revision, acoustic.model_revision, semantic.created_at, acoustic.created_at
        ORDER BY t.id LIMIT %s
        """,
        (user_id, user_id, limit),
    )


def fusion_counts(db: Connection) -> dict[str, int]:
    """How many current semantic fusion, whole-song lyrics and audio embeddings exist."""
    return cast(
        dict[str, int],
        fetch_one(
            db,
            """SELECT count(*) AS fused,
                      (SELECT count(*) FROM current_embeddings l WHERE l.embedding_type='lyrics'
                        AND l.window_index IS NULL) AS lyrics,
                      (SELECT count(*) FROM current_embeddings a WHERE a.embedding_type='audio-track'
                        AND a.window_index IS NULL) AS audio
               FROM current_embeddings e WHERE e.embedding_type='semantic_fusion'""",
        ),
    )


def audio_profile_coverage(
    db: Connection, user_id: uuid.UUID, model: str, revision: str
) -> tuple[int, int]:
    """The user's songs with a ``model`` audio embedding, and how many have a current profile.

    A profile counts when it is of ``revision`` and built from the song's newest embedding.
    """
    row = fetch_one(
        db,
        """WITH latest_source AS (
             SELECT DISTINCT ON (e.track_id) e.track_id, e.run_id
             FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
             WHERE e.embedding_type='audio-track' AND e.window_index IS NULL
               AND ar.model_name=%s
               AND EXISTS (SELECT 1 FROM user_track_links utl
                           WHERE utl.user_id=%s AND utl.track_id=e.track_id)
             ORDER BY e.track_id, ar.created_at DESC
           )
           SELECT count(*) AS total,
                  count(*) FILTER (WHERE EXISTS (
                    SELECT 1 FROM current_audio_profiles tap
                    JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
                    WHERE tap.track_id=latest_source.track_id
                      AND tap.source_run_id=latest_source.run_id
                      AND tap.model_name=%s
                      AND profile_run.kind='audio_profile'
                      AND profile_run.model_revision=%s
                  )) AS current
           FROM latest_source""",
        (model, user_id, model, revision),
    )
    return (int(row["total"]), int(row["current"])) if row else (0, 0)


def songs_with_audio_embeddings(
    db: Connection, user_id: uuid.UUID, models: list[str]
) -> list[uuid.UUID]:
    """The user's songs that have a whole-song audio embedding from any of ``models``."""
    rows = fetch_all(
        db,
        """SELECT DISTINCT e.track_id, ar.model_name
           FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
           WHERE e.embedding_type='audio-track' AND e.window_index IS NULL
             AND ar.model_name=ANY(%s)
             AND EXISTS (SELECT 1 FROM user_track_links utl
                         WHERE utl.user_id=%s AND utl.track_id=e.track_id)
           ORDER BY e.track_id, ar.model_name""",
        (models, user_id),
    )
    return sorted({row["track_id"] for row in rows})


def journey_corpus(db: Connection) -> list[dict[str, Any]]:
    """Every song with both a MuQ-MuLan and a MERT audio embedding, for journeys.

    Rows are the song (``id``, ``title``, ``artist``, ``album``, ``duration_seconds``,
    ``source_id``, ``cover_art``, ``recording_group_id``) with ``semantic_embedding`` and
    ``acoustic_embedding`` in text form, by song id. It is not limited to one user's library.
    """
    return fetch_all(
        db,
        """
        WITH semantic AS (
          SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding
          FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
          WHERE e.embedding_type='audio-track' AND ar.model_name='muq_mulan'
          ORDER BY e.track_id, ar.created_at DESC
        ), acoustic AS (
          SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding
          FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
          WHERE e.embedding_type='audio-track' AND ar.model_name='mert'
          ORDER BY e.track_id, ar.created_at DESC
        )
        SELECT t.id, t.title, t.artist, t.album, t.duration_seconds,
               semantic.embedding AS semantic_embedding, acoustic.embedding AS acoustic_embedding,
               max(ts.external_id) AS source_id, max(ts.source_data->>'coverArt') AS cover_art,
               max(member.group_id::text) AS recording_group_id
        FROM tracks t JOIN semantic ON semantic.track_id=t.id JOIN acoustic ON acoustic.track_id=t.id
        LEFT JOIN track_sources ts ON ts.track_id=t.id
        LEFT JOIN recording_group_members member ON member.track_id=t.id
        GROUP BY t.id, semantic.embedding, acoustic.embedding ORDER BY t.id
        """,
    )


# The current whole-song embeddings each per-user corpus is built from.
_CORPUS_FILTERS = {
    "semantic": "e.embedding_type='audio-track' AND ar.model_name='muq_mulan'",
    "lyrics": "e.embedding_type='lyrics' AND e.window_index IS NULL AND ar.model_name='bge_m3'",
}


def corpus_digest(db: Connection, kind: str, user_id: uuid.UUID) -> str:
    """A digest of the user's current (song, run) pairs for the corpus kind.

    Kinds are ``semantic`` (MuQ-MuLan audio) and ``lyrics`` (BGE-M3).
    """
    row = fetch_one(
        db,
        f"""SELECT coalesce(md5(string_agg(e.track_id::text || ':' || e.run_id::text, ','
                                        ORDER BY e.track_id, e.run_id)), '') AS digest
            FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
            WHERE {_CORPUS_FILTERS[kind]}
              AND EXISTS (SELECT 1 FROM user_track_links utl
                          WHERE utl.track_id=e.track_id AND utl.user_id=%s)""",
        (user_id,),
    )
    return str(row["digest"]) if row else ""


def user_corpus(db: Connection, kind: str, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """The user's songs with their newest embedding of the corpus kind.

    Rows are ``id``, ``title``, ``artist``, ``album``, ``embedding`` (text) and ``run_id``.
    """
    return fetch_all(
        db,
        f"""SELECT DISTINCT ON (e.track_id) t.id, t.title, t.artist, t.album,
                   e.embedding::text AS embedding, ar.id AS run_id
            FROM current_embeddings e
            JOIN analysis_runs ar ON ar.id=e.run_id
            JOIN tracks t ON t.id=e.track_id
            WHERE {_CORPUS_FILTERS[kind]}
              AND EXISTS (SELECT 1 FROM user_track_links utl
                          WHERE utl.track_id=t.id AND utl.user_id=%s)
            ORDER BY e.track_id, ar.created_at DESC""",
        (user_id,),
    )


# Audio embeddings written by library ingest.


def audio_embedding_run(
    db: Connection,
    model_name: str,
    model_revision: str,
    config_hash: str,
    config: dict[str, Any],
    environment: dict[str, Any],
    device: str,
) -> uuid.UUID:
    """The audio embedding run of this model configuration, started now when new."""
    row = fetch_one(
        db,
        """
        INSERT INTO analysis_runs
          (kind, model_name, model_revision, config_hash, config, environment,
           device, precision, status, started_at)
        VALUES ('audio_embedding', %s, %s, %s, %s, %s, %s, 'float32', 'running', now())
        ON CONFLICT (kind, model_name, model_revision, config_hash)
        DO UPDATE SET id = analysis_runs.id
        RETURNING id
        """,
        (model_name, model_revision, config_hash, Jsonb(config), Jsonb(environment), device),
    )
    return cast(dict, row)["id"]


def has_audio_track_embedding(db: Connection, track_id: uuid.UUID, run_id: uuid.UUID) -> bool:
    """Whether the run embedded the whole song."""
    return (
        fetch_one(
            db,
            "SELECT 1 FROM embeddings WHERE track_id=%s AND run_id=%s AND embedding_type='audio-track'",
            (track_id, run_id),
        )
        is not None
    )


def save_audio_embeddings(
    db: Connection,
    track_id: uuid.UUID,
    run_id: uuid.UUID,
    track: dict[str, Any],
    windows: list[tuple[Any, ...]],
) -> None:
    """Store the song's whole-song embedding and its window embeddings (kept when present).

    ``track`` has ``dimension``, ``embedding`` (a pgvector literal), ``inference_ms`` and
    ``peak_vram_bytes``; each window is ``(window_index, start, end, dimension, embedding)``.
    """
    with db.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO embeddings
              (track_id, run_id, embedding_type, dimension, aggregation, embedding,
               inference_ms, peak_vram_bytes)
            VALUES (%s, %s, 'audio-track', %s, 'normalized_mean', %s::vector, %s, %s)
            ON CONFLICT (track_id, run_id, embedding_type, window_index) DO NOTHING
            """,
            (
                track_id,
                run_id,
                track["dimension"],
                track["embedding"],
                track["inference_ms"],
                track["peak_vram_bytes"],
            ),
        )
        cursor.executemany(
            """
            INSERT INTO embeddings
              (track_id, run_id, embedding_type, window_index, window_start_seconds,
               window_end_seconds, dimension, aggregation, embedding)
            VALUES (%s, %s, 'audio-window', %s, %s, %s, %s, 'none', %s::vector)
            ON CONFLICT (track_id, run_id, embedding_type, window_index) DO NOTHING
            """,
            [(track_id, run_id, *window) for window in windows],
        )
