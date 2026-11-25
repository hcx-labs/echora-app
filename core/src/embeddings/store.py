"""Queries on stored embeddings: vectors and nearest neighbours (pgvector).

Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from psycopg import Connection

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
