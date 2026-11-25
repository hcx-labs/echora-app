"""Queries on audio profiles: a song's whole-song summaries, temporal segments and modes.

Profiles (``track_audio_profiles``) are derived from a source run's window embeddings by a
profile run; ``audio_temporal_segments``, ``audio_modes`` and ``audio_mode_intervals`` belong
to one profile. Every function takes an open connection; the caller owns the transaction.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from core.db.connection import fetch_all, fetch_one


def profile_run(
    db: Connection,
    model_name: str,
    revision: str,
    config_hash: str,
    config: dict[str, Any],
    environment: dict[str, Any],
) -> uuid.UUID:
    """The audio profile run of these parameters, started now when new."""
    row = fetch_one(
        db,
        """INSERT INTO analysis_runs
             (kind, model_name, model_revision, config_hash, config, environment,
              device, precision, status, started_at)
           VALUES ('audio_profile',%s,%s,%s,%s,%s,'cpu','float32','running',now())
           ON CONFLICT (kind, model_name, model_revision, config_hash)
           DO UPDATE SET id=analysis_runs.id
           RETURNING id""",
        (model_name, revision, config_hash, Jsonb(config), Jsonb(environment)),
    )
    return cast(dict, row)["id"]


def songs_without_profile(
    db: Connection, model_name: str, run_id: uuid.UUID, track_ids: list
) -> list[dict[str, Any]]:
    """Songs whose newest whole-song embedding of the model has no profile from this run.

    Rows are ``track_id``, ``source_run_id``, ``duration_seconds`` and the source run's
    ``window_seconds`` and ``stride_seconds`` (text, possibly ``None``).
    """
    return fetch_all(
        db,
        """SELECT DISTINCT ON (e.track_id)
                  e.track_id, e.run_id AS source_run_id, t.duration_seconds,
                  ar.config->>'window_seconds' AS window_seconds,
                  ar.config->>'stride_seconds' AS stride_seconds
            FROM current_embeddings e
            JOIN analysis_runs ar ON ar.id=e.run_id
            JOIN tracks t ON t.id=e.track_id
            WHERE e.embedding_type='audio-track' AND e.window_index IS NULL
              AND ar.model_name=%s
              AND NOT EXISTS (
                SELECT 1 FROM track_audio_profiles tap
                JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
                WHERE tap.track_id=e.track_id
                  AND tap.source_run_id=e.run_id
                  AND tap.model_name=%s
                  AND profile_run.kind='audio_profile'
                  AND profile_run.id=%s
              ) AND e.track_id=ANY(%s)
            ORDER BY e.track_id, ar.created_at DESC""",
        [model_name, model_name, run_id, track_ids],
    )


def window_embeddings(
    db: Connection, track_id: uuid.UUID, run_id: uuid.UUID
) -> list[dict[str, Any]]:
    """The run's window embeddings of the song, in order.

    Rows are ``window_index``, ``window_start_seconds``, ``window_end_seconds`` and
    ``embedding`` (text).
    """
    return fetch_all(
        db,
        """SELECT window_index, window_start_seconds, window_end_seconds,
                  embedding::text AS embedding
           FROM embeddings
           WHERE track_id=%s AND run_id=%s AND embedding_type='audio-window'
           ORDER BY window_index""",
        (track_id, run_id),
    )


def save_profile(
    db: Connection,
    profile: tuple[Any, ...],
    segments: list[tuple[Any, ...]],
    modes: list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]],
) -> None:
    """Store a song's profile and replace its segments and modes.

    ``profile`` is the ``track_audio_profiles`` row: ``(track_id, profile_run_id,
    source_run_id, model_name, dimension, global_overlap_embedding,
    global_decorrelated_embedding, resultant_length, mean_global_similarity,
    p05_global_similarity, adjacent_change_mean, adjacent_change_p95, timestamps_exact,
    mode_count)`` with vectors as pgvector literals. Each segment is ``(segment_index,
    start_seconds, end_seconds, dimension, embedding, cohesion, representative_window_index)``;
    each mode is ``((mode_index, dimension, embedding, duration_weight, cohesion,
    representative_window_index), [(start_seconds, end_seconds), ...])``.
    """
    with db.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """INSERT INTO track_audio_profiles
                 (track_id, profile_run_id, source_run_id, model_name, dimension,
                  global_overlap_embedding, global_decorrelated_embedding,
                  resultant_length, mean_global_similarity, p05_global_similarity,
                  adjacent_change_mean, adjacent_change_p95, timestamps_exact, mode_count)
               VALUES (%s,%s,%s,%s,%s,%s::vector,%s::vector,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (track_id, profile_run_id, source_run_id) DO UPDATE SET
                 dimension=EXCLUDED.dimension,
                 model_name=EXCLUDED.model_name,
                 global_overlap_embedding=EXCLUDED.global_overlap_embedding,
                 global_decorrelated_embedding=EXCLUDED.global_decorrelated_embedding,
                 resultant_length=EXCLUDED.resultant_length,
                 mean_global_similarity=EXCLUDED.mean_global_similarity,
                 p05_global_similarity=EXCLUDED.p05_global_similarity,
                 adjacent_change_mean=EXCLUDED.adjacent_change_mean,
                 adjacent_change_p95=EXCLUDED.adjacent_change_p95,
                 timestamps_exact=EXCLUDED.timestamps_exact,
                 mode_count=EXCLUDED.mode_count,
                 created_at=now()
               RETURNING id""",
            profile,
        )
        profile_id = cursor.fetchone()["id"]
        cursor.execute(
            "DELETE FROM audio_temporal_segments WHERE profile_id=%s",
            (profile_id,),
        )
        cursor.execute(
            "DELETE FROM audio_modes WHERE profile_id=%s",
            (profile_id,),
        )
        cursor.executemany(
            """INSERT INTO audio_temporal_segments
                 (profile_id, segment_index, start_seconds, end_seconds,
                  dimension, embedding, cohesion, representative_window_index)
               VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s)""",
            [(profile_id, *segment) for segment in segments],
        )
        for mode, intervals in modes:
            cursor.execute(
                """INSERT INTO audio_modes
                     (profile_id, mode_index, dimension, embedding,
                      duration_weight, cohesion, representative_window_index)
                   VALUES (%s,%s,%s,%s::vector,%s,%s,%s) RETURNING id""",
                (profile_id, *mode),
            )
            mode_id = cursor.fetchone()["id"]
            cursor.executemany(
                """INSERT INTO audio_mode_intervals (mode_id, interval_index, start_seconds, end_seconds)
                   VALUES (%s,%s,%s,%s)""",
                [(mode_id, index, *interval) for index, interval in enumerate(intervals)],
            )
