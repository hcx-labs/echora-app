"""Queries that plan which of a library's songs need analysis, from what is already stored.

Each function reads only; it takes an open connection and the caller owns the transaction.
Songs are named by their Subsonic source ids (``track_sources.external_id``).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from psycopg import Connection

from core.db.connection import fetch_all, fetch_one


def _restriction(
    external_ids: Iterable[str] | None, library_id: uuid.UUID | None
) -> tuple[str, list[object]]:
    restriction, parameters = "", []
    if external_ids is not None:
        restriction, parameters = " AND ts.external_id=ANY(%s)", [list(external_ids)]
    if library_id is not None:
        restriction += " AND ts.library_id=%s"
        parameters.append(library_id)
    return restriction, parameters


def libraries_at(db: Connection, url: str) -> list[uuid.UUID]:
    """The libraries whose root is the URL (case and trailing slashes ignored)."""
    rows = fetch_all(
        db,
        "SELECT id FROM libraries WHERE lower(rtrim(root_path, '/'))=lower(rtrim(%s, '/'))",
        (url,),
    )
    return [row["id"] for row in rows]


def library_with_namespace(db: Connection, namespace: uuid.UUID) -> uuid.UUID | None:
    """The library of the namespace, if one is stored."""
    row = fetch_one(db, "SELECT id FROM libraries WHERE namespace=%s", (namespace,))
    return row["id"] if row else None


def songs_needing_lyrics(
    db: Connection,
    external_ids: Iterable[str] | None,
    library_id: uuid.UUID | None,
    revision: str,
) -> list[str]:
    """Songs without lyrics text or without a current BGE-M3 lyrics vector of the revision.

    ``None`` for the ids or the library leaves that restriction out.
    """
    restriction, parameters = _restriction(external_ids, library_id)
    rows = fetch_all(
        db,
        f"""SELECT DISTINCT ts.external_id
            FROM track_sources ts LEFT JOIN lyrics l ON l.track_id=ts.track_id
            WHERE ts.source_type='subsonic'{restriction}
              AND (l.track_id IS NULL OR l.text IS NULL OR NOT EXISTS (
                SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                WHERE e.track_id=ts.track_id AND e.embedding_type='lyrics'
                  AND e.window_index IS NULL AND ar.model_name='bge_m3'
                  AND ar.model_revision=%s
              )) ORDER BY ts.external_id""",
        [*parameters, revision],
    )
    return [str(row["external_id"]) for row in rows]


def songs_needing_karaoke(
    db: Connection,
    external_ids: Iterable[str] | None,
    library_id: uuid.UUID | None,
    pipeline_revision: str,
    model_revision: str | None,
) -> list[str]:
    """Songs with lyrics text but no current unbounded karaoke.

    Supplied word timing is current when its digest matches; an alignment when its pipeline
    revision does and, given one, its model revision. ``None`` ids or library is unrestricted.
    """
    restriction, parameters = _restriction(external_ids, library_id)
    rows = fetch_all(
        db,
        f"""SELECT DISTINCT ts.external_id
            FROM track_sources ts JOIN lyrics l ON l.track_id=ts.track_id
            WHERE ts.source_type='subsonic'{restriction}
              AND NULLIF(btrim(l.text), '') IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM karaoke_lyrics_variants kv
                WHERE kv.track_id=l.track_id AND kv.bounded=false
                  AND CASE WHEN l.provenance ? 'word_timing_digest'
                    -- Word timing supplied with the lyrics is stored as is, never aligned.
                    THEN kv.model='navidrome'
                      AND kv.provenance->>'source_digest'=l.provenance->>'word_timing_digest'
                    ELSE kv.model<>'navidrome'
                      AND kv.provenance->>'pipeline_revision'=%s
                      AND (%s::text IS NULL OR kv.model_revision=%s::text)
                  END
              ) ORDER BY ts.external_id""",
        [*parameters, pipeline_revision, model_revision, model_revision],
    )
    return [str(row["external_id"]) for row in rows]


def _requested_sources(
    library_id: uuid.UUID, ids: list[str], resolved_track_ids: dict | None
) -> tuple[str, list[object]]:
    # Execution can pin identities resolved from actual downloaded bytes. Do not
    # let a concurrent source remap change the artifact plan for those bytes.
    if resolved_track_ids is None:
        source_join = """LEFT JOIN track_sources ts ON ts.library_id=%s AND ts.source_type='subsonic'
                        AND ts.external_id=requested.external_id"""
        return source_join, [library_id]
    source_join = """LEFT JOIN unnest(%s::text[],%s::uuid[]) AS ts(external_id,track_id)
                        ON ts.external_id=requested.external_id"""
    return source_join, [ids, [resolved_track_ids[item] for item in ids]]


def audio_artifacts(
    db: Connection,
    library_id: uuid.UUID,
    ids: list[str],
    resolved_track_ids: dict | None,
    revisions: dict[str, str],
) -> list[dict[str, Any]]:
    """Which current audio artifacts each requested song has.

    ``resolved_track_ids`` (source id to song) replaces the library's source mapping.
    ``revisions`` holds ``muq``, ``mert``, ``melody``, ``descriptors``, ``waveform`` and
    ``visual_features``. Rows are ``external_id``, ``track_id`` and ``has_muq``, ``has_mert``,
    ``has_fingerprint``, ``has_melody``, ``has_descriptors``, ``has_waveform`` and
    ``has_visual_features``.
    """
    source_join, source_parameters = _requested_sources(library_id, ids, resolved_track_ids)
    return fetch_all(
        db,
        f"""SELECT requested.external_id,
                  ts.track_id,
                  EXISTS (SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                          WHERE e.track_id=ts.track_id AND e.embedding_type='audio-track'
                            AND e.window_index IS NULL AND ar.model_name='muq_mulan'
                            AND ar.model_revision=%s
                            AND ar.config->>'coverage'='full-track'
                            AND ar.config->>'window_seconds'='10'
                            AND ar.config->>'stride_seconds'='5'
                            AND ar.config->>'store_window_embeddings'='true') AS has_muq,
                  EXISTS (SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                          WHERE e.track_id=ts.track_id AND e.embedding_type='audio-track'
                            AND e.window_index IS NULL AND ar.model_name='mert'
                            AND ar.model_revision=%s
                            AND ar.config->>'coverage'='full-track'
                            AND ar.config->>'window_seconds'='10'
                            AND ar.config->>'stride_seconds'='5'
                            AND ar.config->>'store_window_embeddings'='true') AS has_mert,
                  EXISTS (SELECT 1 FROM track_fingerprints tf WHERE tf.track_id=ts.track_id) AS has_fingerprint,
                  EXISTS (SELECT 1 FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
                          WHERE mc.track_id=ts.track_id AND ar.model_name='melody_contour'
                            AND ar.model_revision=%s) AS has_melody,
                  EXISTS (SELECT 1 FROM track_audio_descriptors ad
                          WHERE ad.track_id=ts.track_id AND ad.revision=%s
                            AND ad.status='complete') AS has_descriptors,
                  EXISTS (SELECT 1 FROM track_waveforms w
                          WHERE w.track_id=ts.track_id AND w.revision=%s) AS has_waveform,
                  EXISTS (SELECT 1 FROM track_visual_features vf
                          WHERE vf.track_id=ts.track_id AND vf.revision=%s
                            AND vf.status IN ('complete', 'unsupported')) AS has_visual_features
           FROM unnest(%s::text[]) requested(external_id)
           {source_join}""",
        (
            revisions["muq"],
            revisions["mert"],
            revisions["melody"],
            revisions["descriptors"],
            revisions["waveform"],
            revisions["visual_features"],
            ids,
            *source_parameters,
        ),
    )


def songs_needing_recording_fingerprints(
    db: Connection,
    library_id: uuid.UUID,
    ids: list[str],
    resolved_track_ids: dict | None,
    representation_id: str,
) -> list[str]:
    """Requested songs without recording fingerprints of the representation (as ``audio_artifacts``)."""
    source_join, source_parameters = _requested_sources(library_id, ids, resolved_track_ids)
    rows = fetch_all(
        db,
        f"""SELECT requested.external_id FROM unnest(%s::text[]) requested(external_id)
            {source_join}
            WHERE NOT EXISTS (SELECT 1 FROM recording_fingerprints f
                WHERE f.track_id=ts.track_id AND f.representation_id=%s)""",
        (ids, *source_parameters, representation_id),
    )
    return [str(row["external_id"]) for row in rows]


def known_sources(db: Connection, library_id: uuid.UUID, ids: list[str]) -> set[str]:
    """The ids that are already Subsonic sources of the library."""
    rows = fetch_all(
        db,
        "SELECT external_id FROM track_sources WHERE library_id=%s AND source_type='subsonic' AND external_id=ANY(%s)",
        (library_id, ids),
    )
    return {row["external_id"] for row in rows}


def songs_needing_backfill(db: Connection, library_id: uuid.UUID, ids: list[str]) -> set[str]:
    """Known songs missing lyrics, a lyrics vector, an audio profile or voice evidence."""
    rows = fetch_all(
        db,
        """SELECT DISTINCT ts.external_id
        FROM track_sources ts LEFT JOIN lyrics l ON l.track_id=ts.track_id
        WHERE ts.library_id=%s AND ts.source_type='subsonic' AND ts.external_id=ANY(%s)
        AND (
            l.track_id IS NULL
            OR (l.provenance->>'manual_status')='missing'
            OR (l.text IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                WHERE e.track_id=ts.track_id AND e.embedding_type='lyrics'
                  AND e.window_index IS NULL AND ar.model_name='bge_m3'))
            OR EXISTS (
                SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                WHERE e.track_id=ts.track_id AND e.embedding_type='audio-track'
                  AND e.window_index IS NULL AND ar.model_name IN ('muq_mulan','mert')
                  AND NOT EXISTS (SELECT 1 FROM current_audio_profiles p
                      WHERE p.track_id=e.track_id AND p.source_run_id=e.run_id
                        AND p.model_name=ar.model_name))
            OR NOT EXISTS (
                SELECT 1 FROM current_embeddings e
                JOIN track_vocal_activity activity ON activity.track_id=e.track_id AND activity.run_id=e.run_id
                WHERE e.track_id=ts.track_id AND e.embedding_type='voice-gender')
        )""",
        (library_id, ids),
    )
    return {row["external_id"] for row in rows}


def songs_needing_transcription(db: Connection, library_id: uuid.UUID, ids: list[str]) -> set[str]:
    """Known songs without lyrics text that are not instrumental."""
    rows = fetch_all(
        db,
        """SELECT DISTINCT ts.external_id FROM track_sources ts
        LEFT JOIN lyrics l ON l.track_id=ts.track_id
        WHERE ts.library_id=%s AND ts.source_type='subsonic' AND ts.external_id=ANY(%s)
          AND NULLIF(btrim(l.text),'') IS NULL
          AND coalesce(l.availability_status,'missing') != 'instrumental'""",
        (library_id, ids),
    )
    return {row["external_id"] for row in rows}


def source_freshness(
    db: Connection, interval_seconds: int, library_id: uuid.UUID, ids: list[str]
) -> list[dict[str, Any]]:
    """Each known source's stored ``source_data`` and whether its audio check is ``stale``."""
    return fetch_all(
        db,
        """SELECT external_id,source_data,
        audio_verified_at IS NULL OR audio_verified_at <= now() - (%s * interval '1 second') AS stale
        FROM track_sources WHERE library_id=%s AND source_type='subsonic'
            AND external_id=ANY(%s)""",
        (interval_seconds, library_id, ids),
    )
