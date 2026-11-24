"""Select songs needing work before allocating analysis batches.

Workers recheck artifacts at execution time. This selection never loads models.
Known missing/instrumental/unavailable lyrics are terminal for ordinary sync;
explicit lyrics backfills remain the way to retry retrieval for those songs.
"""

from __future__ import annotations

from core.settings import get_settings

import uuid

from worker.artwork.settings import sync_selection
from worker.planning.audio import plan_audio, plan_karaoke
from worker.planning.freshness import sources_needing_refresh


def select_sync_tracks(
    connection,
    url: str,
    external_ids: list[str],
    mode: str = "all",
    *,
    catalog=(),
    verify_audio_hashes=True,
) -> list[str]:
    if mode not in {"all", "missing"}:
        raise ValueError("Unknown sync mode")
    ids = list(dict.fromkeys(external_ids))
    if not ids:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id FROM libraries WHERE namespace=%s",
            (uuid.uuid5(uuid.NAMESPACE_URL, url.rstrip("/")),),
        )
        library = cursor.fetchone()
        if library is None:
            return ids
        library_id = library[0]
        cursor.execute(
            "SELECT external_id FROM track_sources WHERE library_id=%s AND source_type='subsonic' AND external_id=ANY(%s)",
            (library_id, ids),
        )
        known = {row[0] for row in cursor.fetchall()}
    selected = set(ids) - known
    if mode == "missing":
        return [item for item in ids if item in selected]

    # Entire-library refresh must visit known tracks for metadata, lyrics, and translations.
    # Identity verification is independently controlled by the ingest flag.
    if not verify_audio_hashes:
        return ids
    from core.lyrics.translations import load_settings

    translation_settings, _ = load_settings()
    if translation_settings.enabled and translation_settings.language_pairs:
        return ids

    selected.update(sources_needing_refresh(connection, library_id, ids, catalog))
    selected.update(plan_audio(connection, library_id, ids).download_external_ids)
    with connection.cursor() as cursor:
        cursor.execute(
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
        selected.update(row[0] for row in cursor.fetchall())

    from worker.transcription.config import transcription_model, transcription_enabled

    if transcription_model() and transcription_enabled(connection):
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT DISTINCT ts.external_id FROM track_sources ts
                LEFT JOIN lyrics l ON l.track_id=ts.track_id
                WHERE ts.library_id=%s AND ts.source_type='subsonic' AND ts.external_id=ANY(%s)
                  AND NULLIF(btrim(l.text),'') IS NULL
                  AND coalesce(l.availability_status,'missing') != 'instrumental'""",
                (library_id, ids),
            )
            selected.update(row[0] for row in cursor.fetchall())

    # Share the exact karaoke compatibility contract with the executing pipeline.
    from worker.kara.pipeline import (
        KARAOKE_PIPELINE_REVISION,
        DEFAULT_MODEL_REVISION,
        _stored_model_revision,
        current_model_revision,
    )

    karaoke = plan_karaoke(
        connection,
        KARAOKE_PIPELINE_REVISION,
        ids,
        _stored_model_revision(current_model_revision()),
        library_id=library_id,
    )
    selected.update(karaoke.karaoke_external_ids)
    selected.update(sync_selection(connection, library_id, ids))
    return [item for item in ids if item in selected]
