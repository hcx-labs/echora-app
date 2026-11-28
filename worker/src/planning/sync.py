"""Select songs needing work before allocating analysis batches.

Workers recheck artifacts at execution time. This selection never loads models.
Known missing/instrumental/unavailable lyrics are terminal for ordinary sync;
explicit lyrics backfills remain the way to retry retrieval for those songs. Songs with a resync
request (``core.resync``) join partial and full syncs alike.
"""

from __future__ import annotations

import uuid

from core.planning import store
from core.resync import store as resync
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
    library_id = store.library_with_namespace(
        connection, uuid.uuid5(uuid.NAMESPACE_URL, url.rstrip("/"))
    )
    if library_id is None:
        return ids
    known = store.known_sources(connection, library_id, ids)
    selected = set(ids) - known
    # Songs asked to be redone (a pipeline of theirs, whatever its revision) join every sync.
    selected.update(resync.requested_sources(connection, library_id, ids))
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
    selected.update(store.songs_needing_backfill(connection, library_id, ids))

    from worker.transcription.config import transcription_enabled, transcription_model

    if transcription_model() and transcription_enabled(connection):
        selected.update(store.songs_needing_transcription(connection, library_id, ids))

    # Share the exact karaoke compatibility contract with the executing pipeline.
    from worker.kara.pipeline import (
        KARAOKE_PIPELINE_REVISION,
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
    selected.update(songs_needing_stems(connection, library_id, ids))
    selected.update(sync_selection(connection, library_id, ids))
    return [item for item in ids if item in selected]


def songs_needing_stems(connection, library_id, ids: list[str]) -> set[str]:
    """Songs without karaoke stems, when stems are on anywhere (here or on Modal)."""
    from core.features import feature_enabled
    from core.stems import store as stems
    from worker.stems.pipeline import stems_recipe

    if not (
        feature_enabled(connection, "karaoke_stems", "local")
        or feature_enabled(connection, "karaoke_stems", "modal")
    ):
        return set()
    rows = stems.songs_without_stems(connection, library_id, ids, stems_recipe())
    return {str(row["external_id"]) for row in rows}
