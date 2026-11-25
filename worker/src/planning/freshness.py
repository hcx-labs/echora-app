"""Revalidate source identity without discarding valid canonical artifacts."""

from __future__ import annotations

from worker.library.config import LibraryConfig

# Ignore per-user playback/star/rating fields. Catalog metadata is only a change
# hint, not proof of byte identity: periodic rehashing handles unreported edits.
_CONTENT_HINTS = (
    "size",
    "created",
    "modified",
    "lastModified",
    "updated",
    "etag",
    "path",
    "suffix",
    "contentType",
    "duration",
    "bitRate",
    "samplingRate",
    "bitDepth",
    "channelCount",
    "title",
    "artist",
    "album",
    "year",
    "track",
    "discNumber",
    "coverArt",
)


def sources_needing_refresh(connection, library_id, external_ids, catalog=()):
    interval = int(LibraryConfig.load().source_recheck_seconds)
    if interval < 0:
        raise ValueError("ECHORA_SOURCE_RECHECK_SECONDS must be nonnegative")
    current = {track.id: getattr(track, "raw", {}) or {} for track in catalog}
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT external_id,source_data,
            audio_verified_at IS NULL OR audio_verified_at <= now() - (%s * interval '1 second')
            FROM track_sources WHERE library_id=%s AND source_type='subsonic'
                AND external_id=ANY(%s)""",
            (interval, library_id, list(external_ids)),
        )
        rows = cursor.fetchall()
    return {
        str(external_id)
        for external_id, previous, stale in rows
        if stale
        or any(
            key in current.get(str(external_id), {})
            and current[str(external_id)][key] != (previous or {}).get(key)
            for key in _CONTENT_HINTS
        )
    }
