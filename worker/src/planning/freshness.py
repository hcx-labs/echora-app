"""Revalidate source identity without discarding valid canonical artifacts."""

from __future__ import annotations

from core.planning import store
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
    rows = store.source_freshness(connection, interval, library_id, list(external_ids))
    return {
        str(row["external_id"])
        for row in rows
        if row["stale"]
        or any(
            key in current.get(str(row["external_id"]), {})
            and current[str(row["external_id"])][key] != (row["source_data"] or {}).get(key)
            for key in _CONTENT_HINTS
        )
    }
