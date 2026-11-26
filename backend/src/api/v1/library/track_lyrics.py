"""``GET /library/tracks/{track_id}/lyrics``."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.plugin.lyrics import smoothed
from backend.settings.plugin_integration import profile_from
from core.db.connection import connect
from core.errors import NotFoundError
from core.lyrics import store
from core.lyrics.translations import load_translations
from core.plugin import store as plugin_store
from worker.translation.pipeline import source_lines as translation_source_lines


class LyricsTranslation(ResponseModel):
    """A ready translation, line by line, with the lines it translates."""

    source_language: str | None = None
    target_language: str
    lines: list[dict[str, Any]]
    source_lines: list[Any]


class TrackLyrics(ResponseModel):
    """The song's lyrics, its karaoke timing when aligned, and its translations."""

    available: bool
    availability_status: str | None = None
    text: str | None = None
    language: str | None = None
    source: str | None = None
    provenance: dict[str, Any] | None = None
    karaoke_lines: list[Any] | None = None
    karaoke_model: str | None = None
    karaoke_model_revision: str | None = None
    karaoke_bounded: bool | None = None
    lines: list[Any]
    """Karaoke lines when aligned, otherwise the synced source lines (or none)."""
    karaoke: bool
    translations: list[LyricsTranslation]


class TrackLyricsEndpoint(Endpoint):
    """Lyrics, karaoke lines and translations of one of the user's songs."""

    method = "GET"
    path = "/tracks/{track_id}/lyrics"
    response = TrackLyrics

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            row = store.user_lyrics(connection, user["id"], track_id)
            # TEMPORARY: preview the Navidrome plugin's karaoke smoothing in the web player.
            # Remove with the import of ``smoothed`` once the switches are tuned.
            profile = profile_from(plugin_store.integration(connection, user["id"]))
        if row is None:
            raise NotFoundError("Track is not in your library")
        row = smoothed(row, profile)  # TEMPORARY, see above
        provenance = row.get("provenance") or {}
        source_lines = provenance.get("lines") or []
        karaoke_lines = row.get("karaoke_lines") or []
        original_lines = translation_source_lines(row)
        translations = [
            {
                "source_language": item["source_language"],
                "target_language": item["target_language"],
                "lines": item["lines"],
                "source_lines": original_lines,
            }
            for item in load_translations(track_id, original_lines)
            # A mixed-language song's translation covers only its foreign lines; one where the
            # model left every line as it was has nothing to show.
            if item["status"] == "ready"
            and any(line.get("text", "").strip() for line in item["lines"])
        ]
        return {
            "available": bool(row.get("text")),
            "availability_status": row.get("availability_status"),
            **row,
            "lines": karaoke_lines or source_lines,
            "karaoke": bool(karaoke_lines),
            "translations": translations,
        }
