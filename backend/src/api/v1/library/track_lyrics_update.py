"""``PUT /library/tracks/{track_id}/lyrics``."""

from __future__ import annotations

import uuid

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from core.db.connection import connect
from core.errors import InvalidQueryError
from core.library import store as library
from core.lyrics import store as lyrics


class LyricsUpdate(PutRequest):
    """Lyrics typed by the user, and optionally their language."""

    text: str = Field(min_length=1, max_length=100_000)
    language: str | None = Field(default=None, max_length=12)


class LyricsUpdated(ResponseModel):
    """Whether karaoke and the lyrics embedding are redone (only when the text changed)."""

    track_id: str
    karaoke_pending: bool
    lyrics_embedding_pending: bool


class TrackLyricsUpdate(Endpoint):
    """Replace a song's lyrics with the user's own."""

    method = "PUT"
    path = "/tracks/{track_id}/lyrics"
    request = LyricsUpdate
    response = LyricsUpdated

    def handle(
        self, track_id: uuid.UUID, body: LyricsUpdate, user: SessionUser = Depends(current_user)
    ) -> dict:
        lyric_text = body.text.strip()
        if not lyric_text:
            raise InvalidQueryError("Lyrics cannot be empty")
        language = body.language.strip().lower() if body.language else None
        with connect() as connection:
            library.require_visible(
                connection, user["id"], track_id, "Track is not in your library"
            )
            existing = lyrics.lyrics_text(connection, track_id)
            changed = existing is None or existing["text"] != lyric_text
            lyrics.save_manual_lyrics(connection, track_id, lyric_text, language)
            if changed:
                lyrics.discard_derived(connection, track_id)
        return {
            "track_id": str(track_id),
            "karaoke_pending": changed,
            "lyrics_embedding_pending": changed,
        }
