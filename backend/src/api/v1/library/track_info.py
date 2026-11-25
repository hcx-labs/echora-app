"""``GET /library/tracks/{track_id}/info``."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.library.track_info import track_info
from core.db.connection import connect
from core.errors import NotFoundError
from core.library import store as library
from core.lyrics import store as lyrics
from worker.themes.pipeline import PROMPT_REVISION as LYRIC_THEMES_REVISION


class TrackTags(ResponseModel):
    """Navidrome's tags, as Navidrome reports them."""

    year: Any = None
    genres: list[str]
    moods: list[str]
    track: Any = None
    disc: Any = None
    album_artists: list[str]
    credits: dict[str, list[str]]
    """Contributor names by role (composer, lyricist, ...)."""
    isrc: Any = None
    explicit: bool
    comment: Any = None
    play_count: Any = None
    last_played: Any = None
    added: Any = None
    size_bytes: Any = None
    musicbrainz_id: Any = None


class TrackAudio(ResponseModel):
    """The file's format."""

    codec: str | None = None
    lossless: bool
    bit_rate_kbps: Any = None
    bit_depth: Any = None
    sample_rate_hz: Any = None
    channels: Any = None
    replay_gain: Any = None


class TrackAnalysis(ResponseModel):
    """Echora's audio descriptors."""

    bpm: Any = None
    key: str | None = None
    integrated_lufs: Any = None
    loudness_range_lu: Any = None
    true_peak_dbtp: Any = None


class TrackLyricsFacts(ResponseModel):
    """What lyrics the song has."""

    available: bool
    language: str | None = None
    source: str | None = None
    """``ai`` for generated lyrics, otherwise the provider."""
    synced: bool
    karaoke: bool


class TrackThemes(ResponseModel):
    """The lyrics' description and theme tags."""

    description: str
    tags: Any


class TrackInfo(ResponseModel):
    """The song-info panel."""

    track_id: str
    tags: TrackTags
    audio: TrackAudio
    analysis: TrackAnalysis
    lyrics: TrackLyricsFacts
    themes: TrackThemes | None


class TrackInfoEndpoint(Endpoint):
    """Tags, format, analysis, lyrics and themes of one of the user's songs."""

    method = "GET"
    path = "/tracks/{track_id}/info"
    response = TrackInfo

    def handle(self, track_id: uuid.UUID, user: SessionUser = Depends(current_user)) -> dict:
        with connect() as connection:
            source = library.source_data(connection, user["id"], track_id)
            if source is None:
                raise NotFoundError("Track is not in your library")
            descriptors = library.latest_descriptors(connection, track_id)
            lyrics_row = lyrics.lyrics_facts(connection, track_id)
            # Only a description written for these lyrics and the current prompt.
            themes = lyrics.current_themes(connection, track_id, LYRIC_THEMES_REVISION)
        return {"track_id": str(track_id), **track_info(source, descriptors, lyrics_row, themes)}
