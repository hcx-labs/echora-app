"""``GET /library/tracks/{track_id}/info``."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.library.db import connect
from backend.library.track_info import track_info
from core.errors import NotFoundError
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
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT ts.source_data FROM user_track_links link
                   JOIN track_sources ts ON ts.track_id=link.track_id AND ts.library_id=link.library_id
                                        AND ts.external_id=link.external_id
                   WHERE link.user_id=%s AND link.track_id=%s ORDER BY ts.id LIMIT 1""",
                (user["id"], track_id),
            )
            source = cursor.fetchone()
            if source is None:
                raise NotFoundError("Track is not in your library")
            cursor.execute(
                """SELECT descriptors FROM track_audio_descriptors
                   WHERE track_id=%s AND status IN ('complete', 'partial')
                   ORDER BY created_at DESC LIMIT 1""",
                (track_id,),
            )
            descriptors = cursor.fetchone()
            cursor.execute(
                """SELECT text, language, provenance, karaoke_lines IS NOT NULL
                          AND jsonb_array_length(karaoke_lines) > 0 AS karaoke_lines
                   FROM lyrics WHERE track_id=%s""",
                (track_id,),
            )
            lyrics = cursor.fetchone()
            # Only a description written for these lyrics and the current prompt.
            cursor.execute(
                """SELECT th.description, th.themes FROM track_lyric_themes th
                   JOIN lyrics ly ON ly.track_id=th.track_id
                   WHERE th.track_id=%s AND th.status='ready' AND th.prompt_revision=%s
                     AND th.source_checksum=encode(sha256(convert_to(ly.text, 'UTF8')), 'hex')""",
                (track_id, LYRIC_THEMES_REVISION),
            )
            themes = cursor.fetchone()
        return {"track_id": str(track_id), **track_info(source, descriptors, lyrics, themes)}
