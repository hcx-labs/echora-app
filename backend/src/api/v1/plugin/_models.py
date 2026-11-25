"""Request and response models of the plugin category (Navidrome provider envelopes).

Requests come from the Navidrome plugin, which forwards the PDK's request structs: they carry
metadata Echora does not use (names, MusicBrainz ids). Unknown fields are therefore ignored at
every level instead of rejected.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from backend.api.requests import PostRequest
from backend.api.responses import ResponseModel

_TOLERANT = ConfigDict(extra="ignore")


class PluginRequest(PostRequest):
    """A body from the plugin: unknown fields are ignored."""

    model_config = _TOLERANT


class Song(PluginRequest):
    """A song by its Navidrome id."""

    id: str = Field(min_length=1, max_length=256)


class SimilarRequest(PluginRequest):
    """Songs like ``song``."""

    song: Song
    count: int = Field(default=20, ge=1, le=500)


class PathRequest(PluginRequest):
    """A path of songs from ``startSong`` to ``endSong``."""

    startSong: Song  # noqa: N815 - the PDK's field name
    endSong: Song  # noqa: N815 - the PDK's field name
    count: int = Field(default=25, ge=2, le=500)


class ArtistRequest(PluginRequest):
    """An artist by its Navidrome id."""

    id: str = Field(min_length=1, max_length=256)
    count: int = Field(default=20, ge=1, le=500)
    limit: int = Field(default=10, ge=1, le=100)


class LyricsRequest(PluginRequest):
    """Lyrics for ``track``."""

    track: Song


class SongRef(ResponseModel):
    """A song as Navidrome's provider API expects it."""

    id: str
    name: str | None
    artist: str
    album: str
    durationMs: int  # noqa: N815 - Navidrome's field name


class SonicMatch(ResponseModel):
    """A ranked song; ``similarity`` is -1 for path steps."""

    song: SongRef
    similarity: float


class SonicMatches(ResponseModel):
    """Ranked songs."""

    matches: list[SonicMatch]


class ArtistRef(ResponseModel):
    """An artist by Navidrome id."""

    id: str
    name: str | None


class SimilarArtistList(ResponseModel):
    """Similar artists, most similar first."""

    artists: list[ArtistRef]


class SongList(ResponseModel):
    """Songs for an artist radio."""

    songs: list[SongRef]


class LyricsText(ResponseModel):
    """One lyrics document (TTML, or LRC per language)."""

    lang: str
    text: str


class LyricsList(ResponseModel):
    """Empty when Echora has no lyrics to serve (Navidrome falls back to other providers)."""

    lyrics: list[LyricsText]
