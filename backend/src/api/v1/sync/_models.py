"""Request and response models of the sync category."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from backend.api.navidrome import Credentials
from backend.api.requests import PostRequest
from backend.api.responses import ResponseModel


class DiscoverRequest(Credentials):
    """A server to connect, and how many random songs to preview."""

    limit: int = Field(default=100, ge=1, le=500)


class IngestRequest(Credentials):
    """A server and the songs on it to analyze."""

    track_ids: list[str] = Field(min_length=1, max_length=20000)


class SampleImportRequest(PostRequest):
    """Songs of a saved connection to analyze."""

    track_ids: list[str] = Field(min_length=1, max_length=20000)


class SyncRequest(PostRequest):
    """How to sync a connection."""

    verify_audio_hashes: bool = True
    mode: str = Field(default="all", pattern="^(all|missing)$")
    compute: str | None = Field(default=None, pattern="^(local|modal)$")
    """Where model computation runs (docs/modal-compute.md); omitted uses the instance default."""


class TrackSummary(ResponseModel):
    """A Navidrome song."""

    id: str
    title: str
    artist: str | None = None
    album: str | None = None
    duration: float
    genre: str | None = None
    cover_art: Any = None


class ServerInfo(ResponseModel):
    """The server that answered."""

    version: str
    url: str


class Discovered(ResponseModel):
    """The saved connection and a preview of its songs."""

    connection_id: str
    server: ServerInfo
    tracks: list[TrackSummary]


class Sample(ResponseModel):
    """Random songs of a connection."""

    tracks: list[TrackSummary]


class Catalog(ResponseModel):
    """Navidrome's albums and artists, as Subsonic describes them."""

    albums: list[dict[str, Any]]
    artists: list[dict[str, Any]]


class RecordingIndex(ResponseModel):
    """Recording fingerprint coverage of the connection's songs."""

    enabled: bool
    total: int
    indexed: int
    unsupported: int
    missing: int
    reason: str | None = None


class MissingTrack(ResponseModel):
    """A song Echora has not analyzed yet."""

    id: str
    title: str
    artist: str | None = None
    album: str | None = None
    duration: float
    cover_art: Any = None


class SyncStatus(ResponseModel):
    """How much of the connection's library is analyzed; the first 50 missing songs."""

    server: str
    total: int
    processed: int
    missing: int
    recording_index: RecordingIndex
    tracks: list[MissingTrack]


class SemanticFusionCounts(ResponseModel):
    """Fused vectors, and the lyrics and audio vectors they are built from."""

    fused: int
    lyrics: int
    audio: int


class ProfileCounts(ResponseModel):
    """Songs with a source embedding, and how many have a current profile."""

    total: int
    current: int
    missing: int


class AudioProfileCoverage(ResponseModel):
    """Audio profile coverage per model and overall."""

    revision: str
    models: dict[str, ProfileCounts]
    total: int
    current: int
    missing: int
