"""Response models of the artist endpoints."""

from __future__ import annotations

from backend.api.responses import ResponseModel


class FacetTrack(ResponseModel):
    """The song closest to a facet's centre."""

    id: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None


class ArtistFacet(ResponseModel):
    """One side of an artist's sound: its weight and how many songs it covers."""

    index: int
    weight: float
    track_count: int
    representative_track: FacetTrack
