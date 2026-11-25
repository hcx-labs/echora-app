"""Response models shared by several library endpoints."""

from __future__ import annotations

from typing import Any

from backend.api.responses import ResponseModel


class NameCount(ResponseModel):
    """A name and how many of the user's songs have it."""

    name: str
    tracks: int


class MapPayload(ResponseModel):
    """The library map: projected points, communities and neighbour edges.

    A map without embeddings has only ``points`` (empty) and ``model``. Points, communities
    and edges are documented in the web app's map; they are passed through as computed.
    """

    points: list[dict[str, Any]]
    clusters: list[dict[str, Any]] | None = None
    communities: list[dict[str, Any]] | None = None
    """The same list as ``clusters``."""
    edges: list[dict[str, Any]] | None = None
    model: str
    semantic_weight: float | None = None
    clustering: dict[str, Any] | None = None
    corpus_hash: str | None = None
    snapshot_id: str | None = None
    snapshot_created_at: Any = None
    cache_hit: bool | None = None
