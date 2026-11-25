"""``GET /media/connections/{connection_id}/cover/{cover_id}``."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Response

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.media.cache import cache_key, media_cache
from backend.sync.connections import owned_connection
from core.integrations.navidrome import media_navidrome_client

_CACHE_CONTROL = {"Cache-Control": "private, max-age=3600"}


class Cover(Endpoint):
    """Cover art of a song or album, ``size`` pixels square (64 to 1600), through the cache."""

    method = "GET"
    path = "/connections/{connection_id}/cover/{cover_id:path}"

    def handle(
        self,
        connection_id: str,
        cover_id: str,
        size: int = 160,
        user: SessionUser = Depends(current_user),
    ) -> Response:
        credentials = owned_connection(connection_id, user["id"])
        bounded_size = min(max(size, 64), 1600)
        cache = media_cache()
        key = cache_key("cover", user["id"], connection_id, cover_id, bounded_size)
        cached = cache.get(key) if cache else None
        if cached:
            return Response(
                content=cached.content, media_type=cached.content_type, headers=_CACHE_CONTROL
            )
        try:
            with media_navidrome_client(*credentials) as client:
                content, content_type = client.cover_art(cover_id, bounded_size)
        except Exception as error:
            raise HTTPException(status_code=404, detail="Artwork unavailable") from error
        if cache:
            cache.set(key, content, content_type)
        return Response(content=content, media_type=content_type, headers=_CACHE_CONTROL)
