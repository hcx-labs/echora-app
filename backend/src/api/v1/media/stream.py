"""``GET /media/connections/{connection_id}/stream/{song_id}``."""

from __future__ import annotations

import logging

import httpx
from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.media.cache import cache_key, media_cache
from backend.sync.connections import owned_connection
from core.integrations.navidrome import media_navidrome_client

logger = logging.getLogger(__name__)


class Stream(Endpoint):
    """Stream a song, with range requests; ``cache=player`` keeps it in the media cache.

    ``quality`` is ``original`` (no transcoding), ``320`` or ``120`` kbit/s.
    """

    method = "GET"
    path = "/connections/{connection_id}/stream/{song_id}"

    def handle(
        self,
        connection_id: str,
        song_id: str,
        request: Request,
        quality: str = "original",
        cache: str | None = None,
        user: SessionUser = Depends(current_user),
    ) -> Response:
        credentials = owned_connection(connection_id, user["id"])
        if quality not in {"original", "320", "120"}:
            raise HTTPException(status_code=422, detail="Quality must be original, 320, or 120")
        player_cache = media_cache() if cache == "player" else None
        bit_rate = None if quality == "original" else int(quality)
        key = cache_key("stream", user["id"], connection_id, song_id, quality)
        cached = player_cache.get(key) if player_cache else None
        range_header = request.headers.get("range")
        starts_at_zero = not range_header or range_header in {"bytes=0-", "bytes=0"}
        if cached is None:
            client = media_navidrome_client(*credentials)
            try:
                upstream = client.open_transcode_stream(song_id, bit_rate, range_header)
            except Exception as error:
                client.close()
                raise HTTPException(status_code=502, detail="Could not open this track") from error
            content_type = upstream.headers.get("content-type", "audio/mpeg")
            writer = (
                player_cache.stream_writer(key, content_type)
                if player_cache and starts_at_zero
                else None
            )

            def stream_and_cache():
                completed = False
                try:
                    try:
                        for chunk in upstream.iter_raw(64 * 1024):
                            if writer:
                                writer.append(chunk)
                            yield chunk
                    except httpx.RemoteProtocolError:
                        logger.warning("Navidrome closed %s while streaming", song_id)
                    else:
                        completed = True
                finally:
                    upstream.close()
                    client.close()
                    if writer:
                        if completed:
                            writer.commit()
                        else:
                            writer.abort()

            forwarded = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() in {"content-range", "accept-ranges"}
            }
            return StreamingResponse(
                stream_and_cache(),
                status_code=upstream.status_code,
                media_type=content_type,
                headers={
                    **forwarded,
                    "Cache-Control": "private, max-age=3600" if player_cache else "no-store",
                },
            )
        content, content_type = cached.content, cached.content_type
        total = len(content)
        headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
        if range_header and range_header.startswith("bytes="):
            try:
                specification = range_header[6:].split(",", 1)[0]
                start_text, end_text = specification.split("-", 1)
                if start_text:
                    start = int(start_text)
                    end = min(int(end_text) if end_text else total - 1, total - 1)
                else:
                    suffix = min(int(end_text), total)
                    start, end = total - suffix, total - 1
                if start < 0 or start >= total or end < start:
                    raise ValueError
            except ValueError as error:
                raise HTTPException(
                    status_code=416, detail="Requested audio range is unavailable"
                ) from error
            body = content[start : end + 1]
            headers.update(
                {"Content-Range": f"bytes {start}-{end}/{total}", "Content-Length": str(len(body))}
            )
            return Response(body, status_code=206, media_type=content_type, headers=headers)
        headers["Content-Length"] = str(total)
        return Response(content, media_type=content_type, headers=headers)
