"""A Redis cache of cover art and player streams, shared by every API process.

Failures never reach the request: a Redis error is a cache miss, and a stream is published
only once it was read to the end (``StreamCacheWriter``).
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid
from dataclasses import dataclass
from functools import lru_cache

import redis

from backend.config import MediaCacheConfig

_STAGING_TTL_SECONDS = 3600


def cache_key(kind: str, *parts: object) -> str:
    """A key that does not reveal its parts (user, connection, song)."""
    value = "\x1f".join(str(part) for part in parts).encode()
    return f"echora:media:{kind}:{hashlib.sha256(value).hexdigest()}"


@dataclass
class CachedMedia:
    """A cached body and its content type."""

    content: bytes
    content_type: str


class StreamCacheWriter:
    """Collects a stream under a staging key and publishes it whole, or not at all."""

    def __init__(self, client: redis.Redis[bytes], key: str, content_type: str) -> None:
        self.client = client
        self.key = key
        self.staging_key = f"{key}:staging:{uuid.uuid4().hex}"
        self.enabled = False
        try:
            self.client.set(self.staging_key, f"{content_type}\0".encode(), ex=_STAGING_TTL_SECONDS)
            self.enabled = True
        except redis.RedisError:
            pass

    def append(self, chunk: bytes) -> None:
        if not self.enabled:
            return
        try:
            if not self.client.exists(self.staging_key):
                self.abort()
                return
            self.client.append(self.staging_key, chunk)
            self.client.expire(self.staging_key, _STAGING_TTL_SECONDS)
        except redis.RedisError:
            self.abort()

    def commit(self) -> None:
        if not self.enabled:
            return
        try:
            self.client.rename(self.staging_key, self.key)
            self.client.expire(self.key, MediaCacheConfig.load().ttl_seconds)
        except redis.RedisError:
            self.abort()
        else:
            self.enabled = False

    def abort(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        with contextlib.suppress(redis.RedisError):
            self.client.delete(self.staging_key)


class MediaCache:
    """Get and set cached media; every Redis error is a miss."""

    def __init__(self, url: str) -> None:
        self.client: redis.Redis[bytes] = redis.Redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=MediaCacheConfig.load().connect_timeout_seconds,
            socket_timeout=MediaCacheConfig.load().timeout_seconds,
        )

    def get(self, key: str) -> CachedMedia | None:
        try:
            value = self.client.get(key)
        except redis.RedisError:
            return None
        if not value:
            return None
        content_type, separator, content = value.partition(b"\0")
        if not separator or not content_type:
            return None
        return CachedMedia(content=content, content_type=content_type.decode("ascii", "replace"))

    def set(self, key: str, content: bytes, content_type: str) -> None:
        with contextlib.suppress(redis.RedisError):
            self.client.set(
                key,
                f"{content_type}\0".encode() + content,
                ex=MediaCacheConfig.load().ttl_seconds,
            )

    def stream_writer(self, key: str, content_type: str) -> StreamCacheWriter:
        return StreamCacheWriter(self.client, key, content_type)


@lru_cache(maxsize=1)
def media_cache() -> MediaCache | None:
    """The process's cache, or ``None`` without ``ECHORA_MEDIA_CACHE_REDIS_URL``."""
    url = MediaCacheConfig.load().redis_url
    return MediaCache(url) if url else None
