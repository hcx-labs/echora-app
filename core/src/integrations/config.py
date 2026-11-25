"""Configuration of the external services: Navidrome and Last.fm."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from core.config import EchoraConfig, setting


class NavidromeConfig(EchoraConfig, env_prefix="ECHORA_NAVIDROME_"):
    """HTTP limits of the Navidrome client, and the batch audio cache."""

    timeout_seconds: int = setting(30, gt=0, description="Connect and API timeout.")
    read_timeout_seconds: int = setting(300, gt=0, description="Audio download read timeout.")
    max_connections: int = setting(100, gt=0)
    max_keepalive_connections: int = setting(30, gt=0)
    keepalive_expiry_seconds: int = setting(30, gt=0)
    audio_cache_bytes: int = setting(
        2147483648,
        ge=0,
        legacy="ECHORA_AUDIO_CACHE_BYTES",
        description="Disk a batch may use for downloaded songs (0: no cache).",
    )

    @model_validator(mode="after")
    def _keepalive_within_maximum(self) -> Self:
        if self.max_keepalive_connections > self.max_connections:
            raise ValueError("Navidrome keepalive connections must not exceed maximum")
        return self


class LastfmConfig(EchoraConfig, env_prefix="ECHORA_LASTFM_"):
    """Last.fm requests."""

    validation_timeout_seconds: float = setting(15, gt=0, description="Checking an API key.")
    history_timeout_seconds: int = setting(
        20,
        gt=0,
        legacy="ECHORA_LISTENING_HISTORY_TIMEOUT_SECONDS",
        description="Reading listening history.",
    )
