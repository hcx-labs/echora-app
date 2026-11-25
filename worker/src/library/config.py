"""Library ingest configuration."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class LibraryConfig(EchoraConfig, env_prefix="ECHORA_LIBRARY_"):
    """Fingerprinting and source freshness during sync."""

    fingerprint_timeout_seconds: float = setting(
        120, gt=0, legacy="ECHORA_FINGERPRINT_TIMEOUT_SECONDS"
    )
    source_recheck_seconds: int = setting(
        0,
        ge=0,
        legacy="ECHORA_SOURCE_RECHECK_SECONDS",
        description="Revalidate a song's source after this long (0: on every full sync).",
    )
