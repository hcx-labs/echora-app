"""Prepared-audio cache configuration."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class PreprocessConfig(EchoraConfig, env_prefix="ECHORA_PREPROCESS_"):
    """The content-addressed cache of decoded and separated audio."""

    dir: str = setting("/data/preprocessed")
    max_bytes: int = setting(21474836480, ge=0, description="Cache size (0: no limit).")
    ttl_seconds: float = setting(604800.0, gt=0)
    lock_timeout_seconds: float = setting(1800, gt=0)
    lock_poll_seconds: float = setting(0.05, gt=0)
