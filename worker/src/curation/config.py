"""Curation refresh configuration."""

from __future__ import annotations

from core.config import EchoraConfig, setting


class CurationConfig(EchoraConfig, env_prefix="ECHORA_CURATION_"):
    """The scheduled worker's curation refreshes."""

    due_limit: int = setting(100, gt=0, description="Due curations queued per check.")
