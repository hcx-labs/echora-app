"""AI lyric transcription configuration: an optional published MOSS checkpoint."""

from __future__ import annotations

import re

from core.config import EchoraConfig, compute
from core.errors import ConfigurationError


class TranscriptionConfig(EchoraConfig, env_prefix="ECHORA_TRANSCRIPTION_"):
    """The MOSS checkpoint that transcribes lyrics. Unset: AI lyrics are unavailable."""

    moss_model_id: str = compute("", legacy="MOSS_MODEL_ID")
    moss_revision: str = compute("", legacy="MOSS_REVISION", description="40-character commit.")


def transcription_model() -> tuple[str, str] | None:
    """The configured checkpoint, or ``None``. There is no implicit upstream fallback."""
    config = TranscriptionConfig.load()
    model, revision = config.moss_model_id.strip(), config.moss_revision.strip()
    if not model and not revision:
        return None
    if not model or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ConfigurationError(
            "Set both MOSS_MODEL_ID and an immutable 40-character MOSS_REVISION"
        )
    return model, revision


def transcription_enabled(connection, where: str | None = None):
    """AI lyric generation is switched on for the running job's location."""
    from core.features import feature_enabled

    return feature_enabled(connection, "transcription", where)
