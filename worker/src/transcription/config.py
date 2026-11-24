"""Optional published MOSS checkpoint. No implicit upstream-model fallback."""

from core.settings import get_settings
import re


def transcription_model() -> tuple[str, str] | None:
    model = get_settings().moss_model_id.strip()
    revision = get_settings().moss_revision.strip()
    if not model and not revision:
        return None
    if not model or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Set both MOSS_MODEL_ID and an immutable 40-character MOSS_REVISION")
    return model, revision


def transcription_enabled(connection, where: str | None = None):
    """AI lyric generation is switched on for the running job's location."""
    from core.features import feature_enabled

    return feature_enabled(connection, "transcription", where)
