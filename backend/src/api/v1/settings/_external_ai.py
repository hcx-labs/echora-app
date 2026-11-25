"""Views of the External AI settings shared by its section endpoints.

The settings are one stored record (``core.lyrics.translations``); each section endpoint reads and
writes its own fields. The API key is never returned, only ``has_key``.
"""

from __future__ import annotations

from typing import Any

from backend.api.responses import ResponseModel
from core.integrations.external_ai import DEFAULT_TRANSCRIPTION_PROMPT


class LanguagePairView(ResponseModel):
    """A translation direction."""

    source: str
    target: str


class ExternalAIView(ResponseModel):
    """Every External AI setting, and whether a key is stored."""

    enabled: bool
    provider: str
    url: str
    model: str
    prompt: str
    language_pairs: list[LanguagePairView]
    transcription_refinement: bool
    transcription_prompt: str
    transcription_model: str
    lyric_themes: bool
    has_key: bool


class EndpointView(ResponseModel):
    """The provider, its URL and model, and whether a key is stored."""

    enabled: bool
    provider: str
    url: str
    model: str
    has_key: bool


class TranslationView(ResponseModel):
    """The translation prompt and language pairs."""

    prompt: str
    language_pairs: list[LanguagePairView]


class TranscriptionView(ResponseModel):
    """Transcript refinement; ``enabled`` and ``model`` are read-only here."""

    enabled: bool
    transcription_refinement: bool
    transcription_prompt: str
    transcription_model: str
    model: str
    default_transcription_prompt: str


class ThemesView(ResponseModel):
    """Lyric themes; ``enabled`` is read-only here."""

    enabled: bool
    lyric_themes: bool


def current() -> dict[str, Any]:
    """The stored settings and ``has_key``."""
    import core.lyrics.translations as storage

    settings, encrypted = storage.load_settings()
    return {**settings.model_dump(), "has_key": bool(encrypted)}


def endpoint_view(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("enabled", "provider", "url", "model", "has_key")}


def translation_view(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ("prompt", "language_pairs")}


def transcription_view(value: dict[str, Any]) -> dict[str, Any]:
    # `enabled` is read-only here: the section only acts while External AI is on.
    return {
        **{
            key: value[key]
            for key in (
                "enabled",
                "transcription_refinement",
                "transcription_prompt",
                "transcription_model",
                "model",
            )
        },
        "default_transcription_prompt": DEFAULT_TRANSCRIPTION_PROMPT,
    }


def themes_view(value: dict[str, Any]) -> dict[str, Any]:
    # `enabled` is read-only here, as for transcription.
    return {key: value[key] for key in ("enabled", "lyric_themes")}
