"""Instance-wide, admin-trusted External AI configuration.

The provider is an OpenAI-compatible endpoint ("openai", the default) or Claude through the
official Anthropic SDK ("anthropic", see claude_client.py). For Claude the URL is optional and
defaults to the Anthropic API.

Local/private HTTP destinations are intentional: admins control network egress.
Use network policy to restrict it where needed. HTTP exposes keys/lyrics on the
network; prefer HTTPS off-host. Never log request bodies or authorization headers.
URL is the API base (e.g. http://localhost:8000/v1), not a completion URL.
"""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_PROMPT = "Translate the song into natural, expressive lyrics that feel at home in the target language. Use the whole song for context. Aim to carry its meaning, emotion, imagery, voice, and energy rather than translating word for word. You may reshape phrasing, rhythm, rhyme, slang, and punctuation to make each line flow and sing naturally. Preserve recurring hooks in a recognizable way, but adapt them when the context calls for it.\n\nThe one structural constraint: each source line must produce exactly one translated line, in the same order. Do not merge, split, skip, or add lines. Preserve any supplied line IDs and follow the requested output format. Return only the translation."

# Second pass over AI lyric transcriptions. The response contract is appended in code
# (transcription_refinement.py), so an edited prompt cannot break the output format.
DEFAULT_TRANSCRIPTION_PROMPT = """You proofread automatic lyric transcriptions of songs. The transcript was produced by a speech model listening to the song. It is mostly right, but some words are mis-heard.

Work in two steps. First read the whole transcript and work out what the song is about: its story or theme, who is speaking to whom, recurring images and phrases, the chorus. Write this as a short summary. Then go line by line. A line must make sense on its own and in the context of the whole song. Where it does not, fix the mis-heard words with words that sound alike and fit the line, the neighbouring lines and the song (its theme, who is addressed, tense and person, recurring phrases). Fix as much of a line as that needs, but no more.

Keep the words that are right; never rephrase a line that already makes sense, and never translate. Spell repeated lines (choruses, hooks) identically each time they are sung the same way. Write every line in the song's main script, keeping the same words. If you know this song's lyrics, they can help you choose between sound-alike words; never replace a line with what you remember when it does not sound like the transcript. Keep backing vocals in parentheses. Leave casing and punctuation as they are unless a fix needs it. If a line is right or you are unsure, return it unchanged."""


class LanguagePair(BaseModel):
    """A translation direction: two different language tags."""

    model_config = ConfigDict(extra="forbid")
    source: str
    target: str

    @field_validator("source", "target")
    @classmethod
    def normalize(cls, value: str) -> str:
        value = value.strip().replace("_", "-").lower()
        if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", value):
            raise ValueError("Expected a language tag")
        return value

    @model_validator(mode="after")
    def different(self):
        if self.source == self.target:
            raise ValueError("Source and target must differ")
        return self


Provider = Literal["openai", "anthropic"]


class ExternalAISettings(BaseModel):
    """The administrator's settings for the external AI provider."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    provider: Provider = "openai"
    url: str = Field(default="", max_length=2048)
    model: str = Field(default="", max_length=200)
    prompt: str = Field(default=DEFAULT_PROMPT, min_length=1, max_length=8000)
    language_pairs: list[LanguagePair] = Field(default_factory=list, max_length=64)
    # On by default, but only acts while External AI itself is enabled.
    transcription_refinement: bool = True
    transcription_prompt: str = Field(
        default=DEFAULT_TRANSCRIPTION_PROMPT, min_length=1, max_length=8000
    )
    # Its own model (a stronger one fixes mixed-language refrains); empty uses `model`.
    transcription_model: str = Field(default="", max_length=200)
    # Off by default: it sends every song's lyrics to the provider once.
    lyric_themes: bool = False

    @field_validator("url")
    @classmethod
    def endpoint(cls, value: str) -> str:
        if not value:
            return value
        try:
            parsed = urlsplit(value)
            if (
                any(c.isspace() or ord(c) < 32 for c in value)
                or "\\" in value
                or parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or "?" in value
                or "#" in value
            ):
                raise ValueError()
            _ = parsed.port
        except ValueError:
            raise ValueError(
                "Expected an HTTP(S) base URL without userinfo, query or fragment"
            ) from None
        return value.rstrip("/")

    @field_validator("model", "prompt", "transcription_prompt", "transcription_model")
    @classmethod
    def trim(cls, value):
        return value.strip()

    @model_validator(mode="after")
    def coherent(self):
        pairs = [(p.source, p.target) for p in self.language_pairs]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate language pair")
        if (
            not self.prompt
            or not self.transcription_prompt
            or (self.enabled and self.provider == "openai" and not self.url)
        ):
            raise ValueError("Enabled API requires a URL; prompts cannot be blank")
        return self

    def ready(self) -> bool:
        """Enabled with everything a request needs (Claude's URL defaults to the Anthropic API)."""
        return bool(self.enabled and self.model and (self.url or self.provider == "anthropic"))
