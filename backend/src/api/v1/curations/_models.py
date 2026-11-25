"""Request and response models of the curations category."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import Field

from backend.api.requests import PatchRequest, PostRequest, PutRequest
from backend.api.responses import ResponseModel
from core.curation.preview import CurationRecipe


class CurationPreviewRequest(PostRequest, CurationRecipe):
    """A recipe to preview without saving."""


class CurationCreateRequest(PostRequest, CurationRecipe):
    """A recipe, its name and whether it refreshes on a schedule."""

    name: str = Field(min_length=1, max_length=120)
    refresh_enabled: bool = True


class CurationReplaceRequest(PutRequest, CurationRecipe):
    """The complete new recipe of a saved curation."""

    name: str = Field(min_length=1, max_length=120)
    refresh_enabled: bool = True


class CurationUpdateRequest(PatchRequest):
    """Schedule and size settings; fields left out (or null) stay as they are."""

    refresh_enabled: bool | None = None
    refresh_mode: str | None = Field(default=None, pattern="^(stable|fresh)$")
    track_limit: int | None = Field(default=None, ge=5, le=200)
    familiarity_percent: int | None = Field(default=None, ge=0, le=100)


class RefreshQueued(ResponseModel):
    """The refresh job of a curation."""

    job_id: str
    status: str
    curation_id: str


class CurationSaved(ResponseModel, CurationRecipe):
    """The saved recipe, its id and its queued first refresh (``tracks`` fill in later)."""

    name: str
    refresh_enabled: bool
    id: str
    tracks: list[Any]
    job_id: str
    status: str
    curation_id: str


class CurationChanged(ResponseModel):
    """The curation id and the fields that changed."""

    id: str
    refresh_enabled: bool | None = None
    refresh_mode: str | None = None
    track_limit: int | None = None
    familiarity_percent: int | None = None


class Curation(ResponseModel):
    """A saved curation: its settings, latest recipe and the latest revision's songs."""

    id: uuid.UUID
    user_id: uuid.UUID
    navidrome_connection_id: uuid.UUID
    name: str
    positive_prompt: str
    negative_prompt: str
    track_limit: int
    refresh_mode: str
    refresh_enabled: bool
    refresh_interval_hours: int
    navidrome_playlist_id: str | None
    status: str
    last_error: str | None
    last_refreshed_at: datetime | None
    next_refresh_at: datetime | None
    created_at: datetime
    updated_at: datetime
    positive_track_ids: list[uuid.UUID]
    negative_track_ids: list[uuid.UUID]
    familiarity_percent: int
    curation_type: str
    period_start: str | None
    period_end: str | None
    lookback_days: int
    sound_prompts: list[str]
    themes_prompts: list[str]
    sound_negative_prompts: list[str]
    themes_negative_prompts: list[str]
    sound_weight: int
    target_language: str
    language_strictness: str
    time_of_day_enabled: bool
    journey_start_track_id: uuid.UUID | None
    journey_stop_track_ids: list[uuid.UUID]
    journey_end_track_id: uuid.UUID | None
    journey_lyrics_weight: int
    sound_profile: dict[str, Any]
    recipe: dict[str, Any] | None
    positive_tracks: list[dict[str, Any]]
    negative_tracks: list[dict[str, Any]]
    tracks: list[dict[str, Any]]
    """Position, score, percentile, evidence and playback source of each song."""


class CurationList(ResponseModel):
    """The user's curations, most recently changed first."""

    curations: list[Curation]


class CurationPreview(ResponseModel):
    """The songs a recipe selects now, and how it weighed its signals.

    Sonic journeys report ``journey`` instead of the familiarity, language and coverage keys.
    """

    tracks: list[dict[str, Any]]
    references: dict[str, Any]
    corpus_size: int
    selection: dict[str, Any]
    curation_type: str
    model: str
    weights: dict[str, float]
    scoring_revision: Any
    lyrics_coverage: int
    representation_runs: dict[str, list[Any]] | None = None
    signal_weights: dict[str, float] | None = None
    example_component_weights: dict[str, Any] | None = None
    audio_profile_coverage: dict[str, int] | None = None
    theme_coverage: int | None = None
    lyrics_representation: str | None = None
    shuffle_seed: Any = None
    language: dict[str, Any] | None = None
    familiarity: dict[str, Any] | None = None
    journey: dict[str, Any] | None = None


class JourneyRequest(PostRequest):
    """Two songs and how to travel between them."""

    start_track_id: uuid.UUID
    end_track_id: uuid.UUID
    mode: str = Field(default="semantic", pattern="^(semantic|acoustic|blend)$")
    semantic_weight: float = Field(default=0.5, ge=0, le=1)
    length: int = Field(default=15, ge=3, le=50)


class JourneyStep(ResponseModel):
    """One song of a journey."""

    id: uuid.UUID
    title: str | None
    artist: str | None
    album: str | None
    duration_seconds: float | None
    source_id: str | None
    cover_art: str | None
    recording_group_id: str | None
    position: int
    target_progress: float
    target_similarity: float
    transition_similarity: float


class JourneyPreview(ResponseModel):
    """A path of songs from the start song to the end song."""

    mode: str
    semantic_weight: float
    requested_length: int
    steps: list[JourneyStep]
