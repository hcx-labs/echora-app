"""Helpers shared by the curation endpoints."""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator

from sqlalchemy.exc import IntegrityError

import worker.curation.jobs as curation_jobs
from core.errors import ConflictError


def refresh(curation_id: uuid.UUID, user_id: uuid.UUID) -> dict[str, object]:
    """Queue a curation refresh (``{job_id, status, curation_id}``)."""
    return curation_jobs.enqueue(curation_id, user_id)


@contextlib.contextmanager
def unique_name() -> Iterator[None]:
    """A duplicate curation name is a 409."""
    try:
        yield
    except IntegrityError as error:
        if "curations_user_name_idx" in str(error.orig):
            raise ConflictError("A curation with this name already exists") from error
        raise


def recipe_columns(request) -> dict[str, object]:
    """The curation columns a create or replace request sets."""
    return {
        "name": request.name.strip(),
        "curation_type": request.curation_type,
        "positive_prompt": request.positive_prompt.strip(),
        "negative_prompt": request.negative_prompt.strip(),
        "sound_prompts": request.sound_prompts,
        "themes_prompts": request.themes_prompts,
        "sound_negative_prompts": request.sound_negative_prompts,
        "themes_negative_prompts": request.themes_negative_prompts,
        "sound_weight": request.sound_weight,
        "sound_profile": request.sound_profile.model_dump(exclude_none=True, exclude_defaults=True),
        "positive_track_ids": request.positive_track_ids,
        "negative_track_ids": request.negative_track_ids,
        "familiarity_percent": request.familiarity_percent,
        "period_start": request.period_start,
        "period_end": request.period_end,
        "lookback_days": request.lookback_days,
        "time_of_day_enabled": request.time_of_day_enabled,
        "track_limit": request.track_limit,
        "refresh_mode": request.refresh_mode,
        "target_language": request.target_language,
        "language_strictness": request.language_strictness,
        "journey_start_track_id": request.journey_start_track_id,
        "journey_stop_track_ids": request.journey_stop_track_ids,
        "journey_end_track_id": request.journey_end_track_id,
        "journey_lyrics_weight": request.journey_lyrics_weight,
        "refresh_enabled": request.refresh_enabled,
    }


def saved(request, curation_id: uuid.UUID, queued: dict[str, object]) -> dict[str, object]:
    """The response of a create or replace: the recipe, the id and the queued refresh."""
    return {**request.model_dump(mode="json"), "id": str(curation_id), "tracks": [], **queued}
