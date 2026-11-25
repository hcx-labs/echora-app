"""Models of the concept endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field

from backend.api.requests import PostRequest
from backend.api.responses import ResponseModel


class ConceptExamples(PostRequest):
    """Text prompts and example songs for and against a concept."""

    positive_prompts: list[str] = Field(default_factory=list, max_length=12)
    negative_prompts: list[str] = Field(default_factory=list, max_length=12)
    positive_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    negative_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)


class Concept(ResponseModel):
    """A user's own concept."""

    id: uuid.UUID
    name: str
    description: str | None = None
    positive_prompts: list[str]
    negative_prompts: list[str]
    positive_track_ids: list[uuid.UUID]
    negative_track_ids: list[uuid.UUID]
    enabled: bool
    created_at: datetime
    updated_at: datetime | None = None
