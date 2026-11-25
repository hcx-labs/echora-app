"""Response models shared by several categories.

Responses are Pydantic models so the OpenAPI schema is complete. ``ResponseModel`` rejects
fields it does not declare: a handler that returns more than its model says fails loudly in
tests instead of silently dropping data. Endpoints serialize with ``exclude_unset``, so a key a
handler leaves out stays absent.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ResponseModel(BaseModel):
    """Base of every response body."""

    model_config = ConfigDict(extra="forbid")


class JobSnapshot(ResponseModel):
    """A job as its owner sees it (``core.jobs.queue``): state, progress and summary.

    ``phase``, ``completed``, ``total`` and, when the job has reported them, ``message``,
    ``unit`` and ``track`` repeat the latest progress at the top level.
    """

    id: str
    job_id: str
    kind: str
    worker_type: str | None = None
    connection_id: str | None = None
    parent_id: str | None = None
    status: str
    cancel_requested: bool
    progress: dict[str, Any]
    summary: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str | None = None
    finished_at: str | None = None
    dismissed_at: str | None = None
    phase: str
    completed: int | float
    total: int | float
    message: str | None = None
    unit: str | None = None
    track: dict[str, Any] | None = None
    existing: bool
    """An equivalent job was already active, so no new one was queued."""
