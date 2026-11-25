"""The External processing (Modal) settings view shared by its endpoints."""

from __future__ import annotations

from backend.api.responses import ResponseModel


class SetupJob(ResponseModel):
    """The latest Modal preparation job."""

    id: str
    status: str
    error: str | None
    phase: str | None
    message: str | None
    completed: int | float | None
    total: int | float | None


class ExternalProcessingView(ResponseModel):
    """The Modal settings (never the token secret) and its deployment state."""

    enabled: bool
    provider: str
    token_id: str
    has_secret: bool
    gpu: str
    has_hf_token: bool
    default_compute: str
    allow_users: bool
    workspace: str | None
    status: str
    status_detail: str | None
    checked_at: str | None
    image: str
    gpu_types: list[str]
    setup: SetupJob | None
