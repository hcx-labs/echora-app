"""Response models of the auth category."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import PlainSerializer

from backend.api.responses import ResponseModel

IsoDatetime = Annotated[datetime, PlainSerializer(lambda value: value.isoformat(), return_type=str)]
"""A timestamp as ``datetime.isoformat()`` writes it (``+00:00``, as the legacy API did)."""


class OidcState(ResponseModel):
    """Whether sign-in through the provider is available."""

    configured: bool
    provider: str


class Me(ResponseModel):
    """The signed-in user."""

    username: str
    email: str | None
    display_name: str | None
    is_admin: bool
    onboarding_complete: bool
    navidrome_connection_id: uuid.UUID | None


class SessionState(ResponseModel):
    """The session's deadlines and the browser's activity schedule."""

    renewed: bool
    expires_at: IsoDatetime
    absolute_expires_at: IsoDatetime
    renew_after_seconds: int
    activity_check_seconds: int
    activity_window_seconds: int
    request_timeout_seconds: int
