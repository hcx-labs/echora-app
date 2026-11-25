"""``PUT /settings/timezone``."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from core.db.models import UserPreference
from core.db.session import session_scope
from core.errors import InvalidQueryError, NotFoundError


class TimezoneUpdate(PutRequest):
    """An IANA time zone name, such as ``Asia/Tokyo``."""

    timezone: str = Field(min_length=1, max_length=100)


class TimezoneState(ResponseModel):
    """The stored time zone."""

    timezone: str


class UpdateTimezone(Endpoint):
    """Change the time zone listening statistics use."""

    method = "PUT"
    path = "/timezone"
    request = TimezoneUpdate
    response = TimezoneState

    def handle(
        self, request: TimezoneUpdate, user: SessionUser = Depends(current_user)
    ) -> dict[str, str]:
        try:
            ZoneInfo(request.timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise InvalidQueryError("Unknown timezone") from error
        with session_scope() as session:
            preference = session.get(UserPreference, user["id"])
            if preference is None:
                raise NotFoundError("User preferences are unavailable")
            preference.timezone = request.timezone
            preference.updated_at = datetime.now(UTC)
        return {"timezone": request.timezone}
