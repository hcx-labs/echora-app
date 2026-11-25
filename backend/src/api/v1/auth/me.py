"""``GET /auth/me``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.auth._models import Me

_FIELDS = (
    "username",
    "email",
    "display_name",
    "is_admin",
    "onboarding_complete",
    "navidrome_connection_id",
)


class CurrentUser(Endpoint):
    """The signed-in user."""

    method = "GET"
    path = "/me"
    response = Me

    def handle(self, user: SessionUser = Depends(current_user)) -> dict[str, object]:
        return {key: user[key] for key in _FIELDS}
