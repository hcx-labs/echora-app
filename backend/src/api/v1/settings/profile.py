"""``PUT /settings/profile``."""

from __future__ import annotations

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from core.db.models import User
from core.db.session import session_scope
from core.errors import NotFoundError


class ProfileUpdate(PutRequest):
    """The display name (surrounding whitespace is dropped)."""

    display_name: str = Field(min_length=1, max_length=120)


class ProfileState(ResponseModel):
    """The stored display name."""

    display_name: str


class UpdateProfile(Endpoint):
    """Change the user's display name."""

    method = "PUT"
    path = "/profile"
    request = ProfileUpdate
    response = ProfileState

    def handle(
        self, request: ProfileUpdate, user: SessionUser = Depends(current_user)
    ) -> dict[str, str]:
        with session_scope() as session:
            stored = session.get(User, user["id"])
            if stored is None:
                raise NotFoundError("User not found")
            stored.display_name = request.display_name.strip()
        return {"display_name": request.display_name.strip()}
