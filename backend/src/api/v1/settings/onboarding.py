"""``PUT /settings/onboarding``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from core.db.session import session_scope
from core.errors import NotFoundError
from core.library.membership import attach_user_library
from core.users import store as users


class OnboardingPreference(PutRequest):
    """Whether onboarding is done, and the connection the user chose (if any)."""

    complete: bool = True
    connection_id: uuid.UUID | None = None


class OnboardingState(ResponseModel):
    """The stored choice."""

    onboarding_complete: bool
    navidrome_connection_id: uuid.UUID | None


class SaveOnboarding(Endpoint):
    """Finish (or reopen) onboarding and select the user's Navidrome connection."""

    method = "PUT"
    path = "/onboarding"
    request = OnboardingPreference
    response = OnboardingState

    def handle(
        self, preference: OnboardingPreference, user: SessionUser = Depends(current_user)
    ) -> dict[str, object]:
        with session_scope() as session:
            stored = users.require_preference(session, user["id"])
            stored.onboarding_complete = preference.complete
            selected = users.navidrome_connection(session, preference.connection_id)
            if selected is not None and selected.owner_user_id != user["id"]:
                raise NotFoundError("Connection not found")
            stored.navidrome_connection_id = preference.connection_id
            users.touch(stored)
            selected_url = selected.url if selected else None
        if selected_url is not None:
            attach_user_library(user["id"], selected_url)
        return {
            "onboarding_complete": preference.complete,
            "navidrome_connection_id": preference.connection_id,
        }
