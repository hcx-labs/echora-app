"""``PATCH /settings/oidc/users/{user_id}``."""

from __future__ import annotations

import uuid

from fastapi import Depends

from backend.api.deps import SessionUser, admin_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PatchRequest
from backend.api.responses import ResponseModel
from core.db.session import session_scope
from core.errors import InvalidQueryError
from core.users import store as users


class AccountUpdate(PatchRequest):
    """Promote, demote, block or unblock; omitted or null fields stay."""

    is_admin: bool | None = None
    is_blocked: bool | None = None


class AccountState(ResponseModel):
    """The account and the fields that changed."""

    id: str
    is_admin: bool | None = None
    is_blocked: bool | None = None


class UpdateOidcUser(Endpoint):
    """Change an account's role or block it (administrators only).

    Blocking ends the account's sessions. Administrators cannot demote or block themselves, and
    the last active administrator cannot be demoted.
    """

    method = "PATCH"
    path = "/oidc/users/{user_id}"
    request = AccountUpdate
    response = AccountState

    def handle(
        self, user_id: uuid.UUID, request: AccountUpdate, admin: SessionUser = Depends(admin_user)
    ) -> dict[str, object]:
        values = request.model_dump(exclude_none=True)
        with session_scope() as session:
            target = users.require_user(session, user_id)
            if user_id == admin["id"] and (
                values.get("is_admin") is False or values.get("is_blocked") is True
            ):
                raise InvalidQueryError("You cannot demote or block your own account")
            if (
                target.is_admin
                and values.get("is_admin") is False
                and users.active_admin_count(session) <= 1
            ):
                raise InvalidQueryError("Echora must retain at least one active administrator")
            for key, value in values.items():
                setattr(target, key, value)
            if values.get("is_blocked") is True:
                users.end_sessions(session, user_id)
            return {"id": str(target.id), **values}
