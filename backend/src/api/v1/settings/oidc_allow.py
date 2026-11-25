"""``POST /settings/oidc/allowed-emails``."""

from __future__ import annotations

from fastapi import Depends
from pydantic import Field

from backend.api.deps import SessionUser, admin_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PostRequest
from backend.api.responses import ResponseModel
from core.db.session import session_scope
from core.errors import ConflictError, InvalidQueryError
from core.users import store as users


class AllowEmail(PostRequest):
    """An email to approve (compared case-insensitively)."""

    email: str = Field(min_length=3, max_length=320)


class AllowedEmail(ResponseModel):
    """The approved email, normalised."""

    email: str


class AllowOidcEmail(Endpoint):
    """Approve an email for sign-up while auto-provisioning is off (administrators only)."""

    method = "POST"
    path = "/oidc/allowed-emails"
    request = AllowEmail
    response = AllowedEmail
    status_code = 201

    def handle(
        self, request: AllowEmail, admin: SessionUser = Depends(admin_user)
    ) -> dict[str, str]:
        email = request.email.strip().casefold()
        if "@" not in email:
            raise InvalidQueryError("Enter a valid email address")
        with session_scope() as session:
            if users.email_in_use(session, email):
                raise ConflictError("This user already exists")
            users.allow_email(session, email, admin["id"])
        return {"email": email}
