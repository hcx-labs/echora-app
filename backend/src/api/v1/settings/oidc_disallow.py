"""``DELETE /settings/oidc/allowed-emails/{email}``."""

from __future__ import annotations

from fastapi import Depends, Response

from backend.api.deps import SessionUser, admin_user
from backend.api.endpoint import Endpoint
from core.db.session import session_scope
from core.users import store as users


class DisallowOidcEmail(Endpoint):
    """Withdraw an approval (administrators only); unknown emails are ignored."""

    method = "DELETE"
    path = "/oidc/allowed-emails/{email}"
    status_code = 204

    def handle(self, email: str, admin: SessionUser = Depends(admin_user)) -> Response:
        with session_scope() as session:
            users.disallow_email(session, email.casefold())
        return Response(status_code=204)
