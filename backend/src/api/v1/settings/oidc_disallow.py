"""``DELETE /settings/oidc/allowed-emails/{email}``."""

from __future__ import annotations

from fastapi import Depends, Response

from backend.api.deps import SessionUser, admin_user
from backend.api.endpoint import Endpoint
from core.db.models import OidcAllowedEmail
from core.db.session import session_scope


class DisallowOidcEmail(Endpoint):
    """Withdraw an approval (administrators only); unknown emails are ignored."""

    method = "DELETE"
    path = "/oidc/allowed-emails/{email}"
    status_code = 204

    def handle(self, email: str, admin: SessionUser = Depends(admin_user)) -> Response:
        with session_scope() as session:
            allowed = session.get(OidcAllowedEmail, email.casefold())
            if allowed is not None:
                session.delete(allowed)
        return Response(status_code=204)
