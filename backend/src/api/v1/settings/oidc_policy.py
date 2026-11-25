"""``PUT /settings/oidc/policy``."""

from __future__ import annotations

from fastapi import Depends

from backend.api.deps import SessionUser, admin_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.responses import ResponseModel
from core.db.session import session_scope
from core.users import store as users


class OidcPolicy(PutRequest):
    """Whether anyone the provider signs in gets an account, or only approved emails."""

    auto_provision: bool


class OidcPolicyState(ResponseModel):
    """The stored policy."""

    auto_provision: bool


class UpdateOidcPolicy(Endpoint):
    """Change the sign-up policy (administrators only)."""

    method = "PUT"
    path = "/oidc/policy"
    request = OidcPolicy
    response = OidcPolicyState

    def handle(
        self, request: OidcPolicy, admin: SessionUser = Depends(admin_user)
    ) -> dict[str, bool]:
        with session_scope() as session:
            users.set_auto_provision(session, request.auto_provision)
        return {"auto_provision": request.auto_provision}
