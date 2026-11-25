"""``GET /settings/oidc``."""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy import func, select

from backend.api.deps import SessionUser, admin_user
from backend.api.endpoint import Endpoint
from backend.api.responses import ResponseModel
from backend.config import OidcConfig
from backend.sessions.oidc import issuer, oidc_client
from core.db.models import OidcAllowedEmail, OidcSetting, User
from core.db.session import session_scope


class OidcAccount(ResponseModel):
    """An account that signs in through OIDC."""

    id: str
    email: str | None
    display_name: str | None
    is_admin: bool
    is_blocked: bool


class OidcAdministration(ResponseModel):
    """The provider, the sign-up policy, the accounts and the approved emails."""

    configured: bool
    issuer: str | None
    require_verified_email: bool
    auto_provision: bool
    users: list[OidcAccount]
    allowed_emails: list[str]


class OidcSettings(Endpoint):
    """OIDC sign-in administration (administrators only)."""

    method = "GET"
    path = "/oidc"
    response = OidcAdministration

    def handle(self, admin: SessionUser = Depends(admin_user)) -> dict[str, object]:
        with session_scope() as session:
            policy = session.get(OidcSetting, True)
            users = session.scalars(
                select(User).where(User.email.is_not(None)).order_by(func.lower(User.email))
            ).all()
            allowed = session.scalars(
                select(OidcAllowedEmail).order_by(OidcAllowedEmail.email)
            ).all()
            return {
                "configured": oidc_client() is not None,
                "issuer": issuer() or None,
                "require_verified_email": OidcConfig.load().require_verified_email,
                "auto_provision": policy.auto_provision if policy else True,
                "users": [
                    {
                        "id": str(item.id),
                        "email": item.email,
                        "display_name": item.display_name,
                        "is_admin": item.is_admin,
                        "is_blocked": item.is_blocked,
                    }
                    for item in users
                ],
                "allowed_emails": [item.email for item in allowed],
            }
