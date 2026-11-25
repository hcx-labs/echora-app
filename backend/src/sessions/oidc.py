"""Sign-in through an OpenID Connect provider: the client and account provisioning.

The first account must be the bootstrap administrator (``ECHORA_OIDC_BOOTSTRAP_ADMIN_EMAIL``).
After that, new accounts are created automatically while the policy allows it
(``oidc_settings.auto_provision``), or for emails an administrator allowed.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from authlib.integrations.starlette_client import OAuth
from fastapi import HTTPException

from backend.config import OidcConfig
from backend.sessions.store import create_session
from core.db.session import session_scope
from core.users import store as users


def issuer() -> str:
    """The configured issuer URL without a trailing slash ("" when unset)."""
    return OidcConfig.load().issuer_url.rstrip("/")


@functools.cache
def _registry(issuer_url: str, client_id: str, client_secret: str, scopes: str) -> OAuth:
    oauth = OAuth()
    if issuer_url and client_id and client_secret:
        oauth.register(
            name="oidc",
            client_id=client_id,
            client_secret=client_secret,
            server_metadata_url=f"{issuer_url}/.well-known/openid-configuration",
            client_kwargs={"scope": scopes},
        )
    return oauth


def oidc_client() -> Any | None:
    """The provider's client, or ``None`` when OIDC is not configured."""
    config = OidcConfig.load()
    registry = _registry(
        issuer(), config.client_id, config.client_secret.get_secret_value(), config.scopes
    )
    return registry.create_client("oidc")


def sign_in(claims: Mapping[str, Any]) -> tuple[str, datetime]:
    """Find or provision the account for the provider's claims and open a session.

    Returns the session token and its idle deadline. Refusals are ``HTTPException``s with the
    message shown to the user.
    """
    subject = str(claims.get("sub") or "").strip()
    email = str(claims.get("email") or "").strip().casefold()
    if not subject or not email or "@" not in email:
        raise HTTPException(status_code=422, detail="OIDC must provide subject and email claims")
    config = OidcConfig.load()
    if config.require_verified_email and claims.get("email_verified") is not True:
        raise HTTPException(status_code=403, detail="OIDC email is not verified")
    bootstrap_email = config.bootstrap_admin_email.strip().casefold()
    if not bootstrap_email:
        raise HTTPException(status_code=503, detail="OIDC_BOOTSTRAP_ADMIN_EMAIL is not configured")
    with session_scope() as session:
        user = users.identity(session, subject, email)
        admin_exists = users.admin_exists(session)
        if user is None:
            policy = users.oidc_policy(session)
            allowed = users.allowed_email(session, email)
            if not admin_exists and email != bootstrap_email:
                raise HTTPException(
                    status_code=403, detail="The bootstrap administrator must sign in first"
                )
            if admin_exists and not (policy and policy.auto_provision) and allowed is None:
                raise HTTPException(
                    status_code=403, detail="Your email has not been approved by an administrator"
                )
            display_name = str(
                claims.get("name") or claims.get("preferred_username") or email
            ).strip()
            user = users.add_user(
                session,
                username=email,
                email=email,
                display_name=display_name,
                password_hash=None,
                oidc_subject=subject,
                is_admin=not admin_exists and email == bootstrap_email,
            )
            if allowed is not None:
                users.disallow_email(session, email)
        else:
            if user.is_blocked:
                raise HTTPException(status_code=403, detail="This Echora account is blocked")
            if user.oidc_subject and user.oidc_subject != subject:
                raise HTTPException(
                    status_code=409, detail="This email is linked to another OIDC identity"
                )
            user.oidc_subject = subject
            user.email = email
            user.username = email
        users.drop_expired_sessions(session, datetime.now(UTC))
        token, expires_at, _ = create_session(session, user.id)
    return token, expires_at
