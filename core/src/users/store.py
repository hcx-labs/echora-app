"""Accounts, preferences and the sign-in policy, through the ORM models (``core.db.models``).

Every function takes an open ``Session`` (``core.db.session.session_scope``); the caller owns the
transaction. Changes are made on the loaded models and written when the session commits.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from core.db.models import (
    NavidromeConnection,
    OidcAllowedEmail,
    OidcSetting,
    User,
    UserPreference,
    UserSession,
)
from core.errors import NotFoundError


def user(session: Session, user_id: uuid.UUID) -> User | None:
    """The account."""
    return session.get(User, user_id)


def require_user(session: Session, user_id: uuid.UUID) -> User:
    """The account, or ``NotFoundError("User not found")``."""
    stored = session.get(User, user_id)
    if stored is None:
        raise NotFoundError("User not found")
    return stored


def preference(session: Session, user_id: uuid.UUID) -> UserPreference | None:
    """The user's preferences."""
    return session.get(UserPreference, user_id)


def require_preference(session: Session, user_id: uuid.UUID) -> UserPreference:
    """The user's preferences, or ``NotFoundError("User preferences are unavailable")``."""
    stored = session.get(UserPreference, user_id)
    if stored is None:
        raise NotFoundError("User preferences are unavailable")
    return stored


def touch(stored: UserPreference) -> None:
    """Record that the preferences changed now."""
    stored.updated_at = datetime.now(UTC)


def navidrome_connection(
    session: Session, connection_id: uuid.UUID | None
) -> NavidromeConnection | None:
    """A Navidrome connection by id (``None`` for no id)."""
    return session.get(NavidromeConnection, connection_id) if connection_id else None


def users_with_email(session: Session) -> list[User]:
    """Accounts that have an email address, by address."""
    return list(
        session.scalars(
            select(User).where(User.email.is_not(None)).order_by(func.lower(User.email))
        ).all()
    )


def email_in_use(session: Session, email: str) -> bool:
    """Whether an account has this address (compared case-insensitively; pass it casefolded)."""
    return session.scalar(select(User.id).where(func.lower(User.email) == email)) is not None


def active_admin_count(session: Session) -> int:
    """Administrators who are not blocked."""
    return session.scalar(
        select(func.count()).select_from(User).where(User.is_admin, User.is_blocked.is_(False))
    )


def end_sessions(session: Session, user_id: uuid.UUID) -> None:
    """Sign the user out everywhere."""
    session.execute(delete(UserSession).where(UserSession.user_id == user_id))


def oidc_policy(session: Session) -> OidcSetting | None:
    """The stored OpenID Connect policy (``None`` before an administrator saves one)."""
    return session.get(OidcSetting, True)


def set_auto_provision(session: Session, enabled: bool) -> None:
    """Whether OpenID Connect sign-in creates accounts for new addresses."""
    policy = session.get(OidcSetting, True)
    if policy is None:
        policy = OidcSetting(singleton=True)
        session.add(policy)
    policy.auto_provision = enabled
    policy.updated_at = datetime.now(UTC)


def allowed_emails(session: Session) -> list[OidcAllowedEmail]:
    """Addresses allowed to sign in when accounts are not created automatically, sorted."""
    return list(session.scalars(select(OidcAllowedEmail).order_by(OidcAllowedEmail.email)).all())


def allow_email(session: Session, email: str, created_by: uuid.UUID) -> None:
    """Allow an address to sign in (once)."""
    if session.get(OidcAllowedEmail, email) is None:
        session.add(OidcAllowedEmail(email=email, created_by=created_by))


def disallow_email(session: Session, email: str) -> None:
    """Remove an address from the allowlist, if it is there."""
    allowed = session.get(OidcAllowedEmail, email)
    if allowed is not None:
        session.delete(allowed)
