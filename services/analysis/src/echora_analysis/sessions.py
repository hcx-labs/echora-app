"""Independent app sessions. Only explicit user activity moves the idle deadline."""

from __future__ import annotations

from echora_analysis.config import ApiConfig, OidcConfig, SessionConfig
from echora_analysis.config import SessionConfig
from datetime import datetime, timedelta, timezone
import hashlib
import math
import secrets
from urllib.parse import urlsplit

from fastapi import APIRouter, Cookie, Header, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from core.db.session import session_scope
from core.db.models import User, UserSession


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(session, user_id, *, now: datetime | None = None):
    settings = SessionConfig.load()
    now = now or datetime.now(timezone.utc)
    token = secrets.token_urlsafe(48)
    absolute = now + timedelta(seconds=settings.absolute_timeout_seconds)
    expires = min(absolute, now + timedelta(seconds=settings.idle_timeout_seconds))
    session.add(
        UserSession(
            token_hash=token_hash(token),
            user_id=user_id,
            created_at=now,
            last_renewed_at=now,
            expires_at=expires,
            absolute_expires_at=absolute,
        )
    )
    return token, expires, absolute


def _load(session, token: str | None, now: datetime, *, lock: bool = False):
    if not token:
        raise HTTPException(
            status_code=401, detail="Authentication required", headers={"Cache-Control": "no-store"}
        )
    query = (
        select(UserSession)
        .join(UserSession.user)
        .options(joinedload(UserSession.user).joinedload(User.preference))
        .where(
            UserSession.token_hash == token_hash(token),
            UserSession.expires_at > now,
            UserSession.absolute_expires_at > now,
            User.is_blocked.is_(False),
        )
    )
    if lock:
        # Serialize simultaneous activity from tabs without locking joined user rows.
        query = query.with_for_update(of=UserSession)
    stored = session.scalar(query)
    if stored is None or stored.user.preference is None:
        raise HTTPException(
            status_code=401, detail="Session expired", headers={"Cache-Control": "no-store"}
        )
    return stored


def session_user(token: str | None):
    with session_scope() as session:
        stored = _load(session, token, datetime.now(timezone.utc))
        user, preference = stored.user, stored.user.preference
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "display_name": user.display_name,
            "is_admin": user.is_admin,
            "onboarding_complete": preference.onboarding_complete,
            "navidrome_connection_id": preference.navidrome_connection_id,
        }


def session_status(token: str | None, *, activity: bool = False, now: datetime | None = None):
    settings = SessionConfig.load()
    fixed_now = now is not None
    now = now or datetime.now(timezone.utc)
    with session_scope() as session:
        stored = _load(session, token, now, lock=activity)
        # Lock contention must not let a request revive a session that expired
        # while it waited. Explicit times are only used by deterministic tests.
        if activity and not fixed_now:
            now = datetime.now(timezone.utc)
            if stored.expires_at <= now or stored.absolute_expires_at <= now:
                raise HTTPException(
                    status_code=401, detail="Session expired", headers={"Cache-Control": "no-store"}
                )
        due = stored.last_renewed_at + timedelta(seconds=settings.renew_interval_seconds)
        renewed = activity and now >= due
        if renewed:
            stored.expires_at = min(
                stored.absolute_expires_at,
                now + timedelta(seconds=settings.idle_timeout_seconds),
            )
            stored.last_renewed_at = now
            due = now + timedelta(seconds=settings.renew_interval_seconds)
        return {
            "renewed": renewed,
            "expires_at": stored.expires_at,
            "absolute_expires_at": stored.absolute_expires_at,
            "renew_after_seconds": max(0, math.ceil((due - now).total_seconds())),
            "activity_check_seconds": settings.activity_check_seconds,
            "activity_window_seconds": settings.activity_window_seconds,
            "request_timeout_seconds": settings.request_timeout_seconds,
        }


def set_session_cookie(response: Response, token: str, expires_at: datetime):
    remaining = max(0, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        "echora_session",
        token,
        max_age=remaining,
        expires=expires_at,
        httponly=True,
        samesite="strict",
        secure=SessionConfig.load().cookie_secure,
        path="/",
    )


router = APIRouter(prefix="/auth")


@router.get("/session")
def read_session(response: Response, echora_session: str | None = Cookie(default=None)):
    response.headers["Cache-Control"] = "no-store"
    return session_status(echora_session)


@router.post("/session/activity")
def record_activity(
    response: Response,
    echora_session: str | None = Cookie(default=None),
    x_echora_activity: str | None = Header(default=None),
    origin: str | None = Header(default=None),
):
    if x_echora_activity != "1":
        raise HTTPException(
            status_code=403,
            detail="Activity header required",
            headers={"Cache-Control": "no-store"},
        )
    settings, oidc = SessionConfig.load(), OidcConfig.load()
    allowed = {
        value.strip()
        for value in (settings.trusted_origins + "," + ApiConfig.load().cors_origins).split(",")
        if value.strip()
    }
    for redirect in (oidc.redirect_uri, oidc.post_login_redirect):
        parsed = urlsplit(redirect)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            allowed.add(f"{parsed.scheme}://{parsed.netloc}")
    # Never infer trusted origins from Host or forwarded headers. Enforce this
    # at the API itself, including requests that bypass the Next.js proxy.
    if not origin or origin == "null" or origin not in allowed:
        raise HTTPException(
            status_code=403,
            detail="Untrusted activity origin",
            headers={"Cache-Control": "no-store"},
        )
    result = session_status(echora_session, activity=True)
    # Reissue the authoritative deadline even when another tab already renewed it.
    set_session_cookie(response, echora_session, result["expires_at"])
    response.headers["Cache-Control"] = "no-store"
    return result
