"""Dependencies shared by endpoints: the signed-in user and the administrator check."""

from __future__ import annotations

from typing import Any

from fastapi import Cookie, Depends, HTTPException

from backend.sessions.store import session_user

SessionUser = dict[str, Any]
"""The signed-in user: id, username, email, display_name, is_admin, onboarding_complete and
navidrome_connection_id."""


def current_user(echora_session: str | None = Cookie(default=None)) -> SessionUser:
    """The user of the session cookie (401 without a live session)."""
    return session_user(echora_session)


def admin_user(user: SessionUser = Depends(current_user)) -> SessionUser:
    """The signed-in user, who must be an administrator (403 otherwise)."""
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Administrator access is required")
    return user
