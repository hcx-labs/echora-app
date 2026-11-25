"""The administrator check of the model, External AI, External processing and motion settings.

Its message differs from ``backend.api.deps.admin_user``'s.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException

from backend.api.deps import SessionUser, current_user


def admin_required(user: SessionUser = Depends(current_user)) -> SessionUser:
    """The signed-in administrator (403 "Administrator access required" otherwise)."""
    if not user.get("is_admin"):
        raise HTTPException(403, "Administrator access required")
    return user
