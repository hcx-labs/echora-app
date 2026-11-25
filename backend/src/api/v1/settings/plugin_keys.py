"""``POST /settings/integrations/navidrome/keys``."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Response
from pydantic import Field

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PostRequest
from backend.api.v1.settings._plugin import GeneratedKey
from backend.settings.plugin_integration import connect, issue_key, profile_from


class KeyCreate(PostRequest):
    """A label, and a lifetime in days (the profile's default when omitted)."""

    label: str = Field(default="Navidrome", min_length=1, max_length=100, pattern=r"\S")
    expiry_days: int | None = Field(default=None, ge=1, le=3650)


class CreatePluginKey(Endpoint):
    """Generate another API key for the enabled integration."""

    method = "POST"
    path = "/integrations/navidrome/keys"
    request = KeyCreate
    response = GeneratedKey
    status_code = 201

    def handle(
        self, body: KeyCreate, response: Response, user: SessionUser = Depends(current_user)
    ) -> dict[str, object]:
        response.headers["Cache-Control"] = "no-store"
        with connect() as db:
            db.execute("SELECT id FROM users WHERE id=%s FOR UPDATE", (user["id"],))
            row = db.execute(
                "SELECT * FROM navidrome_integrations WHERE user_id=%s", (user["id"],)
            ).fetchone()
            if not row or not row["enabled"] or not row["connection_id"]:
                raise HTTPException(400, "Enable the integration before generating keys")
            return issue_key(
                db,
                user["id"],
                row["connection_id"],
                body.label,
                body.expiry_days or profile_from(row).key_expiry_days,
            )
