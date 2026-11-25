"""``PUT /settings/integrations/navidrome``."""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, Response

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.v1.settings._plugin import PluginSaved
from backend.settings.plugin_integration import RankingProfile, issue_key, profile_from
from core.db.connection import connect
from core.errors import ConflictError, NotFoundError
from core.lyrics.sources import lyrics_paused
from core.plugin import store


class PluginUpdate(RankingProfile, PutRequest):
    """The ranking profile, whether the integration is on, and its connection."""

    enabled: bool = False
    connection_id: uuid.UUID | None = None


class UpdatePluginSettings(Endpoint):
    """Save the plugin settings; enabling without a live key generates one.

    Changing the connection revokes every key. Lyrics cannot be switched on while the
    connection's sync or lyrics jobs run.
    """

    method = "PUT"
    path = "/integrations/navidrome"
    request = PluginUpdate
    response = PluginSaved

    def handle(
        self, body: PluginUpdate, response: Response, user: SessionUser = Depends(current_user)
    ) -> dict[str, object]:
        response.headers["Cache-Control"] = "no-store"
        with connect(dict_rows=True) as db:
            # Serialize enable/key generation and connection changes for this owner.
            store.lock_owner(db, user["id"])
            connection_id = body.connection_id
            url = store.owned_connection_url(db, connection_id, user["id"])
            if connection_id and url is None:
                raise NotFoundError("Navidrome connection not found")
            if body.enabled and connection_id is None:
                raise HTTPException(400, "Connect Navidrome before enabling the plugin")
            old = store.integration(db, user["id"])
            if connection_id and body.enabled and body.serve_lyrics:
                enabling = (
                    not old
                    or not old["enabled"]
                    or not profile_from(old).serve_lyrics
                    or old["connection_id"] != connection_id
                )
                if enabling and lyrics_paused(db, url):
                    raise ConflictError(
                        "Lyrics cannot be enabled until active sync and lyrics jobs finish"
                    )
            if old and old["connection_id"] != connection_id:
                store.revoke_all_keys(db, user["id"])
            profile = RankingProfile.model_validate(
                body.model_dump(exclude={"enabled", "connection_id"})
            )
            store.save_integration(
                db, user["id"], body.enabled, connection_id, profile.model_dump()
            )
            generated = None
            if body.enabled and not store.has_active_key(db, user["id"], connection_id):
                generated = issue_key(
                    db, user["id"], connection_id, "Navidrome", profile.key_expiry_days
                )
        return {"enabled": body.enabled, "generated_key": generated}
