"""``GET /settings/integrations/navidrome``."""

from __future__ import annotations

from fastapi import Depends, Response

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._plugin import PluginSettingsView
from backend.settings.plugin_integration import profile_from
from core.db.connection import connect
from core.lyrics.sources import lyrics_paused
from core.plugin import store


class PluginSettings(Endpoint):
    """The user's Navidrome plugin settings and keys."""

    method = "GET"
    path = "/integrations/navidrome"
    response = PluginSettingsView

    def handle(
        self, response: Response, user: SessionUser = Depends(current_user)
    ) -> dict[str, object]:
        response.headers["Cache-Control"] = "no-store"
        with connect(dict_rows=True) as db:
            row = store.integration(db, user["id"])
            keys = store.keys(db, user["id"])
            connection_id = row["connection_id"] if row else user.get("navidrome_connection_id")
            url = store.owned_connection_url(db, connection_id, user["id"])
            paused = bool(url and lyrics_paused(db, url))
        return {
            **profile_from(row).model_dump(),
            "enabled": bool(row and row["enabled"]),
            "connection_id": connection_id,
            "lyrics_paused": paused,
            "keys": keys,
        }
