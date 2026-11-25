"""``GET /settings/integrations/navidrome``."""

from __future__ import annotations

from fastapi import Depends, Response

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.v1.settings._plugin import PluginSettingsView
from backend.settings.plugin_integration import profile_from
from core.db.connection import connect
from core.lyrics.sources import lyrics_paused


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
            row = db.execute(
                "SELECT * FROM navidrome_integrations WHERE user_id=%s", (user["id"],)
            ).fetchone()
            keys = db.execute(
                """SELECT id,label,prefix,connection_id,created_at,expires_at,revoked_at,
                          last_used_at,
                CASE WHEN revoked_at IS NOT NULL THEN 'revoked' WHEN expires_at<=now() THEN 'expired'
                     ELSE 'active' END AS status
                FROM navidrome_api_keys WHERE user_id=%s ORDER BY created_at DESC""",
                (user["id"],),
            ).fetchall()
            connection_id = row["connection_id"] if row else user.get("navidrome_connection_id")
            connection = db.execute(
                "SELECT url FROM navidrome_connections WHERE id=%s AND owner_user_id=%s",
                (connection_id, user["id"]),
            ).fetchone()
            paused = bool(connection and lyrics_paused(db, connection["url"]))
        return {
            **profile_from(row).model_dump(),
            "enabled": bool(row and row["enabled"]),
            "connection_id": connection_id,
            "lyrics_paused": paused,
            "keys": keys,
        }
