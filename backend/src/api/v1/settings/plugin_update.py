"""``PUT /settings/integrations/navidrome``."""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, Response
from psycopg.types.json import Jsonb

from backend.api.deps import SessionUser, current_user
from backend.api.endpoint import Endpoint
from backend.api.requests import PutRequest
from backend.api.v1.settings._plugin import PluginSaved
from backend.settings.plugin_integration import RankingProfile, connect, issue_key, profile_from
from core.errors import ConflictError, NotFoundError
from core.lyrics.sources import lyrics_paused


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
        with connect() as db:
            # Serialize enable/key generation and connection changes for this owner.
            db.execute("SELECT id FROM users WHERE id=%s FOR UPDATE", (user["id"],))
            connection_id = body.connection_id
            if connection_id:
                owned = db.execute(
                    "SELECT id FROM navidrome_connections WHERE id=%s AND owner_user_id=%s",
                    (connection_id, user["id"]),
                ).fetchone()
                if owned is None:
                    raise NotFoundError("Navidrome connection not found")
            if body.enabled and connection_id is None:
                raise HTTPException(400, "Connect Navidrome before enabling the plugin")
            old = db.execute(
                "SELECT * FROM navidrome_integrations WHERE user_id=%s", (user["id"],)
            ).fetchone()
            if connection_id and body.enabled and body.serve_lyrics:
                url = db.execute(
                    "SELECT url FROM navidrome_connections WHERE id=%s", (connection_id,)
                ).fetchone()["url"]
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
                db.execute(
                    """UPDATE navidrome_api_keys SET revoked_at=now()
                       WHERE user_id=%s AND revoked_at IS NULL""",
                    (user["id"],),
                )
            profile = RankingProfile.model_validate(
                body.model_dump(exclude={"enabled", "connection_id"})
            )
            db.execute(
                """INSERT INTO navidrome_integrations(user_id,enabled,connection_id,profile)
                VALUES (%s,%s,%s,%s) ON CONFLICT(user_id) DO UPDATE SET enabled=EXCLUDED.enabled,
                connection_id=EXCLUDED.connection_id,profile=EXCLUDED.profile,updated_at=now()""",
                (user["id"], body.enabled, connection_id, Jsonb(profile.model_dump())),
            )
            generated = None
            active = db.execute(
                """SELECT 1 FROM navidrome_api_keys WHERE user_id=%s AND connection_id=%s
                AND revoked_at IS NULL AND expires_at>now() LIMIT 1""",
                (user["id"], connection_id),
            ).fetchone()
            if body.enabled and not active:
                generated = issue_key(
                    db, user["id"], connection_id, "Navidrome", profile.key_expiry_days
                )
        return {"enabled": body.enabled, "generated_key": generated}
