"""OIDC-managed, user-scoped credentials for the Echora Navidrome plugin."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Literal
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Response
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from core.settings import get_settings
from core.lyrics.sources import lyrics_paused


class RankingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    musical_weight: int = Field(default=80, ge=0, le=100)
    musical_semantic_weight: int = Field(default=70, ge=0, le=100)
    missing_lyrics: Literal["audio", "exclude"] = "audio"
    max_per_artist: int = Field(default=2, ge=1, le=20)
    serve_lyrics: bool = True
    include_translations: bool = True
    lyrics_format: Literal["ttml", "lrc"] = "ttml"
    key_expiry_days: int = Field(default=90, ge=1, le=3650)


class IntegrationUpdate(RankingProfile):
    enabled: bool = False
    connection_id: uuid.UUID | None = None


class KeyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(default="Navidrome", min_length=1, max_length=100, pattern=r"\S")
    expiry_days: int | None = Field(default=None, ge=1, le=3650)


def connect():
    return psycopg.connect(get_settings().database_url, row_factory=dict_row)


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def profile_from(row):
    return RankingProfile.model_validate(row["profile"] if row else {})


def issue_key(db, user_id, connection_id, label, days):
    now = datetime.now(timezone.utc)
    secret = "echora_nd_" + secrets.token_urlsafe(32)
    identifier = uuid.uuid4()
    expiry = now + timedelta(days=days)
    db.execute(
        """INSERT INTO navidrome_api_keys
        (id,user_id,connection_id,label,prefix,token_hash,created_at,expires_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            identifier,
            user_id,
            connection_id,
            label.strip(),
            secret[:18],
            digest(secret),
            now,
            expiry,
        ),
    )
    return {"id": str(identifier), "label": label.strip(), "secret": secret, "expires_at": expiry}


def authenticate(authorization: str | None = Header(default=None)):
    unauthorized = HTTPException(
        401, "Invalid or expired integration key", headers={"WWW-Authenticate": "Bearer"}
    )
    if not authorization or not authorization.startswith("Bearer "):
        raise unauthorized
    with connect() as db:
        row = db.execute(
            """SELECT k.id AS key_id,k.user_id,k.connection_id,n.url,i.profile
            FROM navidrome_api_keys k
            JOIN navidrome_integrations i ON i.user_id=k.user_id AND i.enabled
                AND i.connection_id=k.connection_id
            JOIN users u ON u.id=k.user_id AND NOT u.is_blocked
            JOIN navidrome_connections n ON n.id=k.connection_id AND n.owner_user_id=k.user_id
            WHERE k.token_hash=%s AND k.revoked_at IS NULL AND k.expires_at>now()""",
            (digest(authorization[7:]),),
        ).fetchone()
        if row is None:
            raise unauthorized
        db.execute("UPDATE navidrome_api_keys SET last_used_at=now() WHERE id=%s", (row["key_id"],))
    row["profile"] = profile_from(row)
    return row


def settings_router(require_user):
    router = APIRouter(prefix="/settings/integrations/navidrome", tags=["integrations"])

    @router.get("")
    def read(response: Response, user=Depends(require_user)):
        response.headers["Cache-Control"] = "no-store"
        with connect() as db:
            row = db.execute(
                "SELECT * FROM navidrome_integrations WHERE user_id=%s", (user["id"],)
            ).fetchone()
            keys = db.execute(
                """SELECT id,label,prefix,connection_id,created_at,expires_at,revoked_at,last_used_at,
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

    @router.put("")
    def save(body: IntegrationUpdate, response: Response, user=Depends(require_user)):
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
                    raise HTTPException(404, "Navidrome connection not found")
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
                    raise HTTPException(
                        409, "Lyrics cannot be enabled until active sync and lyrics jobs finish"
                    )
            if old and old["connection_id"] != connection_id:
                db.execute(
                    "UPDATE navidrome_api_keys SET revoked_at=now() WHERE user_id=%s AND revoked_at IS NULL",
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

    @router.post("/keys", status_code=201)
    def create(body: KeyCreate, response: Response, user=Depends(require_user)):
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

    @router.delete("/keys/{key_id}", status_code=204)
    def revoke(key_id: uuid.UUID, user=Depends(require_user)):
        with connect() as db:
            row = db.execute(
                """UPDATE navidrome_api_keys SET revoked_at=coalesce(revoked_at,now())
                WHERE id=%s AND user_id=%s RETURNING id""",
                (key_id, user["id"]),
            ).fetchone()
            if row is None:
                raise HTTPException(404, "Integration key not found")
        return Response(status_code=204)

    return router
