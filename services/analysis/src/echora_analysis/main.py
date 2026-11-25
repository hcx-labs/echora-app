from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import core.jobs.queue as jobs
import httpx
import igraph as ig
import numpy as np
import psycopg
from authlib.integrations.starlette_client import OAuth, OAuthError
from core.curation.concepts import (
    combine_concept_percentiles,
    empirical_percentiles,
    predefined_concepts,
    score_concept,
)
from core.curation.journeys import (
    normalize_rows as normalize_journey_rows,
)
from core.curation.journeys import (
    select_journey,
    spherical_targets,
)
from core.curation.listening import (
    TOP_TRACK_PERIODS,
    match_navidrome_play_counts,
    match_top_tracks,
    navidrome_play_counts,
    top_tracks,
)
from core.curation.preview import CurationRecipe
from core.curation.preview import preview_curation as preview_recipe
from core.db.config import DatabaseConfig
from core.db.models import (
    Curation,
    NavidromeConnection,
    OidcAllowedEmail,
    OidcSetting,
    User,
    UserPreference,
    UserSession,
)
from core.db.session import session_scope
from core.embeddings.representations import configure_representations
from core.embeddings.text import shared_lyrics_model
from core.errors import (
    ConflictError,
    EchoraError,
    IntegrationError,
    InvalidQueryError,
    NotFoundError,
)
from core.integrations.config import LastfmConfig
from core.integrations.navidrome import NavidromeClient, media_navidrome_client
from core.library.connections import credential_cipher, load_connection
from core.library.membership import attach_user_library, relink_renumbered_sources
from core.revisions import AUDIO_PROFILE_REVISION, MELODY_CONTOUR_REVISION, SUPPORTED_PROFILE_MODELS
from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, HttpUrl, SecretStr
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse, StreamingResponse
from worker.kara.pipeline import KARAOKE_PIPELINE_REVISION
from worker.library.descriptors import DESCRIPTOR_REVISION
from worker.library.visual_features import VISUAL_FEATURE_REVISION
from worker.library.waveforms import WAVEFORM_REVISION
from worker.melody.index import DEFAULT_CORPUS_SIZE, search_corpus
from worker.remote.settings import compute_for
from worker.themes.pipeline import PROMPT_REVISION as LYRIC_THEMES_REVISION

from echora_analysis.config import ApiConfig, OidcConfig, SessionConfig, UiConfig

from . import sessions
from .artists import (
    fit_artist_profile,
    representative_indices,
    soft_chamfer_similarity,
    weighted_center,
)
from .media_cache import cache_key, media_cache
from .melody_preview import melody_preview
from .navidrome_integration import settings_router as navidrome_integration_router
from .plugin_routes import router as navidrome_plugin_router
from .recording_routes import create_router as recording_router
from .routes.external_ai import router as external_ai_router
from .routes.external_processing import router as external_processing_router
from .routes.motion_artwork import router as motion_artwork_router

app = FastAPI(title="Echora analysis", version="0.3.0")

# Domain errors carry messages written for the user; anything else stays a server error.
_ERROR_STATUS = (
    (NotFoundError, 404),
    (ConflictError, 409),
    (InvalidQueryError, 422),
    (IntegrationError, 502),
)


@app.exception_handler(EchoraError)
async def _echora_error(request: Request, error: EchoraError) -> JSONResponse:
    for kind, status in _ERROR_STATUS:
        if isinstance(error, kind):
            return JSONResponse({"detail": str(error)}, status_code=status)
    raise error


# Browser-facing media (covers, streams) is served cross-origin from the web app
# when NEXT_PUBLIC_ANALYSIS_ORIGIN is set; canvas palette extraction needs CORS.
# Origins are explicit: reflecting arbitrary origins with credentials would let
# any site read responses using the visitor's session cookie.
_cors_origins = [
    origin.strip() for origin in ApiConfig.load().cors_origins.split(",") if origin.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.add_middleware(
    SessionMiddleware,
    secret_key=OidcConfig.load().session_secret.get_secret_value(),
    session_cookie="echora_oidc_state",
    same_site="lax",
    https_only=SessionConfig.load().cookie_secure,
)
_oauth = OAuth()
_oidc_issuer = OidcConfig.load().issuer_url.rstrip("/")
if (
    _oidc_issuer
    and OidcConfig.load().client_id
    and OidcConfig.load().client_secret.get_secret_value()
):
    _oauth.register(
        name="oidc",
        client_id=OidcConfig.load().client_id,
        client_secret=OidcConfig.load().client_secret.get_secret_value(),
        server_metadata_url=f"{_oidc_issuer}/.well-known/openid-configuration",
        client_kwargs={"scope": OidcConfig.load().scopes},
    )
logger = logging.getLogger(__name__)
_COMMUNITY_SNAPSHOT_REVISION = 1


class Credentials(BaseModel):
    url: HttpUrl
    username: str = Field(min_length=1)
    password: SecretStr


class DiscoverRequest(Credentials):
    limit: int = Field(default=100, ge=1, le=500)


class IngestRequest(Credentials):
    track_ids: list[str] = Field(min_length=1, max_length=20000)


class SyncRequest(BaseModel):
    verify_audio_hashes: bool = True
    mode: str = Field(default="all", pattern="^(all|missing)$")
    # Where model computation runs (docs/modal-compute.md); omitted uses the instance default.
    compute: str | None = Field(default=None, pattern="^(local|modal)$")


class OidcPolicyRequest(BaseModel):
    auto_provision: bool


class OidcAllowRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)


class OidcUserUpdateRequest(BaseModel):
    is_admin: bool | None = None
    is_blocked: bool | None = None


class ProfileSettingsRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)


class TimezoneSettingsRequest(BaseModel):
    timezone: str = Field(min_length=1, max_length=100)


class LastFmSettingsRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    api_key: SecretStr


class KaraokeProcessingSettingsRequest(BaseModel):
    enabled: bool
    # Which location's switch: syncs on this server, or syncs on Modal.
    location: str = Field(default="local", pattern="^(local|modal)$")


class LyricsUpdateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    language: str | None = Field(default=None, max_length=12)


class LyricsStatusRequest(BaseModel):
    status: str = Field(pattern="^(instrumental|missing)$")


class TranscriptionLanguageRequest(BaseModel):
    language: str = Field(pattern="^[a-z]{2,3}(?:-[A-Z]{2})?$")


class HuggingFaceTokenRequest(BaseModel):
    # Blank clears the stored token.
    token: SecretStr = Field(max_length=1000)


class HumProcessingSettingsRequest(BaseModel):
    enabled: bool
    location: str = Field(default="local", pattern="^(local|modal)$")


class OnboardingPreference(BaseModel):
    complete: bool = True
    connection_id: uuid.UUID | None = None


class ConceptRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    positive_prompts: list[str] = Field(default_factory=list, max_length=12)
    negative_prompts: list[str] = Field(default_factory=list, max_length=12)
    positive_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    negative_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)


class ConceptPreviewRequest(BaseModel):
    positive_prompts: list[str] = Field(default_factory=list, max_length=12)
    negative_prompts: list[str] = Field(default_factory=list, max_length=12)
    positive_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    negative_track_ids: list[uuid.UUID] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=50, ge=1, le=200)


class ConceptLensRequest(BaseModel):
    concepts: list[str] = Field(min_length=1, max_length=6)
    minimum_percentile: float = Field(default=0.5, ge=0, le=1)
    representation: str = Field(default="hybrid", pattern="^(semantic|lyrics|hybrid)$")


class CurationCreateRequest(CurationRecipe):
    name: str = Field(min_length=1, max_length=120)
    refresh_enabled: bool = True


class CurationUpdateRequest(BaseModel):
    refresh_enabled: bool | None = None
    refresh_mode: str | None = Field(default=None, pattern="^(stable|fresh)$")
    track_limit: int | None = Field(default=None, ge=5, le=200)
    familiarity_percent: int | None = Field(default=None, ge=0, le=100)


class JourneyRequest(BaseModel):
    start_track_id: uuid.UUID
    end_track_id: uuid.UUID
    mode: str = Field(default="semantic", pattern="^(semantic|acoustic|blend)$")
    semantic_weight: float = Field(default=0.5, ge=0, le=1)
    length: int = Field(default=15, ge=3, le=50)


def _credentials(request: Credentials) -> tuple[str, str, str]:
    return str(request.url).rstrip("/"), request.username, request.password.get_secret_value()


def _save_connection(credentials: tuple[str, str, str], user_id: uuid.UUID) -> str:
    url, username, password = credentials
    encrypted = credential_cipher().encrypt(password.encode())
    with session_scope() as session:
        stored = session.scalar(
            select(NavidromeConnection).where(
                NavidromeConnection.owner_user_id == user_id,
                NavidromeConnection.url == url,
                NavidromeConnection.username == username,
            )
        )
        if stored is None:
            stored = NavidromeConnection(
                url=url, username=username, encrypted_password=encrypted, owner_user_id=user_id
            )
            session.add(stored)
            session.flush()
        else:
            stored.encrypted_password = encrypted
            stored.last_used_at = datetime.now(timezone.utc)
        return str(stored.id)


def require_user(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    return _session_user(echora_session)


def _session_user(token: str | None) -> dict[str, object]:
    return sessions.session_user(token)


app.include_router(external_ai_router(require_user, credential_cipher))
app.include_router(external_processing_router(require_user))
app.include_router(motion_artwork_router(require_user))
app.include_router(recording_router(require_user))
app.include_router(sessions.router)
app.include_router(navidrome_integration_router(require_user))
app.include_router(navidrome_plugin_router)


@app.on_event("startup")
def start_background_services() -> None:
    # Worker claims, not API restarts, determine whether execution is interrupted.
    _enforce_secure_cookie_policy()
    with psycopg.connect(DatabaseConfig.load().database_url) as connection:
        configure_representations(connection)


def _enforce_secure_cookie_policy() -> None:
    if SessionConfig.load().cookie_secure:
        return
    redirect = OidcConfig.load().redirect_uri or "http://localhost:3000/analysis/auth/oidc/callback"
    host = urlparse(redirect).hostname or ""
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(
            "COOKIE_SECURE must be true when OIDC_REDIRECT_URI is not localhost; "
            "session cookies would otherwise be sent over plain HTTP"
        )


@app.get("/auth/oidc/status")
def oidc_status() -> dict[str, object]:
    return {"configured": _oauth.create_client("oidc") is not None, "provider": "OIDC"}


@app.get("/auth/oidc/start")
async def oidc_start(request: Request) -> Response:
    client = _oauth.create_client("oidc")
    if client is None:
        raise HTTPException(status_code=503, detail="OIDC is not configured")
    redirect_uri = OidcConfig.load().redirect_uri or str(request.url_for("oidc_callback"))
    return await client.authorize_redirect(request, redirect_uri)


@app.get("/auth/oidc/callback", name="oidc_callback")
async def oidc_callback(request: Request) -> Response:
    client = _oauth.create_client("oidc")
    if client is None:
        raise HTTPException(status_code=503, detail="OIDC is not configured")
    try:
        token_data = await client.authorize_access_token(request)
        claims = token_data.get("userinfo") or await client.userinfo(token=token_data)
    except OAuthError as error:
        logger.warning("OIDC authentication failed: %s", error.error)
        raise HTTPException(status_code=401, detail="OIDC authentication failed") from error
    subject = str(claims.get("sub") or "").strip()
    email = str(claims.get("email") or "").strip().casefold()
    if not subject or not email or "@" not in email:
        raise HTTPException(status_code=422, detail="OIDC must provide subject and email claims")
    require_verified = OidcConfig.load().require_verified_email
    if require_verified and claims.get("email_verified") is not True:
        raise HTTPException(status_code=403, detail="OIDC email is not verified")
    bootstrap_email = OidcConfig.load().bootstrap_admin_email.strip().casefold()
    if not bootstrap_email:
        raise HTTPException(status_code=503, detail="OIDC_BOOTSTRAP_ADMIN_EMAIL is not configured")
    with session_scope() as session:
        user = session.scalar(
            select(User).where((User.oidc_subject == subject) | (func.lower(User.email) == email))
        )
        admin_exists = (
            session.scalar(select(func.count()).select_from(User).where(User.is_admin)) > 0
        )
        if user is None:
            policy = session.get(OidcSetting, True)
            allowed = session.get(OidcAllowedEmail, email)
            if not admin_exists and email != bootstrap_email:
                raise HTTPException(
                    status_code=403, detail="The bootstrap administrator must sign in first"
                )
            if admin_exists and not (policy and policy.auto_provision) and allowed is None:
                raise HTTPException(
                    status_code=403, detail="Your email has not been approved by an administrator"
                )
            display_name = str(
                claims.get("name") or claims.get("preferred_username") or email
            ).strip()
            user = User(
                username=email,
                email=email,
                display_name=display_name,
                password_hash=None,
                oidc_subject=subject,
                is_admin=not admin_exists and email == bootstrap_email,
            )
            user.preference = UserPreference()
            session.add(user)
            session.flush()
            if allowed is not None:
                session.delete(allowed)
        else:
            if user.is_blocked:
                raise HTTPException(status_code=403, detail="This Echora account is blocked")
            if user.oidc_subject and user.oidc_subject != subject:
                raise HTTPException(
                    status_code=409, detail="This email is linked to another OIDC identity"
                )
            user.oidc_subject = subject
            user.email = email
            user.username = email
        session.execute(
            delete(UserSession).where(UserSession.expires_at < datetime.now(timezone.utc))
        )
        token, expires_at, _ = sessions.create_session(session, user.id)
    destination = OidcConfig.load().post_login_redirect
    response = RedirectResponse(destination, status_code=303)
    sessions.set_session_cookie(response, token, expires_at)
    return response


@app.post("/auth/logout", status_code=204)
def logout(response: Response, echora_session: str | None = Cookie(default=None)) -> Response:
    if echora_session:
        with session_scope() as session:
            session.execute(
                delete(UserSession).where(
                    UserSession.token_hash == hashlib.sha256(echora_session.encode()).hexdigest()
                )
            )
    response.delete_cookie("echora_session", path="/")
    return response


@app.get("/auth/me")
def me(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    user = _session_user(echora_session)
    return {
        key: user[key]
        for key in (
            "username",
            "email",
            "display_name",
            "is_admin",
            "onboarding_complete",
            "navidrome_connection_id",
        )
    }


@app.put("/users/me/preferences/onboarding")
def save_onboarding(
    preference: OnboardingPreference, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    with session_scope() as session:
        stored = session.get(UserPreference, user["id"])
        if stored is None:
            raise HTTPException(status_code=404, detail="User preferences are unavailable")
        stored.onboarding_complete = preference.complete
        selected = (
            session.get(NavidromeConnection, preference.connection_id)
            if preference.connection_id
            else None
        )
        if selected is not None and selected.owner_user_id != user["id"]:
            raise HTTPException(status_code=404, detail="Connection not found")
        stored.navidrome_connection_id = preference.connection_id
        stored.updated_at = datetime.now(timezone.utc)
        selected_url = selected.url if selected else None
    if selected_url is not None:
        attach_user_library(user["id"], selected_url)
    return {
        "onboarding_complete": preference.complete,
        "navidrome_connection_id": preference.connection_id,
    }


@app.get("/settings")
def settings(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    user = _session_user(echora_session)
    with session_scope() as session:
        stored_user = session.get(User, user["id"])
        preference = session.get(UserPreference, user["id"])
        connection = (
            session.get(NavidromeConnection, preference.navidrome_connection_id)
            if preference and preference.navidrome_connection_id
            else None
        )
        if stored_user is None or preference is None:
            raise HTTPException(status_code=404, detail="User settings are unavailable")
        model_settings = session.execute(
            text(
                """SELECT karaoke_processing_enabled, hum_processing_enabled, transcription_processing_enabled,
                      karaoke_modal_enabled, hum_modal_enabled, transcription_modal_enabled,
                      hf_token_account, hf_token_encrypted IS NOT NULL
               FROM analysis_settings WHERE singleton=true"""
            )
        ).one_or_none()
        karaoke_enabled = True if model_settings is None else bool(model_settings[0])
        hum_enabled = True if model_settings is None else bool(model_settings[1])
        return {
            "profile": {
                "username": stored_user.username,
                "email": stored_user.email,
                "display_name": stored_user.display_name,
                "is_admin": stored_user.is_admin,
            },
            "models": {
                "karaoke_processing_enabled": karaoke_enabled,
                "hum_processing_enabled": hum_enabled,
                "transcription_processing_enabled": bool(model_settings and model_settings[2]),
                # The same switches for syncs that run on Modal.
                "karaoke_modal_enabled": True
                if model_settings is None
                else bool(model_settings[3]),
                "hum_modal_enabled": True if model_settings is None else bool(model_settings[4]),
                "transcription_modal_enabled": bool(model_settings and model_settings[5]),
                # Gated model downloads, here and on Modal (only the account name, never the token).
                "huggingface_account": model_settings[6] if model_settings else None,
                "has_huggingface_token": bool(model_settings and model_settings[7]),
            },
            "timezone": preference.timezone,
            "navidrome": None
            if connection is None
            else {
                "id": str(connection.id),
                "url": connection.url,
                "username": connection.username,
            },
            "lastfm": {
                "connected": bool(
                    preference.lastfm_username and preference.lastfm_api_key_encrypted
                ),
                "username": preference.lastfm_username,
            },
        }


@app.put("/settings/models/karaoke")
def update_karaoke_processing_settings(
    request: KaraokeProcessingSettingsRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Administrator access required")
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        column = (
            "karaoke_modal_enabled" if request.location == "modal" else "karaoke_processing_enabled"
        )
        cursor.execute(
            f"""INSERT INTO analysis_settings
                 (singleton, {column}, karaoke_bound_to_synced_lines, updated_at)
               VALUES (true,%s,false,now()) ON CONFLICT (singleton) DO UPDATE
               SET {column}=EXCLUDED.{column},
                   karaoke_bound_to_synced_lines=false, updated_at=now()""",
            (request.enabled,),
        )
        pending = 0
        if request.enabled:
            cursor.execute(
                """SELECT count(*) AS pending FROM lyrics l
                   WHERE l.text IS NOT NULL AND coalesce((l.provenance->>'synced')::boolean,false)
                     AND NOT EXISTS (SELECT 1 FROM karaoke_lyrics_variants kv
                                     WHERE kv.track_id=l.track_id AND kv.bounded=false
                                       AND kv.provenance->>'pipeline_revision'=%s)""",
                (KARAOKE_PIPELINE_REVISION,),
            )
            pending = int(cursor.fetchone()["pending"])
    return {"enabled": request.enabled, "location": request.location, "pending": pending}


@app.put("/settings/models/huggingface")
def update_huggingface_token(
    request: HuggingFaceTokenRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    """Store the Hugging Face token for gated model downloads, after checking it."""
    user = _session_user(echora_session)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Administrator access required")
    from worker.models.huggingface_token import TokenRejected, save

    # Never echo the token or Hugging Face's response.
    try:
        account = save(request.token.get_secret_value())
    except TokenRejected:
        raise HTTPException(
            status_code=422, detail="Hugging Face did not accept this token"
        ) from None
    except httpx.HTTPError:
        raise HTTPException(
            status_code=503, detail="Could not reach Hugging Face to check the token. Try again."
        ) from None
    return {"has_token": account is not None, "account": account}


@app.put("/settings/models/transcription")
def update_transcription_processing_settings(
    request: KaraokeProcessingSettingsRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Administrator access required")
    with (
        psycopg.connect(DatabaseConfig.load().database_url) as connection,
        connection.cursor() as cursor,
    ):
        column = (
            "transcription_modal_enabled"
            if request.location == "modal"
            else "transcription_processing_enabled"
        )
        cursor.execute(
            f"""INSERT INTO analysis_settings (singleton, {column}, updated_at)
               VALUES (true,%s,now()) ON CONFLICT (singleton) DO UPDATE
               SET {column}=EXCLUDED.{column}, updated_at=now()""",
            (request.enabled,),
        )
    return {"enabled": request.enabled, "location": request.location}


@app.put("/settings/models/hum")
def update_hum_processing_settings(
    request: HumProcessingSettingsRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Administrator access required")
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        column = "hum_modal_enabled" if request.location == "modal" else "hum_processing_enabled"
        cursor.execute(
            f"""INSERT INTO analysis_settings
                 (singleton, karaoke_bound_to_synced_lines, {column}, updated_at)
               VALUES (true,false,%s,now()) ON CONFLICT (singleton) DO UPDATE
               SET {column}=EXCLUDED.{column},
                   karaoke_bound_to_synced_lines=false, updated_at=now()""",
            (request.enabled,),
        )
        pending = 0
        if request.enabled:
            cursor.execute(
                """SELECT count(DISTINCT ts.track_id) AS pending
                   FROM track_sources ts
                   WHERE ts.source_type='subsonic'
                     AND NOT EXISTS (
                       SELECT 1 FROM melody_contours mc
                       JOIN analysis_runs ar ON ar.id=mc.run_id
                       WHERE mc.track_id=ts.track_id
                         AND ar.model_name='melody_contour' AND ar.model_revision=%s
                     )""",
                (MELODY_CONTOUR_REVISION,),
            )
            pending = int(cursor.fetchone()["pending"])
    return {"enabled": request.enabled, "location": request.location, "pending": pending}


@app.put("/settings/profile")
def update_profile(
    request: ProfileSettingsRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, str]:
    user = _session_user(echora_session)
    with session_scope() as session:
        stored = session.get(User, user["id"])
        if stored is None:
            raise HTTPException(status_code=404, detail="User not found")
        stored.display_name = request.display_name.strip()
    return {"display_name": request.display_name.strip()}


@app.put("/settings/timezone")
def update_timezone(
    request: TimezoneSettingsRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, str]:
    user = _session_user(echora_session)
    try:
        ZoneInfo(request.timezone)
    except ZoneInfoNotFoundError as error:
        raise HTTPException(status_code=422, detail="Unknown timezone") from error
    with session_scope() as session:
        preference = session.get(UserPreference, user["id"])
        if preference is None:
            raise HTTPException(status_code=404, detail="User preferences are unavailable")
        preference.timezone = request.timezone
        preference.updated_at = datetime.now(timezone.utc)
    return {"timezone": request.timezone}


@app.put("/settings/navidrome")
def update_navidrome(
    request: Credentials, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    credentials = _credentials(request)
    try:
        with NavidromeClient(*credentials) as client:
            version = client.ping()
    except Exception as error:
        logger.warning("Navidrome connection test failed for %s: %s", request.url, error)
        raise HTTPException(status_code=422, detail="Could not connect to Navidrome") from error
    connection_id = uuid.UUID(_save_connection(credentials, user["id"]))
    with session_scope() as session:
        preference = session.get(UserPreference, user["id"])
        if preference is None:
            raise HTTPException(status_code=404, detail="User preferences are unavailable")
        preference.navidrome_connection_id = connection_id
        preference.updated_at = datetime.now(timezone.utc)
    attach_user_library(user["id"], credentials[0])
    return {
        "id": connection_id,
        "url": credentials[0],
        "username": credentials[1],
        "server": version,
    }


@app.put("/settings/lastfm")
def update_lastfm(
    request: LastFmSettingsRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    api_key = request.api_key.get_secret_value().strip()
    try:
        response = httpx.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "user.getrecenttracks",
                "user": request.username.strip(),
                "api_key": api_key,
                "format": "json",
                "limit": 1,
            },
            timeout=LastfmConfig.load().validation_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError(str(payload.get("message") or "Last.fm rejected these settings"))
        total = int(payload.get("recenttracks", {}).get("@attr", {}).get("total") or 0)
    except Exception as error:
        logger.warning("Last.fm verification failed for %s: %s", request.username, error)
        raise HTTPException(
            status_code=422, detail="Could not verify the Last.fm account"
        ) from error
    with session_scope() as session:
        preference = session.get(UserPreference, user["id"])
        if preference is None:
            raise HTTPException(status_code=404, detail="User preferences are unavailable")
        preference.lastfm_username = request.username.strip()
        preference.lastfm_api_key_encrypted = credential_cipher().encrypt(api_key.encode())
        preference.updated_at = datetime.now(timezone.utc)
    return {"connected": True, "username": request.username.strip(), "scrobbles": total}


@app.delete("/settings/lastfm", status_code=204)
def disconnect_lastfm(
    response: Response, echora_session: str | None = Cookie(default=None)
) -> Response:
    user = _session_user(echora_session)
    with session_scope() as session:
        preference = session.get(UserPreference, user["id"])
        if preference is not None:
            preference.lastfm_username = None
            preference.lastfm_api_key_encrypted = None
            preference.updated_at = datetime.now(timezone.utc)
    return response


def _admin_user(echora_session: str | None) -> dict[str, object]:
    user = _session_user(echora_session)
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Administrator access is required")
    return user


@app.get("/settings/runtime")
def runtime_settings(response: Response) -> dict[str, int]:
    # Only this explicit, non-secret allowlist is browser-visible.
    response.headers["Cache-Control"] = "no-store"
    return UiConfig.load().public_ui()


@app.get("/settings/oidc")
def oidc_admin_settings(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    _admin_user(echora_session)
    with session_scope() as session:
        policy = session.get(OidcSetting, True)
        users = session.scalars(
            select(User).where(User.email.is_not(None)).order_by(func.lower(User.email))
        ).all()
        allowed = session.scalars(select(OidcAllowedEmail).order_by(OidcAllowedEmail.email)).all()
        return {
            "configured": _oauth.create_client("oidc") is not None,
            "issuer": _oidc_issuer or None,
            "require_verified_email": OidcConfig.load().require_verified_email,
            "auto_provision": policy.auto_provision if policy else True,
            "users": [
                {
                    "id": str(item.id),
                    "email": item.email,
                    "display_name": item.display_name,
                    "is_admin": item.is_admin,
                    "is_blocked": item.is_blocked,
                }
                for item in users
            ],
            "allowed_emails": [item.email for item in allowed],
        }


@app.put("/settings/oidc/policy")
def update_oidc_policy(
    request: OidcPolicyRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, bool]:
    _admin_user(echora_session)
    with session_scope() as session:
        policy = session.get(OidcSetting, True)
        if policy is None:
            policy = OidcSetting(singleton=True)
            session.add(policy)
        policy.auto_provision = request.auto_provision
        policy.updated_at = datetime.now(timezone.utc)
    return {"auto_provision": request.auto_provision}


@app.post("/settings/oidc/allowed-emails", status_code=201)
def allow_oidc_email(
    request: OidcAllowRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, str]:
    admin = _admin_user(echora_session)
    email = request.email.strip().casefold()
    if "@" not in email:
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    with session_scope() as session:
        if session.scalar(select(User.id).where(func.lower(User.email) == email)):
            raise HTTPException(status_code=409, detail="This user already exists")
        if session.get(OidcAllowedEmail, email) is None:
            session.add(OidcAllowedEmail(email=email, created_by=admin["id"]))
    return {"email": email}


@app.delete("/settings/oidc/allowed-emails/{email}", status_code=204)
def remove_allowed_oidc_email(
    email: str, response: Response, echora_session: str | None = Cookie(default=None)
) -> Response:
    _admin_user(echora_session)
    with session_scope() as session:
        allowed = session.get(OidcAllowedEmail, email.casefold())
        if allowed is not None:
            session.delete(allowed)
    return response


@app.patch("/settings/oidc/users/{user_id}")
def update_oidc_user(
    user_id: uuid.UUID,
    request: OidcUserUpdateRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    admin = _admin_user(echora_session)
    values = request.model_dump(exclude_none=True)
    with session_scope() as session:
        target = session.get(User, user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="User not found")
        if user_id == admin["id"] and (
            values.get("is_admin") is False or values.get("is_blocked") is True
        ):
            raise HTTPException(
                status_code=422, detail="You cannot demote or block your own account"
            )
        if target.is_admin and values.get("is_admin") is False:
            admin_count = session.scalar(
                select(func.count())
                .select_from(User)
                .where(User.is_admin, User.is_blocked.is_(False))
            )
            if admin_count <= 1:
                raise HTTPException(
                    status_code=422, detail="Echora must retain at least one active administrator"
                )
        for key, value in values.items():
            setattr(target, key, value)
        if values.get("is_blocked") is True:
            session.execute(delete(UserSession).where(UserSession.user_id == user_id))
        return {"id": str(target.id), **values}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "echora-analysis"}


@app.get("/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "models": {
            "muq_mulan": {"status": "available", "audio": True, "text": True},
            "mert": {"status": "available", "audio": True, "text": False},
            "heartclap": {"status": "blocked-upstream-release", "audio": False, "text": False},
        }
    }


@app.post("/navidrome/discover", dependencies=[Depends(require_user)])
def discover(
    request: DiscoverRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    try:
        with NavidromeClient(*_credentials(request)) as client:
            version = client.ping()
            tracks = client.random_tracks(request.limit)
    except Exception as error:
        logger.warning("Navidrome discovery failed for %s: %s", request.url, error)
        raise HTTPException(
            status_code=400, detail="Could not connect to the Navidrome server"
        ) from error
    connection_id = _save_connection(_credentials(request), user["id"])
    return {
        "connection_id": connection_id,
        "server": {"version": version, "url": str(request.url).rstrip("/")},
        "tracks": [_track_summary(track) for track in tracks],
    }


def _track_summary(track) -> dict[str, object]:
    return {
        "id": track.id,
        "title": track.title,
        "artist": track.artist,
        "album": track.album,
        "duration": track.duration,
        "genre": track.genre,
        "cover_art": track.raw.get("coverArt"),
    }


class SampleImportRequest(BaseModel):
    track_ids: list[str] = Field(min_length=1, max_length=20000)


@app.get("/navidrome/connections/{connection_id}/sample", dependencies=[Depends(require_user)])
def navidrome_sample(
    connection_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    """Random tracks from a saved connection, so onboarding needs no password after connecting."""
    user = _session_user(echora_session)
    credentials = load_connection(connection_id, user["id"])
    if credentials is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    try:
        with NavidromeClient(*credentials) as client:
            tracks = client.random_tracks(limit)
    except Exception as error:
        raise HTTPException(
            status_code=502, detail="Could not read tracks from the Navidrome server"
        ) from error
    return {"tracks": [_track_summary(track) for track in tracks]}


@app.post(
    "/navidrome/connections/{connection_id}/import",
    status_code=202,
    dependencies=[Depends(require_user)],
)
def start_connection_import(
    connection_id: str,
    request: SampleImportRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    """Analyze chosen tracks of a saved connection: the onboarding sample."""
    user = _session_user(echora_session)
    track_ids = sorted(set(request.track_ids))
    selection = hashlib.sha256("\n".join(track_ids).encode()).hexdigest()
    if load_connection(connection_id, user["id"]) is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return jobs.enqueue(
        "import",
        "analysis",
        user["id"],
        connection_id=connection_id,
        payload={"track_ids": track_ids},
        dedupe_key=f"import:{user['id']}:{connection_id}:{selection}",
    )


@app.get("/navidrome/connections/{connection_id}/catalog", dependencies=[Depends(require_user)])
def navidrome_catalog(
    connection_id: str, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    credentials = load_connection(connection_id, user["id"])
    if credentials is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    try:
        with NavidromeClient(*credentials) as client:
            return client.catalog()
    except Exception as error:
        raise HTTPException(
            status_code=502, detail="Could not read the Navidrome catalog"
        ) from error


@app.get("/navidrome/connections/{connection_id}/sync/status", dependencies=[Depends(require_user)])
def navidrome_sync_status(
    connection_id: str, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    credentials = load_connection(connection_id, user["id"])
    if credentials is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    try:
        with NavidromeClient(*credentials) as client:
            tracks = client.all_tracks()
    except Exception as error:
        raise HTTPException(
            status_code=502, detail="Could not scan the Navidrome library"
        ) from error
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, credentials[0].rstrip("/"))
    with (
        psycopg.connect(DatabaseConfig.load().database_url) as connection,
        connection.cursor() as cursor,
    ):
        # Heal a renumbered library here too, so it never shows as unanalyzed.
        relink_renumbered_sources(connection, credentials[0], tracks)
        cursor.execute(
            """
            SELECT ts.external_id FROM track_sources ts
            JOIN libraries l ON l.id=ts.library_id
            WHERE l.namespace=%s AND ts.source_type='subsonic'
            """,
            (namespace,),
        )
        processed = {str(row[0]) for row in cursor.fetchall()}
        from worker.recordings.search import index_status

        recording_index = index_status(
            connection, namespace, [track.id for track in tracks], user["id"]
        )
    missing = [track for track in tracks if track.id not in processed]
    return {
        "server": credentials[0],
        "total": len(tracks),
        "processed": len(tracks) - len(missing),
        "missing": len(missing),
        "recording_index": recording_index,
        "tracks": [
            {
                "id": track.id,
                "title": track.title,
                "artist": track.artist,
                "album": track.album,
                "duration": track.duration,
                "cover_art": track.raw.get("coverArt"),
            }
            for track in missing[:50]
        ],
    }


def _enqueue_connection_job(
    kind: str, connection_id: str, user: dict[str, object], payload: dict[str, object] | None = None
) -> dict[str, object]:
    # Validate ownership without putting decrypted credentials into job payloads.
    if load_connection(connection_id, user["id"]) is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return jobs.enqueue(
        kind,
        "analysis",
        user["id"],
        connection_id=connection_id,
        payload=payload or {},
        dedupe_key=f"{kind}:{user['id']}:{connection_id}",
    )


@app.post("/navidrome/connections/{connection_id}/sync", status_code=202)
def start_navidrome_sync(
    connection_id: str,
    request: SyncRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    try:
        compute = compute_for(user["id"], request.compute)
    except PermissionError:
        raise HTTPException(
            status_code=403, detail="Modal processing is not available to you"
        ) from None
    return _enqueue_connection_job(
        "navidrome_sync",
        connection_id,
        user,
        payload={
            "mode": request.mode,
            "verify_audio_hashes": request.verify_audio_hashes,
            "compute": compute,
        },
    )


@app.post(
    "/navidrome/connections/{connection_id}/recordings/backfill",
    status_code=202,
    dependencies=[Depends(require_user)],
)
def start_recording_backfill(
    connection_id: str,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    return _enqueue_connection_job(
        "recordings_backfill", connection_id, _session_user(echora_session)
    )


@app.get("/library/lyrics/status", dependencies=[Depends(require_user)])
def lyrics_status(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT count(*) AS total,
                      count(*) FILTER (WHERE l.availability_status='available') AS available,
                      count(*) FILTER (WHERE l.availability_status='missing') AS missing,
                      count(*) FILTER (WHERE l.availability_status='unavailable') AS unavailable,
                      count(*) FILTER (WHERE l.availability_status='instrumental') AS instrumental,
                      count(*) FILTER (WHERE EXISTS (
                        SELECT 1 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                        WHERE e.track_id=t.id AND e.embedding_type='lyrics' AND e.window_index IS NULL
                          AND ar.model_name='bge_m3'
                      )) AS embedded
               FROM tracks t LEFT JOIN lyrics l ON l.track_id=t.id"""
        )
        return cursor.fetchone()


@app.post(
    "/navidrome/connections/{connection_id}/lyrics/backfill",
    status_code=202,
    dependencies=[Depends(require_user)],
)
def start_lyrics_backfill(
    connection_id: str,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    return _enqueue_connection_job("lyrics_backfill", connection_id, _session_user(echora_session))


@app.post(
    "/navidrome/connections/{connection_id}/voice/backfill",
    status_code=202,
    dependencies=[Depends(require_user)],
)
def start_voice_backfill(
    connection_id: str,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    return _enqueue_connection_job("voice_backfill", connection_id, _session_user(echora_session))


@app.post(
    "/navidrome/connections/{connection_id}/lyrics/karaoke/backfill",
    status_code=202,
    dependencies=[Depends(require_user)],
)
def start_karaoke_backfill(
    connection_id: str,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    return _enqueue_connection_job("karaoke_backfill", connection_id, _session_user(echora_session))


@app.post("/library/semantic-fusion/rebuild", status_code=202, dependencies=[Depends(require_user)])
def start_semantic_fusion_build(
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    return jobs.enqueue(
        "semantic_fusion_build",
        "analysis",
        _session_user(echora_session)["id"],
        dedupe_key="semantic_fusion_build",
    )


@app.get("/library/semantic-fusion/status", dependencies=[Depends(require_user)])
def semantic_fusion_status(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT count(*) AS fused,
                      (SELECT count(*) FROM current_embeddings l WHERE l.embedding_type='lyrics'
                        AND l.window_index IS NULL) AS lyrics,
                      (SELECT count(*) FROM current_embeddings a WHERE a.embedding_type='audio-track'
                        AND a.window_index IS NULL) AS audio
               FROM current_embeddings e WHERE e.embedding_type='semantic_fusion'"""
        )
        return cursor.fetchone()


@app.get("/library/tracks/{track_id}/similar", dependencies=[Depends(require_user)])
def similar_tracks(
    track_id: uuid.UUID,
    limit: int = 20,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT e.embedding FROM current_embeddings e
               WHERE e.track_id=%s AND e.embedding_type='semantic_fusion'""",
            (track_id,),
        )
        seed = cursor.fetchone()
        if seed is None:
            raise HTTPException(status_code=404, detail="Track has no semantic fusion vector")
        cursor.execute(
            """SELECT t.id, t.title, t.artist, t.album,
                      1 - (e.embedding <=> %s::vector) AS similarity
               FROM current_embeddings e
               JOIN tracks t ON t.id=e.track_id
               WHERE e.embedding_type='semantic_fusion' AND e.track_id != %s
               ORDER BY e.embedding <=> %s::vector LIMIT %s""",
            (seed["embedding"], track_id, seed["embedding"], max(1, min(limit, 100))),
        )
        return {"track_id": str(track_id), "mode": "semantic_fusion", "results": cursor.fetchall()}


@app.get("/library/tracks/{track_id}/recording-group", dependencies=[Depends(require_user)])
def track_recording_group(
    track_id: uuid.UUID,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT rg.id, rg.status, rg.canonical_track_id, rg.created_at, rg.updated_at
               FROM recording_group_members member JOIN recording_groups rg ON rg.id=member.group_id
               WHERE member.track_id=%s""",
            (track_id,),
        )
        group = cursor.fetchone()
        if group is None:
            cursor.execute("SELECT 1 FROM track_fingerprints WHERE track_id=%s", (track_id,))
            return {"group": None, "fingerprinted": cursor.fetchone() is not None}
        cursor.execute(
            """SELECT t.id, t.title, t.artist, t.album, t.duration_seconds, member.confidence,
                      member.membership_status, array_agg(DISTINCT ts.external_id) AS source_ids
               FROM recording_group_members member JOIN tracks t ON t.id=member.track_id
               LEFT JOIN track_sources ts ON ts.track_id=t.id WHERE member.group_id=%s
               GROUP BY t.id, member.confidence, member.membership_status ORDER BY t.title""",
            (group["id"],),
        )
        members = cursor.fetchall()
        cursor.execute(
            """SELECT id, left_track_id, right_track_id, decision, chromaprint_score,
                      duration_delta_seconds, semantic_similarity, acoustic_similarity,
                      matcher_revision, created_at FROM recording_match_evidence
               WHERE left_track_id=ANY(%s) AND right_track_id=ANY(%s) ORDER BY created_at DESC""",
            ([member["id"] for member in members], [member["id"] for member in members]),
        )
        evidence = cursor.fetchall()
    return {"group": {**group, "members": members, "evidence": evidence}, "fingerprinted": True}


@app.get("/library/tracks/{track_id}/audio-descriptors")
def track_audio_descriptors(
    track_id: uuid.UUID,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s",
            (user["id"], track_id),
        )
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Track not found")
        cursor.execute(
            """SELECT revision, status, descriptors, created_at FROM track_audio_descriptors
               WHERE track_id=%s AND revision=%s""",
            (track_id, DESCRIPTOR_REVISION),
        )
        descriptors = cursor.fetchone()
        cursor.execute(
            """SELECT activity.activity, activity.run_id
               FROM track_vocal_activity activity
               JOIN current_analysis_runs ar ON ar.id=activity.run_id
               WHERE activity.track_id=%s AND ar.kind='voice_classification'
               ORDER BY activity.created_at DESC LIMIT 1""",
            (track_id,),
        )
        vocal = cursor.fetchone()
    return {
        "track_id": str(track_id),
        "status": descriptors["status"] if descriptors else "pending",
        "descriptors": descriptors,
        "vocal_activity": vocal,
        "used_in_curation": {"sound_profile": True},
    }


@app.get("/library/tracks/{track_id}/waveform")
def track_waveform(
    track_id: uuid.UUID, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s",
            (user["id"], track_id),
        )
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Track not found")
        cursor.execute(
            "SELECT revision, duration_seconds, peaks FROM track_waveforms WHERE track_id=%s AND revision=%s",
            (track_id, WAVEFORM_REVISION),
        )
        waveform = cursor.fetchone()
        cursor.execute(
            """SELECT mc.source, mc.pitch, mc.voiced, mc.hop_seconds
               FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
               WHERE mc.track_id=%s AND ar.status IN ('complete','running')
               ORDER BY CASE mc.source WHEN 'full-mix' THEN 0 WHEN 'vocals' THEN 1 ELSE 2 END,
                        ar.created_at DESC LIMIT 1""",
            (track_id,),
        )
        contour = cursor.fetchone()
    melody = (
        melody_preview(
            contour["source"], contour["pitch"], contour["voiced"], float(contour["hop_seconds"])
        )
        if contour
        else None
    )
    return {
        "track_id": str(track_id),
        "status": "complete" if waveform else "pending",
        "waveform": waveform,
        "melody": melody,
    }


@app.get("/library/tracks/{track_id}/visual-features")
def track_visual_features(
    track_id: uuid.UUID, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s",
            (user["id"], track_id),
        )
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Track not found")
        cursor.execute(
            """SELECT revision, status, duration_seconds, hop_seconds, features, created_at
               FROM track_visual_features WHERE track_id=%s AND revision=%s""",
            (track_id, VISUAL_FEATURE_REVISION),
        )
        cache = cursor.fetchone()
        # Read enrichment at request time: ingest ordering never requires a DSP
        # rerun, and this authenticated endpoint never schedules model work.
        cursor.execute(
            """SELECT revision, status, descriptors, created_at FROM track_audio_descriptors
               WHERE track_id=%s AND revision=%s""",
            (track_id, DESCRIPTOR_REVISION),
        )
        descriptors = cursor.fetchone()
        cursor.execute(
            """SELECT activity.activity, activity.run_id FROM track_vocal_activity activity
               JOIN current_analysis_runs ar ON ar.id=activity.run_id
               WHERE activity.track_id=%s AND ar.kind='voice_classification'
               ORDER BY activity.created_at DESC LIMIT 1""",
            (track_id,),
        )
        vocal = cursor.fetchone()
        cursor.execute(
            """SELECT mc.source, mc.pitch, mc.voiced, mc.hop_seconds
               FROM melody_contours mc JOIN analysis_runs ar ON ar.id=mc.run_id
               WHERE mc.track_id=%s AND ar.model_revision=%s AND ar.status IN ('complete','running')
               ORDER BY CASE mc.source WHEN 'full-mix' THEN 0 WHEN 'vocals' THEN 1 ELSE 2 END,
                        ar.created_at DESC LIMIT 1""",
            (track_id, MELODY_CONTOUR_REVISION),
        )
        contour = cursor.fetchone()
    melody = (
        melody_preview(
            contour["source"], contour["pitch"], contour["voiced"], float(contour["hop_seconds"])
        )
        if contour
        else None
    )
    status = cache["status"] if cache else "pending"
    features = cache if status == "complete" else None
    return {
        "track_id": str(track_id),
        "status": status,
        "visual_features": features,
        "enrichment": {"descriptors": descriptors, "vocal_activity": vocal, "melody": melody},
    }


@app.get("/library/tracks/{track_id}/audio-quality")
def track_audio_quality(
    track_id: uuid.UUID, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT ts.source_data->>'suffix' AS codec,
                      ts.source_data->>'contentType' AS content_type,
                      (ts.source_data->>'bitRate')::integer AS bit_rate_kbps,
                      (ts.source_data->>'bitDepth')::integer AS bit_depth,
                      (ts.source_data->>'samplingRate')::integer AS sample_rate_hz,
                      (ts.source_data->>'channelCount')::integer AS channels
               FROM user_track_links link
               JOIN track_sources ts ON ts.track_id=link.track_id AND ts.library_id=link.library_id
                                    AND ts.external_id=link.external_id
               WHERE link.user_id=%s AND link.track_id=%s
               ORDER BY ts.id LIMIT 1""",
            (user["id"], track_id),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Track is not in your library")
    codec = str(row.get("codec") or "").lower()
    return {**row, "lossless": codec in {"flac", "alac", "wav", "aiff", "ape", "wv"}}


def _track_info(source: dict, descriptors: dict | None, lyrics: dict | None, themes: dict | None):
    """The song-info panel's facts, from Navidrome's tags, audio analysis, lyrics and themes."""
    data = source.get("source_data") or {}

    def names(value):
        return [
            str(item.get("name") if isinstance(item, dict) else item)
            for item in (value or [])
            if (item.get("name") if isinstance(item, dict) else item)
        ]

    credits: dict[str, list[str]] = {}
    for contributor in data.get("contributors") or []:
        role = str(contributor.get("role") or "").strip()
        name = str((contributor.get("artist") or {}).get("name") or "").strip()
        if role and name and name not in credits.setdefault(role, []):
            credits[role].append(name)
    codec = str(data.get("suffix") or "").lower()
    found = (descriptors or {}).get("descriptors") or {}
    key = found.get("key") or {}
    loudness = found.get("loudness") or {}
    provenance = (lyrics or {}).get("provenance") or {}
    return {
        "tags": {
            "year": data.get("year") or None,
            "genres": names(data.get("genres")) or ([data["genre"]] if data.get("genre") else []),
            "moods": names(data.get("moods")),
            "track": data.get("track") or None,
            "disc": data.get("discNumber") or None,
            "album_artists": names(data.get("albumArtists")),
            "credits": credits,
            "isrc": (data.get("isrc") or [None])[0]
            if isinstance(data.get("isrc"), list)
            else data.get("isrc"),
            "explicit": data.get("explicitStatus") == "explicit",
            "comment": data.get("comment") or None,
            "play_count": data.get("playCount"),
            "last_played": data.get("played"),
            "added": data.get("created"),
            "size_bytes": data.get("size"),
            "musicbrainz_id": data.get("musicBrainzId") or None,
        },
        "audio": {
            "codec": codec or None,
            "lossless": codec in {"flac", "alac", "wav", "aiff", "ape", "wv"},
            "bit_rate_kbps": data.get("bitRate"),
            "bit_depth": data.get("bitDepth"),
            "sample_rate_hz": data.get("samplingRate"),
            "channels": data.get("channelCount"),
            "replay_gain": data.get("replayGain") or None,
        },
        "analysis": {
            "bpm": (found.get("rhythm") or {}).get("bpm"),
            "key": f"{key['tonic']} {key.get('scale') or ''}".strip() if key.get("tonic") else None,
            "integrated_lufs": loudness.get("integrated_lufs"),
            "loudness_range_lu": loudness.get("range_lu"),
            "true_peak_dbtp": found.get("true_peak_dbtp"),
        },
        "lyrics": {
            "available": bool(lyrics and str(lyrics.get("text") or "").strip()),
            "language": (lyrics or {}).get("language"),
            "source": "ai" if provenance.get("ai_generated") else provenance.get("provider"),
            "synced": bool(provenance.get("synced")),
            "karaoke": bool((lyrics or {}).get("karaoke_lines")),
        },
        "themes": {"description": themes["description"], "tags": themes["themes"]}
        if themes
        else None,
    }


@app.get("/library/tracks/{track_id}/info")
def track_info(
    track_id: uuid.UUID, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT ts.source_data FROM user_track_links link
               JOIN track_sources ts ON ts.track_id=link.track_id AND ts.library_id=link.library_id
                                    AND ts.external_id=link.external_id
               WHERE link.user_id=%s AND link.track_id=%s ORDER BY ts.id LIMIT 1""",
            (user["id"], track_id),
        )
        source = cursor.fetchone()
        if source is None:
            raise HTTPException(status_code=404, detail="Track is not in your library")
        cursor.execute(
            """SELECT descriptors FROM track_audio_descriptors
               WHERE track_id=%s AND status IN ('complete', 'partial')
               ORDER BY created_at DESC LIMIT 1""",
            (track_id,),
        )
        descriptors = cursor.fetchone()
        cursor.execute(
            """SELECT text, language, provenance, karaoke_lines IS NOT NULL
                      AND jsonb_array_length(karaoke_lines) > 0 AS karaoke_lines
               FROM lyrics WHERE track_id=%s""",
            (track_id,),
        )
        lyrics = cursor.fetchone()
        # Only a description written for these lyrics and the current prompt.
        cursor.execute(
            """SELECT th.description, th.themes FROM track_lyric_themes th
               JOIN lyrics ly ON ly.track_id=th.track_id
               WHERE th.track_id=%s AND th.status='ready' AND th.prompt_revision=%s
                 AND th.source_checksum=encode(sha256(convert_to(ly.text, 'UTF8')), 'hex')""",
            (track_id, LYRIC_THEMES_REVISION),
        )
        themes = cursor.fetchone()
    return {"track_id": str(track_id), **_track_info(source, descriptors, lyrics, themes)}


@app.get("/library/tracks/{track_id}/lyrics")
def track_lyrics(
    track_id: uuid.UUID, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT l.text, l.language, l.source, l.provenance, l.availability_status,
                      karaoke.lines AS karaoke_lines, karaoke.ass AS karaoke_ass,
                      karaoke.lrc AS karaoke_lrc, karaoke.model AS karaoke_model,
                      karaoke.model_revision AS karaoke_model_revision,
                      karaoke.bounded AS karaoke_bounded
               FROM user_track_links link
               LEFT JOIN lyrics l ON l.track_id=link.track_id
               LEFT JOIN LATERAL (
                 SELECT kv.* FROM karaoke_lyrics_variants kv
                 WHERE kv.track_id=link.track_id AND kv.bounded=false LIMIT 1
               ) karaoke ON true
               WHERE link.user_id=%s AND link.track_id=%s LIMIT 1""",
            (user["id"], track_id),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Track is not in your library")
    provenance = row.get("provenance") or {}
    source_lines = provenance.get("lines") or []
    karaoke_lines = row.get("karaoke_lines") or []
    from core.lyrics.translations import load_translations
    from worker.translation.pipeline import source_lines as translation_source_lines

    original_lines = translation_source_lines(row)
    translations = [
        {
            "source_language": item["source_language"],
            "target_language": item["target_language"],
            "lines": item["lines"],
            "source_lines": original_lines,
        }
        for item in load_translations(track_id, original_lines)
        # A mixed-language song's translation covers only its foreign lines; one where the model
        # left every line as it was has nothing to show.
        if item["status"] == "ready" and any(line.get("text", "").strip() for line in item["lines"])
    ]
    return {
        "available": bool(row.get("text")),
        "availability_status": row.get("availability_status"),
        **row,
        "lines": karaoke_lines or source_lines,
        "karaoke": bool(karaoke_lines),
        "translations": translations,
    }


@app.put("/library/tracks/{track_id}/lyrics", dependencies=[Depends(require_user)])
def update_track_lyrics(
    track_id: uuid.UUID,
    request: LyricsUpdateRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    lyric_text = request.text.strip()
    if not lyric_text:
        raise HTTPException(status_code=422, detail="Lyrics cannot be empty")
    language = request.language.strip().lower() if request.language else None
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s",
            (user["id"], track_id),
        )
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Track is not in your library")
        cursor.execute("SELECT text, language FROM lyrics WHERE track_id=%s", (track_id,))
        existing = cursor.fetchone()
        changed = existing is None or existing["text"] != lyric_text
        cursor.execute(
            """INSERT INTO lyrics (track_id, source, text, language, provenance, availability_status)
               VALUES (%s,'manual',%s,%s,jsonb_build_object('manual',true,'edited_at',now()::text),'available')
               ON CONFLICT (track_id) DO UPDATE SET source='manual', text=EXCLUDED.text,
                 language=EXCLUDED.language, provenance=(CASE WHEN lyrics.text IS DISTINCT FROM EXCLUDED.text THEN
                     coalesce(lyrics.provenance,'{}'::jsonb) - 'lines' - 'synced'
                   ELSE coalesce(lyrics.provenance,'{}'::jsonb) END)
                   || jsonb_build_object('manual',true,'edited_at',now()::text),
                 availability_status='available', created_at=now()""",
            (track_id, lyric_text, language),
        )
        if changed:
            cursor.execute("DELETE FROM karaoke_lyrics_variants WHERE track_id=%s", (track_id,))
            cursor.execute(
                "DELETE FROM embeddings WHERE track_id=%s AND embedding_type='lyrics'", (track_id,)
            )
    return {
        "track_id": str(track_id),
        "karaoke_pending": changed,
        "lyrics_embedding_pending": changed,
    }


@app.put(
    "/library/tracks/{track_id}/lyrics/transcription-language", dependencies=[Depends(require_user)]
)
def force_transcription_language(
    track_id: uuid.UUID,
    request: TranscriptionLanguageRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    language = request.language.strip()
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s",
            (user["id"], track_id),
        )
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Track is not in your library")
        # The track is transcribed by the next sync that runs where generation is on.
        cursor.execute(
            """SELECT transcription_processing_enabled OR transcription_modal_enabled AS enabled
               FROM analysis_settings WHERE singleton=true"""
        )
        setting = cursor.fetchone()
        if not setting or not setting["enabled"]:
            raise HTTPException(
                status_code=409,
                detail="Enable AI lyric generation in Settings before forcing transcription",
            )
        cursor.execute(
            """INSERT INTO lyrics (track_id, source, text, provenance, availability_status)
               VALUES (%s,'none',NULL,jsonb_build_object('transcription_language',%s::text,'forced_transcription',true),'missing')
               ON CONFLICT (track_id) DO UPDATE SET source='none', text=NULL, language=NULL,
                 provenance=(coalesce(lyrics.provenance,'{}'::jsonb) - 'manual' - 'lines' - 'synced')
                   || jsonb_build_object('transcription_language',%s::text,'forced_transcription',true),
                 availability_status='missing', created_at=now()""",
            (track_id, language, language),
        )
        cursor.execute("DELETE FROM karaoke_lyrics_variants WHERE track_id=%s", (track_id,))
        cursor.execute(
            "DELETE FROM embeddings WHERE track_id=%s AND embedding_type='lyrics'", (track_id,)
        )
    return {"track_id": str(track_id), "language": language, "transcription_pending": True}


@app.put("/library/tracks/{track_id}/lyrics/status", dependencies=[Depends(require_user)])
def update_track_lyrics_status(
    track_id: uuid.UUID,
    request: LyricsStatusRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT 1 FROM user_track_links WHERE user_id=%s AND track_id=%s",
            (user["id"], track_id),
        )
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Track is not in your library")
        cursor.execute(
            """INSERT INTO lyrics (track_id, source, text, provenance, availability_status)
               VALUES (%s,'none',NULL,jsonb_build_object('manual_status',%s::text,'edited_at',now()::text),%s)
               ON CONFLICT (track_id) DO UPDATE SET source='none', text=NULL, language=NULL,
                 provenance=(coalesce(lyrics.provenance,'{}'::jsonb)
                   - 'manual' - 'lines' - 'synced' - 'forced_transcription' - 'transcription_language')
                   || jsonb_build_object('manual_status',EXCLUDED.availability_status,'edited_at',now()::text),
                 availability_status=EXCLUDED.availability_status, created_at=now()""",
            (track_id, request.status, request.status),
        )
        cursor.execute("DELETE FROM karaoke_lyrics_variants WHERE track_id=%s", (track_id,))
        cursor.execute(
            "DELETE FROM embeddings WHERE track_id=%s AND embedding_type='lyrics'", (track_id,)
        )
    return {"track_id": str(track_id), "status": request.status}


@app.get(
    "/navidrome/connections/{connection_id}/stream/{song_id}", dependencies=[Depends(require_user)]
)
def stream_track(
    connection_id: str,
    song_id: str,
    request: Request,
    echora_session: str | None = Cookie(default=None),
) -> Response:
    user = _session_user(echora_session)
    credentials = load_connection(connection_id, user["id"])
    if credentials is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    quality = request.query_params.get("quality", "original")
    if quality not in {"original", "320", "120"}:
        raise HTTPException(status_code=422, detail="Quality must be original, 320, or 120")
    bit_rate = None if quality == "original" else int(quality)
    cache = media_cache() if request.query_params.get("cache") == "player" else None
    key = cache_key("stream", user["id"], connection_id, song_id, quality)
    cached = cache.get(key) if cache else None
    range_header = request.headers.get("range")
    starts_at_zero = not range_header or range_header in {"bytes=0-", "bytes=0"}
    if cached is None:
        client = media_navidrome_client(*credentials)
        try:
            upstream = client.open_transcode_stream(song_id, bit_rate, range_header)
        except Exception as error:
            client.close()
            raise HTTPException(status_code=502, detail="Could not open this track") from error
        content_type = upstream.headers.get("content-type", "audio/mpeg")
        writer = cache.stream_writer(key, content_type) if cache and starts_at_zero else None

        def stream_and_cache():
            completed = False
            try:
                try:
                    for chunk in upstream.iter_raw(64 * 1024):
                        if writer:
                            writer.append(chunk)
                        yield chunk
                except httpx.RemoteProtocolError:
                    logger.warning("Navidrome closed %s while streaming", song_id)
                else:
                    completed = True
            finally:
                upstream.close()
                client.close()
                if writer:
                    if completed:
                        writer.commit()
                    else:
                        writer.abort()

        forwarded = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() in {"content-range", "accept-ranges"}
        }
        return StreamingResponse(
            stream_and_cache(),
            status_code=upstream.status_code,
            media_type=content_type,
            headers={
                **forwarded,
                "Cache-Control": "private, max-age=3600" if cache else "no-store",
            },
        )
    content, content_type = cached.content, cached.content_type
    total = len(content)
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
    if range_header and range_header.startswith("bytes="):
        try:
            specification = range_header[6:].split(",", 1)[0]
            start_text, end_text = specification.split("-", 1)
            if start_text:
                start = int(start_text)
                end = min(int(end_text) if end_text else total - 1, total - 1)
            else:
                suffix = min(int(end_text), total)
                start, end = total - suffix, total - 1
            if start < 0 or start >= total or end < start:
                raise ValueError
        except ValueError as error:
            raise HTTPException(
                status_code=416, detail="Requested audio range is unavailable"
            ) from error
        body = content[start : end + 1]
        headers.update(
            {"Content-Range": f"bytes {start}-{end}/{total}", "Content-Length": str(len(body))}
        )
        return Response(body, status_code=206, media_type=content_type, headers=headers)
    headers["Content-Length"] = str(total)
    return Response(content, media_type=content_type, headers=headers)


class ScrobbleRequest(BaseModel):
    connection_id: str
    song_id: str = Field(min_length=1, max_length=200)
    submission: bool = True


@app.post("/navidrome/scrobble", status_code=204, dependencies=[Depends(require_user)])
def scrobble_play(
    request: ScrobbleRequest, echora_session: str | None = Cookie(default=None)
) -> Response:
    """Report playback to the media server (Subsonic scrobble). Fire-and-forget
    from the player: failures must never interrupt listening, hence 204-empty
    on success and a bare 502 the client ignores on failure."""
    user = _session_user(echora_session)
    credentials = load_connection(request.connection_id, user["id"])
    if credentials is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    try:
        with media_navidrome_client(*credentials) as client:
            client.scrobble(request.song_id, request.submission)
    except Exception as error:
        raise HTTPException(status_code=502, detail="Scrobble failed") from error
    return Response(status_code=204)


@app.get(
    "/navidrome/connections/{connection_id}/cover/{cover_id:path}",
    dependencies=[Depends(require_user)],
)
def cover_art(
    connection_id: str,
    cover_id: str,
    request: Request,
    size: int = 160,
    echora_session: str | None = Cookie(default=None),
) -> Response:
    user = _session_user(echora_session)
    credentials = load_connection(connection_id, user["id"])
    if credentials is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    bounded_size = min(max(size, 64), 1600)
    cache = media_cache()
    key = cache_key("cover", user["id"], connection_id, cover_id, bounded_size)
    cached = cache.get(key) if cache else None
    if cached:
        return Response(
            content=cached.content,
            media_type=cached.content_type,
            headers={"Cache-Control": "private, max-age=3600"},
        )
    try:
        with media_navidrome_client(*credentials) as client:
            content, content_type = client.cover_art(cover_id, bounded_size)
    except Exception as error:
        raise HTTPException(status_code=404, detail="Artwork unavailable") from error
    if cache:
        cache.set(key, content, content_type)
    return Response(
        content=content, media_type=content_type, headers={"Cache-Control": "private, max-age=3600"}
    )


@app.post("/ingest/navidrome", status_code=202, dependencies=[Depends(require_user)])
def start_navidrome_ingest(
    request: IngestRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    connection_id = _save_connection(_credentials(request), user["id"])
    track_ids = sorted(set(request.track_ids))
    selection = hashlib.sha256("\n".join(track_ids).encode()).hexdigest()
    return jobs.enqueue(
        "import",
        "analysis",
        user["id"],
        connection_id=connection_id,
        payload={"track_ids": track_ids},
        dedupe_key=f"import:{user['id']}:{connection_id}:{selection}",
    )


@app.get("/library/hum/index", dependencies=[Depends(require_user)])
def hum_index_status(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT count(DISTINCT mc.track_id) AS indexed_tracks
               FROM melody_contours mc
               WHERE EXISTS (SELECT 1 FROM user_track_links utl
                             WHERE utl.user_id=%s AND utl.track_id=mc.track_id)""",
            (user["id"],),
        )
        indexed = int(cursor.fetchone()["indexed_tracks"])
    return {
        "status": "complete" if indexed else "missing",
        "indexed_tracks": indexed,
        "track_limit": indexed,
    }


@app.post("/library/hum/index", status_code=202, dependencies=[Depends(require_user)])
def start_hum_index(
    track_limit: int = DEFAULT_CORPUS_SIZE,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, str]:
    user = _session_user(echora_session)
    track_limit = min(max(track_limit, 1), 500)
    connection_id = user.get("navidrome_connection_id")
    credentials = load_connection(str(connection_id), user["id"]) if connection_id else None
    if credentials is None:
        raise HTTPException(status_code=409, detail="Connect Navidrome before building a hum index")
    return jobs.enqueue(
        "hum_corpus",
        "analysis",
        user["id"],
        connection_id=connection_id,
        payload={"track_limit": track_limit},
        dedupe_key=f"hum_corpus:{user['id']}",
    )


@app.post("/library/hum/search", dependencies=[Depends(require_user)])
async def hum_search(
    request: Request,
    limit: int = 10,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=422, detail="The recording is empty")
    if len(audio) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The recording exceeds 8 MB")
    try:
        return await run_in_threadpool(
            search_corpus,
            user["id"],
            audio,
            min(max(limit, 1), 25),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Hum search failed")
        raise HTTPException(
            status_code=500, detail="Hum search could not process this recording"
        ) from error


@app.get("/library/tracks")
def library_tracks(
    limit: int = 10,
    offset: int = 0,
    q: str = "",
    artist: str = "",
    album: str = "",
    sort_by: str = "name",
    sort_direction: str = "asc",
    echora_session: str | None = Cookie(default=None),
    track_id: uuid.UUID | None = None,
    vocals: list[str] | None = Query(default=None),
    language: list[str] | None = Query(default=None),
    lyrics: list[str] | None = Query(default=None),
    translation: list[str] | None = Query(default=None),
    genre: list[str] | None = Query(default=None),
    year_from: int | None = None,
    year_to: int | None = None,
) -> dict[str, object]:
    user = _session_user(echora_session)
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)
    sort_columns = {
        "name": "lower(t.title)",
        "artist": "lower(t.artist)",
        "released": "t.year",
        "date_added": "source.date_added",
    }
    sort_column = sort_columns.get(sort_by)
    if sort_column is None:
        raise HTTPException(status_code=422, detail="Unknown track sort order")
    if sort_direction not in {"asc", "desc"}:
        raise HTTPException(status_code=422, detail="Unknown sort direction")
    direction = sort_direction.upper()
    nulls = " NULLS LAST" if sort_by in {"artist", "released", "date_added"} else ""
    ordering = f"{sort_column} {direction}{nulls}, lower(t.title), t.id"
    clauses: list[str] = [
        "EXISTS (SELECT 1 FROM user_track_links visible_links "
        "WHERE visible_links.track_id=t.id AND visible_links.user_id=%s)"
    ]
    parameters: list[object] = [user["id"]]
    if track_id is not None:
        clauses.append("t.id = %s")
        parameters.append(track_id)
    if q.strip():
        clauses.append(
            "(t.title ILIKE %s OR t.artist ILIKE %s OR t.album ILIKE %s OR array_to_string(t.genres, ' ') ILIKE %s)"
        )
        parameters.extend([f"%{q.strip()}%"] * 4)
    if artist:
        clauses.append("t.artist = %s")
        parameters.append(artist)
    if album:
        clauses.append("t.album = %s")
        parameters.append(album)
    from .browse_filters import predicates

    filter_clauses, filter_parameters = predicates(
        vocals=vocals,
        language=language,
        lyrics=lyrics,
        translation=translation,
        genre=genre,
        year_from=year_from,
        year_to=year_to,
    )
    clauses.extend(filter_clauses)
    parameters.extend(filter_parameters)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with session_scope() as session:
        named_parameters = {f"p{index}": value for index, value in enumerate(parameters)}
        parameter_index = iter(named_parameters)
        named_where = where
        for name in parameter_index:
            named_where = named_where.replace("%s", f":{name}", 1)
        total = session.execute(
            text(f"SELECT count(*) AS total FROM tracks t {named_where}"), named_parameters
        ).scalar_one()
        source_user_parameter = "source_user_id"
        rows = (
            session.execute(
                text(f"""
            SELECT t.id, t.title, t.artist, t.album, t.year, t.duration_seconds,
                   t.genres, t.ingested_at, l.availability_status AS lyrics_status,
                   source.date_added,
                   (SELECT count(DISTINCT e.run_id) FROM current_embeddings e
                    WHERE e.track_id=t.id AND e.embedding_type='audio-track') AS embedding_runs,
                   source.external_id AS source_id, source.album_id, source.cover_art
            FROM tracks t
            LEFT JOIN lyrics l ON l.track_id=t.id
            LEFT JOIN LATERAL (
              SELECT ts.external_id, ts.source_data->>'albumId' AS album_id,
                     ts.source_data->>'coverArt' AS cover_art,
                     (ts.source_data->>'created')::timestamptz AS date_added
              FROM track_sources ts
              JOIN user_track_links source_links
                ON source_links.user_id=:source_user_id AND source_links.library_id=ts.library_id
               AND source_links.track_id=ts.track_id AND source_links.external_id=ts.external_id
              WHERE ts.track_id=t.id AND ts.source_type='subsonic'
              ORDER BY ts.id LIMIT 1
            ) source ON true
            {named_where}
            ORDER BY {ordering}
            LIMIT :limit OFFSET :offset
            """),
                {
                    **named_parameters,
                    source_user_parameter: user["id"],
                    "limit": limit,
                    "offset": offset,
                },
            )
            .mappings()
            .all()
        )
        tracks = [dict(row) for row in rows]
    return {"tracks": tracks, "total": total, "limit": limit, "offset": offset}


def _lyrics_concept_corpus(user_id: uuid.UUID) -> tuple[list[dict[str, object]], np.ndarray, str]:
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT DISTINCT ON (e.track_id) t.id, t.title, t.artist, t.album,
                      e.embedding::text AS embedding, ar.id AS run_id
               FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id JOIN tracks t ON t.id=e.track_id
               WHERE e.embedding_type='lyrics' AND e.window_index IS NULL AND ar.model_name='bge_m3'
                 AND EXISTS (SELECT 1 FROM user_track_links utl
                             WHERE utl.track_id=t.id AND utl.user_id=%s)
               ORDER BY e.track_id, ar.created_at DESC""",
            (user_id,),
        )
        rows = cursor.fetchall()
    if not rows:
        raise HTTPException(status_code=409, detail="No lyrics embeddings are available")
    matrix = np.stack([np.fromstring(row.pop("embedding").strip("[]"), sep=",") for row in rows])
    run_id = str(rows[0].pop("run_id"))
    return rows, matrix, run_id


def _semantic_concept_corpus(user_id: uuid.UUID) -> tuple[list[dict[str, object]], np.ndarray, str]:
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """
            SELECT DISTINCT ON (e.track_id) t.id, t.title, t.artist, t.album,
                   e.embedding::text AS embedding, ar.id AS run_id
            FROM current_embeddings e
            JOIN analysis_runs ar ON ar.id=e.run_id
            JOIN tracks t ON t.id=e.track_id
            WHERE e.embedding_type='audio-track' AND ar.model_name='muq_mulan'
              AND EXISTS (SELECT 1 FROM user_track_links utl
                          WHERE utl.track_id=t.id AND utl.user_id=%s)
            ORDER BY e.track_id, ar.created_at DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
    if not rows:
        raise HTTPException(status_code=409, detail="No MuQ-MuLan embeddings are available")
    matrix = np.stack([np.fromstring(row.pop("embedding").strip("[]"), sep=",") for row in rows])
    run_id = str(rows[0].pop("run_id"))
    return rows, matrix, run_id


@app.get("/library/concepts")
def library_concepts(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT id, name, description, positive_prompts, negative_prompts,
                      positive_track_ids, negative_track_ids, enabled, created_at, updated_at
               FROM concepts WHERE user_id=%s ORDER BY lower(name)""",
            (user["id"],),
        )
        personal = cursor.fetchall()
    return {"predefined": predefined_concepts(), "personal": personal}


@app.post("/library/concepts", status_code=201)
def create_concept(
    request: ConceptRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    if not request.positive_prompts and not request.positive_track_ids:
        raise HTTPException(
            status_code=422, detail="A concept needs a positive prompt or positive track"
        )
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """
            INSERT INTO concepts (user_id, name, description, positive_prompts, negative_prompts,
                                  positive_track_ids, negative_track_ids)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, name, description, positive_prompts, negative_prompts,
                      positive_track_ids, negative_track_ids, enabled, created_at, updated_at
            """,
            (
                user["id"],
                request.name.strip(),
                request.description.strip(),
                request.positive_prompts,
                request.negative_prompts,
                request.positive_track_ids,
                request.negative_track_ids,
            ),
        )
        return cursor.fetchone()


@app.post("/library/concepts/preview")
def preview_concept(
    request: ConceptPreviewRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    if not request.positive_prompts and not request.positive_track_ids:
        raise HTTPException(
            status_code=422, detail="A concept needs a positive prompt or positive track"
        )
    rows, matrix, run_id = _semantic_concept_corpus(user["id"])
    row_index = {str(row["id"]): index for index, row in enumerate(rows)}

    def examples(identifiers: list[uuid.UUID]) -> np.ndarray | None:
        indices = [
            row_index[str(identifier)] for identifier in identifiers if str(identifier) in row_index
        ]
        return matrix[indices] if indices else None

    raw, percentiles = score_concept(
        matrix,
        request.positive_prompts,
        request.negative_prompts,
        examples(request.positive_track_ids),
        examples(request.negative_track_ids),
    )
    ranked = np.argsort(raw)[::-1][: request.limit]
    return {
        "run_id": run_id,
        "corpus_size": len(rows),
        "calibration": "empirical-library-percentile",
        "tracks": [
            {**rows[index], "raw_score": float(raw[index]), "percentile": float(percentiles[index])}
            for index in ranked
        ],
    }


@app.post("/library/concepts/lens")
def concept_lens(
    request: ConceptLensRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    requested = list(dict.fromkeys(name.strip() for name in request.concepts if name.strip()))
    definitions = {str(item["name"]).casefold(): item for item in predefined_concepts()}
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT name, positive_prompts, negative_prompts, positive_track_ids, negative_track_ids
               FROM concepts WHERE user_id=%s AND enabled""",
            (user["id"],),
        )
        for item in cursor.fetchall():
            definitions[str(item["name"]).casefold()] = item
    missing = [name for name in requested if name.casefold() not in definitions]
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown concepts: {', '.join(missing)}")
    rows, matrix, run_id = (
        _lyrics_concept_corpus(user["id"])
        if request.representation == "lyrics"
        else _semantic_concept_corpus(user["id"])
    )
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    lyrics_matrix: np.ndarray | None = None
    lyrics_available: np.ndarray | None = None
    if request.representation == "hybrid":
        lyrics_rows, available_lyrics, lyrics_run_id = _lyrics_concept_corpus(user["id"])
        lyrics_by_id = {
            str(row["id"]): available_lyrics[index] for index, row in enumerate(lyrics_rows)
        }
        lyrics_matrix = np.zeros((len(rows), available_lyrics.shape[1]), dtype=np.float32)
        lyrics_available = np.asarray([str(row["id"]) in lyrics_by_id for row in rows], dtype=bool)
        for index, row in enumerate(rows):
            if lyrics_available[index]:
                lyrics_matrix[index] = lyrics_by_id[str(row["id"])]
        run_id = f"{run_id}+{lyrics_run_id}"
    row_index = {str(row["id"]): index for index, row in enumerate(rows)}
    result: dict[str, list[dict[str, object]]] = {str(row["id"]): [] for row in rows}
    for name in requested:
        definition = definitions[name.casefold()]
        positive_ids = definition.get("positive_track_ids") or []
        negative_ids = definition.get("negative_track_ids") or []
        positive_indices = [
            row_index[str(identifier)]
            for identifier in positive_ids
            if str(identifier) in row_index
        ]
        negative_indices = [
            row_index[str(identifier)]
            for identifier in negative_ids
            if str(identifier) in row_index
        ]
        positive_prompts = definition.get("positive_prompts") or []
        negative_prompts = definition.get("negative_prompts") or []
        semantic_evidence: np.ndarray | None = None
        lyrics_evidence: np.ndarray | None = None
        if request.representation == "lyrics":
            positive_parts = (
                list(shared_lyrics_model().embed_queries(positive_prompts))
                if positive_prompts
                else []
            )
            negative_parts = (
                list(shared_lyrics_model().embed_queries(negative_prompts))
                if negative_prompts
                else []
            )
            positive_parts.extend(matrix[positive_indices] if positive_indices else [])
            negative_parts.extend(matrix[negative_indices] if negative_indices else [])
            if not positive_parts:
                raise HTTPException(
                    status_code=422, detail=f"Concept {name} has no usable lyrics evidence"
                )
            positive = np.mean(positive_parts, axis=0)
            positive /= max(float(np.linalg.norm(positive)), 1e-8)
            raw = matrix @ positive
            if negative_parts:
                negative = np.mean(negative_parts, axis=0)
                negative /= max(float(np.linalg.norm(negative)), 1e-8)
                raw -= matrix @ negative
            percentiles = empirical_percentiles(raw)
            lyrics_evidence = percentiles
        else:
            semantic_raw, percentiles = score_concept(
                matrix,
                positive_prompts,
                negative_prompts,
                matrix[positive_indices] if positive_indices else None,
                matrix[negative_indices] if negative_indices else None,
            )
            raw = semantic_raw
            semantic_evidence = empirical_percentiles(semantic_raw)
            if (
                request.representation == "hybrid"
                and lyrics_matrix is not None
                and lyrics_available is not None
            ):
                positive_parts = (
                    list(shared_lyrics_model().embed_queries(positive_prompts))
                    if positive_prompts
                    else []
                )
                negative_parts = (
                    list(shared_lyrics_model().embed_queries(negative_prompts))
                    if negative_prompts
                    else []
                )
                positive_parts.extend(
                    lyrics_matrix[index] for index in positive_indices if lyrics_available[index]
                )
                negative_parts.extend(
                    lyrics_matrix[index] for index in negative_indices if lyrics_available[index]
                )
                if positive_parts:
                    positive = np.mean(positive_parts, axis=0)
                    positive /= max(float(np.linalg.norm(positive)), 1e-8)
                    lyrics_raw = lyrics_matrix @ positive
                    if negative_parts:
                        negative = np.mean(negative_parts, axis=0)
                        negative /= max(float(np.linalg.norm(negative)), 1e-8)
                        lyrics_raw -= lyrics_matrix @ negative
                    lyrics_evidence = empirical_percentiles(lyrics_raw, lyrics_available)
                    raw, percentiles = combine_concept_percentiles(
                        semantic_raw, lyrics_raw, lyrics_available
                    )
        for index, row in enumerate(rows):
            if percentiles[index] >= request.minimum_percentile:
                result[str(row["id"])].append(
                    {
                        "name": name,
                        "raw_score": float(raw[index]),
                        "percentile": float(percentiles[index]),
                        "semantic_percentile": float(semantic_evidence[index])
                        if semantic_evidence is not None
                        else None,
                        "lyrics_percentile": float(lyrics_evidence[index])
                        if lyrics_evidence is not None
                        else None,
                        "lyrics_available": bool(lyrics_available[index])
                        if lyrics_available is not None
                        else request.representation == "lyrics",
                    }
                )
    return {
        "run_id": run_id,
        "corpus_size": len(rows),
        "calibration": "empirical-library-percentile",
        "minimum_percentile": request.minimum_percentile,
        "representation": request.representation,
        "scores": result,
    }


def _refresh_curation(curation_id: uuid.UUID, user_id: uuid.UUID) -> dict[str, object]:
    import worker.curation.jobs as curation_jobs

    return curation_jobs.enqueue(curation_id, user_id)


@app.post("/library/curations/preview")
def preview_curation(
    request: CurationRecipe, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    connection_id = user.get("navidrome_connection_id")
    if connection_id is None:
        raise HTTPException(status_code=409, detail="No Navidrome connection is configured")
    return preview_recipe(user["id"], connection_id, request)


@app.get("/library/curations")
def list_curations(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT c.*, latest.recipe,
                   coalesce((SELECT jsonb_agg(jsonb_build_object('id', pt.id, 'title', pt.title, 'artist', pt.artist, 'album', pt.album)) FROM tracks pt WHERE pt.id=ANY(c.positive_track_ids)), '[]') AS positive_tracks,
                   coalesce((SELECT jsonb_agg(jsonb_build_object('id', nt.id, 'title', nt.title, 'artist', nt.artist, 'album', nt.album)) FROM tracks nt WHERE nt.id=ANY(c.negative_track_ids)), '[]') AS negative_tracks,
                   coalesce(jsonb_agg(jsonb_build_object(
                     'id', t.id, 'title', t.title, 'artist', t.artist, 'album', t.album,
                     'duration_seconds', t.duration_seconds, 'source_id', crt.source_id,
                     'position', crt.position, 'score', crt.score,
                     'percentile', nullif(crt.evidence->>'percentile','')::real,
                     'retained', coalesce((crt.evidence->>'retained')::boolean, false),
                     'evidence', crt.evidence,
                     'cover_art', (
                       SELECT max(ts.source_data->>'coverArt')
                       FROM user_track_links utl
                       JOIN track_sources ts ON ts.library_id=utl.library_id
                                             AND ts.track_id=utl.track_id
                                             AND ts.external_id=crt.source_id
                       WHERE utl.user_id=c.user_id AND utl.track_id=crt.track_id
                     )
                   ) ORDER BY crt.position) FILTER (WHERE crt.track_id IS NOT NULL), '[]') AS tracks
               FROM curations c
               LEFT JOIN LATERAL (
                 SELECT id, recipe FROM curation_revisions
                 WHERE curation_id=c.id ORDER BY revision_number DESC LIMIT 1
               ) latest ON true
               LEFT JOIN curation_revision_tracks crt ON crt.revision_id=latest.id
               LEFT JOIN tracks t ON t.id=crt.track_id
               WHERE c.user_id=%s GROUP BY c.id, latest.recipe ORDER BY c.updated_at DESC""",
            (user["id"],),
        )
        return {"curations": cursor.fetchall()}


@app.post("/library/curations", status_code=201)
def create_curation(
    request: CurationCreateRequest, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    connection_id = user.get("navidrome_connection_id")
    if connection_id is None:
        raise HTTPException(status_code=409, detail="No Navidrome connection is configured")
    try:
        with session_scope() as session:
            curation = Curation(
                user_id=user["id"],
                navidrome_connection_id=connection_id,
                name=request.name.strip(),
                curation_type=request.curation_type,
                positive_prompt=request.positive_prompt.strip(),
                negative_prompt=request.negative_prompt.strip(),
                sound_prompts=request.sound_prompts,
                themes_prompts=request.themes_prompts,
                sound_negative_prompts=request.sound_negative_prompts,
                themes_negative_prompts=request.themes_negative_prompts,
                sound_weight=request.sound_weight,
                sound_profile=request.sound_profile.model_dump(
                    exclude_none=True, exclude_defaults=True
                ),
                positive_track_ids=request.positive_track_ids,
                negative_track_ids=request.negative_track_ids,
                familiarity_percent=request.familiarity_percent,
                period_start=request.period_start,
                period_end=request.period_end,
                lookback_days=request.lookback_days,
                time_of_day_enabled=request.time_of_day_enabled,
                track_limit=request.track_limit,
                refresh_mode=request.refresh_mode,
                target_language=request.target_language,
                language_strictness=request.language_strictness,
                journey_start_track_id=request.journey_start_track_id,
                journey_stop_track_ids=request.journey_stop_track_ids,
                journey_end_track_id=request.journey_end_track_id,
                journey_lyrics_weight=request.journey_lyrics_weight,
                refresh_enabled=request.refresh_enabled,
            )
            session.add(curation)
            session.flush()
            curation_id = curation.id
    except IntegrityError as error:
        if "curations_user_name_idx" in str(error.orig):
            raise HTTPException(
                status_code=409, detail="A curation with this name already exists"
            ) from error
        raise
    queued = _refresh_curation(curation_id, user["id"])
    return {**request.model_dump(mode="json"), "id": str(curation_id), "tracks": [], **queued}


@app.put("/library/curations/{curation_id}", status_code=202)
def replace_curation(
    curation_id: uuid.UUID,
    request: CurationCreateRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    """Replace a saved recipe and generate a new playlist revision."""
    user = _session_user(echora_session)
    import worker.curation.jobs as curation_jobs

    with curation_jobs.locked(curation_id) as lock_connection:
        try:
            with session_scope() as session:
                curation = session.scalar(
                    select(Curation).where(
                        Curation.id == curation_id, Curation.user_id == user["id"]
                    )
                )
                if curation is None:
                    raise HTTPException(status_code=404, detail="Curation not found")
                curation_jobs.assert_mutable(lock_connection, curation_id)
                curation.name = request.name.strip()
                curation.curation_type = request.curation_type
                curation.positive_prompt = request.positive_prompt.strip()
                curation.negative_prompt = request.negative_prompt.strip()
                curation.sound_prompts = request.sound_prompts
                curation.themes_prompts = request.themes_prompts
                curation.sound_negative_prompts = request.sound_negative_prompts
                curation.themes_negative_prompts = request.themes_negative_prompts
                curation.sound_weight = request.sound_weight
                curation.sound_profile = request.sound_profile.model_dump(
                    exclude_none=True, exclude_defaults=True
                )
                curation.positive_track_ids = request.positive_track_ids
                curation.negative_track_ids = request.negative_track_ids
                curation.familiarity_percent = request.familiarity_percent
                curation.period_start = request.period_start
                curation.period_end = request.period_end
                curation.time_of_day_enabled = request.time_of_day_enabled
                curation.lookback_days = request.lookback_days
                curation.track_limit = request.track_limit
                curation.refresh_mode = request.refresh_mode
                curation.target_language = request.target_language
                curation.language_strictness = request.language_strictness
                curation.journey_start_track_id = request.journey_start_track_id
                curation.journey_stop_track_ids = request.journey_stop_track_ids
                curation.journey_end_track_id = request.journey_end_track_id
                curation.journey_lyrics_weight = request.journey_lyrics_weight
                curation.refresh_enabled = request.refresh_enabled
                curation.next_refresh_at = (
                    datetime.now(timezone.utc) + timedelta(hours=curation.refresh_interval_hours)
                    if request.refresh_enabled
                    else None
                )
                curation.updated_at = datetime.now(timezone.utc)
        except IntegrityError as error:
            if "curations_user_name_idx" in str(error.orig):
                raise HTTPException(
                    status_code=409, detail="A curation with this name already exists"
                ) from error
            raise
    queued = _refresh_curation(curation_id, user["id"])
    return {**request.model_dump(mode="json"), "id": str(curation_id), "tracks": [], **queued}


@app.patch("/library/curations/{curation_id}")
def update_curation(
    curation_id: uuid.UUID,
    request: CurationUpdateRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    values = request.model_dump(exclude_none=True)
    import worker.curation.jobs as curation_jobs

    with curation_jobs.locked(curation_id) as lock_connection:
        with session_scope() as session:
            curation = session.scalar(
                select(Curation).where(Curation.id == curation_id, Curation.user_id == user["id"])
            )
            if curation is None:
                raise HTTPException(status_code=404, detail="Curation not found")
            curation_jobs.assert_mutable(lock_connection, curation_id)
            for key, value in values.items():
                setattr(curation, key, value)
            if "refresh_enabled" in values:
                curation.next_refresh_at = (
                    datetime.now(timezone.utc) + timedelta(hours=curation.refresh_interval_hours)
                    if values["refresh_enabled"]
                    else None
                )
            curation.updated_at = datetime.now(timezone.utc)
        return {"id": str(curation_id), **values}


@app.post("/library/curations/{curation_id}/refresh", status_code=202)
def refresh_curation(
    curation_id: uuid.UUID, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    return _refresh_curation(curation_id, user["id"])


@app.delete("/library/curations/{curation_id}", status_code=204)
def delete_curation(
    curation_id: uuid.UUID,
    delete_navidrome: bool = True,
    echora_session: str | None = Cookie(default=None),
) -> Response:
    user = _session_user(echora_session)
    import worker.curation.jobs as curation_jobs

    with curation_jobs.locked(curation_id) as lock_connection:
        with session_scope() as session:
            curation = session.scalar(
                select(Curation).where(Curation.id == curation_id, Curation.user_id == user["id"])
            )
            if curation is None:
                raise HTTPException(status_code=404, detail="Curation not found")
            curation_jobs.assert_mutable(
                lock_connection, curation_id, deleting=True, delete_remote=delete_navidrome
            )
            connection_id, playlist_id = (
                curation.navidrome_connection_id,
                curation.navidrome_playlist_id,
            )

        # Delete remotely first so a Navidrome failure does not leave an orphaned
        # playlist after Echora has already forgotten its ID.
        if delete_navidrome and playlist_id:
            credentials = load_connection(str(connection_id))
            if credentials is None:
                raise HTTPException(
                    status_code=409,
                    detail="Navidrome connection is unavailable; keep the Navidrome playlist or reconnect the server",
                )
            with NavidromeClient(*credentials) as client:
                client.delete_playlist(str(playlist_id))

        with session_scope() as session:
            curation = session.scalar(
                select(Curation).where(Curation.id == curation_id, Curation.user_id == user["id"])
            )
            if curation is not None:
                session.delete(curation)
        return Response(status_code=204)


def _curation_scheduler() -> None:
    # Compatibility hook only: scheduled workers now own polling/execution.
    # Startup may still reference this function while API lifecycle is migrated.
    return None


def _cluster_embeddings(
    normalized: np.ndarray, similarities: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    count = len(normalized)
    if count < 3:
        return np.zeros(count, dtype=int), {
            "algorithm": "SNN-Leiden",
            "clusters": 1,
            "neighbors": max(0, count - 1),
        }
    neighbor_count = min(count - 1, max(5, round(np.sqrt(count))))
    nearest = np.argsort(similarities, axis=1)[:, -(neighbor_count + 1) : -1]
    neighbor_sets = [set(map(int, row)) for row in nearest]
    edge_weights: dict[tuple[int, int], float] = {}
    for left in range(count):
        for right_value in nearest[left]:
            right = int(right_value)
            edge = (min(left, right), max(left, right))
            shared = len(neighbor_sets[left] & neighbor_sets[right])
            weight = max(0.01, shared / neighbor_count) * max(
                0.01, float(similarities[left, right])
            )
            edge_weights[edge] = max(edge_weights.get(edge, 0.0), weight)
    graph = ig.Graph(n=count, edges=list(edge_weights), directed=False)
    weights = list(edge_weights.values())
    resolutions = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.75]
    candidates: list[dict[str, object]] = []
    for resolution in resolutions:
        runs: list[tuple[np.ndarray, float]] = []
        for seed in range(6):
            ig.set_random_number_generator(random.Random(104729 + seed))
            partition = graph.community_leiden(
                objective_function="modularity",
                weights=weights,
                resolution=resolution,
                n_iterations=-1,
            )
            runs.append((np.asarray(partition.membership, dtype=int), float(partition.quality)))
        best_labels, quality = max(runs, key=lambda item: item[1])
        comparisons = [
            adjusted_rand_score(runs[left][0], runs[right][0])
            for left in range(len(runs))
            for right in range(left + 1, len(runs))
        ]
        stability = float(np.mean(comparisons)) if comparisons else 1.0
        clusters = len(np.unique(best_labels))
        silhouette = (
            float(silhouette_score(normalized, best_labels, metric="cosine"))
            if 1 < clusters < count
            else -1.0
        )
        candidates.append(
            {
                "resolution": resolution,
                "labels": best_labels,
                "clusters": clusters,
                "silhouette": silhouette,
                "stability": stability,
            }
        )
    has_multiple = any(int(candidate["clusters"]) > 1 for candidate in candidates)
    for index, candidate in enumerate(candidates):
        plateau = sum(
            index + offset in range(len(candidates))
            and candidates[index + offset]["clusters"] == candidate["clusters"]
            for offset in (-1, 1)
        )
        candidate["score"] = (
            float(candidate["silhouette"]) + 0.35 * float(candidate["stability"]) + 0.1 * plateau
        )
        if has_multiple and int(candidate["clusters"]) == 1:
            candidate["score"] = -10.0
    selected = max(candidates, key=lambda candidate: float(candidate["score"]))
    labels = np.asarray(selected["labels"], dtype=int)
    ordered = sorted(
        np.unique(labels), key=lambda label: (-int(np.sum(labels == label)), int(label))
    )
    remap = {old: new for new, old in enumerate(ordered)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=int)
    return labels, {
        "algorithm": "SNN-Leiden",
        "clusters": len(ordered),
        "neighbors": neighbor_count,
        "resolution": selected["resolution"],
        "silhouette": selected["silhouette"],
        "seed_stability_ari": selected["stability"],
        "resolutions_tested": resolutions,
    }


@app.get("/library/audio-profiles/status")
def audio_profile_status(echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    user = _session_user(echora_session)
    models: dict[str, dict[str, int]] = {}
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        for model_name in SUPPORTED_PROFILE_MODELS:
            cursor.execute(
                """WITH latest_source AS (
                 SELECT DISTINCT ON (e.track_id) e.track_id, e.run_id
                 FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                 WHERE e.embedding_type='audio-track' AND e.window_index IS NULL
                   AND ar.model_name=%s
                   AND EXISTS (SELECT 1 FROM user_track_links utl
                               WHERE utl.user_id=%s AND utl.track_id=e.track_id)
                 ORDER BY e.track_id, ar.created_at DESC
               )
               SELECT count(*) AS total,
                      count(*) FILTER (WHERE EXISTS (
                        SELECT 1 FROM current_audio_profiles tap
                        JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
                        WHERE tap.track_id=latest_source.track_id
                          AND tap.source_run_id=latest_source.run_id
                          AND tap.model_name=%s
                          AND profile_run.kind='audio_profile'
                          AND profile_run.model_revision=%s

                      )) AS current
               FROM latest_source""",
                (model_name, user["id"], model_name, AUDIO_PROFILE_REVISION),
            )
            value = cursor.fetchone()
            total, current = int(value["total"]), int(value["current"])
            models[model_name] = {"total": total, "current": current, "missing": total - current}
    return {
        "revision": AUDIO_PROFILE_REVISION,
        "models": models,
        "total": sum(value["total"] for value in models.values()),
        "current": sum(value["current"] for value in models.values()),
        "missing": sum(value["missing"] for value in models.values()),
    }


@app.post("/library/audio-profiles/rebuild", status_code=202)
def start_audio_profile_rebuild(
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT DISTINCT e.track_id, ar.model_name
               FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
               WHERE e.embedding_type='audio-track' AND e.window_index IS NULL
                 AND ar.model_name=ANY(%s)
                 AND EXISTS (SELECT 1 FROM user_track_links utl
                             WHERE utl.user_id=%s AND utl.track_id=e.track_id)
               ORDER BY e.track_id, ar.model_name""",
            (list(SUPPORTED_PROFILE_MODELS), user["id"]),
        )
        source_rows = cursor.fetchall()
        track_ids = sorted({row[0] for row in source_rows})
    return jobs.enqueue(
        "audio_profiles",
        "analysis",
        user["id"],
        payload={"track_ids": [str(track_id) for track_id in track_ids]},
        dedupe_key=f"audio_profiles:{user['id']}",
    )


@app.get("/library/tracks/{track_id}/audio-profile")
def track_audio_profile(
    track_id: uuid.UUID,
    model: str = "muq_mulan",
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    if model not in SUPPORTED_PROFILE_MODELS:
        raise HTTPException(status_code=422, detail="Audio profiles support muq_mulan or mert")
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT tap.*, profile_run.model_revision AS profile_revision,
                      profile_run.config AS profile_config, profile_run.created_at AS profile_created_at,
                      source_run.model_revision AS source_model_revision,
                      source_run.config AS source_config
               FROM current_audio_profiles tap
               JOIN analysis_runs profile_run ON profile_run.id=tap.profile_run_id
               JOIN analysis_runs source_run ON source_run.id=tap.source_run_id
               WHERE tap.track_id=%s AND tap.model_name=%s
                 AND EXISTS (SELECT 1 FROM user_track_links utl
                             WHERE utl.user_id=%s AND utl.track_id=tap.track_id)
               ORDER BY profile_run.created_at DESC, tap.created_at DESC LIMIT 1""",
            (track_id, model, user["id"]),
        )
        profile = cursor.fetchone()
        if profile is None:
            raise HTTPException(status_code=404, detail="Audio profile not found")
        cursor.execute(
            """SELECT segment_index, start_seconds, end_seconds, cohesion,
                      representative_window_index
               FROM audio_temporal_segments
               WHERE profile_id=%s ORDER BY segment_index""",
            (profile["id"],),
        )
        segments = cursor.fetchall()
        cursor.execute(
            """SELECT am.id, am.mode_index, am.duration_weight, am.cohesion,
                      am.representative_window_index
               FROM audio_modes am
               WHERE am.profile_id=%s ORDER BY am.mode_index""",
            (profile["id"],),
        )
        modes = cursor.fetchall()
        for mode in modes:
            cursor.execute(
                """SELECT start_seconds, end_seconds FROM audio_mode_intervals
                   WHERE mode_id=%s ORDER BY interval_index""",
                (mode["id"],),
            )
            mode["intervals"] = cursor.fetchall()
            mode.pop("id")
    return jsonable_encoder(
        {
            "track_id": track_id,
            "profile_run_id": profile["profile_run_id"],
            "source_run_id": profile["source_run_id"],
            "profile_revision": profile["profile_revision"],
            "profile_created_at": profile["profile_created_at"],
            "profile_config": profile["profile_config"],
            "source": {
                "model": profile["model_name"],
                "revision": profile["source_model_revision"],
                "config": profile["source_config"],
            },
            "timestamps_exact": profile["timestamps_exact"],
            "diagnostics": {
                "resultant_length": profile["resultant_length"],
                "mean_global_similarity": profile["mean_global_similarity"],
                "p05_global_similarity": profile["p05_global_similarity"],
                "adjacent_change_mean": profile["adjacent_change_mean"],
                "adjacent_change_p95": profile["adjacent_change_p95"],
            },
            "segments": segments,
            "modes": modes,
        }
    )


@app.get("/library/map")
def library_map(
    model: str = "muq_mulan",
    limit: int = 1500,
    semantic_weight: float | None = None,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    limit = min(max(limit, 10), 3000)
    if model not in {"muq_mulan", "mert", "lyrics", "blend"}:
        raise HTTPException(status_code=422, detail="Unknown representation")
    if model == "blend":
        semantic_weight = 0.5 if semantic_weight is None else semantic_weight
        if not 0 <= semantic_weight <= 1:
            raise HTTPException(status_code=422, detail="semantic_weight must be between 0 and 1")
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        if model == "blend":
            cursor.execute(
                """
                WITH semantic AS (
                  SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding,
                         ar.id AS run_id, ar.model_revision, ar.created_at
                  FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                  WHERE e.embedding_type='audio-track' AND ar.model_name='muq_mulan'
                  ORDER BY e.track_id, ar.created_at DESC
                ), acoustic AS (
                  SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding,
                         ar.id AS run_id, ar.model_revision, ar.created_at
                  FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
                  WHERE e.embedding_type='audio-track' AND ar.model_name='mert'
                  ORDER BY e.track_id, ar.created_at DESC
                )
                SELECT t.id, t.title, t.artist, t.album, t.duration_seconds, t.genres,
                       semantic.embedding AS semantic_embedding, acoustic.embedding AS acoustic_embedding,
                       semantic.run_id::text || ':' || acoustic.run_id::text AS run_id,
                       semantic.model_revision || ':' || acoustic.model_revision AS model_revision,
                       greatest(semantic.created_at, acoustic.created_at) AS run_created_at,
                       max(ts.external_id) AS source_id, max(ts.source_data->>'coverArt') AS cover_art
                FROM semantic JOIN acoustic USING (track_id) JOIN tracks t ON t.id=semantic.track_id
                LEFT JOIN track_sources ts ON ts.track_id=t.id
                  AND EXISTS (SELECT 1 FROM user_track_links source_links
                              WHERE source_links.user_id=%s AND source_links.library_id=ts.library_id
                                AND source_links.track_id=ts.track_id AND source_links.external_id=ts.external_id)
                WHERE EXISTS (SELECT 1 FROM user_track_links visible_links
                              WHERE visible_links.track_id=t.id AND visible_links.user_id=%s)
                GROUP BY t.id, semantic.embedding, acoustic.embedding, semantic.run_id, acoustic.run_id,
                         semantic.model_revision, acoustic.model_revision, semantic.created_at, acoustic.created_at
                ORDER BY t.id LIMIT %s
                """,
                (user["id"], user["id"], limit),
            )
        else:
            cursor.execute(
                """
                WITH selected_embeddings AS (
                  SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding,
                         ar.id AS run_id, ar.model_revision, ar.created_at AS run_created_at
                  FROM current_embeddings e JOIN analysis_runs ar ON ar.id = e.run_id
                  WHERE e.embedding_type = %s AND ar.model_name = %s AND e.window_index IS NULL
                  ORDER BY e.track_id, ar.created_at DESC
                )
                SELECT t.id, t.title, t.artist, t.album, t.duration_seconds, t.genres,
                       selected_embeddings.embedding, selected_embeddings.run_id,
                       selected_embeddings.model_revision, selected_embeddings.run_created_at,
                       max(ts.external_id) AS source_id, max(ts.source_data->>'coverArt') AS cover_art
                FROM selected_embeddings JOIN tracks t ON t.id = selected_embeddings.track_id
                LEFT JOIN track_sources ts ON ts.track_id = t.id
                  AND EXISTS (SELECT 1 FROM user_track_links source_links
                              WHERE source_links.user_id=%s AND source_links.library_id=ts.library_id
                                AND source_links.track_id=ts.track_id AND source_links.external_id=ts.external_id)
                WHERE EXISTS (SELECT 1 FROM user_track_links visible_links
                              WHERE visible_links.track_id=t.id AND visible_links.user_id=%s)
                GROUP BY t.id, selected_embeddings.embedding, selected_embeddings.run_id,
                         selected_embeddings.model_revision, selected_embeddings.run_created_at
                ORDER BY t.id LIMIT %s
                """,
                (
                    "lyrics" if model == "lyrics" else "audio-track",
                    "bge_m3" if model == "lyrics" else model,
                    user["id"],
                    user["id"],
                    limit,
                ),
            )
        rows = cursor.fetchall()
    if not rows:
        return {"points": [], "model": model}
    if model == "blend":
        semantic = np.stack(
            [np.fromstring(row.pop("semantic_embedding").strip("[]"), sep=",") for row in rows]
        )
        acoustic = np.stack(
            [np.fromstring(row.pop("acoustic_embedding").strip("[]"), sep=",") for row in rows]
        )
        semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-8)
        acoustic /= np.maximum(np.linalg.norm(acoustic, axis=1, keepdims=True), 1e-8)
        matrix = np.concatenate(
            [np.sqrt(semantic_weight) * semantic, np.sqrt(1 - semantic_weight) * acoustic], axis=1
        )
    else:
        matrix = np.stack(
            [np.fromstring(row.pop("embedding").strip("[]"), sep=",") for row in rows]
        )
    effective_weight = float(
        semantic_weight if model == "blend" else (1 if model == "muq_mulan" else 0)
    )
    corpus_material = "|".join(
        [
            model,
            f"{effective_weight:.4f}",
            str(_COMMUNITY_SNAPSHOT_REVISION),
            *[f"{row['id']}:{row['run_id']}" for row in rows],
        ]
    )
    corpus_hash = hashlib.sha256(corpus_material.encode()).hexdigest()
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT id, payload, created_at FROM community_snapshots
               WHERE model_name=%s AND semantic_weight=%s AND corpus_hash=%s AND algorithm_revision=%s""",
            (model, effective_weight, corpus_hash, _COMMUNITY_SNAPSHOT_REVISION),
        )
        cached = cursor.fetchone()
    if cached:
        payload = dict(cached["payload"])
        payload["snapshot_id"] = str(cached["id"])
        payload["snapshot_created_at"] = cached["created_at"]
        payload["cache_hit"] = True
        return payload
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    similarities = normalized @ normalized.T
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    left_vectors, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    projection = left_vectors[:, :2] * singular_values[:2]
    scale = np.percentile(np.abs(projection), 97, axis=0)
    projection = projection / np.maximum(scale, 1e-6)
    labels, clustering = _cluster_embeddings(normalized, similarities)
    cluster_count = len(np.unique(labels))
    embedding_centers = np.stack(
        [normalized[labels == index].mean(axis=0) for index in range(cluster_count)]
    )
    embedding_centers /= np.maximum(np.linalg.norm(embedding_centers, axis=1, keepdims=True), 1e-8)
    membership_logits = normalized @ embedding_centers.T / 0.08
    membership_logits -= membership_logits.max(axis=1, keepdims=True)
    memberships = np.exp(membership_logits)
    memberships /= memberships.sum(axis=1, keepdims=True)
    if cluster_count:
        center_matrix = np.stack(
            [projection[labels == index].mean(axis=0) for index in range(cluster_count)]
        )
        layout_centers = center_matrix.copy()
        for _ in range(80):
            for left in range(cluster_count):
                for right in range(left + 1, cluster_count):
                    delta = layout_centers[right] - layout_centers[left]
                    distance = float(np.linalg.norm(delta))
                    if distance < 0.9:
                        if distance < 1e-6:
                            angle = (left * 2.399963 + right) % (2 * np.pi)
                            delta = np.array([np.cos(angle), np.sin(angle)])
                            distance = 1.0
                        shift = delta / distance * (0.9 - distance) * 0.08
                        layout_centers[left] -= shift
                        layout_centers[right] += shift
        packed = projection.copy()
        clustered = labels >= 0
        packed[clustered] = (
            layout_centers[labels[clustered]]
            + (projection[clustered] - center_matrix[labels[clustered]]) * 0.34
        )
        projection = packed
    projection -= projection.mean(axis=0, keepdims=True)
    packed_scale = np.percentile(np.abs(projection), 97, axis=0)
    projection /= np.maximum(packed_scale, 1e-6)
    center_matrix = (
        np.stack([projection[labels == index].mean(axis=0) for index in range(cluster_count)])
        if cluster_count
        else np.empty((0, 2))
    )
    communities = []
    for cluster in range(cluster_count):
        member_indices = np.flatnonzero(labels == cluster)
        genres = Counter(
            genre for item in member_indices for genre in (rows[item].get("genres") or []) if genre
        )
        top_genres = [{"name": name, "tracks": count} for name, count in genres.most_common(3)]
        representatives = sorted(
            member_indices,
            key=lambda item: float(normalized[item] @ embedding_centers[cluster]),
            reverse=True,
        )[:3]
        label_parts = [item["name"] for item in top_genres[:2]]
        label = " / ".join(label_parts) if label_parts else f"Community {cluster + 1}"
        cohesion = float(np.mean(normalized[member_indices] @ embedding_centers[cluster]))
        communities.append(
            {
                "id": cluster,
                "label": label,
                "x": float(center_matrix[cluster, 0]),
                "y": float(center_matrix[cluster, 1]),
                "size": len(member_indices),
                "cohesion": cohesion,
                "top_genres": top_genres,
                "representative_tracks": [
                    {
                        "id": str(rows[item]["id"]),
                        "title": rows[item]["title"],
                        "artist": rows[item].get("artist"),
                    }
                    for item in representatives
                ],
            }
        )
    nearest_count = min(6, max(0, len(rows) - 1))
    nearest_sets = [
        set(map(int, np.argsort(similarities[index])[::-1][1 : nearest_count + 1]))
        for index in range(len(rows))
    ]
    edge_map: dict[tuple[int, int], dict[str, object]] = {}
    for left, neighbors_for_left in enumerate(nearest_sets):
        for right in neighbors_for_left:
            edge = (min(left, right), max(left, right))
            edge_map[edge] = {
                "source_id": str(rows[edge[0]]["id"]),
                "target_id": str(rows[edge[1]]["id"]),
                "similarity": float(similarities[edge[0], edge[1]]),
                "mutual": left in nearest_sets[right],
                "cross_community": bool(labels[edge[0]] != labels[edge[1]]),
            }
    provenance = {
        "run_id": str(rows[0]["run_id"]),
        "model_revision": rows[0].get("model_revision"),
        "run_created_at": rows[0].get("run_created_at"),
        "track_count": len(rows),
    }
    points = []
    for index, row in enumerate(rows):
        cluster = int(labels[index])
        affinity = float(memberships[index, cluster])
        entropy = float(-np.sum(memberships[index] * np.log(np.maximum(memberships[index], 1e-12))))
        bridge_score = entropy / max(np.log(cluster_count), 1e-8) if cluster_count > 1 else 0.0
        strongest = np.argsort(memberships[index])[::-1][:3]
        cluster_memberships = [
            {"cluster": int(value), "strength": float(memberships[index, value])}
            for value in strongest
        ]
        nearest = np.argsort(similarities[index])[::-1]
        neighbors = [
            {
                "id": str(rows[item]["id"]),
                "title": rows[item]["title"],
                "artist": rows[item]["artist"],
                "similarity": float(similarities[index, item]),
            }
            for item in nearest
            if item != index
        ][:4]
        public_row = {
            key: value
            for key, value in row.items()
            if key not in {"run_id", "model_revision", "run_created_at"}
        }
        points.append(
            {
                **public_row,
                "x": float(projection[index, 0]),
                "y": float(projection[index, 1]),
                "cluster": cluster,
                "cluster_affinity": affinity,
                "cluster_memberships": cluster_memberships,
                "bridge_score": bridge_score,
                "neighbors": neighbors,
            }
        )
    payload = jsonable_encoder(
        {
            "points": points,
            "clusters": communities,
            "communities": communities,
            "edges": list(edge_map.values()),
            "model": model,
            "semantic_weight": effective_weight,
            "clustering": {**clustering, **provenance},
            "corpus_hash": corpus_hash,
        }
    )
    parameters = {
        "algorithm": clustering["algorithm"],
        "algorithm_revision": _COMMUNITY_SNAPSHOT_REVISION,
        "neighbors": clustering["neighbors"],
        "resolution": clustering["resolution"],
        "resolutions_tested": clustering["resolutions_tested"],
        "semantic_weight": effective_weight,
    }
    metrics = {
        "silhouette": clustering["silhouette"],
        "seed_stability_ari": clustering["seed_stability_ari"],
    }
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """INSERT INTO community_snapshots
                 (model_name, semantic_weight, corpus_hash, algorithm_revision, track_count, parameters, metrics, payload)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (model_name, semantic_weight, corpus_hash, algorithm_revision)
               DO UPDATE SET payload=EXCLUDED.payload, parameters=EXCLUDED.parameters, metrics=EXCLUDED.metrics
               RETURNING id, created_at""",
            (
                model,
                effective_weight,
                corpus_hash,
                _COMMUNITY_SNAPSHOT_REVISION,
                len(rows),
                Jsonb(parameters),
                Jsonb(metrics),
                Jsonb(payload),
            ),
        )
        snapshot = cursor.fetchone()
    return {
        **payload,
        "snapshot_id": str(snapshot["id"]),
        "snapshot_created_at": snapshot["created_at"],
        "cache_hit": False,
    }


@app.get("/library/community-snapshots", dependencies=[Depends(require_user)])
def community_snapshots(
    limit: int = 20,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    _session_user(echora_session)
    limit = min(max(limit, 1), 100)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT id, model_name, semantic_weight, corpus_hash, algorithm_revision,
                      track_count, parameters, metrics, created_at
               FROM community_snapshots ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        )
        snapshots = cursor.fetchall()
    return {"snapshots": snapshots}


@app.get("/library/community-snapshots/{snapshot_id}", dependencies=[Depends(require_user)])
def community_snapshot(
    snapshot_id: uuid.UUID,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    _session_user(echora_session)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT payload, created_at FROM community_snapshots WHERE id=%s", (snapshot_id,)
        )
        snapshot = cursor.fetchone()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Community snapshot not found")
    return {
        **snapshot["payload"],
        "snapshot_id": str(snapshot_id),
        "snapshot_created_at": snapshot["created_at"],
        "cache_hit": True,
    }


def _artist_embedding_corpus(model: str) -> tuple[list[dict[str, object]], np.ndarray, str]:
    if model not in {"muq_mulan", "mert"}:
        raise HTTPException(status_code=422, detail="Artist profiles support muq_mulan or mert")
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT DISTINCT ON (e.track_id) t.id, t.title, t.artist, t.album,
                      e.embedding::text AS embedding, ar.id AS run_id
               FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id JOIN tracks t ON t.id=e.track_id
               WHERE e.embedding_type='audio-track' AND ar.model_name=%s AND t.artist IS NOT NULL
               ORDER BY e.track_id, ar.created_at DESC""",
            (model,),
        )
        rows = cursor.fetchall()
    if not rows:
        raise HTTPException(status_code=409, detail="No artist embeddings are available")
    matrix = np.stack([np.fromstring(row.pop("embedding").strip("[]"), sep=",") for row in rows])
    corpus_hash = hashlib.sha256(
        "|".join(f"{row['id']}:{row['run_id']}" for row in rows).encode()
    ).hexdigest()
    for row in rows:
        row.pop("run_id", None)
    return rows, matrix, corpus_hash


def _artist_payload(
    name: str, rows: list[dict[str, object]], matrix: np.ndarray, indices: list[int]
) -> tuple[dict[str, object], object]:
    artist_matrix = matrix[indices]
    profile = fit_artist_profile(artist_matrix)
    representatives = representative_indices(artist_matrix, profile)
    facets = []
    for component, representative in enumerate(representatives):
        members = np.flatnonzero(profile.component_labels == component)
        track = rows[indices[representative]]
        facets.append(
            {
                "index": component,
                "weight": float(profile.weights[component]),
                "track_count": int(len(members)),
                "representative_track": {
                    "id": str(track["id"]),
                    "title": track["title"],
                    "artist": track["artist"],
                    "album": track["album"],
                },
            }
        )
    return {
        "artist": name,
        "track_count": len(indices),
        "component_count": len(profile.weights),
        "facets": facets,
    }, profile


@app.get("/library/artists/profile", dependencies=[Depends(require_user)])
def artist_profile(
    artist: str,
    model: str = "muq_mulan",
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    _session_user(echora_session)
    rows, matrix, corpus_hash = _artist_embedding_corpus(model)
    indices = [
        index
        for index, row in enumerate(rows)
        if str(row["artist"]).casefold() == artist.casefold()
    ]
    if not indices:
        raise HTTPException(status_code=404, detail="Artist not found")
    display_name = str(rows[indices[0]]["artist"])
    payload, profile = _artist_payload(display_name, rows, matrix, indices)
    stored = jsonable_encoder({**payload, "model": model, "corpus_hash": corpus_hash})
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """INSERT INTO artist_profiles
                 (artist_key, artist_name, model_name, corpus_hash, track_count, component_count, payload)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (artist_key, model_name, corpus_hash) DO UPDATE SET payload=EXCLUDED.payload
               RETURNING id, created_at""",
            (
                display_name.casefold(),
                display_name,
                model,
                corpus_hash,
                len(indices),
                len(profile.weights),
                Jsonb(stored),
            ),
        )
        saved = cursor.fetchone()
    return {**stored, "profile_id": str(saved["id"]), "created_at": saved["created_at"]}


@app.get("/library/artists/similar", dependencies=[Depends(require_user)])
def similar_artists(
    artist: str,
    model: str = "muq_mulan",
    limit: int = 12,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    _session_user(echora_session)
    limit = min(max(limit, 1), 30)
    rows, matrix, _ = _artist_embedding_corpus(model)
    groups: dict[str, list[int]] = {}
    names: dict[str, str] = {}
    for index, row in enumerate(rows):
        key = str(row["artist"]).casefold()
        groups.setdefault(key, []).append(index)
        names.setdefault(key, str(row["artist"]))
    target_key = artist.casefold()
    if target_key not in groups:
        raise HTTPException(status_code=404, detail="Artist not found")
    _, target = _artist_payload(names[target_key], rows, matrix, groups[target_key])
    target_center = weighted_center(target)
    centroid_candidates = []
    for key, indices in groups.items():
        if key == target_key:
            continue
        center = np.mean(matrix[indices], axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-8)
        centroid_candidates.append((key, float(target_center @ center)))
    candidates = sorted(centroid_candidates, key=lambda item: item[1], reverse=True)[
        : max(limit * 3, 30)
    ]
    results = []
    for key, coarse_similarity in candidates:
        payload, profile = _artist_payload(names[key], rows, matrix, groups[key])
        similarity, forward, backward, component_matrix = soft_chamfer_similarity(target, profile)
        target_component, candidate_component = np.unravel_index(
            np.argmax(component_matrix), component_matrix.shape
        )
        results.append(
            {
                **payload,
                "similarity": similarity,
                "target_coverage": forward,
                "candidate_coverage": backward,
                "coarse_similarity": coarse_similarity,
                "strongest_facet_match": {
                    "target_facet": int(target_component),
                    "candidate_facet": int(candidate_component),
                    "similarity": float(component_matrix[target_component, candidate_component]),
                },
            }
        )
    results.sort(key=lambda item: float(item["similarity"]), reverse=True)
    return {"artist": names[target_key], "model": model, "results": results[:limit]}


@app.post("/library/journeys/preview", dependencies=[Depends(require_user)])
def preview_journey(
    request: JourneyRequest,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    _session_user(echora_session)
    if request.start_track_id == request.end_track_id:
        raise HTTPException(status_code=422, detail="Journey endpoints must be different tracks")
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """
            WITH semantic AS (
              SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding
              FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
              WHERE e.embedding_type='audio-track' AND ar.model_name='muq_mulan'
              ORDER BY e.track_id, ar.created_at DESC
            ), acoustic AS (
              SELECT DISTINCT ON (e.track_id) e.track_id, e.embedding::text AS embedding
              FROM current_embeddings e JOIN analysis_runs ar ON ar.id=e.run_id
              WHERE e.embedding_type='audio-track' AND ar.model_name='mert'
              ORDER BY e.track_id, ar.created_at DESC
            )
            SELECT t.id, t.title, t.artist, t.album, t.duration_seconds,
                   semantic.embedding AS semantic_embedding, acoustic.embedding AS acoustic_embedding,
                   max(ts.external_id) AS source_id, max(ts.source_data->>'coverArt') AS cover_art,
                   max(member.group_id::text) AS recording_group_id
            FROM tracks t JOIN semantic ON semantic.track_id=t.id JOIN acoustic ON acoustic.track_id=t.id
            LEFT JOIN track_sources ts ON ts.track_id=t.id
            LEFT JOIN recording_group_members member ON member.track_id=t.id
            GROUP BY t.id, semantic.embedding, acoustic.embedding ORDER BY t.id
            """
        )
        rows = cursor.fetchall()
    identifiers = {str(row["id"]): index for index, row in enumerate(rows)}
    start_index, end_index = (
        identifiers.get(str(request.start_track_id)),
        identifiers.get(str(request.end_track_id)),
    )
    if start_index is None or end_index is None:
        raise HTTPException(
            status_code=404, detail="One or both journey endpoints lack required embeddings"
        )
    start_group = rows[start_index].get("recording_group_id")
    if start_group and start_group == rows[end_index].get("recording_group_id"):
        raise HTTPException(
            status_code=422,
            detail="Journey endpoints must be different recordings",
        )
    semantic = normalize_journey_rows(
        np.stack(
            [np.fromstring(row.pop("semantic_embedding").strip("[]"), sep=",") for row in rows]
        )
    )
    acoustic = normalize_journey_rows(
        np.stack(
            [np.fromstring(row.pop("acoustic_embedding").strip("[]"), sep=",") for row in rows]
        )
    )
    if request.mode == "semantic":
        matrix = semantic
        semantic_weight = 1.0
    elif request.mode == "acoustic":
        matrix = acoustic
        semantic_weight = 0.0
    else:
        semantic_weight = request.semantic_weight
        matrix = np.concatenate(
            [np.sqrt(semantic_weight) * semantic, np.sqrt(1 - semantic_weight) * acoustic], axis=1
        )
    targets = spherical_targets(matrix[start_index], matrix[end_index], request.length)
    selected = select_journey(
        matrix,
        targets,
        start_index,
        end_index,
        [row.get("artist") for row in rows],
        [row.get("recording_group_id") for row in rows],
    )
    steps = []
    for position, (index, target_similarity, transition_similarity) in enumerate(selected):
        row = rows[index]
        steps.append(
            {
                **row,
                "position": position,
                "target_progress": position / max(len(selected) - 1, 1),
                "target_similarity": target_similarity,
                "transition_similarity": transition_similarity,
            }
        )
    return {
        "mode": request.mode,
        "semantic_weight": semantic_weight,
        "requested_length": request.length,
        "steps": steps,
    }


@app.get("/library/filter-options")
def browse_filter_options(echora_session: str | None = Cookie(default=None)):
    user = _session_user(echora_session)
    with psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as db:
        rows = db.execute(
            """WITH visible AS MATERIALIZED (
            SELECT DISTINCT track_id FROM user_track_links WHERE user_id=%s
        ), options AS (
            SELECT 'language' AS kind, coalesce(nullif(l.language,''),'unknown') AS value, v.track_id
            FROM visible v LEFT JOIN lyrics l ON l.track_id=v.track_id
            UNION ALL
            SELECT 'genre', g, v.track_id FROM visible v JOIN tracks t ON t.id=v.track_id
            CROSS JOIN LATERAL unnest(t.genres) g WHERE g<>''
            UNION ALL
            SELECT 'translation', lt.target_language, v.track_id FROM visible v
            JOIN lyric_translations lt ON lt.track_id=v.track_id WHERE lt.status='ready'
        ) SELECT kind,value,count(DISTINCT track_id) AS tracks FROM options
          GROUP BY kind,value ORDER BY kind,lower(value)""",
            (user["id"],),
        ).fetchall()
    return {
        kind: [
            {"name": row["value"], "tracks": row["tracks"]} for row in rows if row["kind"] == kind
        ]
        for kind in ("language", "genre", "translation")
    }


# Navidrome play counts need a full catalog scan, so keep each user's ranking briefly.
_NAVIDROME_TOP_TTL_SECONDS = 300
_navidrome_top_cache: dict[tuple[str, str], tuple[float, list[tuple[str, int]]]] = {}
_navidrome_top_lock = threading.Lock()


def _navidrome_ranking(user_id: str, connection_id: str) -> list[tuple[str, int]] | None:
    key = (user_id, connection_id)
    with _navidrome_top_lock:
        cached = _navidrome_top_cache.get(key)
    if cached and time.monotonic() - cached[0] < _NAVIDROME_TOP_TTL_SECONDS:
        return cached[1]
    credentials = load_connection(connection_id, uuid.UUID(user_id))
    if credentials is None:
        return None
    with NavidromeClient(*credentials) as client:
        ranking = navidrome_play_counts([track.raw for track in client.all_tracks()])
    with _navidrome_top_lock:
        _navidrome_top_cache[key] = (time.monotonic(), ranking)
    return ranking


def _library_track_rows(user_id: object) -> list[dict[str, object]]:
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT t.id::text AS id, t.title, t.artist, t.album, t.duration_seconds,
                      source.external_id AS source_id, source.cover_art
               FROM tracks t
               LEFT JOIN LATERAL (
                 SELECT ts.external_id, ts.source_data->>'coverArt' AS cover_art
                 FROM track_sources ts
                 JOIN user_track_links links
                   ON links.user_id=%s AND links.library_id=ts.library_id
                  AND links.track_id=ts.track_id AND links.external_id=ts.external_id
                 WHERE ts.track_id=t.id AND ts.source_type='subsonic'
                 ORDER BY ts.id LIMIT 1
               ) source ON true
               WHERE EXISTS (SELECT 1 FROM user_track_links utl WHERE utl.track_id=t.id AND utl.user_id=%s)""",
            (user_id, user_id),
        )
        return cursor.fetchall()


@app.get("/library/top-tracks")
def library_top_tracks(
    limit: int = 10,
    period: str = "1month",
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    """The user's most played library tracks. Echora records no plays itself.

    Last.fm supplies ranked listening periods. Without Last.fm, Navidrome's
    per-song play counts give an all-time ranking (``source`` says which).
    """
    user = _session_user(echora_session)
    limit = min(max(limit, 1), 50)
    if period not in TOP_TRACK_PERIODS:
        raise HTTPException(status_code=422, detail="Unknown listening period")
    with session_scope() as session:
        preference = session.get(UserPreference, user["id"])
        username = preference.lastfm_username if preference else None
        encrypted_key = (
            bytes(preference.lastfm_api_key_encrypted)
            if preference and preference.lastfm_api_key_encrypted
            else None
        )
    if username and encrypted_key:
        try:
            # Ask for more than we show: some top tracks are not in this library.
            entries = top_tracks(
                username,
                credential_cipher().decrypt(encrypted_key).decode(),
                period,
                min(limit * 5, 200),
            )
        except Exception as error:
            raise HTTPException(
                status_code=502, detail=f"Could not read Last.fm listening history: {error}"
            ) from error
        matched = match_top_tracks(_library_track_rows(user["id"]), entries)
        return {
            "available": True,
            "source": "lastfm",
            "period": period,
            "tracks": [{**row, "play_count": count} for row, count in matched[:limit]],
        }
    connection_id = user.get("navidrome_connection_id")
    if not connection_id:
        return {"available": False, "source": None, "period": period, "tracks": []}
    try:
        ranking = _navidrome_ranking(str(user["id"]), str(connection_id))
    except Exception as error:
        raise HTTPException(
            status_code=502, detail=f"Could not read Navidrome play counts: {error}"
        ) from error
    if ranking is None:
        return {"available": False, "source": None, "period": period, "tracks": []}
    matched = match_navidrome_play_counts(_library_track_rows(user["id"]), ranking)
    return {
        "available": True,
        "source": "navidrome",
        "period": "overall",
        "tracks": [{**row, "play_count": count} for row, count in matched[:limit]],
    }


@app.get("/library/facets")
def library_facets(
    artist_query: str = "",
    album_query: str = "",
    artist: str = "",
    limit: int = 20,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    limit = min(max(limit, 1), 50)
    with (
        psycopg.connect(DatabaseConfig.load().database_url, row_factory=dict_row) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """SELECT artist AS name, count(*) AS tracks FROM tracks t
               WHERE artist IS NOT NULL AND artist ILIKE %s
                 AND EXISTS (SELECT 1 FROM user_track_links utl
                             WHERE utl.track_id=t.id AND utl.user_id=%s)
               GROUP BY artist ORDER BY artist LIMIT %s""",
            (f"%{artist_query.strip()}%", user["id"], limit),
        )
        artists = cursor.fetchall()
        album_clauses = [
            "album IS NOT NULL",
            "album ILIKE %s",
            "EXISTS (SELECT 1 FROM user_track_links utl WHERE utl.track_id=t.id AND utl.user_id=%s)",
        ]
        album_parameters: list[object] = [f"%{album_query.strip()}%", user["id"]]
        if artist:
            album_clauses.append("artist = %s")
            album_parameters.append(artist)
        cursor.execute(
            f"SELECT album AS name, count(*) AS tracks FROM tracks t WHERE {' AND '.join(album_clauses)} GROUP BY album ORDER BY album LIMIT %s",
            [*album_parameters, limit],
        )
        albums = cursor.fetchall()
    return {"artists": artists, "albums": albums}


@app.get("/jobs")
def list_user_jobs(
    connection_id: uuid.UUID | None = None,
    active_only: bool = False,
    limit: int = 20,
    library_only: bool = False,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    return {
        "jobs": jobs.list_jobs(
            user["id"],
            connection_id=connection_id,
            active_only=active_only,
            limit=min(max(limit, 1), 100),
            library_only=library_only,
        )
    }


@app.get("/jobs/{job_id}")
def job(job_id: uuid.UUID, echora_session: str | None = Cookie(default=None)) -> dict[str, object]:
    user = _session_user(echora_session)
    value = jobs.get_job(job_id, user["id"])
    if value is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return value


@app.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: uuid.UUID, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    value = jobs.cancel(job_id, user["id"])
    if value is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return value


@app.post("/jobs/{job_id}/dismiss")
def dismiss_job(
    job_id: uuid.UUID, echora_session: str | None = Cookie(default=None)
) -> dict[str, object]:
    user = _session_user(echora_session)
    try:
        value = jobs.dismiss(job_id, user["id"])
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if value is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return value


@app.get("/jobs/{job_id}/batches")
def job_batches(
    job_id: uuid.UUID,
    limit: int = 25,
    offset: int = 0,
    echora_session: str | None = Cookie(default=None),
) -> dict[str, object]:
    user = _session_user(echora_session)
    result = jobs.list_batches(job_id, user["id"], limit=limit, offset=offset)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result
