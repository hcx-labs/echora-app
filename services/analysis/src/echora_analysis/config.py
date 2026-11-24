"""API configuration: origins, sessions, OIDC, the media cache and the browser's polling."""

from __future__ import annotations

import secrets
from typing import Self

from pydantic import SecretStr, model_validator

from core.config import EchoraConfig, secret, setting

# Without a configured secret, sessions survive only as long as this process.
_SESSION_SECRET = secrets.token_urlsafe(48)


class ApiConfig(EchoraConfig, env_prefix="ECHORA_"):
    """Cross-origin access to the API."""

    cors_origins: str = setting("", description="Comma-separated origins that may read responses.")

    @model_validator(mode="after")
    def _explicit_origins(self) -> Self:
        if "*" in self.cors_origins:
            raise ValueError("Credentialed requests require explicit origins, not wildcards")
        return self


class SessionConfig(EchoraConfig, env_prefix="ECHORA_SESSION_"):
    """App sessions. Only explicit user activity moves the idle deadline."""

    trusted_origins: str = setting("")
    idle_timeout_seconds: int = setting(604800, gt=0)
    absolute_timeout_seconds: int = setting(2592000, gt=0)
    renew_interval_seconds: int = setting(3600, gt=0)
    activity_check_seconds: int = setting(60, gt=0)
    activity_window_seconds: int = setting(300, gt=0)
    request_timeout_seconds: int = setting(10, gt=0)
    cookie_secure: bool = setting(False, legacy="COOKIE_SECURE")

    @model_validator(mode="after")
    def _intervals(self) -> Self:
        if "*" in self.trusted_origins:
            raise ValueError("Credentialed requests require explicit origins, not wildcards")
        if not (
            self.activity_check_seconds
            < self.renew_interval_seconds
            < self.idle_timeout_seconds
            <= self.absolute_timeout_seconds
        ):
            raise ValueError("Session intervals require check < renew < idle <= absolute")
        if self.activity_window_seconds < self.activity_check_seconds:
            raise ValueError("Session activity window must be >= activity check")
        return self


class OidcConfig(EchoraConfig, env_prefix="ECHORA_OIDC_"):
    """Sign-in through an OpenID Connect provider."""

    session_secret: SecretStr = secret(_SESSION_SECRET, legacy="OIDC_SESSION_SECRET")
    issuer_url: str = setting("", legacy="OIDC_ISSUER_URL")
    client_id: str = setting("", legacy="OIDC_CLIENT_ID")
    client_secret: SecretStr = secret(legacy="OIDC_CLIENT_SECRET")
    scopes: str = setting("openid profile email", legacy="OIDC_SCOPES")
    redirect_uri: str = setting("", legacy="OIDC_REDIRECT_URI")
    post_login_redirect: str = setting(
        "http://localhost:3000/home", legacy="OIDC_POST_LOGIN_REDIRECT"
    )
    require_verified_email: bool = setting(False, legacy="OIDC_REQUIRE_VERIFIED_EMAIL")
    bootstrap_admin_email: str = setting("", legacy="OIDC_BOOTSTRAP_ADMIN_EMAIL")


class MediaCacheConfig(EchoraConfig, env_prefix="ECHORA_MEDIA_CACHE_"):
    """The Redis cache of cover art and other small media."""

    redis_url: str = setting("", repr=False, exclude=True, legacy="REDIS_URL")
    ttl_seconds: int = setting(3600, gt=0)
    connect_timeout_seconds: float = setting(0.5, gt=0)
    timeout_seconds: int = setting(2, gt=0)


_MAX_MS = 2147483647


class UiConfig(EchoraConfig, env_prefix="ECHORA_"):
    """Browser polling and timeouts, exported to the web app (``public_ui``)."""

    job_poll_ms: int = setting(1200, gt=0, le=_MAX_MS)
    batch_poll_ms: int = setting(1500, gt=0, le=_MAX_MS)
    hum_index_poll_ms: int = setting(2500, gt=0, le=_MAX_MS)
    recording_poll_ms: int = setting(1500, gt=0, le=_MAX_MS)
    recording_search_timeout_ms: int = setting(120000, gt=0, le=_MAX_MS)
    recording_request_timeout_ms: int = setting(10000, gt=0, le=_MAX_MS)
    recording_upload_timeout_ms: int = setting(60000, gt=0, le=_MAX_MS)

    def public_ui(self) -> dict[str, int]:
        """Every field: all of them are safe for the browser."""
        return self.model_dump()
