"""The FastAPI application: middleware, error handlers, startup checks and ``/api/v1``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from backend.api import v1
from backend.api.errors import install_error_handlers
from backend.config import ApiConfig, OidcConfig, SessionConfig
from core.db.connection import connect
from core.embeddings.representations import configure_representations

DEFAULT_REDIRECT_URI = "http://localhost:3000/api/v1/auth/oidc/callback"


def create_app(*, startup: bool = True) -> FastAPI:
    """A new application with every v1 endpoint.

    ``startup=False`` skips the startup checks and database setup (unit tests).
    """
    app = FastAPI(title="Echora", version="1.0.0", lifespan=_lifespan if startup else None)
    install_error_handlers(app)
    # Browser-facing media (covers, streams) can be served cross-origin from the web app;
    # canvas palette extraction needs CORS. Origins are explicit: reflecting arbitrary origins
    # with credentials would let any site read responses with the visitor's session cookie.
    origins = [o.strip() for o in ApiConfig.load().cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # Holds the OIDC state between the start and the callback, not the app session.
    app.add_middleware(
        SessionMiddleware,
        secret_key=OidcConfig.load().session_secret.get_secret_value(),
        session_cookie="echora_oidc_state",
        same_site="lax",
        https_only=SessionConfig.load().cookie_secure,
    )
    app.include_router(v1.router())
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    _startup()
    yield


def _startup() -> None:
    enforce_secure_cookie_policy()
    with connect() as connection:
        configure_representations(connection)


def enforce_secure_cookie_policy() -> None:
    """Refuse to start with insecure cookies unless sign-in returns to localhost."""
    if SessionConfig.load().cookie_secure:
        return
    redirect = OidcConfig.load().redirect_uri or DEFAULT_REDIRECT_URI
    if (urlparse(redirect).hostname or "") not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(
            "COOKIE_SECURE must be true when OIDC_REDIRECT_URI is not localhost; "
            "session cookies would otherwise be sent over plain HTTP"
        )
