"""An isolated schema with the account and session tables, patched into the session modules."""

from __future__ import annotations

import os
from contextlib import contextmanager
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.config import reset_config_cache

OIDC_SETTINGS = (
    "ISSUER_URL",
    "CLIENT_ID",
    "CLIENT_SECRET",
    "REDIRECT_URI",
    "POST_LOGIN_REDIRECT",
    "BOOTSTRAP_ADMIN_EMAIL",
    "REQUIRE_VERIFIED_EMAIL",
)


@pytest.fixture(autouse=True)
def no_oidc(monkeypatch):
    """OIDC unconfigured, whatever the environment running the tests sets."""
    for name in OIDC_SETTINGS:
        monkeypatch.delenv(f"ECHORA_OIDC_{name}", raising=False)
        monkeypatch.delenv(f"OIDC_{name}", raising=False)
    reset_config_cache()


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("ECHORA_SESSION_TRUSTED_ORIGINS", "http://testserver")
    reset_config_cache()
    import backend.sessions.store as store

    return store


@pytest.fixture
def accounts(store, monkeypatch):
    """``(scope, owner_id)``: a session factory on a fresh schema and one signed-up user."""
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for isolated PostgreSQL session tests")
    import backend.sessions.oidc as oidc
    from core.db.models import (
        NavidromeConnection,
        OidcAllowedEmail,
        OidcSetting,
        User,
        UserPreference,
        UserSession,
    )

    schema = "test_sessions_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    engine = create_engine(
        url.replace("postgresql://", "postgresql+psycopg://", 1),
        connect_args={"options": f"-c search_path={schema}"},
    )
    tables = [
        User.__table__,
        UserPreference.__table__,
        UserSession.__table__,
        NavidromeConnection.__table__,
        OidcSetting.__table__,
        OidcAllowedEmail.__table__,
    ]
    User.metadata.create_all(engine, tables=tables)
    factory = sessionmaker(engine, expire_on_commit=False)

    @contextmanager
    def scope():
        with factory.begin() as session:
            yield session

    monkeypatch.setattr(store, "session_scope", scope)
    monkeypatch.setattr(oidc, "session_scope", scope)
    owner = uuid4()
    with scope() as session:
        user = User(
            id=owner, username="session-test", display_name="Session test", is_blocked=False
        )
        user.preference = UserPreference()
        session.add(user)
    try:
        yield scope, owner
    finally:
        engine.dispose()
        with psycopg.connect(url, autocommit=True) as db:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
