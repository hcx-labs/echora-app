"""Shared fixtures: an app client with a fake signed-in user."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from backend.api.deps import current_user
from backend.app import create_app
from core.config import reset_config_cache

USER = {
    "id": uuid.UUID("00000000-0000-0000-0000-000000000001"),
    "username": "listener",
    "email": "listener@example.test",
    "display_name": "Listener",
    "is_admin": False,
    "onboarding_complete": True,
    "navidrome_connection_id": None,
}


@pytest.fixture(autouse=True)
def reset_settings_cache():
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture
def app():
    return create_app(startup=False)


@pytest.fixture
def user():
    return dict(USER)


@pytest.fixture
def client(app, user):
    """A client signed in as ``user`` (no database session needed)."""
    app.dependency_overrides[current_user] = lambda: user
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def anonymous(app):
    """A client without a session."""
    with TestClient(app) as test_client:
        yield test_client
