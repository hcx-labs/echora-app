"""Fixtures of the search endpoint tests: a recording schema and a search-category app."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.deps import current_user
from backend.api.errors import install_error_handlers
from backend.api.v1.search import CATEGORY
from core.testing import apply_migrations, database  # noqa: F401 - the shared fixture


@pytest.fixture
def search_app():
    """Build a client of the search category alone with a given signed-in-user dependency."""
    return _search_app


def _search_app(user_dependency) -> TestClient:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(CATEGORY.router(), prefix="/api/v1")
    app.dependency_overrides[current_user] = user_dependency
    return TestClient(app)


@pytest.fixture
def recording_db(database):  # noqa: F811 - the shared fixture
    with database() as db:
        db.execute("""CREATE TABLE users(id uuid PRIMARY KEY);
            CREATE TABLE tracks(id uuid PRIMARY KEY,title text,artist text,album text,duration_seconds float);
            CREATE TABLE user_track_links(user_id uuid,track_id uuid,external_id text,library_id uuid);
            CREATE TABLE libraries(id uuid PRIMARY KEY,namespace uuid,root_path text);
            CREATE TABLE navidrome_connections(id uuid PRIMARY KEY,owner_user_id uuid,url text);
            CREATE TABLE user_preferences(user_id uuid PRIMARY KEY,navidrome_connection_id uuid);
            CREATE TABLE track_sources(track_id uuid,library_id uuid,external_id text,source_type text,source_data jsonb);
            CREATE TABLE recording_group_members(track_id uuid PRIMARY KEY,group_id uuid);""")
        apply_migrations(db, "0044_recording_search", "0046_recording_diagnostics")
    return database
