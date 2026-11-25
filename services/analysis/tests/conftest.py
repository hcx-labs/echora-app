"""Keep cached process settings from leaking between tests."""

import pytest
from core.config import reset_config_cache
from core.testing import apply_migrations, database  # noqa: F401 - shared fixture


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_compute_choice: use the database-backed compute choice"
    )


@pytest.fixture(autouse=True)
def reset_settings_cache():
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture(autouse=True)
def local_compute_by_default(monkeypatch, request):
    """Jobs without a recorded location ask the database for the instance default.
    Unit tests have no database, so they compute locally unless a test opts out."""
    if "real_compute_choice" in request.keywords:
        yield
        return
    import worker.remote.settings as external_processing

    monkeypatch.setattr(
        external_processing, "compute_for", lambda user_id, requested: requested or "local"
    )
    yield


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
