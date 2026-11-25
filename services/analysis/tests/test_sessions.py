"""Independent session policy, HTTP contract, and PostgreSQL concurrency checks."""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from uuid import uuid4

import psycopg
import pytest
from core.config import reset_config_cache
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from psycopg import sql
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def auth(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", os.getenv("TEST_DATABASE_URL", "postgresql://localhost/unused")
    )
    monkeypatch.setenv("ECHORA_SESSION_TRUSTED_ORIGINS", "http://testserver")
    from core.config import reset_config_cache

    reset_config_cache()
    import echora_analysis.sessions as sessions

    yield sessions
    reset_config_cache()


@pytest.fixture
def database(auth, monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for isolated PostgreSQL session tests")
    from core.db.models import NavidromeConnection, User, UserPreference, UserSession

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
    ]
    User.metadata.create_all(engine, tables=tables)
    factory = sessionmaker(engine, expire_on_commit=False)

    @contextmanager
    def scope():
        with factory.begin() as session:
            yield session

    monkeypatch.setattr(auth, "session_scope", scope)
    owner = uuid4()
    with scope() as session:
        user = User(
            id=owner, username="session-test", display_name="Session test", is_blocked=False
        )
        user.preference = UserPreference()
        session.add(user)
    try:
        yield scope, owner, User, UserSession
    finally:
        engine.dispose()
        with psycopg.connect(url, autocommit=True) as db:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def issue(auth, database, now):
    scope, owner, _, _ = database
    with scope() as session:
        return auth.create_session(session, owner, now=now)


def test_new_login_has_idle_and_absolute_deadlines(auth, database):
    now = datetime.now(timezone.utc)
    token, expires, absolute = issue(auth, database, now)
    assert expires == now + timedelta(days=7)
    assert absolute == now + timedelta(days=30)
    assert len(token) >= 48
    scope, _, _, model = database
    with scope() as session:
        stored = session.scalar(select(model))
        assert stored.token_hash == auth.token_hash(token)
        assert stored.token_hash != token
        assert stored.last_renewed_at == now


def test_passive_reads_do_not_move_idle_deadline(auth, database):
    now = datetime.now(timezone.utc)
    token, expires, absolute = issue(auth, database, now)
    for elapsed in (60, 3600, 86400, 6 * 86400):
        result = auth.session_status(token, now=now + timedelta(seconds=elapsed))
        assert result["renewed"] is False
        assert result["expires_at"] == expires
        assert result["absolute_expires_at"] == absolute


def test_activity_renews_only_when_due(auth, database):
    now = datetime.now(timezone.utc)
    token, expires, absolute = issue(auth, database, now)
    early = auth.session_status(token, activity=True, now=now + timedelta(seconds=3599))
    assert early["renewed"] is False and early["expires_at"] == expires
    due = auth.session_status(token, activity=True, now=now + timedelta(hours=1))
    assert due["renewed"] is True
    assert due["expires_at"] == expires + timedelta(hours=1)
    assert due["absolute_expires_at"] == absolute
    again = auth.session_status(token, activity=True, now=now + timedelta(hours=1, seconds=1))
    assert again["renewed"] is False
    assert again["renew_after_seconds"] == 3599


def test_rolling_activity_cannot_extend_absolute_lifetime(auth, database):
    now = datetime.now(timezone.utc)
    token, _, absolute = issue(auth, database, now)
    for day in (6, 12, 18, 24, 29):
        result = auth.session_status(token, activity=True, now=now + timedelta(days=day))
    assert result["expires_at"] == absolute
    with pytest.raises(HTTPException) as exc:
        auth.session_status(token, activity=True, now=absolute)
    assert exc.value.status_code == 401


def test_expired_or_deleted_sessions_are_not_revived(auth, database):
    now = datetime.now(timezone.utc)
    token, expires, _ = issue(auth, database, now)
    with pytest.raises(HTTPException) as exc:
        auth.session_status(token, activity=True, now=expires)
    assert exc.value.status_code == 401
    scope, _, _, model = database
    with scope() as session:
        session.delete(session.get(model, auth.token_hash(token)))
    with pytest.raises(HTTPException):
        auth.session_status(token, activity=True, now=now)
    with pytest.raises(HTTPException):
        auth.session_status(None, activity=True, now=now)


def test_blocking_user_invalidates_reads_and_activity(auth, database):
    now = datetime.now(timezone.utc)
    token, _, _ = issue(auth, database, now)
    scope, owner, user_model, _ = database
    with scope() as session:
        session.get(user_model, owner).is_blocked = True
    for activity in (False, True):
        with pytest.raises(HTTPException) as exc:
            auth.session_status(token, activity=activity, now=now + timedelta(hours=1))
        assert exc.value.status_code == 401
    with pytest.raises(HTTPException):
        auth.session_user(token)


def test_concurrent_tabs_renew_once(auth, database):
    now = datetime.now(timezone.utc)
    token, expires, _ = issue(auth, database, now)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: auth.session_status(token, activity=True, now=now + timedelta(hours=1)),
                range(4),
            )
        )
    assert sum(result["renewed"] for result in results) == 1
    assert all(result["expires_at"] == expires + timedelta(hours=1) for result in results)


def test_activity_contract_and_cookie_security(auth, monkeypatch):
    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)
    client.cookies.set("echora_session", "opaque-test-token")
    expires = datetime.now(timezone.utc) + timedelta(days=7)
    snapshot = {
        "renewed": False,
        "expires_at": expires,
        "absolute_expires_at": expires + timedelta(days=23),
        "renew_after_seconds": 3600,
        "activity_check_seconds": 60,
        "activity_window_seconds": 300,
        "request_timeout_seconds": 10,
    }
    status = Mock(return_value=snapshot)
    monkeypatch.setattr(auth, "session_status", status)
    response = client.get("/auth/session")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers
    status.assert_called_once_with("opaque-test-token")
    assert client.post("/auth/session/activity").status_code == 403
    assert status.call_count == 1
    response = client.post(
        "/auth/session/activity", headers={"X-Echora-Activity": "1", "Origin": "http://testserver"}
    )
    status.assert_called_with("opaque-test-token", activity=True)
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Path=/" in cookie
    assert response.headers["cache-control"] == "no-store"
    assert "opaque-test-token" not in response.text
    monkeypatch.setenv("COOKIE_SECURE", "true")
    reset_config_cache()
    secure = client.post(
        "/auth/session/activity", headers={"X-Echora-Activity": "1", "Origin": "http://testserver"}
    )
    assert "Secure" in secure.headers["set-cookie"]


def test_migration_preserves_legacy_deadlines():
    import runpy
    from pathlib import Path
    from unittest.mock import patch

    module = runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "core/src/db/migrations/versions/0050_session_activity.py"
        )
    )
    commands = []
    with patch("alembic.op.execute", side_effect=commands.append):
        module["upgrade"]()
    assert any(
        "absolute_expires_at=expires_at, last_renewed_at=created_at" in command
        for command in commands
    )
    assert not any("SET expires_at=" in command for command in commands)


@pytest.mark.parametrize(
    "origin", [None, "null", "https://untrusted.example", "http://testserver.attacker.example"]
)
def test_activity_rejects_untrusted_origins(auth, monkeypatch, origin):
    app = FastAPI()
    app.include_router(auth.router)
    status = Mock()
    monkeypatch.setattr(auth, "session_status", status)
    headers = {"X-Echora-Activity": "1"}
    if origin is not None:
        headers["Origin"] = origin
    response = TestClient(app).post("/auth/session/activity", headers=headers)
    assert response.status_code == 403
    status.assert_not_called()


def test_session_expiring_while_waiting_for_lock_is_not_revived(auth, database, monkeypatch):
    now = datetime.now(timezone.utc)
    token, _, _ = issue(auth, database, now - timedelta(days=7) + timedelta(seconds=1))
    times = iter([now, now + timedelta(seconds=2)])

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(times)

    monkeypatch.setattr(auth, "datetime", Clock)
    with pytest.raises(HTTPException) as error:
        auth.session_status(token, activity=True)
    assert error.value.status_code == 401


def test_legacy_migration_on_postgresql(auth, database):
    import runpy
    from pathlib import Path
    from unittest.mock import patch

    from sqlalchemy import text

    now = datetime.now(timezone.utc)
    scope, _, _, model = database
    old_times = [now - timedelta(hours=3), now - timedelta(hours=1)]
    tokens = [issue(auth, database, created)[0] for created in old_times]
    with scope() as session:
        for token, created in zip(tokens, old_times):
            stored = session.get(model, auth.token_hash(token))
            stored.expires_at = created + timedelta(hours=2)
    with scope() as session:
        session.execute(
            text(
                "ALTER TABLE user_sessions DROP COLUMN absolute_expires_at, DROP COLUMN last_renewed_at"
            )
        )
        migration = runpy.run_path(
            str(
                Path(__file__).resolve().parents[3]
                / "core/src/db/migrations/versions/0050_session_activity.py"
            )
        )
        with patch(
            "alembic.op.execute", side_effect=lambda command: session.execute(text(command))
        ):
            migration["upgrade"]()
    with scope() as session:
        for token, created in zip(tokens, old_times):
            stored = session.get(model, auth.token_hash(token))
            assert stored.created_at == stored.last_renewed_at == created
            assert stored.expires_at == stored.absolute_expires_at == created + timedelta(hours=2)
    with pytest.raises(HTTPException):
        auth.session_status(tokens[0], activity=True, now=now)
    result = auth.session_status(tokens[1], activity=True, now=now)
    assert result["expires_at"] == old_times[1] + timedelta(hours=2)


def test_environment_overrides_control_new_sessions_and_renewals(auth, database, monkeypatch):
    for name, value in [
        ("IDLE_TIMEOUT", 7200),
        ("ABSOLUTE_TIMEOUT", 14400),
        ("RENEW_INTERVAL", 600),
    ]:
        monkeypatch.setenv("ECHORA_SESSION_" + name + "_SECONDS", str(value))
    reset_config_cache()
    now = datetime.now(timezone.utc)
    token, expires, absolute = issue(auth, database, now)
    assert expires == now + timedelta(hours=2)
    assert absolute == now + timedelta(hours=4)
    before = auth.session_status(token, activity=True, now=now + timedelta(seconds=599))
    assert before["renewed"] is False and before["renew_after_seconds"] == 1
    renewed = auth.session_status(token, activity=True, now=now + timedelta(seconds=600))
    assert renewed["renewed"] is True
    assert renewed["expires_at"] == expires + timedelta(seconds=600)
