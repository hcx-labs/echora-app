"""Session policy and PostgreSQL concurrency."""

import runpy
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select, text

from core.config import reset_config_cache
from core.db.models import User, UserSession

MIGRATION = (
    Path(__file__).resolve().parents[3] / "core/src/db/migrations/versions/0050_session_activity.py"
)


def issue(store, accounts, now):
    scope, owner = accounts
    with scope() as session:
        return store.create_session(session, owner, now=now)


def test_new_login_has_idle_and_absolute_deadlines(store, accounts):
    now = datetime.now(UTC)
    token, expires, absolute = issue(store, accounts, now)
    assert expires == now + timedelta(days=7)
    assert absolute == now + timedelta(days=30)
    assert len(token) >= 48
    scope, _ = accounts
    with scope() as session:
        stored = session.scalar(select(UserSession))
        assert stored.token_hash == store.token_hash(token)
        assert stored.token_hash != token
        assert stored.last_renewed_at == now


def test_passive_reads_do_not_move_idle_deadline(store, accounts):
    now = datetime.now(UTC)
    token, expires, absolute = issue(store, accounts, now)
    for elapsed in (60, 3600, 86400, 6 * 86400):
        result = store.session_status(token, now=now + timedelta(seconds=elapsed))
        assert result["renewed"] is False
        assert result["expires_at"] == expires
        assert result["absolute_expires_at"] == absolute


def test_activity_renews_only_when_due(store, accounts):
    now = datetime.now(UTC)
    token, expires, absolute = issue(store, accounts, now)
    early = store.session_status(token, activity=True, now=now + timedelta(seconds=3599))
    assert early["renewed"] is False and early["expires_at"] == expires
    due = store.session_status(token, activity=True, now=now + timedelta(hours=1))
    assert due["renewed"] is True
    assert due["expires_at"] == expires + timedelta(hours=1)
    assert due["absolute_expires_at"] == absolute
    again = store.session_status(token, activity=True, now=now + timedelta(hours=1, seconds=1))
    assert again["renewed"] is False
    assert again["renew_after_seconds"] == 3599


def test_rolling_activity_cannot_extend_absolute_lifetime(store, accounts):
    now = datetime.now(UTC)
    token, _, absolute = issue(store, accounts, now)
    for day in (6, 12, 18, 24, 29):
        result = store.session_status(token, activity=True, now=now + timedelta(days=day))
    assert result["expires_at"] == absolute
    with pytest.raises(HTTPException) as exc:
        store.session_status(token, activity=True, now=absolute)
    assert exc.value.status_code == 401


def test_expired_or_deleted_sessions_are_not_revived(store, accounts):
    now = datetime.now(UTC)
    token, expires, _ = issue(store, accounts, now)
    with pytest.raises(HTTPException) as exc:
        store.session_status(token, activity=True, now=expires)
    assert exc.value.status_code == 401
    store.delete_session(token)
    with pytest.raises(HTTPException):
        store.session_status(token, activity=True, now=now)
    with pytest.raises(HTTPException):
        store.session_status(None, activity=True, now=now)


def test_blocking_user_invalidates_reads_and_activity(store, accounts):
    now = datetime.now(UTC)
    token, _, _ = issue(store, accounts, now)
    scope, owner = accounts
    with scope() as session:
        session.get(User, owner).is_blocked = True
    for activity in (False, True):
        with pytest.raises(HTTPException) as exc:
            store.session_status(token, activity=activity, now=now + timedelta(hours=1))
        assert exc.value.status_code == 401
    with pytest.raises(HTTPException):
        store.session_user(token)


def test_session_user_carries_the_preferences(store, accounts):
    token, _, _ = issue(store, accounts, datetime.now(UTC))
    user = store.session_user(token)
    assert user["id"] == accounts[1]
    assert user["onboarding_complete"] is False
    assert user["navidrome_connection_id"] is None


def test_concurrent_tabs_renew_once(store, accounts):
    now = datetime.now(UTC)
    token, expires, _ = issue(store, accounts, now)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: store.session_status(token, activity=True, now=now + timedelta(hours=1)),
                range(4),
            )
        )
    assert sum(result["renewed"] for result in results) == 1
    assert all(result["expires_at"] == expires + timedelta(hours=1) for result in results)


def test_session_expiring_while_waiting_for_lock_is_not_revived(store, accounts, monkeypatch):
    now = datetime.now(UTC)
    token, _, _ = issue(store, accounts, now - timedelta(days=7) + timedelta(seconds=1))
    times = iter([now, now + timedelta(seconds=2)])

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(times)

    monkeypatch.setattr(store, "datetime", Clock)
    with pytest.raises(HTTPException) as error:
        store.session_status(token, activity=True)
    assert error.value.status_code == 401


def test_migration_preserves_legacy_deadlines():
    module = runpy.run_path(str(MIGRATION))
    commands = []
    with patch("alembic.op.execute", side_effect=commands.append):
        module["upgrade"]()
    assert any(
        "absolute_expires_at=expires_at, last_renewed_at=created_at" in command
        for command in commands
    )
    assert not any("SET expires_at=" in command for command in commands)


def test_legacy_migration_on_postgresql(store, accounts):
    now = datetime.now(UTC)
    scope, _ = accounts
    old_times = [now - timedelta(hours=3), now - timedelta(hours=1)]
    tokens = [issue(store, accounts, created)[0] for created in old_times]
    with scope() as session:
        for token, created in zip(tokens, old_times):
            stored = session.get(UserSession, store.token_hash(token))
            stored.expires_at = created + timedelta(hours=2)
    with scope() as session:
        session.execute(
            text(
                "ALTER TABLE user_sessions DROP COLUMN absolute_expires_at, "
                "DROP COLUMN last_renewed_at"
            )
        )
        migration = runpy.run_path(str(MIGRATION))
        with patch(
            "alembic.op.execute", side_effect=lambda command: session.execute(text(command))
        ):
            migration["upgrade"]()
    with scope() as session:
        for token, created in zip(tokens, old_times):
            stored = session.get(UserSession, store.token_hash(token))
            assert stored.created_at == stored.last_renewed_at == created
            assert stored.expires_at == stored.absolute_expires_at == created + timedelta(hours=2)
    with pytest.raises(HTTPException):
        store.session_status(tokens[0], activity=True, now=now)
    result = store.session_status(tokens[1], activity=True, now=now)
    assert result["expires_at"] == old_times[1] + timedelta(hours=2)


def test_environment_overrides_control_new_sessions_and_renewals(store, accounts, monkeypatch):
    for name, value in [
        ("IDLE_TIMEOUT", 7200),
        ("ABSOLUTE_TIMEOUT", 14400),
        ("RENEW_INTERVAL", 600),
    ]:
        monkeypatch.setenv("ECHORA_SESSION_" + name + "_SECONDS", str(value))
    reset_config_cache()
    now = datetime.now(UTC)
    token, expires, absolute = issue(store, accounts, now)
    assert expires == now + timedelta(hours=2)
    assert absolute == now + timedelta(hours=4)
    before = store.session_status(token, activity=True, now=now + timedelta(seconds=599))
    assert before["renewed"] is False and before["renew_after_seconds"] == 1
    renewed = store.session_status(token, activity=True, now=now + timedelta(seconds=600))
    assert renewed["renewed"] is True
    assert renewed["expires_at"] == expires + timedelta(seconds=600)
