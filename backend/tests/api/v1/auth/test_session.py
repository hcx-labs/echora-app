"""The session endpoints' HTTP contract: cache headers, the activity guard and the cookie."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

import backend.sessions.store as store
from core.config import reset_config_cache

TRUSTED = {"X-Echora-Activity": "1", "Origin": "http://testserver"}


@pytest.fixture
def snapshot():
    expires = datetime.now(UTC) + timedelta(days=7)
    return {
        "renewed": False,
        "expires_at": expires,
        "absolute_expires_at": expires + timedelta(days=23),
        "renew_after_seconds": 3600,
        "activity_check_seconds": 60,
        "activity_window_seconds": 300,
        "request_timeout_seconds": 10,
    }


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setenv("ECHORA_SESSION_TRUSTED_ORIGINS", "http://testserver")
    reset_config_cache()


def test_activity_contract_and_cookie_security(anonymous, trusted, snapshot, monkeypatch):
    anonymous.cookies.set("echora_session", "opaque-test-token")
    status = Mock(return_value=snapshot)
    monkeypatch.setattr(store, "session_status", status)
    response = anonymous.get("/api/v1/auth/session")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers
    assert response.json()["expires_at"] == snapshot["expires_at"].isoformat()
    status.assert_called_once_with("opaque-test-token")
    assert anonymous.post("/api/v1/auth/session/activity").status_code == 403
    assert status.call_count == 1
    response = anonymous.post("/api/v1/auth/session/activity", headers=TRUSTED)
    status.assert_called_with("opaque-test-token", activity=True)
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Path=/" in cookie
    assert response.headers["cache-control"] == "no-store"
    assert "opaque-test-token" not in response.text
    monkeypatch.setenv("COOKIE_SECURE", "true")
    reset_config_cache()
    secure = anonymous.post("/api/v1/auth/session/activity", headers=TRUSTED)
    assert "Secure" in secure.headers["set-cookie"]


@pytest.mark.parametrize(
    "origin", [None, "null", "https://untrusted.example", "http://testserver.attacker.example"]
)
def test_activity_rejects_untrusted_origins(anonymous, trusted, monkeypatch, origin):
    status = Mock()
    monkeypatch.setattr(store, "session_status", status)
    headers = {"X-Echora-Activity": "1"}
    if origin is not None:
        headers["Origin"] = origin
    response = anonymous.post("/api/v1/auth/session/activity", headers=headers)
    assert response.status_code == 403
    assert response.headers["cache-control"] == "no-store"
    status.assert_not_called()


def test_oidc_redirect_origins_are_trusted(monkeypatch):
    from backend.api.v1.auth.session_activity import trusted_origins

    monkeypatch.setenv("ECHORA_OIDC_REDIRECT_URI", "https://music.example.test/api/v1/auth/x")
    monkeypatch.setenv("ECHORA_CORS_ORIGINS", "https://cdn.example.test")
    reset_config_cache()
    assert {"https://music.example.test", "https://cdn.example.test"} <= trusted_origins()


def test_missing_session_is_401_and_uncached(anonymous):
    response = anonymous.get("/api/v1/auth/session")
    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
