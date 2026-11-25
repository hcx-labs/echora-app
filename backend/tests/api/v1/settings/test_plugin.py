from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

import backend.settings.plugin_integration as integration
from backend.api.v1.settings.plugin_keys import KeyCreate
from backend.api.v1.settings.plugin_update import PluginUpdate


class FakeDb:
    def __init__(self, rows=None):
        self.rows, self.calls = rows or [], []

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return self

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_issued_keys_store_only_their_digest():
    db = FakeDb()
    key = integration.issue_key(db, "user", "conn", "  Living room ", 30)
    assert key["secret"].startswith("echora_nd_") and key["label"] == "Living room"
    ((_, params),) = db.calls
    assert key["secret"] not in params
    assert params[5] == integration.digest(key["secret"])
    assert key["expires_at"] - datetime.now(UTC) <= timedelta(days=30)


def test_authenticate_rejects_missing_and_unknown_keys(monkeypatch):
    with pytest.raises(HTTPException) as missing:
        integration.authenticate(None)
    assert missing.value.status_code == 401
    monkeypatch.setattr(integration, "connect", lambda **_: FakeDb())
    with pytest.raises(HTTPException) as unknown:
        integration.authenticate("Bearer echora_nd_nope")
    assert unknown.value.headers == {"WWW-Authenticate": "Bearer"}


def test_authenticate_returns_the_profile_and_records_use(monkeypatch):
    db = FakeDb([{"key_id": "k", "user_id": "u", "connection_id": "c", "url": "x", "profile": {}}])
    monkeypatch.setattr(integration, "connect", lambda **_: db)
    row = integration.authenticate("Bearer echora_nd_secret")
    assert row["profile"] == integration.RankingProfile()
    assert db.calls[0][1] == (integration.digest("echora_nd_secret"),)
    assert "last_used_at=now()" in db.calls[1][0]


def test_bodies_match_what_the_web_app_sends():
    profile = integration.RankingProfile().model_dump()
    assert PluginUpdate(**profile, enabled=True, connection_id=None).enabled
    with pytest.raises(ValueError):
        PluginUpdate(**profile, keys=[])
    assert KeyCreate(label="TV", expiry_days=30).expiry_days == 30
    with pytest.raises(ValueError):
        KeyCreate(label="   ")


def test_enabling_without_a_connection_is_refused(client, monkeypatch):
    import backend.api.v1.settings.plugin_update as plugin_update

    monkeypatch.setattr(plugin_update, "connect", lambda **_: FakeDb())
    response = client.put("/api/v1/settings/integrations/navidrome", json={"enabled": True})
    assert response.status_code == 400
