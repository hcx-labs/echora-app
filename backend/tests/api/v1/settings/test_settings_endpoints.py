"""Routing, administrator checks and validation of the settings endpoints (no database)."""

import pytest

from backend.config import UiConfig

ADMIN_ONLY = [
    ("PUT", "/models/karaoke", {"enabled": True}),
    ("PUT", "/models/transcription", {"enabled": True}),
    ("PUT", "/models/hum", {"enabled": True}),
    ("PUT", "/models/huggingface", {"token": ""}),
    ("GET", "/motion-artwork", None),
    ("PUT", "/motion-artwork/location", {"enabled": True}),
    ("GET", "/motion-artwork/status", None),
    ("DELETE", "/motion-artwork/loops", None),
]
OIDC_ADMIN = [
    ("GET", "/oidc", None),
    ("PUT", "/oidc/policy", {"auto_provision": True}),
    ("POST", "/oidc/allowed-emails", {"email": "a@example.test"}),
    ("DELETE", "/oidc/allowed-emails/a@example.test", None),
    ("PATCH", "/oidc/users/00000000-0000-0000-0000-000000000002", {"is_admin": True}),
]


@pytest.mark.parametrize(("method", "path", "body"), ADMIN_ONLY)
def test_instance_settings_need_an_administrator(client, method, path, body):
    response = client.request(method, "/api/v1/settings" + path, json=body)
    assert response.status_code == 403
    assert response.json() == {"detail": "Administrator access required"}


@pytest.mark.parametrize(("method", "path", "body"), OIDC_ADMIN)
def test_oidc_settings_need_an_administrator(client, method, path, body):
    response = client.request(method, "/api/v1/settings" + path, json=body)
    assert response.status_code == 403
    assert response.json() == {"detail": "Administrator access is required"}


def test_runtime_settings_are_public_and_uncached(anonymous):
    response = anonymous.get("/api/v1/settings/runtime")
    assert response.json() == UiConfig.load().public_ui()
    assert response.headers["cache-control"] == "no-store"


def test_model_switch_location_is_validated(client, admin):
    response = client.put(
        "/api/v1/settings/models/karaoke", json={"enabled": True, "location": "mars"}
    )
    assert response.status_code == 422


def test_unknown_fields_are_rejected(client):
    assert (
        client.put("/api/v1/settings/timezone", json={"timezone": "UTC", "x": 1}).status_code == 422
    )


def test_unknown_timezone(client):
    for zone in ("Mars/Olympus", "../etc"):
        response = client.put("/api/v1/settings/timezone", json={"timezone": zone})
        assert response.status_code == 422
        assert response.json() == {"detail": "Unknown timezone"}


def test_allowed_email_must_look_like_one(client, admin):
    response = client.post("/api/v1/settings/oidc/allowed-emails", json={"email": "nobody"})
    assert response.status_code == 422
    assert response.json() == {"detail": "Enter a valid email address"}


def test_administrators_cannot_demote_themselves(client, admin, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import backend.api.v1.settings.oidc_user as oidc_user

    @contextmanager
    def scope():
        yield SimpleNamespace(get=lambda model, key: SimpleNamespace(is_admin=True))

    monkeypatch.setattr(oidc_user, "session_scope", scope)
    response = client.patch(f"/api/v1/settings/oidc/users/{admin['id']}", json={"is_admin": False})
    assert response.status_code == 422
    assert response.json() == {"detail": "You cannot demote or block your own account"}


def test_huggingface_token_errors_never_echo_it(client, admin, monkeypatch):
    import backend.api.v1.settings.models_huggingface as huggingface

    def rejected(token):
        raise huggingface.TokenRejected

    monkeypatch.setattr(huggingface, "save", rejected)
    response = client.put("/api/v1/settings/models/huggingface", json={"token": "hf_private"})
    assert response.status_code == 422
    assert "hf_private" not in response.text
    monkeypatch.setattr(huggingface, "save", lambda token: "someone")
    assert client.put(
        "/api/v1/settings/models/huggingface", json={"token": "hf_private"}
    ).json() == {"has_token": True, "account": "someone"}


def test_deletes_answer_204(client, admin, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import backend.api.v1.settings.lastfm_disconnect as lastfm
    import backend.api.v1.settings.oidc_disallow as disallow

    @contextmanager
    def scope():
        yield SimpleNamespace(get=lambda model, key: None)

    monkeypatch.setattr(lastfm, "session_scope", scope)
    monkeypatch.setattr(disallow, "session_scope", scope)
    assert client.delete("/api/v1/settings/lastfm").status_code == 204
    assert client.delete("/api/v1/settings/oidc/allowed-emails/a@example.test").status_code == 204
