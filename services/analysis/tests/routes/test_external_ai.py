import core.lyrics.translations as translation_storage
import pytest
from core.integrations.external_ai import ExternalAISettings
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from echora_analysis.routes.external_ai import EndpointUpdate, SettingsUpdate, router


def config(**values):
    return ExternalAISettings(
        enabled=True,
        url="http://localhost:8000/v1",
        model="local",
        language_pairs=[{"source": "JA", "target": "en_US"}],
        **values,
    )


@pytest.fixture
def api(monkeypatch):
    state = {"settings": ExternalAISettings(), "encrypted": None, "user": None}
    cipher = Fernet(Fernet.generate_key())

    def require_user():
        if state["user"] is None:
            raise HTTPException(401)
        return state["user"]

    def save(settings, encrypted, *, replace_key):
        state["settings"] = settings
        if replace_key:
            state["encrypted"] = encrypted
        return bool(state["encrypted"])

    monkeypatch.setattr(
        translation_storage, "load_settings", lambda: (state["settings"], state["encrypted"])
    )
    monkeypatch.setattr(translation_storage, "save_settings", save)
    app = FastAPI()
    app.include_router(router(require_user, lambda: cipher))
    return TestClient(app), state, cipher


def test_auth_and_encryption_redaction(api):
    client, state, cipher = api
    path = "/settings/external-ai"
    for user, status in [(None, 401), ({"is_admin": False}, 403)]:
        state["user"] = user
        assert client.get(path).status_code == status
        assert client.put(path, json={}).status_code == status
    state["user"] = {"is_admin": True}
    assert client.get(path).json()["enabled"] is False
    result = client.put(path, json={**config().model_dump(), "api_key": "private-marker"})
    assert result.status_code == 200
    assert result.json()["has_key"]
    assert "private-marker" not in result.text + client.get(path).text
    assert cipher.decrypt(state["encrypted"]) == b"private-marker"
    assert state["encrypted"] != b"private-marker"
    client.put(path, json=config().model_dump())
    assert state["encrypted"]
    assert client.put(path, json={"url": "file:///tmp/x"}).status_code == 422
    client.put(path, json={"api_key": ""})
    assert not client.get(path).json()["has_key"]


def test_invalid_body_does_not_echo_key(api):
    client, state, _ = api
    state["user"] = {"is_admin": True}
    for body in [
        {"enabled": True, "api_key": "private-marker"},
        {"api_key": "private-marker", "unknown": "private-marker"},
        {"api_key": "private-marker" * 1000},
    ]:
        result = client.put("/settings/external-ai", json=body)
        assert result.status_code == 422
        assert "private-marker" not in result.text


@pytest.mark.parametrize(
    "section,keys",
    [
        ("endpoint", {"enabled", "provider", "url", "model", "has_key"}),
        ("translation", {"prompt", "language_pairs"}),
        ("themes", {"enabled", "lyric_themes"}),
        (
            "transcription",
            {
                "enabled",
                "transcription_refinement",
                "transcription_prompt",
                "transcription_model",
                "model",
                "default_transcription_prompt",
            },
        ),
    ],
)
def test_section_contracts(api, monkeypatch, section, keys):
    client, state, _ = api
    path = f"/settings/external-ai/{section}"
    for user, status in [(None, 401), ({"is_admin": False}, 403)]:
        state["user"] = user
        assert client.get(path).status_code == status
        assert client.put(path, json={}).status_code == status
    state["user"] = {"is_admin": True}
    assert set(client.get(path).json()) == keys
    combined = {**ExternalAISettings().model_dump(), "has_key": True}
    monkeypatch.setattr(translation_storage, "save_section", lambda *a, **kw: combined)
    assert set(client.put(path, json={}).json()) == keys
    wrong = {"prompt": "wrong-section"} if section == "endpoint" else {"url": "http://localhost/v1"}
    assert client.put(path, json=wrong).status_code == 422


def test_clear_translations_requires_admin_and_idle_jobs(api, monkeypatch):
    client, state, _ = api
    calls = []
    monkeypatch.setattr(translation_storage, "clear_translations", lambda: calls.append(True) or 7)
    path = "/settings/external-ai/translations"
    for user, status in [(None, 401), ({"is_admin": False}, 403)]:
        state["user"] = user
        assert client.delete(path).status_code == status
    assert calls == []
    state["user"] = {"is_admin": True}
    assert client.delete(path).json() == {"deleted": 7}

    def busy():
        raise RuntimeError("translation_jobs_active")

    monkeypatch.setattr(translation_storage, "clear_translations", busy)
    assert client.delete(path).status_code == 409


def test_key_updates_never_dump_the_key():
    assert SettingsUpdate(api_key="secret").model_dump().get("api_key") is None


def test_endpoint_update_needs_a_url_only_for_openai():
    with pytest.raises(ValidationError):
        EndpointUpdate(enabled=True, provider="openai")
    assert EndpointUpdate(enabled=True, provider="anthropic").provider == "anthropic"
